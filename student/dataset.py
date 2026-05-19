"""Dataset wrapping the TripoSR→Pix3D teacher cache built by teacher/build_cache.py.

Each item:
  - image: (3, H, W) float in [0,1], DinoV2-normalized
  - points: (N, 3) float — target surface point cloud in canonical (Y-up) frame
  - category: str

Augmentation (train only):
  - Horizontal flip: mirror image left↔right AND mirror target points across X.
    Chairs/sofas are left-right symmetric on average, so flipping is label-safe.
  - Color jitter: small brightness/contrast/saturation perturbation on the
    pre-normalize image. Helps the encoder generalize across lighting.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
import trimesh
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet stats — DinoV2 uses these
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


class TeacherCacheDataset(Dataset):
    def __init__(
        self,
        cache_root: Path | str = "data/teacher_cache",
        categories: tuple[str, ...] = ("chair",),
        image_size: int = 224,
        normalize_points: bool = True,
        augment: bool = False,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.2,
        n_color_points: int = 4096,
    ):
        self.cache_root = Path(cache_root)
        self.normalize_points = normalize_points
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.n_color_points = n_color_points

        # Base (no-aug) pipeline used for val and as the final stage during aug.
        self.tf = build_image_transform(image_size)

        # Color jitter applied to the PIL image before ToTensor + normalize.
        self._jitter = (
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
            if augment and color_jitter > 0
            else None
        )

        self.entries: list[Path] = []
        for cat in categories:
            cat_dir = self.cache_root / cat
            if not cat_dir.exists():
                continue
            for d in sorted(cat_dir.iterdir()):
                if (
                    d.is_dir()
                    and (d / "meta.json").exists()
                    and (d / "points.npy").exists()
                ):
                    self.entries.append(d)
        if not self.entries:
            raise RuntimeError(f"No cache entries found under {self.cache_root}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        d = self.entries[i]
        img = Image.open(d / "input.png").convert("RGB")
        pts = np.load(d / "points.npy").astype(np.float32)  # (N, 3)

        # ── colored surface points from canonical textured mesh ──────────────
        # Sample n_color_points on the canonical mesh and look up their RGB
        # via the bound TextureVisuals. Returns zeros + has_color=False when
        # the entry has no canonical_mesh.obj yet (older cache items).
        c_pts, c_rgb, has_color = self._load_color_target(d, pts)

        # ── augmentation (train only) ────────────────────────────────────────
        do_flip = self.augment and random.random() < self.hflip_prob
        if do_flip:
            img = TF.hflip(img)
            pts = pts.copy()
            pts[:, 0] = -pts[:, 0]  # mirror across X to match the flipped image
            if has_color:
                c_pts = c_pts.copy()
                c_pts[:, 0] = -c_pts[:, 0]

        if self._jitter is not None:
            img = self._jitter(img)

        # ── image to tensor (resize + normalize) ─────────────────────────────
        img_t = self.tf(img)

        # ── points to unit sphere ────────────────────────────────────────────
        # Use the SAME normalization (center+scale from pts) for c_pts so they
        # share the canonical frame the model deforms into.
        if self.normalize_points:
            c = pts.mean(0, keepdims=True)
            pts = pts - c
            s = float(np.linalg.norm(pts, axis=1).max())
            if s > 0:
                pts = pts / s
                if has_color:
                    c_pts = (c_pts - c) / s

        return {
            "image": img_t,
            "points": torch.from_numpy(pts),
            "color_points": torch.from_numpy(c_pts.astype(np.float32)),
            "color_rgb": torch.from_numpy(c_rgb.astype(np.float32)),
            "has_color": torch.tensor(has_color, dtype=torch.bool),
            "category": d.parent.name,
            "entry_id": d.name,
        }

    def _load_color_target(
        self, d: Path, pts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Sample N points on canonical_mesh with their per-point RGB in [0,1].

        Falls back to zeros (and has_color=False) when the mesh or texture is
        missing — keeps backwards compat with un-baked cache entries.
        """
        mesh_path = d / "canonical_mesh.obj"
        if not mesh_path.exists():
            n = self.n_color_points
            return np.zeros((n, 3), np.float32), np.zeros((n, 3), np.float32), False
        try:
            mesh = trimesh.load(mesh_path, process=False, force="mesh")
            n = self.n_color_points
            pts_s, face_idx = trimesh.sample.sample_surface(
                mesh, n, seed=abs(hash(d.name)) % (2**31)
            )
            visual = getattr(mesh, "visual", None)
            colors = None
            if visual is not None and hasattr(visual, "to_color"):
                # to_color() yields a ColorVisuals with vertex_colors; sampling
                # on faces → interpolate vertex colors at the sampled barycentrics.
                # Simpler: use trimesh's built-in `face_colors` lookup at face_idx.
                cv = visual.to_color()
                if getattr(cv, "vertex_colors", None) is not None:
                    vc = np.asarray(cv.vertex_colors)[:, :3].astype(np.float32) / 255.0
                    faces = mesh.faces[face_idx]
                    # average the 3 vertex colors per sampled face (cheap approx)
                    colors = vc[faces].mean(axis=1)
            if colors is None:
                colors = np.full((n, 3), 0.5, dtype=np.float32)
            return pts_s.astype(np.float32), colors.astype(np.float32), True
        except Exception:
            n = self.n_color_points
            return np.zeros((n, 3), np.float32), np.zeros((n, 3), np.float32), False
