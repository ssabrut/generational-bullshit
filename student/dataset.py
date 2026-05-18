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
    ):
        self.cache_root = Path(cache_root)
        self.normalize_points = normalize_points
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter

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

        # ── augmentation (train only) ────────────────────────────────────────
        do_flip = self.augment and random.random() < self.hflip_prob
        if do_flip:
            img = TF.hflip(img)
            pts = pts.copy()
            pts[:, 0] = -pts[:, 0]  # mirror across X to match the flipped image

        if self._jitter is not None:
            img = self._jitter(img)

        # ── image to tensor (resize + normalize) ─────────────────────────────
        img_t = self.tf(img)

        # ── points to unit sphere ────────────────────────────────────────────
        if self.normalize_points:
            c = pts.mean(0, keepdims=True)
            pts = pts - c
            s = float(np.linalg.norm(pts, axis=1).max())
            if s > 0:
                pts = pts / s

        return {
            "image": img_t,
            "points": torch.from_numpy(pts),
            "category": d.parent.name,
            "entry_id": d.name,
        }
