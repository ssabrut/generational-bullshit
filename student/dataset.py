"""Dataset wrapping the TripoSR→Pix3D teacher cache built by teacher/build_cache.py.

Each item:
  - image: (3, H, W) float in [0,1], DinoV2-normalized
  - points: (N, 3) float — target surface point cloud in canonical (Y-up) frame
  - category: str
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet stats — DinoV2 uses these
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_image_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class TeacherCacheDataset(Dataset):
    def __init__(
        self,
        cache_root: Path | str = "data/teacher_cache",
        categories: tuple[str, ...] = ("chair",),
        image_size: int = 224,
        normalize_points: bool = True,
    ):
        self.cache_root = Path(cache_root)
        self.normalize_points = normalize_points
        self.tf = build_image_transform(image_size)

        self.entries: list[Path] = []
        for cat in categories:
            cat_dir = self.cache_root / cat
            if not cat_dir.exists():
                continue
            for d in sorted(cat_dir.iterdir()):
                if d.is_dir() and (d / "meta.json").exists() and (d / "points.npy").exists():
                    self.entries.append(d)
        if not self.entries:
            raise RuntimeError(f"No cache entries found under {self.cache_root}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        d = self.entries[i]
        img = Image.open(d / "input.png").convert("RGB")
        img_t = self.tf(img)

        pts = np.load(d / "points.npy").astype(np.float32)  # (N, 3)
        if self.normalize_points:
            # Center and scale to unit sphere — student predicts in this normalized frame.
            c = pts.mean(0, keepdims=True)
            pts = pts - c
            s = float(np.linalg.norm(pts, axis=1).max())
            if s > 0:
                pts = pts / s

        return {
            "image":    img_t,
            "points":   torch.from_numpy(pts),
            "category": d.parent.name,
            "entry_id": d.name,
        }
