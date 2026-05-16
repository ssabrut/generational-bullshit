import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]

OMNI_CATEGORIES = ["chair", "sofa"]


def read_ply_xyz(path):
    pts = []
    in_data = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "end_header":
                in_data = True
                continue
            if in_data and line:
                parts = line.split()
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return torch.tensor(pts, dtype=torch.float32)


def normalise_points(pts):
    centre = (pts.max(0).values + pts.min(0).values) / 2.0
    pts = pts - centre
    scale = pts.norm(dim=-1).max().clamp(min=1e-8)
    return pts / scale


def c2w_to_Rt(c2w):
    R_c2w = c2w[:3, :3]
    t_cam = c2w[:3, 3]
    R = R_c2w.T
    T = -R @ t_cam
    return R, T


def focal_from_angle(camera_angle_x):
    return 1.0 / float(np.tan(camera_angle_x / 2.0))


class OmniObject3DDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        categories: list = None,
        img_size: int = 224,
        n_pts: int = 4096,
        random_view: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.n_pts = n_pts
        self.random_view = random_view
        self.img_tfm = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(DINO_MEAN, DINO_STD),
            ]
        )

        cats = categories or OMNI_CATEGORIES
        self.samples = self._collect_instances(cats, split)
        print(
            f"OmniObject3DDataset [{split}]: {len(self.samples)} instances "
            f"from {cats}"
        )

    def _collect_instances(self, cats, split):
        all_instances = []
        for cat in cats:
            cat_raw_dir = self.root / "raw" / cat
            if not cat_raw_dir.exists():
                continue

            for obj_dir in sorted(cat_raw_dir.iterdir()):
                obj_id = obj_dir.name
                tf_path = obj_dir / "transforms.json"
                pcd_path = self.root / "points" / cat / obj_id / "pcd_4096.ply"
                if tf_path.exists() and pcd_path.exists():
                    all_instances.append((cat, obj_id))

        random.seed(42)
        random.shuffle(all_instances)
        n = len(all_instances)
        cut_val = int(0.8 * n)
        cut_test = int(0.9 * n)
        splits = {
            "train": all_instances[:cut_val],
            "val": all_instances[cut_val:cut_test],
            "test": all_instances[cut_test:],
        }

        return splits[split]

    def _load_transforms(self, cat: str, obj_id: str):
        tf_path = self.root / "raw" / cat / obj_id / "transforms.json"
        with open(tf_path) as fh:
            meta = json.load(fh)
        focal = focal_from_angle(meta["camera_angle_x"])
        frames = meta["frames"]  # list of {file_path, transform_matrix}
        return frames, focal

    def _load_image(self, cat: str, obj_id: str, filename: str):
        img_path = self.root / "raw" / cat / obj_id / filename
        img = Image.open(img_path).convert("RGB")
        return self.img_tfm(img)

    def _load_points(self, cat: str, obj_id: str):
        pcd_path = self.root / "points" / cat / obj_id / "pcd_4096.ply"
        pts = read_ply_xyz(str(pcd_path))  # [4096, 3]
        pts = normalise_points(pts)  # → unit sphere
        if self.n_pts < pts.shape[0]:
            idx = torch.randperm(pts.shape[0])[: self.n_pts]
            pts = pts[idx]
        return pts  # [n_pts, 3]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cat, obj_id = self.samples[idx]

        # 1. Load camera metadata
        frames, focal = self._load_transforms(cat, obj_id)

        # 2. Pick one view
        view_idx = random.randrange(len(frames)) if self.random_view else 0
        frame = frames[view_idx]

        # 3. Load image
        image = self._load_image(cat, obj_id, frame["file_path"])

        # 4. Parse c2w → R, T (world-to-camera)
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)  # [4,4]
        R, T = c2w_to_Rt(c2w)

        # 5. Load point cloud
        gt_pts = self._load_points(cat, obj_id)

        return {
            "image": image,  # [3, H, W]
            "gt_points": gt_pts,  # [n_pts, 3]
            "rot": R,  # [3, 3]
            "trans": T,  # [3]
            "focal": torch.tensor(focal, dtype=torch.float32),  # scalar
            "category": cat,
            "instance": obj_id,
            "view_idx": view_idx,
        }
