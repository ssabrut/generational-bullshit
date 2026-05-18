"""Smoke-test TripoSR on a handful of Pix3D chair images.

Pipeline: load image + mask, composite onto gray background, tight crop,
square pad, resize, run TripoSR, save mesh + metadata for review.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_DIR = Path(__file__).parent / "TripoSR"
sys.path.insert(0, str(REPO_DIR))

from tsr.system import TSR
from tsr.utils import resize_foreground


def pick_chair_entries(pix3d_json_path: Path, category: str, n: int, seed: int) -> list[dict]:
    """Pick N entries with distinct 3D models (avoids 10 photos of the same chair)."""
    with open(pix3d_json_path) as f:
        meta = json.load(f)
    cands = [m for m in meta if m["category"] == category]
    rng = np.random.default_rng(seed)
    by_model: dict[str, list[dict]] = {}
    for m in cands:
        by_model.setdefault(m["model"], []).append(m)
    models = list(by_model.keys())
    rng.shuffle(models)
    return [by_model[mod][0] for mod in models[:n]]


def prepare_image(
    img_path: Path,
    mask_path: Path,
    target_size: int = 512,
    fg_ratio: float = 0.85,
    bg_value: float = 0.5,
) -> Image.Image | None:
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    alpha = (mask > 0.5).astype(np.float32)

    ys, xs = np.where(alpha > 0)
    if len(ys) == 0:
        return None

    composite = img * alpha[..., None] + bg_value * (1.0 - alpha[..., None])

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    crop = composite[y1:y2, x1:x2]
    crop_a = alpha[y1:y2, x1:x2]

    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), bg_value, dtype=np.float32)
    sq_a = np.zeros((s, s), dtype=np.float32)
    oy, ox = (s - h) // 2, (s - w) // 2
    sq[oy : oy + h, ox : ox + w] = crop
    sq_a[oy : oy + h, ox : ox + w] = crop_a

    rgba = np.concatenate([sq, sq_a[..., None]], axis=-1)
    rgba_pil = Image.fromarray((rgba * 255).astype(np.uint8))
    rgba_pil = resize_foreground(rgba_pil, fg_ratio)
    arr = np.array(rgba_pil).astype(np.float32) / 255.0
    final = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * bg_value
    out = Image.fromarray((final * 255).astype(np.uint8))
    return out.resize((target_size, target_size), Image.BICUBIC)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pix3d-root", type=Path, default=Path("data/pix3d"))
    p.add_argument("--output", type=Path, default=Path("data/teacher_meshes/chair_smoke"))
    p.add_argument("--category", default="chair")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mc-resolution", type=int, default=192)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("[warn] MPS not available, falling back to CPU")
        device = "cpu"

    args.output.mkdir(parents=True, exist_ok=True)
    entries = pick_chair_entries(
        args.pix3d_root / "pix3d.json", args.category, args.n, args.seed
    )
    print(f"Selected {len(entries)} {args.category} entries (one per unique model)")

    print("Loading TripoSR (will download ~700MB from HF on first run)...")
    t0 = time.time()
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    print(f"Model loaded in {time.time() - t0:.1f}s on {device}")

    for i, e in enumerate(entries):
        t_start = time.time()
        stem = Path(e["img"]).stem
        sub = args.output / f"{i:03d}_{stem}"
        sub.mkdir(parents=True, exist_ok=True)

        img = prepare_image(args.pix3d_root / e["img"], args.pix3d_root / e["mask"])
        if img is None:
            print(f"[{i:02d}] {stem}: empty mask, skipping")
            continue
        img.save(sub / "input.png")

        with torch.no_grad():
            codes = model([img], device=device)
            meshes = model.extract_mesh(
                codes, has_vertex_color=True, resolution=args.mc_resolution
            )
        mesh = meshes[0]
        mesh.export(sub / "mesh.obj")

        with open(sub / "meta.json", "w") as f:
            json.dump(
                {
                    "img": e["img"],
                    "model": e["model"],
                    "rot_mat": e.get("rot_mat"),
                    "trans_mat": e.get("trans_mat"),
                    "focal_length": e.get("focal_length"),
                    "n_verts": int(len(mesh.vertices)),
                    "n_faces": int(len(mesh.faces)),
                    "elapsed_s": round(time.time() - t_start, 2),
                },
                f,
                indent=2,
            )
        print(
            f"[{i:02d}] {stem}: {len(mesh.vertices)} verts, "
            f"{len(mesh.faces)} faces, {time.time() - t_start:.1f}s"
        )


if __name__ == "__main__":
    main()
