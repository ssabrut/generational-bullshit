"""Build the teacher pseudo-label cache for Pix3D chair/sofa.

Per entry:
  1. Load image + mask → composite on gray bg → run TripoSR
  2. ICP-align TripoSR mesh to Pix3D GT mesh (canonical frame, Y-up)
  3. Poisson-disk sample 10k surface points from the aligned mesh
  4. Save:  canonical_mesh.obj, points.npy, meta.json, input.png

Skip entries where:
  - mask is empty (no foreground pixels)
  - ICP cost > COST_THRESH (TripoSR reconstruction too bad to trust)

Usage:
  python teacher/build_cache.py --category chair --workers 1
  python teacher/build_cache.py --category sofa  --workers 1

The full chair set (~3800 images) takes roughly 4-5h on CPU (M-series).
Run overnight; progress is checkpointed per entry (skip if output exists).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

REPO = Path(__file__).parent / "TripoSR"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from align import align_to_target, apply_transform
from tsr.system import TSR
from tsr.utils import resize_foreground

COST_THRESH = 0.08  # mean NN dist in Pix3D-model units; >8cm is a bad fit
N_POINTS = 10_000  # surface samples saved per entry


# ── image prep ────────────────────────────────────────────────────────────────


def prepare_image(
    img_path: Path, mask_path: Path, size: int = 512, fg_ratio: float = 0.85
) -> Image.Image | None:
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    alpha = (mask > 0.5).astype(np.float32)
    if alpha.sum() < 100:
        return None
    composite = img * alpha[..., None] + 0.5 * (1.0 - alpha[..., None])
    ys, xs = np.where(alpha > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    crop = composite[y1:y2, x1:x2]
    crop_a = alpha[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), 0.5, dtype=np.float32)
    sq_a = np.zeros((s, s), dtype=np.float32)
    oy, ox = (s - h) // 2, (s - w) // 2
    sq[oy : oy + h, ox : ox + w] = crop
    sq_a[oy : oy + h, ox : ox + w] = crop_a
    rgba = np.concatenate([sq, sq_a[..., None]], axis=-1)
    rgba_pil = Image.fromarray((rgba * 255).astype(np.uint8))
    rgba_pil = resize_foreground(rgba_pil, fg_ratio)
    arr = np.array(rgba_pil).astype(np.float32) / 255.0
    final = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * 0.5
    return Image.fromarray((final * 255).astype(np.uint8)).resize(
        (size, size), Image.BICUBIC
    )


# ── point cloud sampling ──────────────────────────────────────────────────────


def sample_points(
    mesh: trimesh.Trimesh, n: int = N_POINTS, seed: int = 0
) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    if len(pts) < n // 2:
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float32)


# ── entry filter ──────────────────────────────────────────────────────────────


def filter_entries(meta: list[dict], category: str) -> list[dict]:
    """One entry per unique (image, model) pair within the requested category."""
    seen: set[str] = set()
    out: list[dict] = []
    for e in meta:
        if e["category"] != category:
            continue
        key = f"{e['img']}|{e['model']}"
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pix3d-root", type=Path, default=Path("data/pix3d"))
    p.add_argument("--output", type=Path, default=Path("data/teacher_cache"))
    p.add_argument("--category", default="chair", choices=["chair", "sofa"])
    p.add_argument("--mc-resolution", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N entries (for testing)",
    )
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    out_dir = args.output / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pix3d_root / "pix3d.json") as f:
        all_meta = json.load(f)
    entries = filter_entries(all_meta, args.category)
    if args.limit:
        entries = entries[: args.limit]
    print(f"[{args.category}] {len(entries)} entries to process → {out_dir}")

    print("Loading TripoSR …")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
    )
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    print(f"Model ready on {device}")

    stats = {"ok": 0, "skip_mask": 0, "skip_cost": 0, "error": 0}

    for idx, e in enumerate(entries):
        stem = Path(e["img"]).stem
        entry_id = f"{idx:05d}_{stem}"
        sub = out_dir / entry_id

        # Checkpoint: skip if already done
        if (sub / "meta.json").exists():
            stats["ok"] += 1
            continue

        sub.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            # ── 1. prepare image ──────────────────────────────────────────
            img = prepare_image(args.pix3d_root / e["img"], args.pix3d_root / e["mask"])
            if img is None:
                stats["skip_mask"] += 1
                sub.rmdir()
                continue
            img.save(sub / "input.png")

            # ── 2. TripoSR ────────────────────────────────────────────────
            with torch.no_grad():
                codes = model([img], device=device)
                meshes = model.extract_mesh(
                    codes, has_vertex_color=True, resolution=args.mc_resolution
                )
            src_mesh = meshes[0]

            # ── 3. ICP align to Pix3D GT ──────────────────────────────────
            tgt_mesh = trimesh.load(
                args.pix3d_root / e["model"], process=False, force="mesh"
            )
            M, cost = align_to_target(src_mesh, tgt_mesh)

            if cost > COST_THRESH:
                stats["skip_cost"] += 1
                sub.rmdir()
                print(f"[{idx:5d}] {stem}: SKIP cost={cost:.4f} > {COST_THRESH}")
                continue

            # ── 4. apply transform → canonical mesh ───────────────────────
            canon_verts = apply_transform(np.asarray(src_mesh.vertices, dtype=float), M)
            canon_mesh = trimesh.Trimesh(
                vertices=canon_verts, faces=src_mesh.faces, process=False
            )
            canon_mesh.export(sub / "canonical_mesh.obj")

            # ── 5. Poisson-disk point cloud ───────────────────────────────
            pts = sample_points(canon_mesh)
            np.save(sub / "points.npy", pts)

            # ── 6. metadata ───────────────────────────────────────────────
            elapsed = time.time() - t0
            with open(sub / "meta.json", "w") as f:
                json.dump(
                    {
                        "img": e["img"],
                        "mask": e["mask"],
                        "model": e["model"],
                        "category": e["category"],
                        "focal_length": e.get("focal_length"),
                        "icp_cost": round(cost, 6),
                        "n_points": int(len(pts)),
                        "n_verts": int(len(canon_verts)),
                        "n_faces": int(len(src_mesh.faces)),
                        "elapsed_s": round(elapsed, 2),
                    },
                    f,
                    indent=2,
                )

            stats["ok"] += 1
            print(
                f"[{idx:5d}] {stem}: cost={cost:.4f}, {int(len(pts))}pts, {elapsed:.1f}s"
            )

        except Exception as exc:
            stats["error"] += 1
            print(f"[{idx:5d}] {stem}: ERROR {exc}")
            # leave sub dir as-is so we can inspect later

    print(
        f"\nDone. ok={stats['ok']}  skip_mask={stats['skip_mask']}  "
        f"skip_cost={stats['skip_cost']}  errors={stats['error']}"
    )


if __name__ == "__main__":
    main()
