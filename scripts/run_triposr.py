"""Run TripoSR inference on one or more images using the local checkpoint.

Preprocessing mirrors `teacher/build_cache.py::prepare_image` (which produced the
good teacher_cache results):
  1. obtain alpha mask (from rembg, or a provided mask file)
  2. composite RGB on 0.5 gray using alpha
  3. crop to mask bbox, center on square of side max(h, w)
  4. resize_foreground to the target ratio (default 0.85)
  5. resize square to (size, size) with BICUBIC (default 512)

Example:
    python scripts/run_triposr.py path/to/image.png
    python scripts/run_triposr.py path/to/image.png --mask path/to/mask.png
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "teacher" / "TripoSR"))

from tsr.system import TSR
from tsr.utils import resize_foreground


def resolve_device(name: str) -> str:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


def get_alpha_from_rembg(img_pil: Image.Image, rembg_session) -> np.ndarray:
    """Return a HxW float32 alpha mask in [0, 1] from rembg."""
    from rembg import remove
    cutout = remove(img_pil.convert("RGBA"), session=rembg_session)
    arr = np.array(cutout, dtype=np.float32) / 255.0
    return arr[..., 3]


def get_alpha_from_file(mask_path: Path) -> np.ndarray:
    return np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0


def prepare_image(rgb: np.ndarray, alpha: np.ndarray, size: int = 512,
                  fg_ratio: float = 0.85) -> Image.Image | None:
    """Mirror of teacher/build_cache.py::prepare_image."""
    alpha_bin = (alpha > 0.5).astype(np.float32)
    if alpha_bin.sum() < 100:
        return None

    composite = rgb * alpha_bin[..., None] + 0.5 * (1.0 - alpha_bin[..., None])

    ys, xs = np.where(alpha_bin > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    crop = composite[y1:y2, x1:x2]
    crop_a = alpha_bin[y1:y2, x1:x2]

    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), 0.5, dtype=np.float32)
    sq_a = np.zeros((s, s), dtype=np.float32)
    oy, ox = (s - h) // 2, (s - w) // 2
    sq[oy:oy + h, ox:ox + w] = crop
    sq_a[oy:oy + h, ox:ox + w] = crop_a

    rgba = np.concatenate([sq, sq_a[..., None]], axis=-1)
    rgba_pil = Image.fromarray((rgba * 255).astype(np.uint8))
    rgba_pil = resize_foreground(rgba_pil, fg_ratio)

    arr = np.array(rgba_pil).astype(np.float32) / 255.0
    final = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * 0.5
    return Image.fromarray((final * 255).astype(np.uint8)).resize(
        (size, size), Image.BICUBIC
    )


def main():
    parser = argparse.ArgumentParser(description="Run TripoSR with build_cache-style preprocessing.")
    parser.add_argument("images", nargs="+", help="Input image path(s).")
    parser.add_argument("--masks", nargs="+", default=None,
                        help="Optional mask path(s), one per image. If given, rembg is skipped.")
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "Models" / "pre-trained" / "triposr"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs" / "triposr_out"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--size", type=int, default=512, help="Final input image size.")
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument("--rembg-model", default="isnet-general-use")
    parser.add_argument("--format", default="obj", choices=["obj", "glb"])
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)

    if args.masks is not None and len(args.masks) != len(args.images):
        raise SystemExit(f"--masks count ({len(args.masks)}) must match images ({len(args.images)})")

    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"device={device} | out={out_dir}")

    t0 = time.time()
    model = TSR.from_pretrained(args.model_dir, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    logging.info(f"model loaded in {time.time() - t0:.1f}s")

    rembg_session = None
    if args.masks is None:
        import rembg
        rembg_session = rembg.new_session(args.rembg_model)
        logging.info(f"rembg model: {args.rembg_model}")

    for i, img_path in enumerate(args.images):
        img_path = Path(img_path)
        sample_dir = out_dir / img_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        if args.masks is not None:
            alpha = get_alpha_from_file(Path(args.masks[i]))
            if alpha.shape != rgb.shape[:2]:
                alpha = np.array(
                    Image.fromarray((alpha * 255).astype(np.uint8)).resize(
                        (rgb.shape[1], rgb.shape[0]), Image.NEAREST
                    ),
                    dtype=np.float32,
                ) / 255.0
        else:
            alpha = get_alpha_from_rembg(Image.open(img_path), rembg_session)

        # Save the alpha for debugging — this is what TripoSR effectively sees as foreground.
        Image.fromarray((alpha * 255).astype(np.uint8)).save(sample_dir / "alpha.png")

        prepped = prepare_image(rgb, alpha, size=args.size, fg_ratio=args.foreground_ratio)
        if prepped is None:
            logging.warning(f"[{i + 1}/{len(args.images)}] {img_path.name}: empty mask, skipping")
            continue
        prepped.save(sample_dir / "input.png")

        logging.info(f"[{i + 1}/{len(args.images)}] {img_path.name}: running TripoSR ...")
        t0 = time.time()
        with torch.no_grad():
            scene_codes = model([prepped], device=device)
        logging.info(f"  forward: {time.time() - t0:.1f}s")

        # torchmcubes lacks MPS support; run extract_mesh on CPU when on MPS.
        t0 = time.time()
        mc_device = "cpu" if device == "mps" else device
        if mc_device != device:
            model.to(mc_device)
            scene_codes = scene_codes.to(mc_device)
        meshes = model.extract_mesh(scene_codes, True, resolution=args.mc_resolution)
        if mc_device != device:
            model.to(device)
        logging.info(f"  marching cubes: {time.time() - t0:.1f}s")

        out_path = sample_dir / f"mesh.{args.format}"
        meshes[0].export(out_path)
        logging.info(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
