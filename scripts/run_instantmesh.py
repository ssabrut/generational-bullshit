"""Run InstantMesh inference on one or more images.

Pipeline:
  1. Remove background (rembg) or use a provided mask
  2. Preprocess image (same gray-composite + crop + resize as run_triposr.py)
  3. Generate 6 multi-view images via Zero123++ diffusion
  4. Reconstruct mesh using the LRM-based model
  5. Export as OBJ or GLB

Requires teacher/InstantMesh to be cloned:
    git clone https://github.com/TencentARC/InstantMesh teacher/InstantMesh
    pip install -r teacher/InstantMesh/requirements.txt

Example:
    python scripts/run_instantmesh.py path/to/image.png
    python scripts/run_instantmesh.py path/to/image.png --mask path/to/mask.png
    python scripts/run_instantmesh.py path/to/image.png --config instant-mesh-large
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
INSTANTMESH_ROOT = REPO_ROOT / "teacher" / "InstantMesh"
sys.path.insert(0, str(INSTANTMESH_ROOT))


def resolve_device(name: str) -> str:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return name


def get_alpha_from_rembg(img_pil: Image.Image, rembg_session) -> np.ndarray:
    from rembg import remove
    cutout = remove(img_pil.convert("RGBA"), session=rembg_session)
    arr = np.array(cutout, dtype=np.float32) / 255.0
    return arr[..., 3]


def get_alpha_from_file(mask_path: Path) -> np.ndarray:
    return np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0


def prepare_image(rgb: np.ndarray, alpha: np.ndarray, size: int = 320,
                  fg_ratio: float = 0.85) -> Image.Image | None:
    """Preprocess input image: composite on gray, crop to mask, resize foreground."""
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

    # Resize foreground within the square using the fg_ratio
    fg_size = int(size * fg_ratio)
    fg_offset = (size - fg_size) // 2

    sq_pil = Image.fromarray((sq * 255).astype(np.uint8)).resize((fg_size, fg_size), Image.BICUBIC)
    final = Image.new("RGB", (size, size), (128, 128, 128))
    final.paste(sq_pil, (fg_offset, fg_offset))
    return final


def load_pipeline(config_name: str, device: str):
    """Load InstantMesh diffusion + reconstruction pipeline."""
    import os
    import importlib

    # InstantMesh config YAMLs live in teacher/InstantMesh/configs/
    config_map = {
        "instant-mesh-large": "instant-mesh-large.yaml",
        "instant-mesh-base": "instant-mesh-base.yaml",
        "instant-nerf-large": "instant-nerf-large.yaml",
        "instant-nerf-base": "instant-nerf-base.yaml",
    }
    if config_name not in config_map:
        raise ValueError(f"Unknown config '{config_name}'. Choose from: {list(config_map)}")

    config_path = INSTANTMESH_ROOT / "configs" / config_map[config_name]
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Make sure InstantMesh is cloned at: {INSTANTMESH_ROOT}"
        )

    # InstantMesh uses OmegaConf for configs
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)

    # Load Zero123++ diffusion pipeline for multi-view generation
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    logging.info("Loading Zero123++ multi-view diffusion pipeline ...")
    mv_pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    mv_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        mv_pipe.scheduler.config, timestep_spacing="trailing"
    )

    # Load InstantMesh reconstruction model
    from huggingface_hub import hf_hub_download

    hf_model_map = {
        "instant-mesh-large": ("TencentARC/InstantMesh", "checkpoints/instant_mesh_large.ckpt"),
        "instant-mesh-base": ("TencentARC/InstantMesh", "checkpoints/instant_mesh_base.ckpt"),
        "instant-nerf-large": ("TencentARC/InstantMesh", "checkpoints/instant_nerf_large.ckpt"),
        "instant-nerf-base": ("TencentARC/InstantMesh", "checkpoints/instant_nerf_base.ckpt"),
    }
    repo_id, filename = hf_model_map[config_name]
    logging.info(f"Downloading reconstruction checkpoint: {filename} ...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Dynamically load the model class defined in the InstantMesh config
    model_cls_path = cfg.model.target  # e.g. "src.models.instantmesh.InstantMesh"
    module_path, cls_name = model_cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, cls_name)

    recon_model = model_cls(**cfg.model.params)
    state = torch.load(ckpt_path, map_location="cpu")
    recon_model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    recon_model = recon_model.to(device)
    recon_model.eval()

    mv_pipe = mv_pipe.to(device)

    return mv_pipe, recon_model, cfg


def generate_multiview(mv_pipe, image: Image.Image, num_steps: int = 75,
                       seed: int = 42) -> list[Image.Image]:
    """Run Zero123++ to produce 6 multi-view images from a single input."""
    generator = torch.Generator(device=mv_pipe.device).manual_seed(seed)
    result = mv_pipe(
        image,
        num_inference_steps=num_steps,
        generator=generator,
    ).images[0]

    # Zero123++ returns a 3x2 grid of 320x320 views — split into individual images
    w, h = result.size
    view_w, view_h = w // 2, h // 3
    views = []
    for row in range(3):
        for col in range(2):
            box = (col * view_w, row * view_h, (col + 1) * view_w, (row + 1) * view_h)
            views.append(result.crop(box))
    return views  # 6 views: front-right, back-right, back-left, front-left, top, bottom


def reconstruct_mesh(recon_model, input_image: Image.Image, mv_images: list[Image.Image],
                     cfg, device: str, export_path: Path, export_format: str = "obj"):
    """Run the LRM reconstruction model and export the mesh."""
    import torchvision.transforms.functional as TF

    def to_tensor(img: Image.Image) -> torch.Tensor:
        return TF.to_tensor(img).unsqueeze(0).to(device)

    # Stack input + 6 views: shape [1, 7, 3, H, W]
    all_images = [input_image] + mv_images
    images_tensor = torch.cat([to_tensor(img) for img in all_images], dim=0).unsqueeze(0)

    with torch.no_grad():
        # Camera poses are fixed for Zero123++ output — InstantMesh defines them internally
        planes = recon_model.forward_planes(images_tensor)
        mesh = recon_model.extract_mesh(planes, use_texture_map=True)

    mesh.export(str(export_path))


def main():
    parser = argparse.ArgumentParser(description="Run InstantMesh inference on input image(s).")
    parser.add_argument("images", nargs="+", help="Input image path(s).")
    parser.add_argument("--masks", nargs="+", default=None,
                        help="Optional mask path(s), one per image. Skips rembg if provided.")
    parser.add_argument("--config", default="instant-mesh-large",
                        choices=["instant-mesh-large", "instant-mesh-base",
                                 "instant-nerf-large", "instant-nerf-base"],
                        help="InstantMesh model variant to use.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs" / "instantmesh_out"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--size", type=int, default=320,
                        help="Input image size (InstantMesh default is 320).")
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument("--diffusion-steps", type=int, default=75,
                        help="Number of Zero123++ denoising steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rembg-model", default="isnet-general-use")
    parser.add_argument("--format", default="obj", choices=["obj", "glb"])
    parser.add_argument("--save-views", action="store_true",
                        help="Save the 6 generated multi-view images alongside the mesh.")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)

    if args.masks is not None and len(args.masks) != len(args.images):
        raise SystemExit(f"--masks count ({len(args.masks)}) must match images ({len(args.images)})")

    if not INSTANTMESH_ROOT.exists():
        raise SystemExit(
            f"InstantMesh not found at {INSTANTMESH_ROOT}\n"
            f"Run: git clone https://github.com/TencentARC/InstantMesh {INSTANTMESH_ROOT}"
        )

    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"device={device} | config={args.config} | out={out_dir}")

    t0 = time.time()
    mv_pipe, recon_model, cfg = load_pipeline(args.config, device)
    logging.info(f"pipeline loaded in {time.time() - t0:.1f}s")

    rembg_session = None
    if args.masks is None:
        import rembg
        rembg_session = rembg.new_session(args.rembg_model)
        logging.info(f"rembg model: {args.rembg_model}")

    for i, img_path in enumerate(args.images):
        img_path = Path(img_path)
        sample_dir = out_dir / img_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"[{i + 1}/{len(args.images)}] processing: {img_path.name}")

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

        Image.fromarray((alpha * 255).astype(np.uint8)).save(sample_dir / "alpha.png")

        prepped = prepare_image(rgb, alpha, size=args.size, fg_ratio=args.foreground_ratio)
        if prepped is None:
            logging.warning(f"  empty mask, skipping")
            continue
        prepped.save(sample_dir / "input.png")

        # Step 1: generate 6 multi-view images
        logging.info(f"  generating multi-view images ({args.diffusion_steps} steps) ...")
        t0 = time.time()
        mv_images = generate_multiview(mv_pipe, prepped, num_steps=args.diffusion_steps, seed=args.seed)
        logging.info(f"  diffusion: {time.time() - t0:.1f}s")

        if args.save_views:
            for j, view in enumerate(mv_images):
                view.save(sample_dir / f"view_{j:02d}.png")

        # Step 2: reconstruct mesh from views
        logging.info(f"  reconstructing mesh ...")
        t0 = time.time()
        out_path = sample_dir / f"mesh.{args.format}"
        reconstruct_mesh(recon_model, prepped, mv_images, cfg, device, out_path, args.format)
        logging.info(f"  reconstruction: {time.time() - t0:.1f}s")
        logging.info(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
