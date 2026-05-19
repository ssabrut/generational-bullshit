"""Inference: image → deformed mesh → .usdz.

Usage:
  # Arbitrary image (recommended: --remove-bg for in-the-wild photos)
  python student/infer.py \
      --checkpoint runs/student_ddp_v2/best.pt \
      --image path/to/chair.png \
      --out out/chair.usdz \
      [--remove-bg]

  # Sanity check on a teacher-cache entry (uses training-distribution input.png)
  python student/infer.py \
      --checkpoint runs/student_ddp_v2/best.pt \
      --cache-entry data/teacher_cache/chair/00000_0001 \
      --out out/chair.usdz

The .usdz is written with usd-core (`pip install usd-core`).
Optional: `pip install rembg onnxruntime` for --remove-bg.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.dataset import build_image_transform  # noqa: E402
from student.model import Student  # noqa: E402
from student.template import build_template  # noqa: E402


def load_student(
    checkpoint: Path,
    device: str,
    subdivisions: int = 3,
    n_stages: int = 1,
    hidden: int = 128,
) -> Student:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # If the checkpoint stored its training args, use them to rebuild the exact geometry.
    saved = ckpt.get("args") if isinstance(ckpt, dict) else None
    if isinstance(saved, dict):
        subdivisions = saved.get("subdivisions", subdivisions)
        n_stages = saved.get("n_stages", n_stages)
        hidden = saved.get("hidden", hidden)
        print(f"  using checkpoint args: subdivisions={subdivisions}  n_stages={n_stages}  hidden={hidden}")

    tpl = build_template(subdivisions=subdivisions)
    tpl.verts = tpl.verts - tpl.verts.mean(dim=0, keepdim=True)
    tpl.verts = tpl.verts / tpl.verts.norm(dim=1).max()
    model = Student(template=tpl, hidden=hidden, n_stages=n_stages).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:4]}{'…' if len(missing) > 4 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:4]}{'…' if len(unexpected) > 4 else ''}")
    model.eval()
    return model


def _remove_background(img: Image.Image) -> Image.Image:
    try:
        from rembg import remove
    except ImportError as exc:
        raise SystemExit(
            "rembg is required for --remove-bg. Install with: pip install rembg onnxruntime"
        ) from exc
    cut = remove(img)  # RGBA with transparent background
    # Composite onto white so DinoV2 sees a clean background like the teacher cache.
    bg = Image.new("RGBA", cut.size, (255, 255, 255, 255))
    bg.paste(cut, mask=cut.split()[-1])
    return _center_crop_to_content(bg.convert("RGB"))


def _center_crop_to_content(img: Image.Image, pad_ratio: float = 0.1) -> Image.Image:
    """Tight-crop around non-white pixels, then pad to a square with white."""
    arr = np.asarray(img)
    mask = (arr < 245).any(axis=-1)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    side = int(max(h, w) * (1 + pad_ratio))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = side // 2
    out = Image.new("RGB", (side, side), (255, 255, 255))
    crop = img.crop((max(0, cx - half), max(0, cy - half),
                     min(arr.shape[1], cx + half), min(arr.shape[0], cy + half)))
    out.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return out


def preprocess_image(image_path: Path, size: int = 224, remove_bg: bool = False) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    if remove_bg:
        img = _remove_background(img)
    tf = build_image_transform(size)
    return tf(img).unsqueeze(0)  # (1, 3, 224, 224)


@torch.no_grad()
def predict_mesh(model: Student, image: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    out = model(image)
    verts = out["verts"][0].cpu().numpy().astype(np.float32)  # (V, 3)
    faces = model.template_faces.cpu().numpy().astype(np.int32)  # (F, 3)
    return verts, faces


def write_usdz(verts: np.ndarray, faces: np.ndarray, out_path: Path) -> None:
    try:
        from pxr import Usd, UsdGeom, Vt
    except ImportError as exc:
        raise SystemExit(
            "usd-core is required to write .usdz. Install with: pip install usd-core"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # USD wants a .usdc payload that we then package into .usdz.
    tmp_usdc = out_path.with_suffix(".usdc")
    stage = Usd.Stage.CreateNew(str(tmp_usdc))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())

    mesh_prim = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    mesh_prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    mesh_prim.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
    )
    mesh_prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    # Per-vertex normals from face geometry (simple area-weighted average).
    v = verts
    f = faces
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn = vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8)
    normals_attr = mesh_prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(vn.astype(np.float32)))
    mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    _ = normals_attr

    stage.GetRootLayer().Save()

    # Package the .usdc into a .usdz archive.
    UsdUtils = __import__("pxr.UsdUtils", fromlist=["UsdUtils"])
    UsdUtils.CreateNewUsdzPackage(str(tmp_usdc), str(out_path))
    tmp_usdc.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt / last.pt")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Input RGB image (in-the-wild)")
    src.add_argument(
        "--cache-entry",
        type=Path,
        help="Path to a teacher-cache entry dir (uses its input.png — training distribution)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output .usdz path")
    p.add_argument(
        "--subdivisions",
        type=int,
        default=3,
        help="Override only if checkpoint doesn't store args. 3=642 verts, 4=2562, 5=10242",
    )
    p.add_argument("--n-stages", type=int, default=1, help="Override only if checkpoint doesn't store args")
    p.add_argument("--hidden", type=int, default=128, help="Override only if checkpoint doesn't store args")
    p.add_argument(
        "--remove-bg",
        action="store_true",
        help="Background-remove + center-crop the input (only useful with --image)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda | cpu | mps",
    )
    args = p.parse_args()

    print(f"loading checkpoint {args.checkpoint}  device={args.device}")
    model = load_student(
        args.checkpoint,
        args.device,
        subdivisions=args.subdivisions,
        n_stages=args.n_stages,
        hidden=args.hidden,
    )

    if args.cache_entry is not None:
        image_path = args.cache_entry / "input.png"
        if not image_path.exists():
            raise SystemExit(f"no input.png in {args.cache_entry}")
        print(f"preprocessing {image_path} (cache entry — no bg removal needed)")
        image = preprocess_image(image_path, remove_bg=False).to(args.device)
    else:
        print(f"preprocessing {args.image}  remove_bg={args.remove_bg}")
        image = preprocess_image(args.image, remove_bg=args.remove_bg).to(args.device)

    print("running student…")
    verts, faces = predict_mesh(model, image)
    print(f"  mesh: {len(verts)} verts, {len(faces)} faces")

    print(f"writing {args.out}")
    write_usdz(verts, faces, args.out)
    print("done")


if __name__ == "__main__":
    main()
