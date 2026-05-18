"""
ONNX Runtime inference and tester for Pixel2MeshDINO.

Modes
-----
infer   Run on a single image, save per-stage OBJ files.
test    Verify output shapes, optionally compare against the PyTorch model,
        and benchmark latency over N runs.

Usage
-----
  # Inference on an image (uses median camera from pix3d if available)
  conda run -n AI python scripts/pix2mesh_onnx_runner.py infer \\
      --onnx  runs/pix2mesh_chair_v3/model.onnx \\
      --faces runs/pix2mesh_chair_v3/model_faces.npz \\
      --image data/pix3d/img/chair/0070.png \\
      --out   GAN/exports/onnx_out.obj

  # Test: shape checks + PyTorch parity + benchmark
  conda run -n AI python scripts/pix2mesh_onnx_runner.py test \\
      --onnx    runs/pix2mesh_chair_v3/model.onnx \\
      --weights runs/pix2mesh_chair_v3/best.pt \\
      --runs    20
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

_to_tensor  = transforms.ToTensor()
_normalise  = transforms.Normalize(IMG_MEAN, IMG_STD)

EXPECTED_SHAPES = [(1, 162, 3), (1, 642, 3), (1, 2562, 3)]


# ── OnnxModel wrapper ─────────────────────────────────────────────────────────

class OnnxModel:
    """Thin wrapper around an onnxruntime InferenceSession."""

    def __init__(self, onnx_path: str, device: str = "cpu"):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # External data (model.onnx.data) must live next to model.onnx;
        # ORT resolves it automatically when given the .onnx path.
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        print(f"  Loaded ONNX model from {onnx_path}")
        print(f"  Provider: {self.sess.get_providers()[0]}")

    def __call__(self, image, rot, trans, focal):
        """All inputs are numpy float32 arrays."""
        feeds = {
            "image": image.astype(np.float32),
            "rot":   rot.astype(np.float32),
            "trans": trans.astype(np.float32),
            "focal": focal.astype(np.float32),
        }
        return self.sess.run(None, feeds)   # [verts1, verts2, verts3]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_image(path: str, mask_path: str | None = None, size: int = 224) -> np.ndarray:
    """Load, optionally mask, resize and normalise → float32 [1, 3, H, W]."""
    img = Image.open(path).convert("RGB")

    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask) > 128
        img_arr  = np.array(img)
        img_arr[~mask_arr] = 0
        img = Image.fromarray(img_arr)

    img = img.resize((size, size), Image.BILINEAR)
    t   = _normalise(_to_tensor(img))         # [3, H, W]
    return t.numpy()[None]                    # [1, 3, H, W]


def save_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Pixel2Mesh-DINO ONNX output\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"  Saved {path}  ({len(verts)} verts, {len(faces)} faces)")


def load_faces(faces_path: str):
    npz = np.load(faces_path)
    return npz["faces1"], npz["faces2"], npz["faces3"]


def median_camera(pix3d_root: str):
    """Return (rot, trans, focal) numpy arrays using training-split medians."""
    import json, random
    with open(os.path.join(pix3d_root, "pix3d.json")) as f:
        anns = json.load(f)
    anns = [a for a in anns if a["category"] == "chair"
            and not a.get("truncated") and not a.get("occluded")]
    rng = random.Random(42)
    rng.shuffle(anns)
    train = anns[:int(0.8 * len(anns))]

    focals, tzs = [], []
    for a in train:
        w, h = a["img_size"]
        focals.append(a["focal_length"] / (max(w, h) / 2.0))
        tzs.append(a["trans_mat"][2])

    focal = float(np.median(focals))
    tz    = float(np.median(tzs))
    rot   = np.eye(3,  dtype=np.float32)[None]          # [1, 3, 3]
    trans = np.array([[0.0, 0.0, tz]], dtype=np.float32) # [1, 3]
    return rot, trans, np.array([focal], dtype=np.float32)


# ── infer ─────────────────────────────────────────────────────────────────────

def cmd_infer(args):
    model  = OnnxModel(args.onnx, args.device)
    f1, f2, f3 = load_faces(args.faces)

    # Camera
    pix3d_root = os.path.join(ROOT, "data", "pix3d")
    if os.path.exists(os.path.join(pix3d_root, "pix3d.json")):
        print("  Using median camera from training split")
        rot, trans, focal = median_camera(pix3d_root)
    else:
        print(f"  Using CLI camera (tz={args.tz}, focal={args.focal})")
        rot   = np.eye(3,  dtype=np.float32)[None]
        trans = np.array([[0.0, 0.0, args.tz]], dtype=np.float32)
        focal = np.array([args.focal], dtype=np.float32)

    img_path = args.image if os.path.isabs(args.image) else os.path.join(ROOT, args.image)
    print(f"  Image: {img_path}")
    image = load_image(img_path)

    t0 = time.perf_counter()
    verts1, verts2, verts3 = model(image, rot, trans, focal)
    print(f"  Inference: {(time.perf_counter() - t0)*1000:.1f} ms")

    out_base = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    base, ext = os.path.splitext(out_base)
    ext = ext or ".obj"
    for i, (v, f) in enumerate([(verts1, f1), (verts2, f2), (verts3, f3)], 1):
        save_obj(f"{base}_stage{i}{ext}", v[0], f)


# ── test ──────────────────────────────────────────────────────────────────────

def cmd_test(args):
    import onnxruntime as ort

    print("\n── Shape / smoke test ───────────────────────────────────────────")
    model = OnnxModel(args.onnx, args.device)

    dummy_image = np.zeros((1, 3, 224, 224), dtype=np.float32)
    dummy_rot   = np.eye(3, dtype=np.float32)[None]
    dummy_trans = np.array([[0.0, 0.0, 2.5]], dtype=np.float32)
    dummy_focal = np.array([2.5], dtype=np.float32)

    outputs = model(dummy_image, dummy_rot, dummy_trans, dummy_focal)
    all_ok  = True
    for name, out, exp in zip(["verts1", "verts2", "verts3"], outputs, EXPECTED_SHAPES):
        ok = out.shape == exp
        all_ok &= ok
        print(f"  {name:8s}  got {str(out.shape):18s}  expected {str(exp):18s}  {'✓' if ok else '✗'}")
    print(f"  Shapes: {'all OK' if all_ok else 'MISMATCH'}")

    # PyTorch parity
    if args.weights:
        print("\n── PyTorch parity ───────────────────────────────────────────────")
        import torch
        sys.path.insert(0, os.path.dirname(__file__))
        from train_pix2mesh_chair import Pixel2MeshDINO

        pt_model = Pixel2MeshDINO()
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        pt_model.load_state_dict(ckpt["model"])
        pt_model.eval()

        # Disable xformers so PT and ORT use the same ops
        for m in pt_model.modules():
            if hasattr(m, "use_xformers"):
                m.use_xformers = False

        with torch.no_grad():
            pt_stages = pt_model(
                torch.from_numpy(dummy_image),
                torch.from_numpy(dummy_rot),
                torch.from_numpy(dummy_trans),
                torch.from_numpy(dummy_focal),
            )

        for i, ((pv, *_), ort_v, name) in enumerate(
                zip(pt_stages, outputs, ["verts1", "verts2", "verts3"])):
            diff = np.abs(pv.numpy() - ort_v).max()
            print(f"  {name:8s}  max|PT − ORT| = {diff:.3e}  {'✓' if diff < 5e-4 else '✗ HIGH'}")

    # Latency benchmark
    print(f"\n── Latency benchmark ({args.runs} runs) ──────────────────────────")
    # Warmup
    for _ in range(3):
        model(dummy_image, dummy_rot, dummy_trans, dummy_focal)

    feeds = {
        "image": dummy_image, "rot": dummy_rot,
        "trans": dummy_trans, "focal": dummy_focal,
    }
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        model.sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"  mean={times.mean():.1f} ms  "
          f"p50={np.percentile(times, 50):.1f} ms  "
          f"p95={np.percentile(times, 95):.1f} ms  "
          f"min={times.min():.1f} ms")

    # Per-input breakdown (ORT profiling)
    print("\n── Input sensitivity ────────────────────────────────────────────")
    real_image = load_image(
        args.image if os.path.isabs(args.image) else os.path.join(ROOT, args.image)
    ) if args.image else dummy_image
    t0 = time.perf_counter()
    model(real_image, dummy_rot, dummy_trans, dummy_focal)
    print(f"  Real image inference: {(time.perf_counter() - t0)*1000:.1f} ms")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pixel2Mesh ONNX runner")
    sub    = parser.add_subparsers(dest="cmd", required=True)

    # -- infer ----------------------------------------------------------------
    p_infer = sub.add_parser("infer", help="Run inference on a single image")
    p_infer.add_argument("--onnx",   required=True)
    p_infer.add_argument("--faces",  required=True)
    p_infer.add_argument("--image",  required=True)
    p_infer.add_argument("--out",    default="GAN/exports/onnx_out.obj")
    p_infer.add_argument("--focal",  type=float, default=2.5)
    p_infer.add_argument("--tz",     type=float, default=2.5)
    p_infer.add_argument("--device", default="cpu")

    # -- test -----------------------------------------------------------------
    p_test = sub.add_parser("test", help="Shape checks, parity test, and benchmark")
    p_test.add_argument("--onnx",    required=True)
    p_test.add_argument("--weights", default=None,
                        help="PyTorch checkpoint for parity comparison (optional)")
    p_test.add_argument("--image",   default=None,
                        help="Optional real image for sensitivity check")
    p_test.add_argument("--runs",    type=int, default=20,
                        help="Number of benchmark iterations")
    p_test.add_argument("--device",  default="cpu")

    args = parser.parse_args()

    if args.cmd == "infer":
        cmd_infer(args)
    elif args.cmd == "test":
        cmd_test(args)


if __name__ == "__main__":
    main()
