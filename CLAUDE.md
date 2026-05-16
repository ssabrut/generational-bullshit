# Generational Bullshit — Single-Image 3D Reconstruction

## Environment

- **Python:** `/Users/michaeleko/miniconda3/envs/quantap/bin/python` (3.11)
- **PyTorch:** 2.11.0, **device: CPU only** (Apple M5, 24 GB unified memory)
- **Installed:** torch, torchvision, numpy, pillow, scikit-learn
- **Not installed:** open3d, imageio, scikit-image, einops, timm
- **DINOv2 cache:** `~/.cache/torch/hub/facebookresearch_dinov2_main`
- **No MPS, no CUDA** — all device logic defaults to CPU

## How to run

```bash
# Training
python -c "from app.src.train import train; train(omni_root='data/OmniObject3D', visualise=True)"

# Inference
python -c "
from app.src.infer import infer
infer(image_path='path/to/image.jpg', ckpt_path='checkpoints/triplane_epoch050.pth')
"
```

Imports use **relative imports** inside `app.src` — always invoke from the project root as above, never as `python -m app.src.train`.

---

## Dataset: OmniObject3D

- **Path:** `data/OmniObject3D/`
- **Categories downloaded:** chair (29 instances), sofa (14 instances)
- **Split:** 80/10/10 train/val/test by instance, `random.seed(42)` → train=34, val=4, test=5

### Modalities per instance

| Path | Contents | Notes |
|---|---|---|
| `camera/{cat}/{obj}/` | `elevation.npy`, `rotation.npy` — shape `(24,)` float64 | Azimuth/elevation per raw view |
| `points/{cat}/{obj}/` | `pcd_4096.ply` — ASCII PLY, 4096 pts | Millimetre scale, properties: x y z |
| `raw/{cat}/{obj}/` | `000.png`…`023.png` (1024×1024 RGBA) + `transforms.json` | NeRF-style c2w 4×4, `camera_angle_x`, `aabb=[[-0.4,-0.4,-0.4],[0.4,0.4,0.4]]` |
| `render/{cat}/{obj}/render/` | `images/r_0.png`…`r_99.png` (800×800 RGB) | Synthetic renders — primary supervision signal |
| `render/{cat}/{obj}/render/` | `depths/r_*_depth.exr`, `normals/r_*_normal.png` | EXR depth skipped (needs `brew install openexr`) |
| `scan/{cat}/{obj}/Scan/` | `Scan.obj` + `Scan.mtl` + `Scan.jpg` | ~300k verts, ~600k faces |

---

## Architecture: Triplane NeRF

**Why triplane over Pixel2Mesh (GCN):** 100 synthetic render views per object enable dense photometric supervision, far stronger than Chamfer distance to a point cloud.

```
Reference image [1, 3, 224, 224]
  → ImageEncoder (DINOv2 ViT-B/14, frozen)  →  CLS token [1, 768]
  → TriplaneGenerator (MLP + Conv refiner)   →  3 planes [1, 3, 32, 64, 64]
  → sample_triplane at 3D query points       →  features [1, N, 96]
  → NeRFMLP                                  →  σ [1,N,1], RGB [1,N,3]
  → volume_render                            →  rgb_map, depth_map
  → photometric loss vs render/images/
```

**Key constants** (`app/src/model/triplane.py`):
- `_PLANE_CH = 32` — feature channels per plane
- `_PLANE_RES = 64` — plane spatial resolution H=W
- `_SCENE_BOUND = 1.2` — scene occupies [-1.2, 1.2]^3
- Default: `n_samples=64, near=0.5, far=4.5`

---

## File Reference

### `app/src/dataloader/omniobject3d.py`

- `read_ply_xyz(path)` — pure-Python ASCII PLY reader, returns `[N,3]` float32
- `normalise_points(pts)` — centre + scale to unit sphere (radius ≤ 1.0)
- `c2w_to_Rt(c2w)` — 4×4 c2w → world-to-camera `R [3,3]`, `T [3]`
- `focal_from_angle(angle)` — `1 / tan(angle/2)` normalised focal
- `OmniObject3DDataset` — one random raw view per `__getitem__`; returns `image, gt_points, rot, trans, focal, category, instance, view_idx`

### `app/src/model/triplane.py`

- **`ImageEncoder`** — DINOv2 frozen; xFormers warnings suppressed via `warnings.catch_warnings()`
- **`TriplaneGenerator`** — MLP → reshape → shared Conv2d refiner → `[B, 3, 32, 64, 64]`
- **`_bilinear_sample`** — manual gather+lerp replacing `F.grid_sample` (avoids backward limitations); fully differentiable
- **`sample_triplane`** — XY/XZ/YZ projection + bilinear sample + concat → `[B,N,96]`
- **`NeRFMLP`** — two heads: density_net and colour_net, 3-layer MLP with GELU
- **`volume_render`** — softplus(σ), sigmoid(RGB), transmittance via exclusive cumprod
- **`cast_rays`** — random pixel → camera-space direction → world-space ray via c2w rotation
- **`TriplaneNeRF`**:
  - `render_full(image, c2w, focal, H, W, chunk, planes=None)` — dense render in chunks; pass `planes=` to skip re-encoding
  - `extract_mesh(image, resolution=64, threshold=10.0)` — marching cubes on CPU (requires `pip install scikit-image`)
- **`make_optimizer`** — AdamW, two groups: generator lr=5e-4, nerf_mlp lr=5e-4; encoder excluded

### `app/src/train.py`

- **`RenderViewLoader`** — caches render `transforms.json`; loads `render/images/*.png` as `[3,H,W]` float32 in [0,1] (no DINOv2 normalisation — these are GT targets)
- **`collate_fn`** — stacks tensor keys, keeps string keys as lists
- **`Visualiser`** — `matplotlib.use("MacOSX")`, fresh figure per step (plt.ion + plt.pause + plt.close):
  - `[0]` Loss curves: total/photo/tv/l2 + val scatter
  - `[1]` Input image (normalisation undone)
  - `[2]` GT render (target view, resized 64×64)
  - `[3]` Predicted — **full 64×64 `render_full` using pre-computed `planes=`** (not sparse ray scatter — that was a bug)
  - `[4]` GT point cloud — **`projection='3d'` interactive axes**, click-drag to rotate
- **`train()`** — per step: encode → triplanes → per-instance render-view photometric loss → TV+L2 reg → backward → clip grad 1.0 → step → visualise

### `app/src/infer.py`

- **`load_model(ckpt_path, device)`** — auto-infers arch from state dict:
  - `plane_ch` from `generator.refine.0.weight.shape[0]`
  - `plane_res` from `sqrt(generator.mlp.6.weight.shape[0] / (3 * plane_ch))`
  - `nerf_h` from `nerf_mlp.density_net.0.weight.shape[0]`
- **`turntable_c2w(n_views, elevation, radius)`** — look-at via cross products, OpenGL convention (+Y up)
- **`infer()`** — encodes once → reuses `planes=` for all N views → saves `frames/`, `overview.png`; optionally `mesh.obj` + `mesh_views.png`

---

## Known Issues Fixed

| Bug | Cause | Fix |
|---|---|---|
| `grid_sampler_2d_backward` error | `F.grid_sample` backward limitations | Replaced with `_bilinear_sample` (gather+lerp) |
| Predicted render panel all black | 512 rays into 256×256 = 0.8% pixels filled | Use `render_full(H=64, W=64)` for dense 64×64 preview |
| `ModuleNotFoundError: src.dataloader` | Absolute imports break when invoked as module | Changed all to relative imports (`.dataloader.omniobject3d`) |
| `focal_from_angle` import error | Linter removed it from triplane.py | Removed stale import in train.py |
| `plane_res` arch inference wrong | `refine.0.weight.shape[2]` = kernel size (3) | Use `mlp.6.weight.shape[0]` → `sqrt(flat/(3*C))` |
| xFormers warnings on every load | CUDA-only attention optimisation unavailable | `warnings.catch_warnings()` in `ImageEncoder.__init__` |
| MPS/CUDA references throughout | Originally targeted Apple Silicon MPS | Removed all — CPU only throughout |
