# Generational Bullshit — Single-Image 3D Reconstruction

Triplane NeRF that reconstructs 3D geometry and appearance from a single image, trained on OmniObject3D using dense synthetic-render photometric supervision.

## Environment

- **Python:** `/Users/michaeleko/miniconda3/envs/quantap/bin/python` (3.11)
- **PyTorch:** 2.11.0, **device: CPU only** (Apple M5, 24 GB unified memory)
- **Installed:** torch, torchvision, numpy, pillow, scikit-learn, matplotlib
- **Not installed:** open3d, imageio, scikit-image (needed for `extract_mesh`), einops, timm
- **DINOv2 cache:** `~/.cache/torch/hub/facebookresearch_dinov2_main`
- No MPS, no CUDA — all device logic resolves to CPU.

## How to run

Imports inside [app/src/](app/src/) are **relative** — always invoke from the project root, not as `python -m app.src.train`.

```bash
# Training
python -c "from app.src.train import train; train(omni_root='data/OmniObject3D', visualise=True)"

# Inference
python -c "
from app.src.infer import infer
infer(image_path='path/to/image.jpg', ckpt_path='checkpoints/triplane_epoch050.pth')
"
```

Both modules also run directly via their `__main__` blocks: `python app/src/train.py` / `python app/src/infer.py`.

## Dataset: OmniObject3D

- **Root:** `data/OmniObject3D/`
- **Categories downloaded:** chair (29 instances), sofa (14 instances) — set in `OMNI_CATEGORIES` at [app/src/dataloader/omniobject3d.py:14](app/src/dataloader/omniobject3d.py#L14)
- **Split:** 80/10/10 train/val/test by instance, `random.seed(42)` → train=34, val=4, test=5

| Path | Contents | Notes |
|---|---|---|
| `camera/{cat}/{obj}/` | `elevation.npy`, `rotation.npy` shape `(24,)` float64 | Azimuth/elevation per raw view |
| `points/{cat}/{obj}/` | `pcd_4096.ply` (ASCII PLY, 4096 pts) | Millimetre scale, `x y z` props |
| `raw/{cat}/{obj}/` | `000.png`…`023.png` (1024×1024 RGBA) + `transforms.json` | NeRF-style c2w 4×4, `camera_angle_x`, `aabb=[[-0.4,-0.4,-0.4],[0.4,0.4,0.4]]` |
| `render/{cat}/{obj}/render/images/` | `r_0.png`…`r_99.png` (800×800 RGB) | Synthetic renders — **primary supervision signal** |
| `render/{cat}/{obj}/render/normals/` | `r_*_normal.png` | Alpha channel used as silhouette mask |
| `render/{cat}/{obj}/render/depths/` | `r_*_depth.exr` | Skipped (would need `brew install openexr`) |
| `scan/{cat}/{obj}/Scan/` | `Scan.obj`, `Scan.mtl`, `Scan.jpg` | ~300k verts, ~600k faces |

## Architecture: Triplane NeRF

**Why triplane over Pixel2Mesh GCN:** 100 synthetic render views per object enable dense photometric supervision, much stronger than Chamfer-only supervision to a 4096-point cloud.

```
Reference image [1, 3, 224, 224]
  → ImageEncoder (DINOv2 ViT-B/14, frozen)  →  CLS token [1, 768]
  → TriplaneGenerator (MLP + Conv refiner)   →  planes [1, 3, 48, 96, 96]
  → sample_triplane at 3D query points       →  features [1, N, 144]
  → NeRFMLP (density + colour heads)         →  σ [1,N,1], RGB [1,N,3]
  → volume_render                            →  rgb_map, depth_map
  → photometric + silhouette loss vs render/images/  +  TV + L2 reg
```

**Key constants in [app/src/model/triplane.py:5-8](app/src/model/triplane.py#L5-L8):**

```python
_DINOV2_MODEL = "dinov2_vitb14"  # embed_dim = 768
_PLANE_CH     = 48               # feature channels per triplane
_PLANE_RES    = 96               # spatial resolution H = W
_SCENE_BOUND  = 0.5              # scene occupies [-0.5, 0.5]^3 (OmniObject3D aabb ±0.4 + margin)
```

Default render config in `TriplaneNeRF`: `n_samples=64, near=0.5, far=4.5`. Training overrides to `n_samples=96, near=3.0, far=5.0` to match the inference camera radius.

**Layer details:**

- **ImageEncoder** — DINOv2 ViT-B/14, `requires_grad_(False)`. xFormers warnings suppressed inside `__init__` via `warnings.catch_warnings()`.
- **TriplaneGenerator** — `Linear(768→1024) → GELU → Linear(1024→1024) → GELU → Linear(1024 → 3·seed_ch·seed_res²)` then reshape to `[B, 3·seed_ch, seed_res, seed_res]` with `seed_res = plane_res // 4`. Refiner: `Conv → GELU → ConvTranspose(×2) → GELU → Conv → GELU → ConvTranspose(×2) → GELU → Conv → GELU → Conv1×1`. Output `[B, 3, 48, 96, 96]`.
- **NeRFMLP** — two heads. `density_net`: `Linear(144→h) → GELU → Linear(h→h) → GELU → Linear(h→h) → Linear(h→1)`. `colour_net`: input is `[triplane_feat, view_pe]`, same depth, final `Linear(h→3)`. Default `h=128`.

## Supervision signal

Each training step:

1. Sample one instance, pick **one random raw view** as the encoder input.
2. Pick **K random synthetic render views** (`render/images/r_*.png`) as targets.
3. Cast `n_rays` rays per target view (`fg_frac=0.5` of them sampled from the silhouette mask).
4. Volume-render → compute losses → backward.

**Losses** (combined in [`compute_loss`](app/src/model/triplane.py#L676)):

| Loss | Form | Weight |
|---|---|---|
| Photometric | `MSE + 0.1·L1` | 1.0 (implicit) |
| Silhouette | BCE on opacity vs `mask > 0.5` | `w_sil = 1.0` |
| Plane TV | `Σ |∂x| + |∂y|` over plane features | `w_tv = 5e-5` |
| Plane L2 | `planes.pow(2).mean()` | `w_l2 = 1e-6` |

Gradient clip = 1.0 in [app/src/train.py](app/src/train.py). `w_tv` was halved from an earlier value (was over-smoothing thin structures), and `w_l2` was dropped 100× (was suppressing useful plane activations).

## File reference

### [app/src/dataloader/omniobject3d.py](app/src/dataloader/omniobject3d.py)

- `read_ply_xyz(path)` — pure-Python ASCII PLY reader, `[N, 3] float32`.
- `normalise_points(pts)` — centre + scale to unit sphere (radius ≤ 1.0); clamp `min=1e-8` to survive degenerate clouds.
- `c2w_to_Rt(c2w)` — 4×4 c2w → world-to-camera `R [3,3]`, `T = -R @ t_cam [3]`.
- `focal_from_angle(camera_angle_x)` — `1 / tan(angle/2)`, normalised focal.
- `OMNI_CATEGORIES = ["chair", "sofa"]`.
- `OmniObject3DDataset` — one random raw view per `__getitem__`. Returns dict:
  ```python
  {"image": [3,224,224], "gt_points": [n_pts,3], "rot": [3,3], "trans": [3],
   "focal": scalar, "category": str, "instance": str, "view_idx": int}
  ```
  Image preproc: resize 224×224, DINOv2 norm `mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]`.

### [app/src/model/triplane.py](app/src/model/triplane.py)

- `ImageEncoder`, `TriplaneGenerator`, `NeRFMLP`, `TriplaneNeRF` — see Architecture above.
- `_bilinear_sample(plane, coords)` — manual `gather + lerp` replacement for `F.grid_sample` (whose backward had backend limits on CPU). Fully differentiable.
- `sample_triplane(planes, xyz)` — project to XY/XZ/YZ, sample each, concat → `[B, N, 3·plane_ch]`.
- `positional_encoding(x, n_freqs=4)` — sin/cos at `2^k`, `[..., D] → [..., 2·n_freqs·D]`.
- `volume_render(sigma, rgb, t)` — `softplus(σ)`, `sigmoid(RGB)`, transmittance via exclusive cumprod with `1e-10` epsilon.
- `cast_rays(c2w, focal, H, W, n_rays, device, mask=None, fg_frac=0.0)` — random pixel sampler, optionally foreground-biased.
- `sample_along_rays(rays_o, rays_d, n_samples, near, far, perturb)` — stratified samples.
- `pts_to_triplane_coords(pts, bound=_SCENE_BOUND)` — clamp to `±bound` then normalise to `[-1, 1]`.
- `photometric_loss`, `plane_tv_loss`, `plane_l2_loss`, `compute_loss` — losses.
- `make_optimizer(model, lr_gen=5e-4, lr_nerf=5e-4, weight_decay=...)` — AdamW, two param groups (`generator` and `nerf_mlp`); **encoder is excluded** (frozen).
- `TriplaneNeRF.render_full(image, c2w, focal, H, W, chunk, planes=None)` — dense render in chunks. **Pass `planes=` to skip re-encoding** when rendering multiple views of the same input.
- `TriplaneNeRF.extract_mesh(image, resolution=64, threshold=10.0)` — marching cubes on CPU (requires `pip install scikit-image`).

### [app/src/train.py](app/src/train.py)

- `RenderViewLoader` — caches each instance's `render/transforms.json`; loads `render/images/r_*.png` resized to `render_size` (default 256) as `[3, H, W] float32` in `[0, 1]` — **no DINOv2 normalisation** (these are GT targets, not encoder inputs). Silhouette masks come from the alpha of `normals/r_*_normal.png`.
- `render_view_loss(model, planes, view, n_rays, device, w_sil, fg_frac)` — cast rays into one target view, render, return `photo + w_sil·sil`.
- `collate_fn(batch)` — stacks tensor keys; strings (`category`, `instance`) become lists.
- `save_checkpoint(model, optimizer, epoch, step, val_loss, path)` — `torch.save` with metadata.
- `load_checkpoint(model, path, optimizer=None, device=None)` — restores model + optimizer.
- `Visualiser` — `matplotlib.use("MacOSX")`, fresh figure each step via `plt.ion + plt.pause + plt.close`. 1 row × 5 cols:
  - `[0]` Loss curves: total / photo / tv / l2 + val scatter.
  - `[1]` Input image (DINOv2 normalisation undone for display).
  - `[2]` GT render (target view, resized 64×64).
  - `[3]` Predicted — **full 64×64 `render_full` using pre-computed `planes=`** (not sparse-ray scatter, which displayed black).
  - `[4]` GT point cloud with `projection='3d'` interactive axes.
- `train(omni_root, ckpt_dir='checkpoints', categories=None, n_epochs=50, batch_size=2, n_render_views=4, n_rays=1024, render_size=256, lr_gen=5e-4, lr_nerf=5e-4, w_tv=5e-5, w_l2=1e-6, w_sil=1.0, fg_frac=0.5, val_every=5, visualise=False, vis_dir='checkpoints/plot', log_path='checkpoints/train_loss.csv', resume_from=None)` — main loop. The `__main__` block overrides defaults to `n_epochs=100, batch_size=4`. Logs per-step to CSV with columns `step,epoch,batch,total,photo,sil,tv,l2,lr`.
- `visualise_reconstruction(model, omni_root, cat, obj_id, device, out_dir, n_views)` — post-training qualitative check on a held-out instance.

### [app/src/infer.py](app/src/infer.py)

- `load_model(ckpt_path, device)` — **auto-infers architecture from state dict**:
  - `plane_ch` from the final refiner conv's out-channels (`generator.refine.{last}.weight.shape[0]`).
  - `seed_ch` from `generator.refine.0.weight.shape[0]` (first refiner conv in-channels).
  - `seed_res = sqrt(generator.mlp.6.weight.shape[0] / (3 · seed_ch))`, then `plane_res = seed_res · 4` (two `ConvTranspose2d(stride=2)`).
  - `nerf_h` from `nerf_mlp.density_net.0.weight.shape[0]`.
- `preprocess(image_path, device)` — any image → `[1, 3, 224, 224]` DINOv2-normalised.
- `turntable_c2w(n_views=36, elevation=20.0, radius=2.5)` — look-at via cross products, OpenGL convention (+Y up, -Z forward). Returns `[N, 4, 4]`.
- `infer(image_path, ckpt_path, out_dir='inference_out', n_views=36, render_h=256, render_w=256, render_chunk=4096, focal=None, elevation=20.0, radius=2.5, extract_mesh=False, mesh_res=64, mesh_thresh=10.0)` — encodes once → reuses `planes=` for all N views → saves `frames/view_*.png`, `overview.png`; optionally `mesh.obj` + `mesh_views.png`. Decorated with `@torch.no_grad()`.

## Known issues fixed

| Bug | Cause | Fix |
|---|---|---|
| `grid_sampler_2d_backward` error on CPU | `F.grid_sample` backward limitations | Replaced with `_bilinear_sample` (gather + lerp). |
| Predicted-view dashboard panel all black | 512 sparse rays into a 256×256 canvas = 0.8% pixel coverage | Use `render_full(H=64, W=64)` with reused `planes=`. |
| `ModuleNotFoundError: src.dataloader` | Absolute imports break when invoked from project root | Switched to relative imports throughout `app.src`. |
| `focal_from_angle` import error | Linter removed it from triplane.py | Removed stale import in train.py; lives only in dataloader. |
| Wrong `plane_res` inference in `load_model` | `refine.0.weight.shape[2]` is kernel size (3) | Compute from `mlp.6.weight.shape[0] / (3 · seed_ch)`. |
| xFormers warnings every load | CUDA-only attention optimisation unavailable | `warnings.catch_warnings()` in `ImageEncoder.__init__`. |
| MPS / CUDA references throughout | Originally targeted Apple Silicon MPS | Removed all — CPU only. |
