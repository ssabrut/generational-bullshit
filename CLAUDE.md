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
  → ImageEncoder (DINOv2 ViT-B/14, frozen)  →  patch tokens [1, 256, 768]
  → TriplaneGenerator (per-plane Linear seed + 3-stage Conv refiner)
                                              →  planes [1, 3, 48, 128, 128]
  → sample_triplane at 3D query points       →  features [1, N, 144]
  → NeRFMLP (density + colour heads)         →  σ [1,N,1], RGB [1,N,3]
  → volume_render                            →  rgb_map, depth_map
  → photometric + silhouette loss vs render/images/  +  TV + L2 reg
```

**Key constants in [app/src/model/triplane.py:5-9](app/src/model/triplane.py#L5-L9):**

```python
_DINOV2_MODEL = "dinov2_vitb14"  # embed_dim = 768
_PLANE_CH     = 48               # feature channels per triplane
_PLANE_RES    = 128              # spatial resolution H = W (= patch_grid × 8)
_PATCH_GRID   = 16               # DINOv2 ViT-B/14 @ 224 → 16×16 = 256 tokens
_SCENE_BOUND  = 0.5              # scene occupies [-0.5, 0.5]^3 (OmniObject3D aabb ±0.4 + margin)
```

Default render config in `TriplaneNeRF`: `n_samples=64, near=0.5, far=4.5`. Training overrides to `n_samples=96, near=3.0, far=5.0` to match the inference camera radius.

**Layer details:**

- **ImageEncoder** — DINOv2 ViT-B/14, `requires_grad_(False)`. Returns `out["x_norm_patchtokens"]` (the 256-token spatial grid), **not** the CLS token. xFormers warnings suppressed inside `__init__` via `warnings.catch_warnings()`.
- **TriplaneGenerator** — patch tokens are projected to per-plane spatial seeds via three independent `Linear(768 → seed_ch=96)` heads (`token_proj_xy`, `token_proj_xz`, `token_proj_yz`). Each plane's seed is reshaped to `[seed_ch, 16, 16]` and upsampled by a refiner with three `ConvTranspose×2` stages (`16 → 32 → 64 → 128`). Output `[B, 3, 48, 128, 128]`. Generator param count dropped from ~170M (old MLP head) to ~0.5M; the spatial inductive bias is what does the work.
- **NeRFMLP** — two heads. `density_net`: `Linear(144→h) → GELU → Linear(h→h) → GELU → Linear(h→h) → Linear(h→1)`. `colour_net`: input is `[triplane_feat, view_pe]`, same depth, final `Linear(h→3)`. Default `h=128`.

## Supervision signal

Each training step:

1. Sample one instance, pick **one random raw view** as the encoder input.
2. Pick **K random synthetic render views** (`render/images/r_*.png`) as targets.
3. Cast `n_rays` rays per target view (`fg_frac=0.5` of them sampled from the silhouette mask).
4. Volume-render → compute losses → backward.

**Losses** assembled inline in the train loop; legacy `compute_loss` in [triplane.py](app/src/model/triplane.py) is unused.

| Loss | Form | Final weight (epoch 31+) |
|---|---|---|
| Photometric | `MSE + 0.1·L1` | 1.0 (implicit) |
| Silhouette | BCE on opacity vs `mask > 0.5` | `w_sil = 0.3` |
| Plane TV | `Σ |∂x| + |∂y|` over plane features | `w_tv = 5e-5` |
| Plane L2 | `planes.pow(2).mean()` | `w_l2 = 1e-6` |

**Photometric warm-up** (via `loss_weights(epoch)` in [app/src/train.py](app/src/train.py)):

| Epoch range | w_sil | w_tv | w_l2 |
|---|---|---|---|
| 1–5 | 0.0 | 0.0 | 0.0 — pure photometric so colour gradient establishes initial density |
| 6–15 | linear 0.0 → 0.6 | 5e-5 | 1e-6 |
| 16+ | 0.6 | 5e-5 | 1e-6 |

Why ramp instead of holding `w_sil=1.0`: silhouette BCE ≈ 0.69 at init vs photometric ≈ 0.28 — silhouette dominates ~2.5× and pushes early training toward mass-in-mask before colour/structure learn anything. But the ramp also can't be *too* gentle: an earlier run with `w_sil_max=0.3` over epochs 11–30 let the model converge to a "predict zero density everywhere" local minimum (high full-image PSNR from matching the black background, but black foreground). Current schedule starts sil sooner (epoch 6) and ends higher (0.6) to force opacity inside the silhouette.

Gradient clip = 1.0 in [app/src/train.py](app/src/train.py).

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
- `TriplaneNeRF.render_full(image, c2w, focal, H, W, chunk, planes=None, near=None, far=None)` — dense render in chunks. **Pass `planes=` to skip re-encoding** when rendering multiple views of the same input. **Pass `near`/`far`** to override the model's defaults (e.g. self-render from a raw OmniObject3D camera at distance ~1.2 needs `near≈0.2, far≈2.2` instead of training's `3.0/5.0`, otherwise every ray samples past the scene volume).
- `TriplaneNeRF.extract_mesh(image, resolution=64, threshold=10.0)` — marching cubes on CPU (requires `pip install scikit-image`).

### [app/src/train.py](app/src/train.py)

- `RenderViewLoader` — caches each instance's `render/transforms.json`; loads `render/images/r_*.png` resized to `render_size` (default 256) as `[3, H, W] float32` in `[0, 1]` — **no DINOv2 normalisation** (these are GT targets, not encoder inputs). Silhouette masks come from the alpha of `normals/r_*_normal.png`.
- `render_view_loss(model, planes, view, n_rays, device, w_sil, fg_frac)` — cast rays into one target view, render, return `photo + w_sil·sil`.
- `collate_fn(batch)` — stacks tensor keys; strings (`category`, `instance`) become lists.
- `save_checkpoint(model, optimizer, epoch, step, val_loss, path)` — `torch.save` with metadata.
- `load_checkpoint(model, path, optimizer=None, device=None)` — restores model + optimizer.
- `loss_weights(epoch, w_sil_max=0.6, w_tv=5e-5, w_l2=1e-6)` — returns `{w_sil, w_tv, w_l2}` per the warm-up schedule above.
- `c2w_from_rot_trans(rot, trans)` — invert the dataset's world-to-camera `R, T` to a 4×4 c2w for the encoder's input view. Used by the self-render diagnostic.
- `psnr_from_mse(mse)` — `-10·log10(mse)` in dB; returns inf for mse=0.
- `fg_psnr(pred, gt, mask)` — PSNR computed *only* on pixels where `mask > 0.5`. Use this instead of full-image PSNR when supervision is on white-background renders: the full-image metric is inflated by easy background matches, which can mask "predict zero density everywhere" failure modes.
- `Visualiser` — `matplotlib.use("MacOSX")`, fresh figure each step via `plt.ion + plt.pause + plt.close`. 1 row × 6 cols:
  - `[0]` Loss curves on **log y-scale** with 5th–95th-percentile y-clip so a single early spike doesn't compress the curve. Lines: total / photo / sil / tv / l2 + val scatter. PSNR is shown in the title.
  - `[1]` Input image (DINOv2 normalisation undone for display).
  - `[2]` GT render (target view, resized 64×64).
  - `[3]` Predicted (novel view) — full 64×64 `render_full` using pre-computed `planes=`.
  - `[4]` **Self-render** — same generator, rendered from the encoder's *own input camera* (via `c2w_from_rot_trans`). The raw camera sits at distance ~1.2, so this call passes `near=max(0.05, ||cam_pos||-1), far=||cam_pos||+1` to bracket the scene from that closer viewpoint (training near/far of 3.0/5.0 would sample entirely past the object). If self-render is sharp but `[3]` is blurry, the bottleneck is generator multi-view consistency; if both are blurry, escalate to NeRFMLP capacity.
  - `[5]` GT point cloud with `projection='3d'` interactive axes.
- `train(omni_root, ckpt_dir='checkpoints', categories=None, n_epochs=50, batch_size=2, n_render_views=4, n_rays=1024, render_size=256, lr_gen=5e-4, lr_nerf=5e-4, w_tv=5e-5, w_l2=1e-6, w_sil=0.6, fg_frac=0.5, val_every=5, resume=None, vis_dir=None, log_path=None, device=None, visualise=True)` — main loop. The `__main__` block overrides defaults to `n_epochs=100, batch_size=4`. Per-epoch `loss_weights(epoch)` returns the actual weights used; `w_sil` here caps the final value (after the ramp). CSV columns: `step,epoch,batch,total,photo,sil,tv,l2,lr,w_sil,w_tv,w_l2,psnr,psnr_fg,fg_opacity` — the visualised-view metrics (`psnr`, `psnr_fg`, `fg_opacity`) are only populated when `visualise=True` (NaN otherwise). **`psnr_fg` is the meaningful one** — full-image `psnr` is inflated by background matches.
- `visualise_reconstruction(model, omni_root, cat, obj_id, device, out_dir, n_views)` — post-training qualitative check on a held-out instance.

### [app/src/infer.py](app/src/infer.py)

- `load_model(ckpt_path, device)` — **auto-infers architecture from state dict** under the new patch-token arch:
  - `plane_ch` from the final refiner conv's out-channels (`generator.refine.{last}.weight.shape[0]`).
  - `seed_ch` from `generator.token_proj_xy.weight.shape[0]`.
  - `plane_res = 128` (fixed: patch grid 16 × three `ConvTranspose2d(stride=2)`).
  - `nerf_h` from `nerf_mlp.density_net.0.weight.shape[0]`.
- `preprocess(image_path, device)` — any image → `[1, 3, 224, 224]` DINOv2-normalised.
- `turntable_c2w(n_views=36, elevation=20.0, radius=2.5)` — look-at via cross products, OpenGL convention (+Y up, -Z forward). Returns `[N, 4, 4]`.
- `infer(image_path, ckpt_path, out_dir='inference_out', n_views=36, render_h=256, render_w=256, render_chunk=4096, focal=None, elevation=20.0, radius=2.5, extract_mesh=False, mesh_res=64, mesh_thresh=10.0)` — encodes once → reuses `planes=` for all N views → saves `frames/view_*.png`, `overview.png`; optionally `mesh.obj` + `mesh_views.png`. Decorated with `@torch.no_grad()`.

## Known issues fixed

| Bug | Cause | Fix |
|---|---|---|
| Mean-shape collapse — predictions are blurry blobs across 39 epochs, almost identical across very different inputs | `ImageEncoder` returned only the CLS token; a 768-dim vector was projected to 165K plane values with no spatial inductive bias. With 34 training instances, the optimal solution is the category-mean shape. | Switched to DINOv2 patch tokens (`x_norm_patchtokens`, 16×16=256 of them). Three per-plane `Linear(768→seed_ch)` heads + 3-stage `ConvTranspose×2` refiner; `_PLANE_RES` raised 96 → 128 to align with the patch grid. Generator param count fell from ~170M to ~0.5M. **Old checkpoints incompatible**; legacy ones moved to `checkpoints/legacy/`. |
| Silhouette loss dominated at init (BCE ≈ 0.69 vs photometric ≈ 0.28) | Fixed `w_sil = 1.0` from epoch 1. Early training optimised mass-in-mask before colour/structure. | `loss_weights(epoch)` warm-up: epochs 1–10 pure photometric, 11–30 linear ramp to `w_sil=0.3`, 31+ hold. |
| Loss plot unreadable — a single ~3000-magnitude spike at step ~30 auto-scaled the y-axis to where everything else looked flat; silhouette line missing entirely. | Linear y-scale and `record(...)` never persisted `sil`. | Log-y scale, y-clip to 1st–95th percentile, and `record(step, total, photo, sil, tv, l2)` now persists silhouette. PSNR shown in the loss-panel title. |
| `grid_sampler_2d_backward` error on CPU | `F.grid_sample` backward limitations | Replaced with `_bilinear_sample` (gather + lerp). |
| Predicted-view dashboard panel all black | 512 sparse rays into a 256×256 canvas = 0.8% pixel coverage | Use `render_full(H=64, W=64)` with reused `planes=`. |
| `ModuleNotFoundError: src.dataloader` | Absolute imports break when invoked from project root | Switched to relative imports throughout `app.src`. |
| `focal_from_angle` import error | Linter removed it from triplane.py | Removed stale import in train.py; lives only in dataloader. |
| Wrong `plane_res` inference in `load_model` | `refine.0.weight.shape[2]` is kernel size (3) | Fixed `plane_res = 128` under the new patch-token arch (16-token grid × three ConvTranspose×2). |
| xFormers warnings every load | CUDA-only attention optimisation unavailable | `warnings.catch_warnings()` in `ImageEncoder.__init__`. |
| MPS / CUDA references throughout | Originally targeted Apple Silicon MPS | Removed all — CPU only. |

## Known issues outstanding

- `TriplaneNeRF.extract_mesh` in [app/src/model/triplane.py](app/src/model/triplane.py) calls `self.nerf_mlp(feats)` with one positional argument, but `NeRFMLP.forward(feats, view_dirs)` requires `view_dirs`. Calling `extract_mesh` will currently fail with a missing-argument error. Pre-existing — fix separately when mesh extraction is needed.
