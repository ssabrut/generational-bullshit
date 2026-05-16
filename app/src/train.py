"""
Training loop for TriplaneNeRF on OmniObject3D.

Supervision signal
------------------
For each training step:
  1. Sample one object instance from OmniObject3D.
  2. Pick one raw captured image as the *input* to the encoder.
  3. Pick K random synthetic render views (from render/images/) as *targets*.
  4. For each target view, cast N rays, volume-render them, compute photometric
     loss vs the GT pixel colours from render/images/*.png.
  5. Add TV + L2 plane regularisation.

Why render views and not raw views as targets?
  Raw images are real photos (1024×1024 RGBA, messy backgrounds).
  Render views are clean synthetic renders on white background with known,
  exact camera poses — ideal photometric supervision.

Notes
-----
  - Runs on CPU.
  - DataLoader num_workers=0 (safest on macOS).
  - Render chunk size is tunable if you hit OOM.
"""

import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataloader.omniobject3d import OmniObject3DDataset
from .model.triplane import (TriplaneNeRF, cast_rays, compute_loss,
                             make_optimizer, sample_triplane)

# ── reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Render-view loader
# Loads target images + c2w poses from render/images/ and transforms.json.
# Kept separate from OmniObject3DDataset so the dataset stays input-only.
# ─────────────────────────────────────────────────────────────────────────────


class RenderViewLoader:
    """
    Caches render metadata per object and loads target view images on demand.

    Usage
    -----
    loader = RenderViewLoader(omni_root, img_size=256)
    views  = loader.sample(cat, obj_id, k=4, device=device)
    # views: list of {"rgb": [3,H,W], "c2w": [4,4], "focal": scalar}
    """

    _IMG_TFM_CACHE: dict = {}

    def __init__(self, root: str, img_size: int = 256):
        self.root = Path(root)
        self.img_size = img_size
        self._meta_cache: dict = {}
        self._tfm = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # → [3, H, W]  float32 in [0, 1]
            ]
        )

    def _load_meta(self, cat: str, obj_id: str) -> dict:
        key = (cat, obj_id)
        if key not in self._meta_cache:
            path = self.root / "render" / cat / obj_id / "render" / "transforms.json"
            with open(path) as fh:
                self._meta_cache[key] = json.load(fh)
        return self._meta_cache[key]

    def sample(
        self,
        cat: str,
        obj_id: str,
        k: int = 4,
        device: torch.device = torch.device("cpu"),
    ) -> list:
        """
        Sample k random render views for (cat, obj_id).

        Returns list of dicts:
          rgb   : [3, H, W]   float32 in [0, 1]
          c2w   : [4, 4]      float32 camera-to-world
          focal : []          float32 normalised focal length
        """
        meta = self._load_meta(cat, obj_id)
        focal = float(1.0 / np.tan(meta["camera_angle_x"] / 2.0))
        frames = random.sample(meta["frames"], k)

        img_dir = self.root / "render" / cat / obj_id / "render" / "images"
        views = []
        for frame in frames:
            img_path = img_dir / f"{frame['file_path']}.png"
            rgb = self._tfm(Image.open(img_path).convert("RGB")).to(device)

            c2w = torch.tensor(
                frame["transform_matrix"], dtype=torch.float32, device=device
            )
            views.append(
                {
                    "rgb": rgb,
                    "c2w": c2w,
                    "focal": torch.tensor(focal, dtype=torch.float32, device=device),
                }
            )
        return views


# ─────────────────────────────────────────────────────────────────────────────
# Per-view rendering loss
# ─────────────────────────────────────────────────────────────────────────────


def render_view_loss(
    model: TriplaneNeRF,
    planes: torch.Tensor,  # [1, 3, C, H, W]  — pre-computed for this instance
    view: dict,  # {"rgb":[3,H,W], "c2w":[4,4], "focal":[]}
    n_rays: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Cast n_rays into one target view, render them, return photometric L2+L1 loss.
    planes is computed once per instance and reused across all target views.
    """
    rgb_gt = view["rgb"]  # [3, H, W]
    c2w = view["c2w"].unsqueeze(0)  # [1, 4, 4]
    focal = view["focal"].unsqueeze(0)  # [1]
    _, H, W = rgb_gt.shape

    rays_o, rays_d, pix_ij = cast_rays(c2w, focal, H, W, n_rays=n_rays, device=device)

    # Volume render
    pts, t = _sample_rays(model, rays_o, rays_d)  # [1, R, S, 3], [1, R, S]
    B, R, S, _ = pts.shape
    pts_norm = pts.clamp(-model.generator.plane_res, model.generator.plane_res)
    pts_norm = pts.clamp(-1.2, 1.2) / 1.2

    pts_flat = pts_norm.view(1, R * S, 3)
    feats = sample_triplane(planes, pts_flat).view(1, R, S, -1)
    sigma, rgb_raw = model.nerf_mlp(feats)

    from .model.triplane import volume_render

    rgb_map, _, _ = volume_render(sigma, rgb_raw, t)  # [1, R, 3]

    # GT pixels at sampled locations
    rgb_gt_rays = rgb_gt[:, pix_ij[:, 0], pix_ij[:, 1]].T  # [R, 3]
    rgb_gt_rays = rgb_gt_rays.unsqueeze(0)  # [1, R, 3]

    return F.mse_loss(rgb_map, rgb_gt_rays) + 0.1 * F.l1_loss(rgb_map, rgb_gt_rays)


def _sample_rays(model, rays_o, rays_d):
    from .model.triplane import sample_along_rays

    return sample_along_rays(
        rays_o,
        rays_d,
        n_samples=model.n_samples,
        near=model.near,
        far=model.far,
        perturb=model.training,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Collate — keeps string fields out of the default tensor collate
# ─────────────────────────────────────────────────────────────────────────────


def collate_fn(batch: list) -> dict:
    tensor_keys = ["image", "gt_points", "rot", "trans", "focal"]
    str_keys = ["category", "instance", "view_idx"]
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out.update({k: [b[k] for b in batch] for k in str_keys})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Live visualiser
# ─────────────────────────────────────────────────────────────────────────────


class Visualiser:
    """
    Live training dashboard — one window per step, same pattern as scripts/train.py.

    Layout (1 row, 5 columns):
      [0] Loss curves  — total / photometric / tv / l2 per step, val as dots
      [1] Input image  — raw captured view fed to the encoder
      [2] GT render    — synthetic target view used this step
      [3] Predicted    — volume-rendered output at the same camera
      [4] GT point cloud — X-Z top-down scatter of the pre-sampled PLY
    """

    def __init__(self, vis_dir: str = None):
        plt.ion()
        self.vis_dir = vis_dir
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)

        self._steps: list = []
        self._total: list = []
        self._photo: list = []
        self._tv: list = []
        self._l2: list = []
        self._val_steps: list = []
        self._val_loss: list = []

    def record(self, step, total, photo, tv, l2):
        self._steps.append(step)
        self._total.append(total)
        self._photo.append(photo)
        self._tv.append(tv)
        self._l2.append(l2)

    def record_val(self, step, val_loss):
        self._val_steps.append(step)
        self._val_loss.append(val_loss)

    def update(
        self,
        step: int,
        epoch: int,
        input_img: torch.Tensor,  # [3, 224, 224]  DINOv2-normalised
        gt_render: torch.Tensor,  # [3, H, W]      float32 in [0, 1]
        pred_render: torch.Tensor,  # [3, H, W]      float32 in [0, 1]
        gt_pts: torch.Tensor,  # [N, 3]
        save_path: str = None,
    ):
        fig = plt.figure(figsize=(22, 4))
        fig.patch.set_facecolor("#1a1a1a")

        gs = fig.add_gridspec(1, 5, width_ratios=[2, 1, 1, 1, 1])
        axes = [fig.add_subplot(gs[i]) for i in range(4)]
        ax3d = fig.add_subplot(gs[4], projection="3d")
        for ax in axes:
            ax.set_facecolor("#1a1a1a")

        # ── [0] loss curves ───────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(self._steps, self._total, color="#4A90D9", lw=1.5, label="total")
        ax.plot(
            self._steps, self._photo, color="#E8672A", lw=1.0, label="photo", alpha=0.85
        )
        ax.plot(self._steps, self._tv, color="#50C878", lw=0.8, label="tv", alpha=0.7)
        ax.plot(self._steps, self._l2, color="#C8A84B", lw=0.8, label="l2", alpha=0.7)
        if self._val_steps:
            ax.scatter(
                self._val_steps,
                self._val_loss,
                color="#FF6B9D",
                s=40,
                zorder=5,
                label="val",
            )
        ax.set_title(f"Loss  (epoch {epoch}, step {step})", color="white", fontsize=9)
        ax.set_xlabel("step", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.legend(
            fontsize=7, facecolor="#2a2a2a", labelcolor="white", loc="upper right"
        )
        ax.grid(True, linewidth=0.3, alpha=0.4, color="#444")

        # ── [1] input image (undo DINOv2 normalisation) ───────────────────────
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        inp = (input_img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[1].imshow(inp)
        axes[1].set_title("Input image", color="white", fontsize=9)
        axes[1].axis("off")

        # ── [2] GT render ─────────────────────────────────────────────────────
        axes[2].imshow(gt_render.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
        axes[2].set_title("GT render", color="white", fontsize=9)
        axes[2].axis("off")

        # ── [3] predicted render ──────────────────────────────────────────────
        axes[3].imshow(pred_render.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
        axes[3].set_title("Predicted", color="white", fontsize=9)
        axes[3].axis("off")

        # ── [4] GT point cloud (interactive 3D) ──────────────────────────────
        p = gt_pts.cpu().numpy()
        ax3d.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.5, c="#20B2AA", alpha=0.5)
        ax3d.set_facecolor("#1a1a1a")
        ax3d.set_title("GT point cloud", color="white", fontsize=9)
        ax3d.tick_params(colors="#aaa", labelsize=5)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor("#333")
        ax3d.yaxis.pane.set_edgecolor("#333")
        ax3d.zaxis.pane.set_edgecolor("#333")
        lim = 1.1
        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)

        fig.suptitle(f"Epoch {epoch} | Step {step}", color="white", fontsize=11)
        fig.tight_layout(pad=1.5)

        _path = save_path or (
            os.path.join(self.vis_dir, f"epoch{epoch:04d}_step{step:07d}.png")
            if self.vis_dir
            else None
        )
        if _path:
            fig.savefig(
                _path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor()
            )

        plt.pause(0.001)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoint(model, optimizer, epoch, step, val_loss, path):
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(model, path, optimizer=None, device=None):
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(
        f"  [ckpt] loaded epoch={ckpt['epoch']} step={ckpt['step']} "
        f"val_loss={ckpt.get('val_loss', 'n/a')}"
    )
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def train(
    omni_root: str,
    ckpt_dir: str = "checkpoints",
    categories: list = None,
    n_epochs: int = 50,
    batch_size: int = 2,
    n_render_views: int = 4,  # target render views sampled per instance
    n_rays: int = 1024,  # rays per render view per step
    render_size: int = 256,  # resize render images to this before cropping
    lr_gen: float = 5e-4,
    lr_nerf: float = 5e-4,
    w_tv: float = 5e-5,  # halved — TV was over-smoothing thin structures
    w_l2: float = 1e-6,  # 100× lower — was suppressing useful plane activations
    val_every: int = 5,  # validate every N epochs
    resume: str = None,  # path to checkpoint to resume from
    vis_dir: str = None,  # dashboard PNGs; defaults to {ckpt_dir}/plots
    log_path: str = None,  # per-step loss CSV; defaults to {ckpt_dir}/train_loss.csv
    device: torch.device = None,
    visualise: bool = True,  # live dashboard via plt.show
):
    if device is None:
        device = torch.device("cpu")
    print(f"Training on: {device}")

    os.makedirs(ckpt_dir, exist_ok=True)
    weights_dir = os.path.join(ckpt_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    if vis_dir is None:
        vis_dir = os.path.join(ckpt_dir, "plots")
    if log_path is None:
        log_path = os.path.join(ckpt_dir, "train_loss.csv")

    # ── datasets ──────────────────────────────────────────────────────────────
    cats = categories or ["chair", "sofa"]
    train_ds = OmniObject3DDataset(omni_root, split="train", categories=cats)
    val_ds = OmniObject3DDataset(
        omni_root, split="val", categories=cats, random_view=False
    )

    # num_workers=0: safest on macOS
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # ── render-view loader ────────────────────────────────────────────────────
    render_loader = RenderViewLoader(omni_root, img_size=render_size)

    # ── model + optimizer ─────────────────────────────────────────────────────
    model = TriplaneNeRF(
        plane_ch=48,
        plane_res=96,
        nerf_hidden=192,
        n_samples=96,
        near=1.5,
        far=4.0,
    ).to(device)
    optimizer = make_optimizer(model, lr_gen=lr_gen, lr_nerf=lr_nerf)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5
    )

    start_epoch = 1
    global_step = 0
    history = {"train": [], "val": []}

    if resume:
        ckpt = load_checkpoint(model, resume, optimizer, device)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("step", 0)

    total_p = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model params: total={total_p:,}  trainable={trainable:,}  "
        f"frozen={total_p-trainable:,}"
    )
    print(f"Train: {len(train_ds)} instances | Val: {len(val_ds)} instances")
    print(
        f"Epochs: {n_epochs} | Batch: {batch_size} | "
        f"Render views/instance: {n_render_views} | Rays/view: {n_rays}"
    )
    print()

    vis = Visualiser(vis_dir=vis_dir) if visualise else None

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_is_new = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log_fh = open(log_path, "a", buffering=1)  # line-buffered
    if log_is_new:
        log_fh.write("step,epoch,batch,total,photo,tv,l2,lr\n")
    print(f"Per-step loss log: {log_path}")

    from .model.triplane import (plane_l2_loss, plane_tv_loss,
                                 sample_along_rays, volume_render)

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            img = batch["image"].to(device)  # [B, 3, 224, 224]
            gt_pts = batch["gt_points"].to(device)  # [B, N, 3]
            cats_ = batch["category"]
            ids = batch["instance"]
            B = img.shape[0]

            optimizer.zero_grad()

            # Encode all images → triplanes [B, 3, C, H, W]
            img_feat = model.encoder(img)
            planes = model.generator(img_feat)

            # Per-instance photometric loss across K render views
            photo_loss = torch.tensor(0.0, device=device)
            # Keep one view's GT + predicted RGB for the dashboard (instance 0)
            _vis_gt_render = None
            _vis_pred_render = None

            for b in range(B):
                plane_b = planes[b : b + 1]
                views = render_loader.sample(
                    cats_[b], ids[b], k=n_render_views, device=device
                )
                for vi, view in enumerate(views):
                    rgb_gt = view["rgb"]  # [3, H, W]
                    c2w = view["c2w"].unsqueeze(0)  # [1, 4, 4]
                    focal = view["focal"].unsqueeze(0)  # [1]
                    _, H, W = rgb_gt.shape

                    rays_o, rays_d, pix_ij = cast_rays(
                        c2w, focal, H, W, n_rays=n_rays, device=device
                    )
                    pts, t = sample_along_rays(
                        rays_o,
                        rays_d,
                        n_samples=model.n_samples,
                        near=model.near,
                        far=model.far,
                        perturb=True,
                    )
                    R, S = pts.shape[1], pts.shape[2]
                    pts_norm = pts.clamp(-1.2, 1.2) / 1.2
                    feats = sample_triplane(plane_b, pts_norm.view(1, R * S, 3))
                    feats = feats.view(1, R, S, -1)
                    sigma, rgb_raw = model.nerf_mlp(feats)
                    rgb_map, _, _ = volume_render(sigma, rgb_raw, t)  # [1, R, 3]

                    rgb_gt_rays = rgb_gt[:, pix_ij[:, 0], pix_ij[:, 1]].T.unsqueeze(0)
                    view_loss = F.mse_loss(rgb_map, rgb_gt_rays) + 0.1 * F.l1_loss(
                        rgb_map, rgb_gt_rays
                    )
                    photo_loss = photo_loss + view_loss

                    # Capture the first view of the first instance for the dashboard
                    if b == 0 and vi == 0 and vis is not None:
                        with torch.no_grad():
                            # Full render at 64×64 — dense enough to evaluate
                            # quality, fast enough to run every step on CPU.
                            # Reuse already-computed planes (no second encoder pass).
                            _VIS_H, _VIS_W = 64, 64
                            rendered = model.render_full(
                                img[0:1],
                                c2w,
                                focal,
                                H=_VIS_H,
                                W=_VIS_W,
                                chunk=_VIS_H * _VIS_W,
                                planes=plane_b.detach(),
                            )
                            _vis_pred_render = (
                                rendered["rgb"].permute(2, 0, 1).clamp(0, 1)
                            )
                            _vis_gt_render = (
                                F.interpolate(
                                    rgb_gt.unsqueeze(0),
                                    size=(_VIS_H, _VIS_W),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                .squeeze(0)
                                .cpu()
                            )

            photo_loss = photo_loss / (B * n_render_views)
            tv_loss = plane_tv_loss(planes)
            l2_loss = plane_l2_loss(planes)
            loss = photo_loss + w_tv * tv_loss + w_l2 * l2_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

            log_fh.write(
                f"{global_step},{epoch},{batch_idx+1},"
                f"{loss.item():.6f},{photo_loss.item():.6f},"
                f"{tv_loss.item():.6f},{l2_loss.item():.6f},"
                f"{optimizer.param_groups[0]['lr']:.3e}\n"
            )

            # ── live dashboard update every step ──────────────────────────────
            if vis is not None:
                vis.record(
                    global_step,
                    loss.item(),
                    photo_loss.item(),
                    tv_loss.item(),
                    l2_loss.item(),
                )
                vis.update(
                    step=global_step,
                    epoch=epoch,
                    input_img=img[0],
                    gt_render=_vis_gt_render,
                    pred_render=_vis_pred_render,
                    gt_pts=gt_pts[0],
                )
            else:
                print(
                    f"  epoch {epoch:3d} | step {global_step:5d} | "
                    f"batch {batch_idx+1}/{len(train_loader)} | "
                    f"loss {loss.item():.5f}  "
                    f"(photo={photo_loss.item():.5f} "
                    f"tv={tv_loss.item():.5f} "
                    f"l2={l2_loss.item():.5f})"
                )

        train_avg = float(np.mean(epoch_losses))
        history["train"].append(train_avg)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d} done | train_loss={train_avg:.5f} | "
            f"time={elapsed:.1f}s | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # ── validation ────────────────────────────────────────────────────────
        val_avg = None
        if epoch % val_every == 0 or epoch == n_epochs:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    img = batch["image"].to(device)
                    cats_ = batch["category"]
                    ids = batch["instance"]
                    B = img.shape[0]

                    img_feat = model.encoder(img)
                    planes = model.generator(img_feat)

                    v_loss = torch.tensor(0.0, device=device)
                    for b in range(B):
                        plane_b = planes[b : b + 1]
                        views = render_loader.sample(
                            cats_[b], ids[b], k=2, device=device
                        )
                        for view in views:
                            v_loss = v_loss + render_view_loss(
                                model, plane_b, view, n_rays, device
                            )
                    val_losses.append((v_loss / (B * 2)).item())

            val_avg = float(np.mean(val_losses))
            history["val"].append(val_avg)

            if vis is not None:
                vis.record_val(global_step, val_avg)

            print(f"  [val]  epoch {epoch:3d} | val_loss={val_avg:.5f}")

        ckpt_path = os.path.join(weights_dir, f"triplane_epoch{epoch:03d}.pth")
        save_checkpoint(model, optimizer, epoch, global_step, val_avg, ckpt_path)

        print()

    log_fh.close()
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Quick render visualisation (call after training)
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def visualise_reconstruction(
    model: TriplaneNeRF,
    omni_root: str,
    cat: str,
    obj_id: str,
    device: torch.device,
    out_dir: str = "renders",
    n_views: int = 4,
):
    """
    Render n_views of a reconstructed object and save PNGs side-by-side
    with GT.  Useful for qualitative inspection after training.
    """
    import torchvision.utils as vutils

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # Input image (raw view 0)
    root = Path(omni_root)
    tf_path = root / "raw" / cat / obj_id / "transforms.json"
    with open(tf_path) as fh:
        raw_meta = json.load(fh)
    raw_frame = raw_meta["frames"][0]

    dino_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_img = (
        dino_tfm(
            Image.open(root / "raw" / cat / obj_id / raw_frame["file_path"]).convert(
                "RGB"
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    # Load render metadata
    render_meta_path = root / "render" / cat / obj_id / "render" / "transforms.json"
    with open(render_meta_path) as fh:
        render_meta = json.load(fh)
    focal_r = float(1.0 / np.tan(render_meta["camera_angle_x"] / 2.0))
    frames = random.sample(render_meta["frames"], n_views)

    img_dir = root / "render" / cat / obj_id / "render" / "images"
    rows = []
    for frame in frames:
        c2w = torch.tensor(
            frame["transform_matrix"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        focal = torch.tensor(focal_r, dtype=torch.float32, device=device).unsqueeze(0)

        rendered = model.render_full(input_img, c2w, focal, H=256, W=256, chunk=1024)
        pred_rgb = rendered["rgb"].permute(2, 0, 1).clamp(0, 1)  # [3, H, W]

        gt_rgb = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )(Image.open(img_dir / f"{frame['file_path']}.png").convert("RGB"))

        rows.append(torch.stack([gt_rgb, pred_rgb.cpu()], dim=0))

    grid = vutils.make_grid(torch.cat(rows, dim=0), nrow=2, padding=4)
    vutils.save_image(grid, os.path.join(out_dir, f"{cat}_{obj_id}.png"))
    print(f"Saved → {out_dir}/{cat}_{obj_id}.png  (GT | pred per row)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, history = train(
        omni_root="data/OmniObject3D",
        ckpt_dir="checkpoints",
        categories=["chair", "sofa"],
        n_epochs=100,
        batch_size=4,
        n_render_views=4,
        n_rays=1024,
        render_size=256,
        lr_gen=5e-4,
        lr_nerf=5e-4,
        w_tv=5e-5,
        w_l2=1e-6,
        val_every=5,
        resume=None,
        vis_dir="checkpoints/plot",
        log_path="checkpoints/train_loss.csv"
    )
