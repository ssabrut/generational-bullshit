"""Training loop for the DinoV2 + GCN student.

- Train/val split (90/10, fixed seed)
- AdamW with cosine LR schedule
- Periodic checkpointing (best-val + last)
- Periodic visualization: input + target + prediction for a few val samples

Run:
  python student/train.py --epochs 10 --batch-size 4
  python student/train.py --resume runs/last.pt  # to continue
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "teacher"))

from render_previews import _cull_and_shade
from student.dataset import TeacherCacheDataset
from student.loss import compute_loss
from student.model import Student
from student.template import build_template

_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


# ── utils ─────────────────────────────────────────────────────────────────────

def chamfer_only(pred: torch.Tensor, target: torch.Tensor) -> float:
    d = torch.cdist(pred, target, p=2)
    return float((d.min(-1).values.mean(-1) + d.min(-2).values.mean(-1)).mean())


def pick_device(name: str) -> str:
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


# ── visualization ─────────────────────────────────────────────────────────────

def render_mesh(verts: np.ndarray, faces: np.ndarray, ax, azim=30, elev=15) -> None:
    v = verts @ _YUP_TO_ZUP.T
    front, shaded = _cull_and_shade(v, faces, azim, elev)
    ax.add_collection3d(Poly3DCollection(v[front], facecolors=shaded, edgecolors="none"))
    b = np.stack([v.min(0), v.max(0)])
    ax.set_xlim(b[0, 0], b[1, 0]); ax.set_ylim(b[0, 1], b[1, 1]); ax.set_zlim(b[0, 2], b[1, 2])
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off()


def render_points(pts: np.ndarray, ax, azim=30, elev=15) -> None:
    p = pts @ _YUP_TO_ZUP.T
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.4, c="#3355aa", alpha=0.5)
    b = np.stack([p.min(0), p.max(0)])
    ax.set_xlim(b[0, 0], b[1, 0]); ax.set_ylim(b[0, 1], b[1, 1]); ax.set_zlim(b[0, 2], b[1, 2])
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off()


def save_val_preview(model: Student, loader: DataLoader, device: str,
                     out_path: Path, n_show: int = 6) -> None:
    model.eval()
    collected = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            tgts = batch["points"]
            out = model(imgs)
            verts = out["verts"].cpu().numpy()
            for b in range(imgs.shape[0]):
                collected.append({
                    "image": batch["image"][b].permute(1, 2, 0).numpy(),
                    "target": tgts[b].numpy(),
                    "pred": verts[b],
                    "entry": batch["entry_id"][b],
                })
                if len(collected) >= n_show:
                    break
            if len(collected) >= n_show:
                break
    model.train()

    faces = model.template_faces.cpu().numpy()
    fig = plt.figure(figsize=(3 * n_show, 9))
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    for i, c in enumerate(collected):
        ax_img = fig.add_subplot(3, n_show, i + 1)
        img = np.clip(c["image"] * std + mean, 0, 1)
        ax_img.imshow(img); ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.set_title(c["entry"][:14], fontsize=7)

        ax_tgt = fig.add_subplot(3, n_show, n_show + i + 1, projection="3d")
        render_points(c["target"], ax_tgt)
        if i == 0:
            ax_tgt.set_title("target", fontsize=8)

        ax_pred = fig.add_subplot(3, n_show, 2 * n_show + i + 1, projection="3d")
        render_mesh(c["pred"], faces, ax_pred)
        if i == 0:
            ax_pred.set_title("prediction", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache",       type=Path, default=ROOT / "data/teacher_cache")
    p.add_argument("--categories",  nargs="+", default=["chair", "sofa"])
    p.add_argument("--out-dir",     type=Path, default=ROOT / "runs/student")
    p.add_argument("--epochs",      type=int, default=10)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split",   type=float, default=0.1)
    p.add_argument("--hidden",      type=int, default=128)
    p.add_argument("--n-stages",    type=int, default=1)
    p.add_argument("--device",      default="auto")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--preview-every", type=int, default=1, help="Save preview every N epochs")
    p.add_argument("--w-chamfer", type=float, default=1.0)
    p.add_argument("--w-edge",    type=float, default=0.01,
                   help="Edge-length regularizer weight (was 0.1 in smoke test)")
    p.add_argument("--w-lap",     type=float, default=0.05,
                   help="Laplacian smoothness regularizer weight (was 0.5 in smoke test)")
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  f, indent=2)

    print(f"device       : {device}")
    print(f"cache        : {args.cache}")
    print(f"categories   : {args.categories}")
    print(f"out_dir      : {args.out_dir}")

    # ── data ──────────────────────────────────────────────────────────────────
    full_ds = TeacherCacheDataset(cache_root=args.cache, categories=tuple(args.categories))
    n_val = max(1, int(round(len(full_ds) * args.val_split)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"dataset      : {len(full_ds)} entries → train {n_train} / val {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # ── model ─────────────────────────────────────────────────────────────────
    tpl = build_template()
    tpl.verts = tpl.verts - tpl.verts.mean(dim=0, keepdim=True)
    tpl.verts = tpl.verts / tpl.verts.norm(dim=1).max()

    model = Student(template=tpl, hidden=args.hidden, n_stages=args.n_stages).to(device)
    laplacian = tpl.laplacian.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in trainable)
    print(f"params       : trainable {n_train_p / 1e6:.2f}M (DinoV2 frozen)")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    n_steps_total = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps_total)

    # ── train ─────────────────────────────────────────────────────────────────
    history: list[dict] = []
    best_val = float("inf")
    print(f"\n{'epoch':>5} {'train':>10} {'val':>10} {'val_chamf':>10} {'lr':>10} {'sec':>6}")

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        t_start = time.time()
        train_losses = []
        for batch in train_loader:
            imgs = batch["image"].to(device)
            tgts = batch["points"].to(device)
            out = model(imgs)
            loss, _ = compute_loss(out["verts"], tgts, model.template_edges, laplacian,
                                    w_chamfer=args.w_chamfer, w_edge=args.w_edge, w_lap=args.w_lap)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            sched.step()
            train_losses.append(float(loss.detach()))

        # Val
        model.eval()
        val_losses, val_chamfers = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                tgts = batch["points"].to(device)
                out = model(imgs)
                loss, comp = compute_loss(out["verts"], tgts, model.template_edges, laplacian,
                                           w_chamfer=args.w_chamfer, w_edge=args.w_edge, w_lap=args.w_lap)
                val_losses.append(float(loss))
                val_chamfers.append(comp["chamfer"])

        tr_avg  = float(np.mean(train_losses))
        val_avg = float(np.mean(val_losses))
        val_chamf = float(np.mean(val_chamfers))
        lr_now = opt.param_groups[0]["lr"]
        elapsed = time.time() - t_start

        history.append({"epoch": epoch, "train": tr_avg, "val": val_avg,
                        "val_chamfer": val_chamf, "lr": lr_now, "sec": elapsed})
        print(f"{epoch:>5d} {tr_avg:>10.4f} {val_avg:>10.4f} {val_chamf:>10.4f} "
              f"{lr_now:>10.2e} {elapsed:>6.1f}")

        # Checkpoint
        torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                   args.out_dir / "last.pt")
        if val_avg < best_val:
            best_val = val_avg
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       args.out_dir / "best.pt")

        # Preview
        if epoch % args.preview_every == 0 or epoch == args.epochs:
            save_val_preview(model, val_loader, device,
                             args.out_dir / f"preview_e{epoch:03d}.png")

        with open(args.out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── final loss-curve plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train"] for h in history], label="train", color="#aa3355")
    ax.plot(epochs, [h["val"]   for h in history], label="val",   color="#3355aa")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"best val: {best_val:.4f}")
    fig.savefig(args.out_dir / "loss_curve.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDone. Best val loss = {best_val:.4f}")
    print(f"Artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
