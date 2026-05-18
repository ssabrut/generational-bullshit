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
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    ax.add_collection3d(
        Poly3DCollection(v[front], facecolors=shaded, edgecolors="none")
    )
    b = np.stack([v.min(0), v.max(0)])
    ax.set_xlim(b[0, 0], b[1, 0])
    ax.set_ylim(b[0, 1], b[1, 1])
    ax.set_zlim(b[0, 2], b[1, 2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.set_axis_off()


def render_points(pts: np.ndarray, ax, azim=30, elev=15) -> None:
    p = pts @ _YUP_TO_ZUP.T
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.4, c="#3355aa", alpha=0.5)
    b = np.stack([p.min(0), p.max(0)])
    ax.set_xlim(b[0, 0], b[1, 0])
    ax.set_ylim(b[0, 1], b[1, 1])
    ax.set_zlim(b[0, 2], b[1, 2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.set_axis_off()


def collect_val_samples(
    model: Student, loader: DataLoader, device: str, n_show: int = 6
) -> list[dict]:
    model.eval()
    collected: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            tgts = batch["points"]
            out = model(imgs)
            verts = out["verts"].cpu().numpy()
            for b in range(imgs.shape[0]):
                collected.append(
                    {
                        "image": batch["image"][b].permute(1, 2, 0).numpy(),
                        "target": tgts[b].numpy(),
                        "pred": verts[b],
                        "entry": batch["entry_id"][b],
                    }
                )
                if len(collected) >= n_show:
                    break
            if len(collected) >= n_show:
                break
    model.train()
    return collected


def render_live_dashboard(
    fig,
    collected: list[dict],
    faces: np.ndarray,
    history: list[dict],
    epoch: int,
    n_show: int = 6,
) -> None:
    """Refresh the persistent dashboard figure in place."""
    fig.clear()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Top 3 rows: input | target | prediction for n_show samples
    # Bottom row spans all columns: live loss curve
    n_rows = 4
    for i, c in enumerate(collected):
        ax_img = fig.add_subplot(n_rows, n_show, i + 1)
        img = np.clip(c["image"] * std + mean, 0, 1)
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(c["entry"][:14], fontsize=7)

        ax_tgt = fig.add_subplot(n_rows, n_show, n_show + i + 1, projection="3d")
        render_points(c["target"], ax_tgt)
        if i == 0:
            ax_tgt.set_title("target", fontsize=8)

        ax_pred = fig.add_subplot(n_rows, n_show, 2 * n_show + i + 1, projection="3d")
        render_mesh(c["pred"], faces, ax_pred)
        if i == 0:
            ax_pred.set_title(f"pred (e{epoch})", fontsize=8)

    # Loss curve across the full bottom row
    ax_loss = fig.add_subplot(n_rows, 1, 4)
    epochs = [h["epoch"] for h in history]
    ax_loss.plot(
        epochs,
        [h["train"] for h in history],
        label="train",
        color="#aa3355",
        marker="o",
        markersize=3,
    )
    ax_loss.plot(
        epochs,
        [h["val"] for h in history],
        label="val",
        color="#3355aa",
        marker="o",
        markersize=3,
    )
    ax_loss.plot(
        epochs,
        [h["val_chamfer"] for h in history],
        label="val chamfer",
        color="#33aa55",
        linestyle="--",
        alpha=0.7,
    )
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(alpha=0.3)
    ax_loss.set_title(
        f"epoch {epoch} | val={history[-1]['val']:.4f} | "
        f"chamf={history[-1]['val_chamfer']:.4f}",
        fontsize=9,
    )

    fig.tight_layout()


def save_val_preview(
    model: Student,
    loader: DataLoader,
    device: str,
    out_path: Path,
    history: list[dict],
    epoch: int,
    live_fig=None,
    n_show: int = 6,
) -> None:
    collected = collect_val_samples(model, loader, device, n_show=n_show)
    faces = model.template_faces.cpu().numpy()

    # Snapshot PNG (single-epoch view, no loss curve)
    snap = plt.figure(figsize=(3 * n_show, 9))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i, c in enumerate(collected):
        ax_img = snap.add_subplot(3, n_show, i + 1)
        img = np.clip(c["image"] * std + mean, 0, 1)
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(c["entry"][:14], fontsize=7)

        ax_tgt = snap.add_subplot(3, n_show, n_show + i + 1, projection="3d")
        render_points(c["target"], ax_tgt)
        if i == 0:
            ax_tgt.set_title("target", fontsize=8)

        ax_pred = snap.add_subplot(3, n_show, 2 * n_show + i + 1, projection="3d")
        render_mesh(c["pred"], faces, ax_pred)
        if i == 0:
            ax_pred.set_title("prediction", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    snap.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(snap)

    # Live dashboard (preds + loss curve), saved AND shown
    if live_fig is not None:
        render_live_dashboard(live_fig, collected, faces, history, epoch, n_show=n_show)
        live_fig.savefig(
            out_path.parent / "live_dashboard.png", dpi=110, bbox_inches="tight"
        )
        live_fig.canvas.draw_idle()
        plt.pause(0.001)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=ROOT / "data/teacher_cache")
    p.add_argument("--categories", nargs="+", default=["chair", "sofa"])
    p.add_argument("--out-dir", type=Path, default=ROOT / "runs/student")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-stages", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--preview-every", type=int, default=1, help="Save preview every N epochs"
    )
    p.add_argument("--w-chamfer", type=float, default=1.0)
    p.add_argument(
        "--w-edge",
        type=float,
        default=0.01,
        help="Edge-length regularizer weight (was 0.1 in smoke test)",
    )
    p.add_argument(
        "--w-lap",
        type=float,
        default=0.05,
        help="Laplacian smoothness regularizer weight (was 0.5 in smoke test)",
    )
    p.add_argument(
        "--augment",
        action="store_true",
        help="Enable H-flip + color jitter on training set (val unchanged)",
    )
    p.add_argument("--hflip-prob", type=float, default=0.5)
    p.add_argument("--color-jitter", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15,
                   help="Early-stop after N epochs with no val improvement (0 = off)")
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            f,
            indent=2,
        )

    print(f"device       : {device}")
    print(f"cache        : {args.cache}")
    print(f"categories   : {args.categories}")
    print(f"out_dir      : {args.out_dir}")

    # ── data ──────────────────────────────────────────────────────────────────
    # Build two dataset views over the SAME cache: aug-on for train, aug-off for
    # val. Then split deterministically by index so the train/val sets stay
    # disjoint regardless of augmentation flag.
    train_full = TeacherCacheDataset(
        cache_root=args.cache,
        categories=tuple(args.categories),
        augment=args.augment,
        hflip_prob=args.hflip_prob,
        color_jitter=args.color_jitter,
    )
    val_full = TeacherCacheDataset(
        cache_root=args.cache,
        categories=tuple(args.categories),
        augment=False,
    )
    n_total = len(train_full)
    n_val = max(1, int(round(n_total * args.val_split)))
    n_train = n_total - n_val

    perm = torch.randperm(
        n_total, generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_ds = torch.utils.data.Subset(train_full, train_idx)
    val_ds = torch.utils.data.Subset(val_full, val_idx)
    print(
        f"dataset      : {n_total} entries → train {n_train} / val {n_val}"
        f"  (augment={args.augment})"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

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
    epochs_no_improve = 0

    # Persistent live figure (preds + loss curve), updated each epoch
    plt.ion()
    live_fig = plt.figure("student-live", figsize=(3 * 6, 12))
    plt.show(block=False)

    print()
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs", unit="ep", position=0)
    for epoch in epoch_bar:
        # Train
        model.train()
        t_start = time.time()
        train_losses = []
        train_bar = tqdm(
            train_loader,
            desc=f"train e{epoch:03d}",
            unit="batch",
            position=1,
            leave=False,
        )
        for batch in train_bar:
            imgs = batch["image"].to(device)
            tgts = batch["points"].to(device)
            out = model(imgs)
            loss, _ = compute_loss(
                out["verts"],
                tgts,
                model.template_edges,
                laplacian,
                w_chamfer=args.w_chamfer,
                w_edge=args.w_edge,
                w_lap=args.w_lap,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            sched.step()
            train_losses.append(float(loss.detach()))
            train_bar.set_postfix(
                loss=f"{train_losses[-1]:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}"
            )

        # Val
        model.eval()
        val_losses, val_chamfers = [], []
        val_bar = tqdm(
            val_loader,
            desc=f"  val e{epoch:03d}",
            unit="batch",
            position=1,
            leave=False,
        )
        with torch.no_grad():
            for batch in val_bar:
                imgs = batch["image"].to(device)
                tgts = batch["points"].to(device)
                out = model(imgs)
                loss, comp = compute_loss(
                    out["verts"],
                    tgts,
                    model.template_edges,
                    laplacian,
                    w_chamfer=args.w_chamfer,
                    w_edge=args.w_edge,
                    w_lap=args.w_lap,
                )
                val_losses.append(float(loss))
                val_chamfers.append(comp["chamfer"])
                val_bar.set_postfix(
                    loss=f"{val_losses[-1]:.4f}", chamf=f"{val_chamfers[-1]:.4f}"
                )

        tr_avg = float(np.mean(train_losses))
        val_avg = float(np.mean(val_losses))
        val_chamf = float(np.mean(val_chamfers))
        lr_now = opt.param_groups[0]["lr"]
        elapsed = time.time() - t_start

        history.append(
            {
                "epoch": epoch,
                "train": tr_avg,
                "val": val_avg,
                "val_chamfer": val_chamf,
                "lr": lr_now,
                "sec": elapsed,
            }
        )
        tqdm.write(
            f"e{epoch:03d} | train {tr_avg:.4f} | val {val_avg:.4f} | "
            f"chamf {val_chamf:.4f} | lr {lr_now:.2e} | {elapsed:.1f}s"
        )
        epoch_bar.set_postfix(
            train=f"{tr_avg:.4f}", val=f"{val_avg:.4f}", chamf=f"{val_chamf:.4f}"
        )

        # Checkpoint
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
            args.out_dir / "last.pt",
        )
        if val_avg < best_val:
            best_val = val_avg
            epochs_no_improve = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                args.out_dir / "best.pt",
            )
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                tqdm.write(
                    f"\nEarly stop: no val improvement for {args.patience} epochs "
                    f"(best {best_val:.4f} at epoch {epoch - args.patience})"
                )
                break

        # Preview (snapshot PNG + live dashboard refresh)
        if epoch % args.preview_every == 0 or epoch == args.epochs:
            save_val_preview(
                model,
                val_loader,
                device,
                args.out_dir / f"preview_e{epoch:03d}.png",
                history=history,
                epoch=epoch,
                live_fig=live_fig,
            )

        with open(args.out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── final loss-curve plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train"] for h in history], label="train", color="#aa3355")
    ax.plot(epochs, [h["val"] for h in history], label="val", color="#3355aa")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f"best val: {best_val:.4f}")
    fig.savefig(args.out_dir / "loss_curve.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    plt.ioff()
    plt.close(live_fig)

    print(f"\nDone. Best val loss = {best_val:.4f}")
    print(f"Artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
