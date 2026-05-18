"""Distributed (DDP) training for the DinoV2 + GCN student on CPU.

Targets a 2-Mac cluster connected over LAN. Uses the gloo backend (the only one
that works for CPU). MPS / CUDA are intentionally not used here — DDP on Apple
GPU is not supported.

Launched by torchrun on each machine; see scripts/launch_ddp_node{0,1}.sh.

Notes on multi-node CPU DDP:
- Each rank holds a full copy of the model. Gradients are all-reduced over the
  network every step (gloo). The 2 macs must reach each other over TCP on
  MASTER_PORT.
- Per-rank batch size is `--batch-size`; effective global batch is
  batch_size * world_size.
- Rank 0 owns all I/O: checkpoints, history.json, previews, tqdm bars.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI on remote workers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "teacher"))

from student.dataset import TeacherCacheDataset
from student.loss import compute_loss
from student.model import Student
from student.template import build_template
from student.train import collect_val_samples, render_mesh, render_points  # reuse helpers


# ── DDP setup ─────────────────────────────────────────────────────────────────

def setup_ddp() -> tuple[int, int, int]:
    """Initialize the process group from torchrun-provided env vars.

    Returns (rank, local_rank, world_size).
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="gloo",  # CPU-compatible; nccl is GPU-only
        rank=rank,
        world_size=world_size,
    )
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ── preview (rank 0 only) ─────────────────────────────────────────────────────

def save_preview_rank0(model: torch.nn.Module, loader: DataLoader, device: str,
                       out_path: Path, n_show: int = 6) -> None:
    """Render val previews on the local rank-0 model only."""
    inner = model.module if isinstance(model, DDP) else model
    collected = collect_val_samples(inner, loader, device, n_show=n_show)
    faces = inner.template_faces.cpu().numpy()

    fig = plt.figure(figsize=(3 * n_show, 9))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
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


# ── all-reduce helpers ────────────────────────────────────────────────────────

def all_reduce_mean(value: float, world_size: int) -> float:
    """Average a scalar loss across all ranks."""
    t = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache",        type=Path, default=ROOT / "data/teacher_cache")
    p.add_argument("--categories",   nargs="+", default=["chair", "sofa"])
    p.add_argument("--out-dir",      type=Path, default=ROOT / "runs/student_ddp")
    p.add_argument("--epochs",       type=int, default=10)
    p.add_argument("--batch-size",   type=int, default=4,
                   help="Per-rank batch size; global batch = batch_size * world_size")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split",    type=float, default=0.1)
    p.add_argument("--hidden",       type=int, default=128)
    p.add_argument("--n-stages",     type=int, default=1)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--num-workers",  type=int, default=0)
    p.add_argument("--preview-every", type=int, default=1)
    p.add_argument("--w-chamfer",    type=float, default=1.0)
    p.add_argument("--w-edge",       type=float, default=0.01)
    p.add_argument("--w-lap",        type=float, default=0.05)
    args = p.parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────────
    rank, local_rank, world_size = setup_ddp()
    device = "cpu"  # CPU-only multi-Mac cluster

    torch.manual_seed(args.seed + rank)  # different shuffles per rank, same model init
    np.random.seed(args.seed + rank)

    if is_main(rank):
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.out_dir / "config.json", "w") as f:
            json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      f, indent=2)
        print(f"[rank {rank}/{world_size}] world_size={world_size}  device={device}")
        print(f"[rank {rank}/{world_size}] cache={args.cache}  cats={args.categories}")
        print(f"[rank {rank}/{world_size}] out_dir={args.out_dir}")

    # ── data ──────────────────────────────────────────────────────────────────
    # Important: all ranks must build IDENTICAL train/val splits. Use a fixed
    # generator seeded the same on every rank, then wrap with DistributedSampler
    # which shards the train set.
    full_ds = TeacherCacheDataset(cache_root=args.cache, categories=tuple(args.categories))
    n_val = max(1, int(round(len(full_ds) * args.val_split)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),  # SAME on every rank
    )
    if is_main(rank):
        print(f"[rank {rank}] dataset: {len(full_ds)} → train {n_train} / val {n_val}")
        print(f"[rank {rank}] per-rank batch={args.batch_size}  "
              f"global batch={args.batch_size * world_size}")

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=args.seed, drop_last=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, drop_last=True,
    )
    # Validation: rank 0 only evaluates the whole val set for stable comparable numbers.
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    tpl = build_template()
    tpl.verts = tpl.verts - tpl.verts.mean(dim=0, keepdim=True)
    tpl.verts = tpl.verts / tpl.verts.norm(dim=1).max()

    model = Student(template=tpl, hidden=args.hidden, n_stages=args.n_stages).to(device)
    laplacian = tpl.laplacian.to(device)

    # Broadcast initial weights so every rank starts identically (defensive;
    # they should already match if the seed is identical).
    for p_ in model.parameters():
        dist.broadcast(p_.data, src=0)

    # find_unused_parameters=True because the frozen DinoV2 encoder has no
    # gradient flow into its parameters in every forward — gloo without this
    # will hang on the all-reduce.
    ddp_model = DDP(model, find_unused_parameters=True)

    trainable = [p_ for p_ in ddp_model.parameters() if p_.requires_grad]
    if is_main(rank):
        n_train_p = sum(p_.numel() for p_ in trainable)
        print(f"[rank {rank}] trainable params: {n_train_p / 1e6:.2f}M (DinoV2 frozen)")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    n_steps_total = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps_total)

    # ── train loop ────────────────────────────────────────────────────────────
    history: list[dict] = []
    best_val = float("inf")

    if is_main(rank):
        print()
    epoch_iter = range(1, args.epochs + 1)
    if is_main(rank):
        epoch_iter = tqdm(epoch_iter, desc="epochs", unit="ep", position=0)

    for epoch in epoch_iter:
        train_sampler.set_epoch(epoch)  # reshuffle differently each epoch
        ddp_model.train()
        t_start = time.time()
        train_losses: list[float] = []

        if is_main(rank):
            batch_iter = tqdm(train_loader, desc=f"train e{epoch:03d}",
                              unit="batch", position=1, leave=False)
        else:
            batch_iter = train_loader

        for batch in batch_iter:
            imgs = batch["image"].to(device)
            tgts = batch["points"].to(device)
            out = ddp_model(imgs)
            loss, _ = compute_loss(
                out["verts"], tgts, model.template_edges, laplacian,
                w_chamfer=args.w_chamfer, w_edge=args.w_edge, w_lap=args.w_lap,
            )
            opt.zero_grad()
            loss.backward()  # DDP all-reduces grads here
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            sched.step()
            train_losses.append(float(loss.detach()))
            if is_main(rank):
                batch_iter.set_postfix(  # type: ignore[union-attr]
                    loss=f"{train_losses[-1]:.4f}",
                    lr=f"{opt.param_groups[0]['lr']:.2e}",
                )

        # Per-rank mean train loss, then global all-reduce
        tr_local = float(np.mean(train_losses)) if train_losses else 0.0
        tr_global = all_reduce_mean(tr_local, world_size)

        # ── validation (rank 0 only, broadcast result) ────────────────────────
        val_avg = 0.0
        val_chamf = 0.0
        if is_main(rank):
            ddp_model.eval()
            val_losses: list[float] = []
            val_chamfers: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    tgts = batch["points"].to(device)
                    out = ddp_model(imgs)
                    loss, comp = compute_loss(
                        out["verts"], tgts, model.template_edges, laplacian,
                        w_chamfer=args.w_chamfer, w_edge=args.w_edge, w_lap=args.w_lap,
                    )
                    val_losses.append(float(loss))
                    val_chamfers.append(comp["chamfer"])
            val_avg = float(np.mean(val_losses))
            val_chamf = float(np.mean(val_chamfers))

        # Broadcast scalar val metrics from rank 0 so all ranks agree on
        # best-val state (only rank 0 saves, but this keeps the loop coherent).
        val_tensor = torch.tensor([val_avg, val_chamf], dtype=torch.float64)
        dist.broadcast(val_tensor, src=0)
        val_avg, val_chamf = float(val_tensor[0]), float(val_tensor[1])

        lr_now = opt.param_groups[0]["lr"]
        elapsed = time.time() - t_start

        if is_main(rank):
            history.append({"epoch": epoch, "train": tr_global, "val": val_avg,
                            "val_chamfer": val_chamf, "lr": lr_now, "sec": elapsed})
            tqdm.write(f"e{epoch:03d} | train {tr_global:.4f} | val {val_avg:.4f} | "
                       f"chamf {val_chamf:.4f} | lr {lr_now:.2e} | {elapsed:.1f}s")
            epoch_iter.set_postfix(  # type: ignore[union-attr]
                train=f"{tr_global:.4f}", val=f"{val_avg:.4f}", chamf=f"{val_chamf:.4f}",
            )

            # Checkpoints (save the underlying module, not the DDP wrapper)
            state = {"model": ddp_model.module.state_dict(),
                     "epoch": epoch, "args": vars(args)}
            torch.save(state, args.out_dir / "last.pt")
            if val_avg < best_val:
                best_val = val_avg
                torch.save(state, args.out_dir / "best.pt")

            if epoch % args.preview_every == 0 or epoch == args.epochs:
                save_preview_rank0(ddp_model, val_loader, device,
                                   args.out_dir / f"preview_e{epoch:03d}.png")

            with open(args.out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        # Barrier so workers wait for rank 0 to finish I/O before the next epoch
        dist.barrier()

    # ── final plot (rank 0 only) ──────────────────────────────────────────────
    if is_main(rank):
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs = [h["epoch"] for h in history]
        ax.plot(epochs, [h["train"] for h in history], label="train", color="#aa3355")
        ax.plot(epochs, [h["val"]   for h in history], label="val",   color="#3355aa")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"best val: {best_val:.4f} (DDP, world_size={world_size})")
        fig.savefig(args.out_dir / "loss_curve.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[rank 0] Done. Best val loss = {best_val:.4f}")
        print(f"[rank 0] Artifacts in {args.out_dir}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
