"""Diagnostic: are gradients actually flowing through cross-attention?

For each named parameter we print:
  - gradient L2 norm
  - parameter L2 norm
  - ratio (grad / param) — proxy for "how much would this change after one step"

If cross-attention gradients are orders of magnitude smaller than e.g. the offset
head, the image conditioning is weak — predictions will be image-agnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.dataset import TeacherCacheDataset
from student.loss import compute_loss
from student.model import Student
from student.template import build_template


def main() -> None:
    torch.manual_seed(0)
    device = "cpu"

    ds = TeacherCacheDataset(cache_root=ROOT / "data/teacher_cache", categories=("chair", "sofa"))
    print(f"dataset: {len(ds)} entries\n")

    # Build template and student
    tpl = build_template()
    tpl.verts = tpl.verts - tpl.verts.mean(dim=0, keepdim=True)
    tpl.verts = tpl.verts / tpl.verts.norm(dim=1).max()
    model = Student(template=tpl, hidden=128, n_stages=1).to(device)
    laplacian = tpl.laplacian.to(device)

    # Run a forward + backward on a batch of 4 samples
    imgs   = torch.stack([ds[i]["image"]  for i in range(4)]).to(device)
    points = torch.stack([ds[i]["points"] for i in range(4)]).to(device)
    out = model(imgs)
    loss, comp = compute_loss(out["verts"], points, model.template_edges, laplacian)
    print(f"loss: {float(loss):.4f}  components: {comp}\n")

    loss.backward()

    # Group params by where they live
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "encoder (DinoV2, frozen)": [],
        "embed (vert init)":        [],
        "cross-attn q":             [],
        "cross-attn k":             [],
        "cross-attn v":             [],
        "cross-attn out":           [],
        "gcn1":                     [],
        "gcn2":                     [],
        "offset head":              [],
    }
    for name, p in model.named_parameters():
        if "encoder" in name:
            groups["encoder (DinoV2, frozen)"].append((name, p))
        elif "embed" in name:
            groups["embed (vert init)"].append((name, p))
        elif "cross.q" in name:
            groups["cross-attn q"].append((name, p))
        elif "cross.k" in name:
            groups["cross-attn k"].append((name, p))
        elif "cross.v" in name:
            groups["cross-attn v"].append((name, p))
        elif "cross.o" in name:
            groups["cross-attn out"].append((name, p))
        elif "gcn1" in name:
            groups["gcn1"].append((name, p))
        elif "gcn2" in name:
            groups["gcn2"].append((name, p))
        elif "offset_head" in name:
            groups["offset head"].append((name, p))

    print(f"{'group':<30} {'#params':>10} {'param_norm':>12} {'grad_norm':>12} {'ratio':>10}")
    print("-" * 80)
    for gname, items in groups.items():
        n = sum(p.numel() for _, p in items)
        if n == 0:
            continue
        pn = sum(float(p.detach().norm() ** 2) for _, p in items) ** 0.5
        gn_sq = 0.0
        for _, p in items:
            if p.grad is None:
                continue
            gn_sq += float(p.grad.detach().norm() ** 2)
        gn = gn_sq ** 0.5
        ratio = gn / pn if pn > 0 else float("nan")
        print(f"{gname:<30} {n:>10d} {pn:>12.4f} {gn:>12.6f} {ratio:>10.4e}")

    # Also: gradient norm on the offset itself (after the head, before mesh)
    # We need to retain_grad on intermediates — easier to just check the deformation
    # is varying across batch:
    pred = out["verts"].detach()  # (B, V, 3)
    deltas = pred - tpl.verts.unsqueeze(0)
    per_sample_norms = deltas.reshape(pred.shape[0], -1).norm(dim=-1)
    print(f"\nPer-sample deformation norm: {per_sample_norms.tolist()}")
    print(f"  std across batch (higher = more image-conditioning): {per_sample_norms.std().item():.6f}")

    # Diff between predictions: how much do two different inputs differ in output?
    diff = (pred[0] - pred[1]).norm()
    diff_self = (pred[0] - pred[0]).norm()
    print(f"  ‖pred[0] - pred[1]‖ = {diff:.4f}   (sanity ‖pred[0] - pred[0]‖ = {diff_self:.4f})")


if __name__ == "__main__":
    main()
