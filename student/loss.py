"""Losses for fixed-template image-to-mesh distillation.

- chamfer_distance: primary supervision (pred verts ↔ teacher point cloud).
- edge_length_reg: penalize stretched/collapsed edges (prevents tearing).
- laplacian_smoothness: encourage local smoothness via graph Laplacian.

All inputs are batched: (B, V, 3) / (B, N, 3).
"""
from __future__ import annotations

import torch


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance, batched. Sum of mean NN distances in both directions."""
    # pred: (B, V, 3), target: (B, N, 3)
    d = torch.cdist(pred, target, p=2)        # (B, V, N)
    p_to_t = d.min(dim=-1).values.mean(dim=-1)  # (B,)
    t_to_p = d.min(dim=-2).values.mean(dim=-1)  # (B,)
    return (p_to_t + t_to_p).mean()


def edge_length_reg(verts: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean squared edge length — discourages mesh from stretching unboundedly.

    Uses only the forward half of the bidirectional edge_index to avoid double-counting.
    """
    src, dst = edge_index[0], edge_index[1]
    # Take only edges where src < dst (deduplicate the bidirectional pairs)
    keep = src < dst
    s, d = src[keep], dst[keep]
    diffs = verts[:, s, :] - verts[:, d, :]   # (B, E/2, 3)
    return (diffs ** 2).sum(dim=-1).mean()


def laplacian_smoothness(verts: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """Encourage each vertex to lie near the mean of its neighbours.

    Loss = mean ‖L @ verts‖²  where L = D - A is the (dense) graph Laplacian.
    """
    # verts: (B, V, 3), laplacian: (V, V)
    lv = torch.einsum("vw,bwd->bvd", laplacian, verts)
    return (lv ** 2).sum(dim=-1).mean()


def compute_loss(
    pred_verts: torch.Tensor,
    target_pts: torch.Tensor,
    edge_index: torch.Tensor,
    laplacian: torch.Tensor,
    w_chamfer: float = 1.0,
    w_edge:    float = 0.1,
    w_lap:     float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Return (total_loss, components_dict)."""
    chamfer = chamfer_distance(pred_verts, target_pts)
    edge    = edge_length_reg(pred_verts, edge_index)
    lap     = laplacian_smoothness(pred_verts, laplacian)
    total   = w_chamfer * chamfer + w_edge * edge + w_lap * lap
    return total, {"chamfer": float(chamfer), "edge": float(edge), "lap": float(lap)}
