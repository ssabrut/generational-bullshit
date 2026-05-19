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
    d = torch.cdist(pred, target, p=2)  # (B, V, N)
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
    diffs = verts[:, s, :] - verts[:, d, :]  # (B, E/2, 3)
    return (diffs**2).sum(dim=-1).mean()


def laplacian_smoothness(verts: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """Encourage each vertex to lie near the mean of its neighbours.

    Loss = mean ‖L @ verts‖²  where L = D - A is the (dense) graph Laplacian.
    """
    # verts: (B, V, 3), laplacian: (V, V)
    lv = torch.einsum("vw,bwd->bvd", laplacian, verts)
    return (lv**2).sum(dim=-1).mean()


def color_l1_loss(
    pred_verts: torch.Tensor,
    pred_colors: torch.Tensor,
    color_pts: torch.Tensor,
    color_rgb: torch.Tensor,
    has_color: torch.Tensor,
) -> torch.Tensor:
    """L1 between predicted per-vertex RGB and the nearest target sample's RGB.

    pred_verts: (B, V, 3), pred_colors: (B, V, 3)
    color_pts:  (B, M, 3) — colored surface samples in the target frame
    color_rgb:  (B, M, 3) in [0, 1]
    has_color:  (B,) bool — which entries actually have texture; others get 0.
    """
    if not has_color.any():
        return pred_colors.new_zeros(())
    d = torch.cdist(pred_verts, color_pts, p=2)  # (B, V, M)
    nn_idx = d.argmin(dim=-1)  # (B, V)
    # gather: tgt_colors[b, v] = color_rgb[b, nn_idx[b, v]]
    tgt = torch.gather(color_rgb, 1, nn_idx.unsqueeze(-1).expand(-1, -1, 3))
    per_sample = (pred_colors - tgt).abs().mean(dim=(1, 2))  # (B,)
    mask = has_color.to(per_sample.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (per_sample * mask).sum() / denom


def compute_loss(
    pred_verts: torch.Tensor,
    target_pts: torch.Tensor,
    edge_index: torch.Tensor,
    laplacian: torch.Tensor,
    w_chamfer: float = 1.0,
    w_edge: float = 0.1,
    w_lap: float = 0.5,
    pred_colors: torch.Tensor | None = None,
    color_pts: torch.Tensor | None = None,
    color_rgb: torch.Tensor | None = None,
    has_color: torch.Tensor | None = None,
    w_color: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Return (total_loss, components_dict).

    Color supervision is opt-in: pass pred_colors + color_{pts,rgb} + has_color
    and w_color > 0 to enable it. Without those, behavior is unchanged.
    """
    chamfer = chamfer_distance(pred_verts, target_pts)
    edge = edge_length_reg(pred_verts, edge_index)
    lap = laplacian_smoothness(pred_verts, laplacian)
    total = w_chamfer * chamfer + w_edge * edge + w_lap * lap
    comps = {
        "chamfer": float(chamfer.detach()),
        "edge": float(edge.detach()),
        "lap": float(lap.detach()),
    }
    if (
        w_color > 0
        and pred_colors is not None
        and color_pts is not None
        and color_rgb is not None
        and has_color is not None
    ):
        col = color_l1_loss(pred_verts, pred_colors, color_pts, color_rgb, has_color)
        total = total + w_color * col
        comps["color"] = float(col.detach())
    return total, comps
