"""ICP-based alignment of TripoSR meshes to their Pix3D ground-truth meshes.

Strategy (empirically validated on 10 chairs):
  1. Apply a fixed axis swap: TripoSR is Z-up → Pix3D is Y-up.
     Swap: (x, y, z) → (x, z, -y)
  2. Try 4 azimuth initializations around Y (0°, 90°, 180°, 270°).
  3. ICP refine each; keep the lowest-cost result.

Returns 4x4 matrix M such that apply_transform(verts, M) puts the mesh
in Pix3D's canonical frame (X-right, Y-up, Z-front).
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# Fixed Z-up → Y-up swap validated on 10 chairs: argmax(extent) = Y after swap for all.
_SWAP_ZY = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

# 4 azimuths around Y axis (0°, 90°, 180°, 270°) as 3x3 rotation matrices.
_AZ_ROTS: list[np.ndarray] = []
for _k in range(4):
    _a = _k * np.pi / 2
    _AZ_ROTS.append(
        np.array([[np.cos(_a), 0, np.sin(_a)],
                  [0,          1, 0         ],
                  [-np.sin(_a),0, np.cos(_a)]], dtype=float)
    )


def _procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Similarity transform (scale + rot + trans) so T@A ≈ B. Returns 4x4 matrix."""
    mu_a, mu_b = A.mean(0), B.mean(0)
    A0, B0 = A - mu_a, B - mu_b
    H = A0.T @ B0
    U, sv, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    var_a = float((A0 ** 2).sum())
    s = float(np.trace(np.diag(sv) @ D) / var_a) if var_a > 0 else 1.0
    t = mu_b - s * R @ mu_a
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def _icp_once(src: np.ndarray, tgt: np.ndarray, n_iter: int = 40) -> tuple[np.ndarray, float]:
    """Single-init similarity-ICP. Returns (4x4 cumulative transform, mean NN dist)."""
    tree = cKDTree(tgt)
    M_cum = np.eye(4)
    pts = src.copy()
    cost = np.inf
    for _ in range(n_iter):
        d, idx = tree.query(pts)
        T = _procrustes(pts, tgt[idx])
        pts = (T[:3, :3] @ pts.T).T + T[:3, 3]
        M_cum = T @ M_cum
        new_cost = float(np.mean(d))
        if abs(cost - new_cost) < 1e-7:
            break
        cost = new_cost
    return M_cum, cost


def _sample(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator) -> np.ndarray:
    seed = int(rng.integers(1 << 30))
    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    if len(pts) < n // 2:
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=float)


def align_to_target(
    src_mesh: trimesh.Trimesh,
    tgt_mesh: trimesh.Trimesh,
    n_samples: int = 1500,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Find M s.t. apply_transform(src_verts, M) puts src into Pix3D canonical frame.

    Tries 4 azimuth initializations after a fixed Z→Y axis swap, returns best by cost.
    Cost = mean nearest-neighbour distance after alignment (lower = better fit).
    """
    rng = np.random.default_rng(seed)
    src_pts = _sample(src_mesh, n_samples, rng)
    tgt_pts = _sample(tgt_mesh, n_samples, rng)

    # Step 1: apply fixed Z-up → Y-up swap
    src_swapped = src_pts @ _SWAP_ZY.T
    src_c = src_swapped.mean(0)
    tgt_c = tgt_pts.mean(0)

    best_M, best_cost = None, np.inf
    for R_az in _AZ_ROTS:
        # Initial placement: azimuth rotation + centroid match
        M_init = np.eye(4)
        R_full = R_az @ _SWAP_ZY           # combined: swap then azimuth
        M_init[:3, :3] = R_full
        M_init[:3, 3] = tgt_c - R_az @ src_c

        warmed = (R_az @ src_swapped.T).T + M_init[:3, 3]
        M_refine, cost = _icp_once(warmed, tgt_pts, n_iter=40)
        M_total = np.eye(4)
        M_total[:3, :3] = M_refine[:3, :3] @ R_full
        M_total[:3, 3] = M_refine[:3, :3] @ M_init[:3, 3] + M_refine[:3, 3]

        if cost < best_cost:
            best_cost, best_M = cost, M_total

    return best_M, best_cost


def apply_transform(verts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 4x4 similarity transform (upper-left 3x3 = s*R, last col = t)."""
    return (M[:3, :3] @ verts.T).T + M[:3, 3]
