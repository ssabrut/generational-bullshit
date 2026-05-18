"""Canonical template mesh for the student GCN decoder.

Starts from an icosphere (642 verts, 1280 faces) and rescales to a chair-ish
ellipsoid in Pix3D's canonical frame (Y-up). The student predicts per-vertex
offsets from this template, in the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh


@dataclass
class Template:
    verts: torch.Tensor  # (V, 3) float — initial template positions
    faces: torch.Tensor  # (F, 3) long  — triangle indices
    edge_index: torch.Tensor  # (2, E)  long — undirected edges (both directions)
    laplacian: (
        torch.Tensor
    )  # (V, V)  sparse float — graph Laplacian (for smoothness loss)


def _edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Return unique undirected edges from a triangle mesh as a (2,E) array (both dirs)."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e_sorted = np.sort(e, axis=1)
    unique = np.unique(e_sorted, axis=0)
    bidir = np.concatenate([unique, unique[:, ::-1]], axis=0).T
    return bidir.astype(np.int64)


def _graph_laplacian(n_verts: int, edge_index: np.ndarray) -> torch.Tensor:
    """Sparse graph Laplacian L = D - A, on N×N. Used for the smoothness loss."""
    src, dst = edge_index
    # Adjacency (sparse)
    vals = np.ones(len(src), dtype=np.float32)
    A = torch.sparse_coo_tensor(
        torch.from_numpy(edge_index),
        torch.from_numpy(vals),
        (n_verts, n_verts),
    ).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    # L = D - A as a dense (N,N) — N=642 is small enough
    L = -A.to_dense()
    L[torch.arange(n_verts), torch.arange(n_verts)] += deg
    return L


def build_template(
    subdivisions: int = 3,
    scale: tuple[float, float, float] = (0.30, 0.40, 0.30),
) -> Template:
    """Build the icosphere template, scaled to chair-ish proportions in Pix3D frame.

    Pix3D canonical chair bbox is roughly X:±0.30, Y:±0.40, Z:±0.30 (chair height ≈ 80cm).
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    verts = np.asarray(mesh.vertices, dtype=np.float32) * np.asarray(
        scale, dtype=np.float32
    )
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edge_index = _edges_from_faces(faces)
    L = _graph_laplacian(len(verts), edge_index)

    return Template(
        verts=torch.from_numpy(verts),
        faces=torch.from_numpy(faces),
        edge_index=torch.from_numpy(edge_index),
        laplacian=L,
    )
