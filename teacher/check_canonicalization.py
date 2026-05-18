"""Render TripoSR meshes after applying rot_mat.T to test canonicalization.

If all canonicalized meshes face the same direction in 'front' view, the scheme works.
If they face different directions, TripoSR's frame is not Pix3D's camera frame and we
need a different alignment strategy.

Also tries a few common axis-swap corrections in case there's a constant offset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.gridspec import GridSpec
from render_previews import _cull_and_shade

# Candidate "global" corrections to try if direct rot_mat.T fails.
# Each is a 3x3 rotation applied AFTER rot_mat.T.
CORRECTIONS = {
    "identity": np.eye(3),
    "swap_yz": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float),
    "flip_180_y": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float),
    "neg_yz_swap": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
}


def render_panel(
    verts: np.ndarray,
    faces: np.ndarray,
    ax,
    bounds: np.ndarray,
    azim: float = 0,
    elev: float = 10,
) -> None:
    front_faces, shaded = _cull_and_shade(verts, faces, azim, elev)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.add_collection3d(
        Poly3DCollection(verts[front_faces], facecolors=shaded, edgecolors="none")
    )
    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 1], bounds[1, 1])
    ax.set_zlim(bounds[0, 2], bounds[1, 2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--meshes-dir", type=Path, default=Path("data/teacher_meshes/chair_smoke")
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/teacher_meshes/canon_check.png")
    )
    args = p.parse_args()

    rows = sorted([d for d in args.meshes_dir.iterdir() if d.is_dir()])
    print(f"Found {len(rows)} mesh dirs")

    col_names = ["raw"] + list(CORRECTIONS)
    n_cols = len(col_names)
    fig = plt.figure(figsize=(n_cols * 2.2, len(rows) * 2.2))
    gs = GridSpec(len(rows), n_cols, figure=fig, wspace=0.02, hspace=0.05)

    # Pass 1: compute per-correction global bounds across all meshes (so scale is shared)
    bounds_by_col: dict[str, list[np.ndarray]] = {c: [] for c in col_names}
    cached_verts: dict[str, list[np.ndarray]] = {c: [] for c in col_names}
    cached_faces: list[np.ndarray] = []

    for d in rows:
        m = trimesh.load(d / "mesh.obj", process=False)
        verts = np.asarray(m.vertices)
        faces = np.asarray(m.faces)
        cached_faces.append(faces)

        cached_verts["raw"].append(verts)
        bounds_by_col["raw"].append(np.stack([verts.min(0), verts.max(0)]))

        with open(d / "meta.json") as f:
            meta = json.load(f)
        R = np.asarray(meta["rot_mat"], dtype=float)
        canon = verts @ R  # equivalent to (R.T @ v.T).T

        for cname, C in CORRECTIONS.items():
            v = canon @ C.T
            cached_verts[cname].append(v)
            bounds_by_col[cname].append(np.stack([v.min(0), v.max(0)]))

    shared_bounds = {
        c: np.stack(
            [
                np.min([b[0] for b in bounds_by_col[c]], axis=0),
                np.max([b[1] for b in bounds_by_col[c]], axis=0),
            ]
        )
        for c in col_names
    }

    for r, d in enumerate(rows):
        for c, cname in enumerate(col_names):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            render_panel(
                cached_verts[cname][r], cached_faces[r], ax, shared_bounds[cname]
            )
            if r == 0:
                ax.set_title(cname, fontsize=10)
            if c == 0:
                ax.text2D(
                    -0.1,
                    0.5,
                    d.name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    fontsize=7,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=110, bbox_inches="tight")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
