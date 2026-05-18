"""Run ICP-to-GT alignment on the smoke-test meshes and render the result.

If all 10 canonicalized meshes face the same way in the 'front' view, the
alignment scheme is good and we can use it for the full cache.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from align import align_to_target, apply_transform
from render_previews import _cull_and_shade

# Canonical frame is Pix3D Y-up; matplotlib is Z-up by default.
# Swap Y↔Z (and negate new-Z to keep right-handedness) before passing to matplotlib.
_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)

VIEWS = [("front", 0, 10), ("side", 90, 10), ("back", 180, 10), ("top-3/4", 45, 35)]


def render_panel(verts, faces, ax, bounds, azim, elev):
    verts = verts @ _YUP_TO_ZUP.T
    bounds = np.stack([verts.min(0), verts.max(0)])
    front_faces, shaded = _cull_and_shade(verts, faces, azim, elev)
    ax.add_collection3d(
        Poly3DCollection(verts[front_faces], facecolors=shaded, edgecolors="none")
    )
    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 1], bounds[1, 1])
    ax.set_zlim(bounds[0, 2], bounds[1, 2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main():
    smoke_dir = Path("data/teacher_meshes/chair_smoke")
    pix3d_root = Path("data/pix3d")
    out_path = Path("data/teacher_meshes/chair_canonical_preview.png")

    rows = sorted([d for d in smoke_dir.iterdir() if d.is_dir()])
    aligned_meshes = []
    costs = []
    for d in rows:
        with open(d / "meta.json") as f:
            meta = json.load(f)
        src = trimesh.load(d / "mesh.obj", process=False)
        tgt = trimesh.load(pix3d_root / meta["model"], process=False, force="mesh")
        t0 = time.time()
        M, cost = align_to_target(src, tgt)
        elapsed = time.time() - t0
        aligned_verts = apply_transform(np.asarray(src.vertices), M)
        aligned_meshes.append((aligned_verts, np.asarray(src.faces)))
        costs.append(cost)
        print(f"{d.name}: cost={cost:.4f}, {elapsed:.2f}s")

    per_bounds = [np.stack([v.min(0), v.max(0)]) for v, _ in aligned_meshes]

    n_cols = 1 + len(VIEWS)
    fig = plt.figure(figsize=(n_cols * 2.4, len(rows) * 2.4))
    gs = GridSpec(len(rows), n_cols, figure=fig, wspace=0.02, hspace=0.05)

    for r, (d, (verts, faces)) in enumerate(zip(rows, aligned_meshes)):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(Image.open(d / "input.png"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"{d.name}\ncost={costs[r]:.3f}", fontsize=7)
        if r == 0:
            ax.set_title("input", fontsize=9)
        for c, (name, az, el) in enumerate(VIEWS, start=1):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            render_panel(verts, faces, ax, per_bounds[r], az, el)
            if r == 0:
                ax.set_title(name, fontsize=9)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    print(
        f"costs: mean={np.mean(costs):.4f}, max={np.max(costs):.4f}, min={np.min(costs):.4f}"
    )


if __name__ == "__main__":
    main()
