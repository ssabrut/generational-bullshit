"""Render TripoSR meshes from 4 fixed viewpoints in the SAME world frame.

This is for visually checking canonicalization: if the meshes all come out
upright and facing the same way, no extra pose alignment is needed; if not,
we need to add an alignment step before using them as student supervision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

VIEWS = [
    ("front", 0, 10),
    ("side", 90, 10),
    ("back", 180, 10),
    ("top-3/4", 45, 35),
]


def _view_dir(azim: float, elev: float) -> np.ndarray:
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _cull_and_shade(
    verts: np.ndarray, faces: np.ndarray, azim: float, elev: float
) -> tuple[np.ndarray, np.ndarray]:
    """Back-face cull, then Lambertian shade the survivors."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    view = _view_dir(azim, elev)
    front = (n @ view) > 0
    faces, n = faces[front], n[front]
    intensity = np.clip(n @ view, 0.2, 1.0)
    base = np.array([0.45, 0.5, 0.6])
    return faces, base[None, :] * intensity[:, None]


def render_one_view(
    verts: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,  # kept for signature compatibility; unused
    azim: float,
    elev: float,
    size_px: int = 240,
    ref_bounds: np.ndarray | None = None,
) -> np.ndarray:
    del face_colors
    fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    front_faces, shaded = _cull_and_shade(verts, faces, azim, elev)
    ax.add_collection3d(
        Poly3DCollection(verts[front_faces], facecolors=shaded, edgecolors="none")
    )
    b = ref_bounds if ref_bounds is not None else np.stack([verts.min(0), verts.max(0)])
    ax.set_xlim(b[0, 0], b[1, 0])
    ax.set_ylim(b[0, 1], b[1, 1])
    ax.set_zlim(b[0, 2], b[1, 2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img


def render_mesh(
    mesh_path: Path,
    ref_bounds: np.ndarray | None = None,
) -> list[np.ndarray]:
    mesh = trimesh.load(mesh_path, process=False)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    return [
        render_one_view(verts, faces, None, az, el, ref_bounds=ref_bounds)
        for _, az, el in VIEWS
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--meshes-dir", type=Path, default=Path("data/teacher_meshes/chair_smoke")
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/teacher_meshes/chair_smoke_preview.png"),
    )
    p.add_argument(
        "--shared-bounds",
        action="store_true",
        help="Use a single world bbox across all rows (shows scale variation across instances)",
    )
    args = p.parse_args()

    rows = sorted([d for d in args.meshes_dir.iterdir() if d.is_dir()])
    print(f"Found {len(rows)} mesh dirs")

    ref_bounds = None
    if args.shared_bounds:
        all_v = []
        for d in rows:
            m = trimesh.load(d / "mesh.obj", process=False)
            all_v.append(np.asarray(m.vertices))
        all_v = np.concatenate(all_v, axis=0)
        ref_bounds = np.stack([all_v.min(0), all_v.max(0)])

    n_cols = 1 + len(VIEWS)
    fig = plt.figure(figsize=(n_cols * 2.4, len(rows) * 2.4))
    gs = GridSpec(len(rows), n_cols, figure=fig, wspace=0.02, hspace=0.05)

    for r, d in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(Image.open(d / "input.png"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(d.name, fontsize=8)
        if r == 0:
            ax.set_title("input", fontsize=9)

        views = render_mesh(d / "mesh.obj", ref_bounds=ref_bounds)
        for c, (name, _, _) in enumerate(VIEWS, start=1):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(views[c - 1])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Saved preview → {args.output}")


if __name__ == "__main__":
    main()
