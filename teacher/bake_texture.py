"""Convert a vertex-colored TripoSR mesh into a UV-mapped OBJ + MTL + PNG.

Steps:
  1. Load the vertex-colored mesh produced by TripoSR (mesh.obj with per-vertex RGB).
  2. UV-unwrap with xatlas.
  3. Bake vertex colors into a texture image by rasterizing UV triangles.
  4. Export OBJ (with `mtllib`), MTL (with `map_Kd`), and PNG.

Usage:
    python teacher/bake_texture.py path/to/dir_with_mesh.obj --tex-size 1024
    python teacher/bake_texture.py data/teacher_meshes/chair_smoke --recursive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
import xatlas
from PIL import Image


def unwrap(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run xatlas. Returns (vmapping, indices, uvs).

    vmapping[i] gives the original vertex index for the i-th unwrapped vertex,
    so vertex colors / positions must be re-indexed via vmapping before export.
    """
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices, mesh.faces)
    atlas.generate()
    vmapping, indices, uvs = atlas[0]
    return vmapping, indices, uvs


def bake_vertex_colors_to_texture(
    uvs: np.ndarray,
    indices: np.ndarray,
    vertex_colors: np.ndarray,
    tex_size: int,
) -> Image.Image:
    """Rasterize each UV triangle and interpolate vertex colors via barycentrics."""
    img = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    mask = np.zeros((tex_size, tex_size), dtype=bool)

    uv_px = uvs.copy()
    uv_px[:, 0] *= tex_size - 1
    uv_px[:, 1] = (1.0 - uv_px[:, 1]) * (tex_size - 1)

    for tri in indices:
        i0, i1, i2 = tri
        p0, p1, p2 = uv_px[i0], uv_px[i1], uv_px[i2]
        c0, c1, c2 = vertex_colors[i0], vertex_colors[i1], vertex_colors[i2]

        min_x = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
        max_x = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), tex_size - 1)
        min_y = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
        max_y = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), tex_size - 1)
        if max_x < min_x or max_y < min_y:
            continue

        xs, ys = np.meshgrid(
            np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1)
        )
        px = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

        v0 = p1 - p0
        v1 = p2 - p0
        v2 = px - p0
        d00 = v0 @ v0
        d01 = v0 @ v1
        d11 = v1 @ v1
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            continue
        d20 = v2 @ v0
        d21 = v2 @ v1
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        inside = (u >= -1e-4) & (v >= -1e-4) & (w >= -1e-4)
        if not inside.any():
            continue
        bary = np.stack([u[inside], v[inside], w[inside]], axis=1)
        colors = bary @ np.stack([c0, c1, c2], axis=0)

        sel_x = px[inside, 0].astype(np.int32)
        sel_y = px[inside, 1].astype(np.int32)
        img[sel_y, sel_x] = colors
        mask[sel_y, sel_x] = True

    # Dilate to fill seams: simple 1-px expand of valid texels.
    for _ in range(2):
        pad = np.pad(img, ((1, 1), (1, 1), (0, 0)))
        pad_m = np.pad(mask, ((1, 1), (1, 1)))
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shift = pad[1 + dy : 1 + dy + tex_size, 1 + dx : 1 + dx + tex_size]
            shift_m = pad_m[1 + dy : 1 + dy + tex_size, 1 + dx : 1 + dx + tex_size]
            fill = (~mask) & shift_m
            img[fill] = shift[fill]
            mask |= fill

    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def bake_one(obj_path: Path, tex_size: int, out_stem: str = "model") -> Path:
    mesh = trimesh.load(obj_path, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"{obj_path} did not load as a single mesh")
    if mesh.visual.kind != "vertex" or mesh.visual.vertex_colors is None:
        raise RuntimeError(
            f"{obj_path} has no vertex colors — re-run TripoSR with has_vertex_color=True"
        )

    vcols = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0

    vmapping, indices, uvs = unwrap(mesh)
    new_vertices = mesh.vertices[vmapping]
    new_vcols = vcols[vmapping]

    texture = bake_vertex_colors_to_texture(uvs, indices, new_vcols, tex_size)

    out_dir = obj_path.parent
    png_path = out_dir / f"{out_stem}.png"
    texture.save(png_path)

    visual = trimesh.visual.TextureVisuals(
        uv=uvs, image=texture, material=trimesh.visual.material.SimpleMaterial(image=texture)
    )
    out_mesh = trimesh.Trimesh(
        vertices=new_vertices, faces=indices, visual=visual, process=False
    )
    obj_out = out_dir / f"{out_stem}.obj"
    out_mesh.export(obj_out)  # writes .obj, .mtl, and image alongside
    return obj_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="mesh.obj file OR directory to scan")
    p.add_argument("--tex-size", type=int, default=1024)
    p.add_argument("--recursive", action="store_true", help="scan dir for mesh files")
    p.add_argument(
        "--pattern",
        default="mesh.obj",
        help="filename to match when scanning (e.g. canonical_mesh.obj)",
    )
    p.add_argument("--out-stem", default="model")
    args = p.parse_args()

    if args.path.is_file():
        targets = [args.path]
    elif args.recursive:
        targets = sorted(args.path.rglob(args.pattern))
    else:
        targets = sorted(args.path.glob(f"*/{args.pattern}"))

    if not targets:
        print(f"No mesh.obj found under {args.path}")
        return

    for obj in targets:
        try:
            out = bake_one(obj, args.tex_size, args.out_stem)
            print(f"[ok] {obj} -> {out} (+ .mtl + .png)")
        except Exception as exc:
            print(f"[fail] {obj}: {exc}")


if __name__ == "__main__":
    main()
