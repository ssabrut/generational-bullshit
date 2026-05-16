"""
Inference — single reference image → novel views + mesh.

Usage
-----
    python -c "
    from app.src.infer import infer
    infer(
        image_path  = 'path/to/chair.jpg',
        ckpt_path   = 'checkpoints/triplane_epoch050.pth',
    )
    "

What it does
------------
1. Load a checkpoint into TriplaneNeRF.
2. Preprocess the reference image (resize → DINOv2 normalise).
3. Encode → triplane features (one forward pass, no gradients).
4. Render a 360° turntable (N evenly-spaced azimuth angles) at full resolution.
5. Optionally extract a mesh via marching cubes.
6. Show all results in a matplotlib window and save PNGs to out_dir.
"""

import math
import os

import matplotlib

matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model.triplane import TriplaneNeRF

# ── DINOv2 normalisation ───────────────────────────────────────────────────────
_DINO_TFM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_model(ckpt_path: str, device: torch.device) -> TriplaneNeRF:
    """Load a TriplaneNeRF from a training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)  # support both raw state_dict and wrapped ckpt

    # Infer arch from state dict so we don't need to hard-code it
    plane_ch = state["generator.refine.0.weight"].shape[0]  # out channels
    flat_dim = state["generator.mlp.6.weight"].shape[0]  # 3*C*H*W
    plane_res = int(math.sqrt(flat_dim // (3 * plane_ch)))  # H = W
    nerf_h = state["nerf_mlp.density_net.0.weight"].shape[0]
    n_samples = 64  # not stored in ckpt — use training default

    model = TriplaneNeRF(
        plane_ch=plane_ch,
        plane_res=plane_res,
        nerf_hidden=nerf_h,
        n_samples=n_samples,
        near=0.5,
        far=4.5,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  plane_ch={plane_ch}  plane_res={plane_res}  nerf_hidden={nerf_h}")
    return model


def preprocess(image_path: str, device: torch.device) -> torch.Tensor:
    """Load any image file → [1, 3, 224, 224] DINOv2-normalised tensor."""
    img = Image.open(image_path).convert("RGB")
    return _DINO_TFM(img).unsqueeze(0).to(device)


def turntable_c2w(
    n_views: int = 36,
    elevation: float = 20.0,  # degrees above horizon
    radius: float = 2.5,  # camera distance from origin
) -> torch.Tensor:
    """
    Generate N evenly-spaced camera-to-world matrices for a 360° turntable.

    Convention: OpenGL-style (+Y up, camera looks toward -Z in camera space).
    Returns [N, 4, 4] float32.
    """
    c2ws = []
    elev_rad = math.radians(elevation)

    for i in range(n_views):
        azim_rad = 2 * math.pi * i / n_views

        # Camera position in world space
        cx = radius * math.cos(elev_rad) * math.sin(azim_rad)
        cy = radius * math.sin(elev_rad)
        cz = radius * math.cos(elev_rad) * math.cos(azim_rad)
        cam_pos = np.array([cx, cy, cz])

        # Look-at: camera looks at origin
        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)  # -Z in cam space
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(world_up, forward)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(forward, right)

        # c2w columns: [right | up | -forward | position]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # camera -Z points toward scene
        c2w[:3, 3] = cam_pos
        c2ws.append(c2w)

    return torch.tensor(np.stack(c2ws), dtype=torch.float32)  # [N, 4, 4]


# ─────────────────────────────────────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def infer(
    image_path: str,
    ckpt_path: str,
    out_dir: str = "inference_out",
    n_views: int = 36,  # turntable frames
    render_h: int = 256,
    render_w: int = 256,
    render_chunk: int = 2048,  # rays per chunk (lower if OOM)
    focal: float = None,  # normalised focal; None → infer from 60° FoV
    elevation: float = 20.0,  # camera elevation in degrees
    radius: float = 2.5,  # camera distance
    extract_mesh: bool = False,  # requires scikit-image
    mesh_res: int = 64,
    mesh_thresh: float = 10.0,
    device: torch.device = None,
):
    """
    Run inference on a single reference image.

    Parameters
    ----------
    image_path   : path to any RGB image (JPG, PNG, …)
    ckpt_path    : path to a training checkpoint (.pth)
    out_dir      : directory for saved renders and mesh
    n_views      : number of turntable viewpoints to render
    render_h/w   : output resolution per frame
    render_chunk : pixels per chunk (reduce if you hit OOM)
    focal        : normalised focal length (1/tan(fov/2)); None → 60° FoV
    elevation    : camera elevation angle in degrees
    radius       : camera-to-origin distance (must be inside [near, far])
    extract_mesh : run marching cubes after rendering
    mesh_res     : voxel grid resolution for marching cubes
    mesh_thresh  : density threshold for marching cubes isosurface
    device       : torch device; defaults to CPU
    """
    # ── device ────────────────────────────────────────────────────────────────
    if device is None:
        device = torch.device("cpu")
    print(f"Device: {device}")

    os.makedirs(out_dir, exist_ok=True)

    # ── model ─────────────────────────────────────────────────────────────────
    model = load_model(ckpt_path, device)

    # ── preprocess reference image ────────────────────────────────────────────
    image = preprocess(image_path, device)  # [1, 3, 224, 224]
    print(f"Reference image: {image_path}  →  {tuple(image.shape)}")

    # ── encode once, reuse planes for all views ───────────────────────────────
    planes = model._planes_from_image(image)  # [1, 3, C, Hp, Wp]
    print(f"Triplane shape: {tuple(planes.shape)}")

    # ── focal length ──────────────────────────────────────────────────────────
    if focal is None:
        focal = 1.0 / math.tan(math.radians(60) / 2)  # 60° FoV
    focal_t = torch.tensor([focal], dtype=torch.float32, device=device)  # [1]
    print(
        f"Focal (normalised): {focal:.4f}  ({math.degrees(2*math.atan(1/focal)):.1f}° FoV)"
    )

    # ── turntable camera poses ────────────────────────────────────────────────
    c2ws = turntable_c2w(n_views, elevation=elevation, radius=radius).to(device)
    print(f"Rendering {n_views} views at {render_h}×{render_w} …")

    # ── render all views ──────────────────────────────────────────────────────
    frames_rgb = []
    frames_depth = []

    for vi in range(n_views):
        c2w = c2ws[vi : vi + 1]  # [1, 4, 4]
        out = model.render_full(
            image,
            c2w,
            focal_t,
            H=render_h,
            W=render_w,
            chunk=render_chunk,
            planes=planes,
        )
        frames_rgb.append(out["rgb"])  # [H, W, 3]  CPU float32
        frames_depth.append(out["depth"])  # [H, W]     CPU float32

        if (vi + 1) % max(1, n_views // 8) == 0:
            print(f"  {vi+1}/{n_views} views done")

    # ── save individual frames ────────────────────────────────────────────────
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for vi, rgb in enumerate(frames_rgb):
        img_pil = Image.fromarray((rgb.numpy() * 255).clip(0, 255).astype(np.uint8))
        img_pil.save(os.path.join(frames_dir, f"view_{vi:03d}.png"))

    print(f"Frames saved → {frames_dir}/")

    # ── display results ───────────────────────────────────────────────────────
    _show_results(
        image_path=image_path,
        frames_rgb=frames_rgb,
        frames_depth=frames_depth,
        n_cols=min(n_views, 8),
        out_dir=out_dir,
    )

    # ── mesh extraction ───────────────────────────────────────────────────────
    if extract_mesh:
        print(f"\nExtracting mesh (res={mesh_res}, thresh={mesh_thresh}) …")
        try:
            mesh = model.extract_mesh(
                image,
                resolution=mesh_res,
                threshold=mesh_thresh,
                device=torch.device("cpu"),  # marching cubes is CPU-only
            )
            _save_obj(
                mesh["vertices"], mesh["faces"], os.path.join(out_dir, "mesh.obj")
            )
            _show_mesh(mesh["vertices"], mesh["faces"], out_dir)
        except ImportError as e:
            print(f"Mesh extraction skipped: {e}")

    print(f"\nDone. All outputs in: {out_dir}/")
    return frames_rgb, frames_depth


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────


def _show_results(image_path, frames_rgb, frames_depth, n_cols, out_dir):
    """
    Show a grid: reference image | evenly-sampled render views | depth strip.
    """
    n_views = len(frames_rgb)
    # Pick n_cols evenly-spaced views to display
    idxs = np.linspace(0, n_views - 1, n_cols, dtype=int)
    sel_rgb = [frames_rgb[i] for i in idxs]
    sel_depth = [frames_depth[i] for i in idxs]

    n_rows = 3  # reference | RGB renders | depth renders
    fig, axes = plt.subplots(
        n_rows, n_cols + 1, figsize=((n_cols + 1) * 2.5, n_rows * 2.5)
    )
    fig.patch.set_facecolor("#1a1a1a")
    for row in axes:
        for ax in row:
            ax.set_facecolor("#1a1a1a")
            ax.axis("off")

    # ── row 0: reference image (col 0) + empty cols ───────────────────────────
    ref = Image.open(image_path).convert("RGB")
    axes[0, 0].imshow(np.array(ref))
    axes[0, 0].set_title("Reference", color="white", fontsize=8)

    # ── row 1: selected RGB renders ───────────────────────────────────────────
    axes[1, 0].set_title("RGB renders", color="white", fontsize=8)
    for col, (idx, rgb) in enumerate(zip(idxs, sel_rgb), start=1):
        axes[1, col].imshow(rgb.numpy().clip(0, 1))
        axes[1, col].set_title(
            f"az={idx*360//len(frames_rgb)}°", color="white", fontsize=7
        )

    # ── row 2: depth maps ─────────────────────────────────────────────────────
    axes[2, 0].set_title("Depth", color="white", fontsize=8)
    # Normalise depth across all selected frames for consistent colour scale
    d_all = np.stack([d.numpy() for d in sel_depth])
    d_min, d_max = d_all.min(), d_all.max()
    for col, depth in enumerate(sel_depth, start=1):
        d_norm = (depth.numpy() - d_min) / (d_max - d_min + 1e-8)
        axes[2, col].imshow(d_norm, cmap="plasma")

    fig.suptitle("TriplaneNeRF — inference", color="white", fontsize=11)
    fig.tight_layout(pad=0.5)

    save_path = os.path.join(out_dir, "overview.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Overview saved → {save_path}")

    plt.pause(0.001)
    plt.show()


def _save_obj(verts: np.ndarray, faces: np.ndarray, path: str):
    """Write a bare-bones .obj file."""
    with open(path, "w") as fh:
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    print(f"Mesh saved → {path}  ({len(verts)} verts, {len(faces)} faces)")


def _show_mesh(verts: np.ndarray, faces: np.ndarray, out_dir: str):
    """Quick 3-angle mesh visualisation using Poly3DCollection."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tris = [verts[f] for f in faces]

    fig = plt.figure(figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a1a")

    for col, (elev, azim, title) in enumerate(
        [
            (20, 45, "Front-right"),
            (20, 180, "Back"),
            (70, 45, "Top"),
        ]
    ):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        ax.set_facecolor("#1a1a1a")
        poly = Poly3DCollection(
            tris, alpha=0.6, facecolor="#4A90D9", edgecolor="#222", linewidth=0.05
        )
        ax.add_collection3d(poly)
        b = 1.3
        ax.set_xlim(-b, b)
        ax.set_ylim(-b, b)
        ax.set_zlim(-b, b)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, color="white", fontsize=9)
        ax.set_axis_off()

    fig.suptitle("Extracted mesh", color="white", fontsize=11)
    fig.tight_layout()

    save_path = os.path.join(out_dir, "mesh_views.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Mesh views saved → {save_path}")

    plt.pause(0.001)
    plt.show()


if __name__ == "__main__":
    infer(
        image_path="data/pix3d/img/chair/0003.png",
        ckpt_path="checkpoints/triplane_epoch050.pth",
        out_dir="inference_out",
        n_views=36,  # 360° turntable, one frame every 10°
        render_h=256,
        render_w=256,
        elevation=20.0,  # camera angle above horizon
        radius=2.5,  # camera distance from object
        extract_mesh=True,  # flip to True after: pip install scikit-image
    )
