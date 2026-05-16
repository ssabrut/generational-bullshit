import torch
import torch.nn as nn
import torch.nn.functional as F

_DINOV2_MODEL = "dinov2_vitb14"  # embed_dim = 768
_PLANE_CH = 32  # feature channels per triplane
_PLANE_RES = 64  # spatial resolution of each plane (H = W)
_SCENE_BOUND = 1.2  # scene occupies [-bound, bound]^3


class ImageEncoder(nn.Module):
    """
    Encode a single image to a global feature vector via DINOv2 CLS token.

    Output: [B, 768]  (ViT-B embed_dim)
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available")
            self.dino = torch.hub.load(
                "facebookresearch/dinov2",
                _DINOV2_MODEL,
                pretrained=True,
                verbose=False,
            )
        if freeze:
            for p in self.dino.parameters():
                p.requires_grad = False
        self.embed_dim = self.dino.embed_dim  # 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 224, 224]  →  [B, 768]"""
        with torch.set_grad_enabled(self.dino.training):
            out = self.dino.forward_features(x)
        return out["x_norm_clstoken"]  # [B, 768]


class TriplaneGenerator(nn.Module):
    """
    Map a global image feature vector → three orthogonal feature planes.

    Architecture
    ------------
    MLP: [B, 768] → [B, C * 3 * H * W]
    Reshape to [B, 3, C, H, W] (XY / XZ / YZ planes).

    A small ConvNet refines each plane independently so the generator can
    learn spatially-structured features rather than just a flat MLP output.

    Parameters
    ----------
    in_dim   : input feature dim (768 for ViT-B)
    plane_ch : feature channels per plane
    plane_res: spatial resolution H = W of each plane
    """

    def __init__(
        self,
        in_dim: int = 768,
        plane_ch: int = _PLANE_CH,
        plane_res: int = _PLANE_RES,
    ):
        super().__init__()
        self.plane_ch = plane_ch
        self.plane_res = plane_res
        flat_dim = 3 * plane_ch * plane_res * plane_res

        # Project image feature → flat triplane token
        hidden = 1024
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, flat_dim),
        )

        # Per-plane spatial refinement (shared weights across the 3 planes)
        self.refine = nn.Sequential(
            nn.Conv2d(plane_ch, plane_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(plane_ch, plane_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(plane_ch, plane_ch, 1),
        )

    def forward(self, img_feat: torch.Tensor) -> torch.Tensor:
        """
        img_feat : [B, in_dim]
        Returns  : [B, 3, C, H, W]  — (XY, XZ, YZ) planes
        """
        B = img_feat.shape[0]
        flat = self.mlp(img_feat)  # [B, 3*C*H*W]
        planes = flat.view(
            B * 3, self.plane_ch, self.plane_res, self.plane_res
        )  # [B*3, C, H, W]
        planes = self.refine(planes)
        return planes.view(
            B, 3, self.plane_ch, self.plane_res, self.plane_res
        )  # [B, 3, C, H, W]


def _bilinear_sample(plane: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Manual bilinear sampling — pure gather + lerp ops, fully differentiable.
    Replaces F.grid_sample to avoid backend-specific backward limitations.

    plane  : [B, C, H, W]
    coords : [B, N, 2]     in [-1, 1]  (u = horizontal, v = vertical)
    Returns: [B, N, C]
    """
    B, C, H, W = plane.shape
    N = coords.shape[1]

    # Map [-1, 1] → [0, H/W - 1]  (align_corners=True convention)
    u = (coords[..., 0] + 1.0) * 0.5 * (W - 1)  # [B, N]
    v = (coords[..., 1] + 1.0) * 0.5 * (H - 1)  # [B, N]

    # Clamp to border (padding_mode="border")
    u = u.clamp(0, W - 1)
    v = v.clamp(0, H - 1)

    u0 = u.floor().long().clamp(0, W - 2)
    v0 = v.floor().long().clamp(0, H - 2)
    u1 = u0 + 1
    v1 = v0 + 1

    # Fractional weights
    wu = (u - u0.float()).unsqueeze(1)  # [B, 1, N]
    wv = (v - v0.float()).unsqueeze(1)  # [B, 1, N]

    # Gather four corner values  [B, C, N]
    def _gather(row, col):
        idx = (row * W + col).unsqueeze(1).expand(B, C, N)  # [B, C, N]
        flat = plane.view(B, C, H * W)
        return flat.gather(2, idx)

    q00 = _gather(v0, u0)  # top-left
    q10 = _gather(v1, u0)  # bottom-left
    q01 = _gather(v0, u1)  # top-right
    q11 = _gather(v1, u1)  # bottom-right

    out = (
        q00 * (1 - wv) * (1 - wu)
        + q10 * wv * (1 - wu)
        + q01 * (1 - wv) * wu
        + q11 * wv * wu
    )  # [B, C, N]

    return out.permute(0, 2, 1)  # [B, N, C]


def sample_triplane(planes: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """
    Bilinearly sample triplane features at 3D query coordinates.

    Parameters
    ----------
    planes : [B, 3, C, H, W]   — (XY, XZ, YZ) feature planes
    xyz    : [B, N, 3]          — query points in [-1, 1]^3

    Returns
    -------
    feats  : [B, N, 3*C]        — concatenated plane features
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]  # [B, N] each

    coords_xy = torch.stack([x, y], dim=-1)
    coords_xz = torch.stack([x, z], dim=-1)
    coords_yz = torch.stack([y, z], dim=-1)

    xy_feat = _bilinear_sample(planes[:, 0], coords_xy)
    xz_feat = _bilinear_sample(planes[:, 1], coords_xz)
    yz_feat = _bilinear_sample(planes[:, 2], coords_yz)

    return torch.cat([xy_feat, xz_feat, yz_feat], dim=-1)  # [B, N, 3*C]


class NeRFMLP(nn.Module):
    """
    Tiny MLP that maps triplane features → (density, RGB).

    Input  : [B, N, 3*C]   triplane features per point
    Output : density [B, N, 1]  (pre-activation, apply softplus outside)
             colour  [B, N, 3]  (pre-activation, apply sigmoid outside)
    """

    def __init__(self, feat_dim: int = 3 * _PLANE_CH, hidden: int = 128):
        super().__init__()
        self.density_net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Colour head takes density features + view direction (optional)
        self.colour_net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, feats: torch.Tensor):
        """feats: [B, N, feat_dim]  →  (sigma [B,N,1], rgb [B,N,3])"""
        sigma = self.density_net(feats)  # [B, N, 1]  raw
        rgb = self.colour_net(feats)  # [B, N, 3]  raw
        return sigma, rgb


def volume_render(
    sigma: torch.Tensor,  # [B, R, S, 1]  raw density (pre-softplus)
    rgb: torch.Tensor,  # [B, R, S, 3]  raw colour  (pre-sigmoid)
    t: torch.Tensor,  # [B, R, S]     sample distances along ray
) -> tuple:
    """
    Classic NeRF volume rendering.  Pure torch.

    Parameters
    ----------
    sigma : [B, R, S, 1]   raw density values
    rgb   : [B, R, S, 3]   raw colour values
    t     : [B, R, S]      sample positions along each ray (world units)

    Returns
    -------
    rgb_map   : [B, R, 3]  rendered colour
    depth_map : [B, R]     expected ray termination depth
    weights   : [B, R, S]  per-sample alpha compositing weights
    """
    # Activate
    sigma = F.softplus(sigma.squeeze(-1))  # [B, R, S]  ≥ 0
    rgb = torch.sigmoid(rgb)  # [B, R, S, 3]  in (0,1)

    # Interval lengths δ_i = t_{i+1} - t_i
    deltas = t[..., 1:] - t[..., :-1]  # [B, R, S-1]
    # Pad last delta with a large value so the last sample can still contribute
    deltas = torch.cat(
        [deltas, torch.full_like(deltas[..., :1], 1e10)], dim=-1
    )  # [B, R, S]

    # Alpha = 1 - exp(-σ·δ)
    alpha = 1.0 - torch.exp(-sigma * deltas)  # [B, R, S]

    # Transmittance T_i = prod_{j<i}(1 - alpha_j)
    # Computed as exclusive cumprod of (1 - alpha)
    ones = torch.ones_like(alpha[..., :1])
    T = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1), dim=-1)[
        ..., :-1
    ]  # [B, R, S]

    weights = T * alpha  # [B, R, S]
    rgb_map = (weights.unsqueeze(-1) * rgb).sum(-2)  # [B, R, 3]
    depth_map = (weights * t).sum(-1)  # [B, R]

    return rgb_map, depth_map, weights


def cast_rays(
    c2w: torch.Tensor,  # [B, 4, 4]  camera-to-world
    focal: torch.Tensor,  # [B]        normalised focal  (= 1/tan(fov/2))
    H: int,
    W: int,
    n_rays: int = 512,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """
    Sample random rays from the camera frustum.

    Returns
    -------
    rays_o : [B, n_rays, 3]   ray origins (camera position in world)
    rays_d : [B, n_rays, 3]   unit ray directions in world space
    pix_ij : [n_rays, 2]      (i, j) pixel indices for gathering GT pixels
    """
    B = c2w.shape[0]

    # Random pixel indices
    i_idx = torch.randint(0, H, (n_rays,), device=device)
    j_idx = torch.randint(0, W, (n_rays,), device=device)

    # Normalised device coords in [-1, 1] (matching grid_sample convention)
    # focal_norm = focal_px / (max(H,W)/2)  →  we stored it that way already
    # NDC: x = (j - W/2) / (W/2), y = -(i - H/2) / (H/2)  (flip y for camera)
    x = (j_idx.float() - W / 2.0) / (W / 2.0)
    y = -(i_idx.float() - H / 2.0) / (H / 2.0)

    # Ray directions in camera space: [x/focal, y/focal, -1] (OpenGL-style)
    # focal is stored normalised so we need: dir_cam = [x, y, -focal]
    # but since focal_norm = focal_px/(W/2) and x is already normalised:
    # dir_cam_x = x / focal_norm, dir_cam_y = y / focal_norm, dir_cam_z = -1
    f = focal.view(B, 1)  # [B, 1]
    dirs_cam = torch.stack(
        [
            x.unsqueeze(0).expand(B, -1) / f,  # [B, n_rays]
            y.unsqueeze(0).expand(B, -1) / f,  # [B, n_rays]
            -torch.ones(B, n_rays, device=device),
        ],
        dim=-1,
    )  # [B, n_rays, 3]

    # Rotate to world space using c2w rotation (upper-left 3×3)
    R = c2w[:, :3, :3]  # [B, 3, 3]
    rays_d = torch.bmm(dirs_cam, R.transpose(1, 2))  # [B, n_rays, 3]
    rays_d = F.normalize(rays_d, dim=-1)

    # Ray origins = camera position in world = c2w[:, :3, 3]
    rays_o = c2w[:, :3, 3].unsqueeze(1).expand(-1, n_rays, -1)  # [B, n_rays, 3]

    return rays_o, rays_d, torch.stack([i_idx, j_idx], dim=-1)


def sample_along_rays(
    rays_o: torch.Tensor,  # [B, R, 3]
    rays_d: torch.Tensor,  # [B, R, 3]
    n_samples: int = 64,
    near: float = 0.5,
    far: float = 4.0,
    perturb: bool = True,
) -> tuple:
    """
    Stratified sampling along each ray.

    Returns
    -------
    pts : [B, R, S, 3]   3D sample positions
    t   : [B, R, S]      sample distances from origin
    """
    B, R, _ = rays_o.shape
    device = rays_o.device

    # Evenly spaced bins
    t_vals = torch.linspace(near, far, n_samples, device=device)  # [S]
    t = t_vals.view(1, 1, n_samples).expand(B, R, -1).clone()  # [B, R, S]

    if perturb:
        mid = 0.5 * (t[..., 1:] + t[..., :-1])
        upper = torch.cat([mid, t[..., -1:]], dim=-1)
        lower = torch.cat([t[..., :1], mid], dim=-1)
        t = lower + (upper - lower) * torch.rand_like(t)

    pts = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * t.unsqueeze(-1)  # [B,R,S,3]
    return pts, t


def pts_to_triplane_coords(pts: torch.Tensor, bound: float = _SCENE_BOUND):
    """Normalise world-space pts to [-1, 1] for triplane sampling."""
    return pts.clamp(-bound, bound) / bound  # [B, R, S, 3]


class TriplaneNeRF(nn.Module):
    """
    Single-image → triplane NeRF model.

    Forward (training)
    ------------------
    image  : [B, 3, 224, 224]   DINOv2-normalised input view
    c2w    : [B, 4, 4]          camera-to-world of the *target* render view
    focal  : [B]                normalised focal length
    H, W   : int                render image resolution
    n_rays : int                rays to sample per image (default 512)

    Returns dict:
      rgb_map   : [B, n_rays, 3]
      depth_map : [B, n_rays]
      pix_ij    : [n_rays, 2]    pixel indices for loss computation
      planes    : [B, 3, C, H, W]  (for optional plane regularisation)

    Forward (inference / full render)
    ----------------------------------
    Call render_full() to get a complete H×W image.
    Call extract_mesh() for marching cubes mesh (CPU).
    """

    def __init__(
        self,
        plane_ch: int = _PLANE_CH,
        plane_res: int = _PLANE_RES,
        nerf_hidden: int = 128,
        n_samples: int = 64,
        near: float = 0.5,
        far: float = 4.0,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.near = near
        self.far = far

        self.encoder = ImageEncoder(freeze=True)
        self.generator = TriplaneGenerator(
            in_dim=self.encoder.embed_dim,
            plane_ch=plane_ch,
            plane_res=plane_res,
        )
        self.nerf_mlp = NeRFMLP(
            feat_dim=3 * plane_ch,
            hidden=nerf_hidden,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _planes_from_image(self, image: torch.Tensor) -> torch.Tensor:
        """image [B,3,224,224] → planes [B,3,C,H,W]"""
        img_feat = self.encoder(image)  # [B, 768]
        return self.generator(img_feat)  # [B, 3, C, H, W]

    def _render_rays(
        self,
        planes: torch.Tensor,  # [B, 3, C, H, W]
        rays_o: torch.Tensor,  # [B, R, 3]
        rays_d: torch.Tensor,  # [B, R, 3]
        perturb: bool = True,
    ) -> tuple:
        pts, t = sample_along_rays(
            rays_o,
            rays_d,
            n_samples=self.n_samples,
            near=self.near,
            far=self.far,
            perturb=perturb,
        )  # [B,R,S,3], [B,R,S]

        B, R, S, _ = pts.shape
        pts_norm = pts_to_triplane_coords(pts)  # [B, R, S, 3]

        # Flatten R*S for batch sampling, then restore
        pts_flat = pts_norm.view(B, R * S, 3)  # [B, R*S, 3]
        feats = sample_triplane(planes, pts_flat)  # [B, R*S, 3C]
        feats = feats.view(B, R, S, -1)  # [B, R, S, 3C]

        # MLP
        sigma, rgb = self.nerf_mlp(feats)  # [B,R,S,1], [B,R,S,3]

        rgb_map, depth_map, weights = volume_render(sigma, rgb, t)
        return rgb_map, depth_map, weights

    # ── public ────────────────────────────────────────────────────────────────

    def forward(
        self,
        image: torch.Tensor,
        c2w: torch.Tensor,
        focal: torch.Tensor,
        H: int,
        W: int,
        n_rays: int = 512,
        perturb: bool = True,
    ) -> dict:
        device = image.device
        planes = self._planes_from_image(image)

        rays_o, rays_d, pix_ij = cast_rays(
            c2w, focal, H, W, n_rays=n_rays, device=device
        )
        rgb_map, depth_map, weights = self._render_rays(
            planes, rays_o, rays_d, perturb=perturb
        )

        return {
            "rgb_map": rgb_map,  # [B, n_rays, 3]
            "depth_map": depth_map,  # [B, n_rays]
            "pix_ij": pix_ij,  # [n_rays, 2]
            "planes": planes,  # [B, 3, C, H, W]  for reg loss
        }

    @torch.no_grad()
    def render_full(
        self,
        image: torch.Tensor,  # [1, 3, 224, 224]
        c2w: torch.Tensor,  # [1, 4, 4]
        focal: torch.Tensor,  # [1]
        H: int,
        W: int,
        chunk: int = 1024,
        planes: torch.Tensor = None,  # [1, 3, C, Hp, Wp]  pre-computed; skips encoder
    ) -> dict:
        """
        Render a complete H×W image in chunks (avoids OOM on CPU).
        Returns rgb [H, W, 3] and depth [H, W] on CPU.
        Pass `planes` to reuse already-computed triplane features (saves one encoder forward).
        """
        device = image.device
        if planes is None:
            planes = self._planes_from_image(image)

        i_grid, j_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        pix_flat = torch.stack([i_grid.reshape(-1), j_grid.reshape(-1)], -1)  # [H*W, 2]
        N = pix_flat.shape[0]

        all_rgb = []
        all_depth = []
        for start in range(0, N, chunk):
            pix_chunk = pix_flat[start : start + chunk]  # [k, 2]
            i_idx = pix_chunk[:, 0]
            j_idx = pix_chunk[:, 1]

            x = (j_idx.float() - W / 2.0) / (W / 2.0)
            y = -(i_idx.float() - H / 2.0) / (H / 2.0)

            f = focal.view(1, 1)
            dirs_cam = torch.stack(
                [
                    x.unsqueeze(0) / f,
                    y.unsqueeze(0) / f,
                    -torch.ones(1, len(i_idx), device=device),
                ],
                dim=-1,
            )  # [1, k, 3]

            R = c2w[:, :3, :3]
            rays_d = torch.bmm(dirs_cam, R.transpose(1, 2))
            rays_d = F.normalize(rays_d, dim=-1)
            rays_o = c2w[:, :3, 3].unsqueeze(1).expand(-1, len(i_idx), -1)

            rgb_c, depth_c, _ = self._render_rays(planes, rays_o, rays_d, perturb=False)
            all_rgb.append(rgb_c[0].cpu())
            all_depth.append(depth_c[0].cpu())

        rgb_img = torch.cat(all_rgb, dim=0).view(H, W, 3)
        depth_img = torch.cat(all_depth, dim=0).view(H, W)
        return {"rgb": rgb_img, "depth": depth_img}

    @torch.no_grad()
    def extract_mesh(
        self,
        image: torch.Tensor,  # [1, 3, 224, 224]
        resolution: int = 64,
        threshold: float = 10.0,
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        """
        Marching cubes mesh extraction (runs on CPU).
        Requires: pip install scikit-image

        Returns dict with 'vertices' [V,3] and 'faces' [F,3] as numpy arrays.
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            raise ImportError("pip install scikit-image  to use extract_mesh()")

        image = image.to(device)
        planes = self._planes_from_image(image)

        # Build a dense 3D grid
        lin = torch.linspace(-1.0, 1.0, resolution, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing="ij")
        pts = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [R,R,R,3]
        pts_flat = pts.view(1, -1, 3)  # [1, R^3, 3]

        # Query density in chunks (avoid OOM)
        chunk = 32768
        sigmas = []
        for start in range(0, pts_flat.shape[1], chunk):
            feats = sample_triplane(planes, pts_flat[:, start : start + chunk])
            s, _ = self.nerf_mlp(feats)
            sigmas.append(F.softplus(s).squeeze(-1))
        sigma_vol = torch.cat(sigmas, dim=1).view(resolution, resolution, resolution)
        sigma_np = sigma_vol.cpu().numpy()

        s_min, s_max = float(sigma_np.min()), float(sigma_np.max())
        print(f"  sigma range: [{s_min:.4f}, {s_max:.4f}]  threshold={threshold:.4f}")
        if not (s_min < threshold < s_max):
            threshold = s_min + 0.5 * (s_max - s_min)
            print(f"  threshold out of range — clamped to median: {threshold:.4f}")

        verts, faces, _, _ = marching_cubes(sigma_np, level=threshold)
        # Rescale verts from [0, resolution-1] to [-bound, bound]
        verts = verts / (resolution - 1) * 2.0 * _SCENE_BOUND - _SCENE_BOUND

        return {"vertices": verts, "faces": faces}


def photometric_loss(rgb_pred: torch.Tensor, rgb_gt: torch.Tensor) -> torch.Tensor:
    """L2 + L1 combined photometric loss.  Both [B, R, 3] or [R, 3]."""
    return F.mse_loss(rgb_pred, rgb_gt) + 0.1 * F.l1_loss(rgb_pred, rgb_gt)


def plane_tv_loss(planes: torch.Tensor) -> torch.Tensor:
    """
    Total variation regularisation on triplane features.
    Encourages smooth, non-noisy feature planes.
    planes: [B, 3, C, H, W]
    """
    B, three, C, H, W = planes.shape
    p = planes.view(B * three, C, H, W)
    tv_h = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean()
    tv_w = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def plane_l2_loss(planes: torch.Tensor) -> torch.Tensor:
    """L2 weight decay on plane values — prevents feature explosion."""
    return planes.pow(2).mean()


def compute_loss(
    rgb_pred: torch.Tensor,
    rgb_gt: torch.Tensor,
    planes: torch.Tensor,
    w_photo: float = 1.0,
    w_tv: float = 1e-4,
    w_l2: float = 1e-4,
) -> tuple:
    photo = photometric_loss(rgb_pred, rgb_gt)
    tv = plane_tv_loss(planes)
    l2 = plane_l2_loss(planes)
    total = w_photo * photo + w_tv * tv + w_l2 * l2
    return total, {"photometric": photo.item(), "tv": tv.item(), "l2": l2.item()}


def make_optimizer(
    model: TriplaneNeRF,
    lr_gen: float = 5e-4,
    lr_nerf: float = 5e-4,
    weight_decay: float = 1e-5,
) -> torch.optim.Optimizer:
    """
    Two parameter groups:
      generator (TriplaneGenerator) — learns the image→plane mapping
      nerf_mlp  (NeRFMLP)          — learns density/colour from features
    Encoder (DINOv2) is frozen and excluded.
    """
    return torch.optim.AdamW(
        [
            {"params": model.generator.parameters(), "lr": lr_gen},
            {"params": model.nerf_mlp.parameters(), "lr": lr_nerf},
        ],
        weight_decay=weight_decay,
    )
