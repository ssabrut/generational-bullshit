# %% [markdown]
# ## Setup

# %%
import torch
torch.backends.quantized.engine = "qnnpack"

import numpy as np

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

torch.manual_seed(42)
np.random.seed(42)
print("Device:", DEVICE)

# %% [markdown]
# ## Mesh utilities
#
# Carried over from `pix2mesh_dinov2.py` unchanged — unit cube template, Loop
# subdivision, adjacency and Laplacian builders.

# %%
import torch.nn.functional as F
from torch import nn


def get_unit_cube():
    v = torch.tensor([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=torch.float32)
    f = torch.tensor([
        [0,1,2],[0,2,3],
        [4,6,5],[4,7,6],
        [0,4,5],[0,5,1],
        [2,6,7],[2,7,3],
        [0,7,4],[0,3,7],
        [1,5,6],[1,6,2],
    ], dtype=torch.long)
    return v, f


def loop_subdivide(vertices, faces):
    V = vertices.shape[0]
    device = vertices.device
    edges_dict, mids = {}, []

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in edges_dict:
            edges_dict[key] = len(mids) + V
            mids.append((vertices[a] + vertices[b]) / 2.0)
        return edges_dict[key]

    new_faces = []
    for tri in faces.numpy():
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        m01, m12, m20 = midpoint(v0,v1), midpoint(v1,v2), midpoint(v2,v0)
        new_faces += [[v0,m01,m20],[v1,m12,m01],[v2,m20,m12],[m01,m12,m20]]

    new_v = torch.cat([vertices, torch.stack(mids).to(device)], 0) if mids else vertices
    new_f = torch.tensor(new_faces, dtype=torch.long, device=device)
    return new_v, new_f


def build_adjacency(vertices, faces):
    V = vertices.shape[0]
    adj = torch.zeros(V, V)
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1)%3])
            adj[a, b] = adj[b, a] = 1.0
    adj += torch.eye(V)
    return adj / adj.sum(1, keepdim=True).clamp(min=1)


def build_laplacian(vertices, faces):
    V = vertices.shape[0]
    adj = torch.zeros(V, V)
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1)%3])
            adj[a, b] = adj[b, a] = 1.0
    return torch.eye(V) - adj / adj.sum(1, keepdim=True).clamp(1)


v0, f0 = get_unit_cube()
v1, f1 = loop_subdivide(v0, f0)
v2, f2 = loop_subdivide(v1, f1)
v3, f3 = loop_subdivide(v2, f2)
for label, v, f in [("Cube     ", v0, f0), ("Subdiv×1 ", v1, f1),
                     ("Subdiv×2 ", v2, f2), ("Subdiv×3 ", v3, f3)]:
    print(f"  {label}: {v.shape[0]:4d} verts, {f.shape[0]:4d} faces")

# %% [markdown]
# ## Vertex projection & feature sampling
#
# Carried over from `pix2mesh_dinov2.py` unchanged.
#
# OmniObject3D uses NeRF-convention c2w matrices; `omniobject3d_dinov2.py`
# converts them to `(R, T)` world-to-camera form so these functions work as-is.

# %%
def project_vertices(vertices, rot, trans, focal):
    """
    Project template vertices onto the image plane.
    vertices : [B, V, 3]  world-space
    rot      : [B, 3, 3]  world-to-camera rotation
    trans    : [B, 3]     world-to-camera translation
    focal    : [B]        normalised focal (focal_px / (img_size / 2))
    Returns  : [B, V, 2]  in [-1, 1] for grid_sample
    """
    v_cam = torch.bmm(vertices, rot.transpose(1, 2)) + trans.unsqueeze(1)
    z = v_cam[..., 2:3].clamp(min=1e-6)
    f = focal.view(-1, 1, 1)
    xy = f * v_cam[..., :2] / z
    return xy.clamp(-1, 1)


def sample_features(feature_maps, coords):
    """
    Bilinearly sample multi-scale feature maps at projected vertex coords.
    feature_maps : list of [B, C_i, H_i, W_i]
    coords       : [B, V, 2]
    Returns      : [B, V, sum_C_i]
    """
    grid = coords.unsqueeze(1)  # [B, 1, V, 2]
    sampled = []
    for fm in feature_maps:
        s = F.grid_sample(fm, grid, align_corners=True, mode="bilinear",
                          padding_mode="border")        # [B, C, 1, V]
        sampled.append(s.squeeze(2).permute(0, 2, 1))  # [B, V, C]
    return torch.cat(sampled, dim=-1)                  # [B, V, sum_C]

# %% [markdown]
# ## DINOv2 + FPN Encoder
#
# Identical to `pix2mesh_dinov2.py`.
#
# ```
# DINOv2 ViT-B/14  (patch=14, embed=768)
#   hooks on blocks [2, 5, 8, 11] → intermediate patch tokens
#   each: [B, 256, 768] → reshape → [B, 768, 16, 16]
#
# FPN lateral convs  (768 → 512 per level)
#   P4: 16×16  (deepest — high semantics)
#   P3: 16×16  → up×2 → 32×32
#   P2: 16×16  → up×4 → 64×64
#   P1: 16×16  → up×8 → 128×128
# ```
#
# Output: 4 maps [B, 512, H, W] at 128², 64², 32², 16².
# Total feature dim per vertex = 512 × 4 = 2048.

# %%
_DINOV2_MODEL = "dinov2_vitb14"
_FPN_CHANNELS = 512
_TAP_BLOCKS   = [2, 5, 8, 11]


class DINOv2FPNEncoder(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        self.dino = torch.hub.load(
            "facebookresearch/dinov2", _DINOV2_MODEL, pretrained=True
        )
        embed_dim = self.dino.embed_dim  # 768 for ViT-B

        self._feats = {}
        self._hooks = []
        for blk_idx in _TAP_BLOCKS:
            hook = self.dino.blocks[blk_idx].register_forward_hook(
                self._make_hook(blk_idx)
            )
            self._hooks.append(hook)

        if freeze_backbone:
            for p in self.dino.parameters():
                p.requires_grad = False

        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, _FPN_CHANNELS, 1, bias=False),
                nn.BatchNorm2d(_FPN_CHANNELS),
                nn.ReLU(inplace=True),
            )
            for _ in _TAP_BLOCKS
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(_FPN_CHANNELS, _FPN_CHANNELS, 3, padding=1, bias=False),
                nn.BatchNorm2d(_FPN_CHANNELS),
                nn.ReLU(inplace=True),
            )
            for _ in _TAP_BLOCKS
        ])
        self.feat_dims = [_FPN_CHANNELS] * 4  # [512, 512, 512, 512]

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.forward(dummy)
        print("DINOv2+FPN feature pyramid:")
        for i, fm in enumerate(out):
            print(f"  Level {i+1}: {tuple(fm.shape)}")
        print(f"feat_dims : {self.feat_dims}  total={sum(self.feat_dims)}")

    def _make_hook(self, blk_idx):
        def hook(module, input, output):
            self._feats[blk_idx] = output[:, 1:, :]
        return hook

    def _patch_tokens_to_spatial(self, tokens, h_patches, w_patches):
        B, N, D = tokens.shape
        return tokens.permute(0, 2, 1).reshape(B, D, h_patches, w_patches)

    def forward(self, x):
        B, C, H, W = x.shape
        self._feats.clear()
        with torch.set_grad_enabled(self.dino.training):
            _ = self.dino(x)
        h_p = H // self.dino.patch_size
        w_p = W // self.dino.patch_size

        laterals = []
        for i, blk_idx in enumerate(_TAP_BLOCKS):
            spatial = self._patch_tokens_to_spatial(
                self._feats[blk_idx], h_p, w_p)
            laterals.append(self.lateral[i](spatial))

        fpn = [None] * 4
        fpn[3] = laterals[3]
        for i in range(2, -1, -1):
            upsampled = F.interpolate(fpn[i+1], size=laterals[i].shape[-2:],
                                      mode="nearest")
            fpn[i] = laterals[i] + upsampled

        target_sizes = [128, 64, 32, 16]
        out = []
        for i in range(4):
            fm = F.interpolate(fpn[i], size=(target_sizes[i], target_sizes[i]),
                               mode="bilinear", align_corners=False)
            out.append(self.output_convs[i](fm))
        return out


encoder = DINOv2FPNEncoder(freeze_backbone=True).to(DEVICE)

# %% [markdown]
# ## GCN
#
# Carried over from `pix2mesh_dinov2.py` unchanged.

# %%
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_self  = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bn   = nn.BatchNorm1d(out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        B, V, _ = x.shape
        neigh = torch.einsum("vw,bwf->bvf", adj, x)
        out = self.fc_self(x) + self.fc_neigh(neigh) + self.bias
        out = self.bn(out.reshape(B*V, -1)).reshape(B, V, -1)
        return F.relu(out)


class GCNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gc1  = GraphConv(in_dim, hidden_dim)
        self.gc2  = GraphConv(hidden_dim, hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim, bias=False) \
                    if in_dim != hidden_dim else nn.Identity()

    def forward(self, x, adj):
        return self.gc2(self.gc1(x, adj), adj) + self.skip(x)


class DeformStage(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            GCNBlock(feat_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_blocks)
        ])
        self.disp_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 3)
        )

    def forward(self, feats, verts, adj):
        h = feats
        for blk in self.blocks:
            h = blk(h, adj)
        return verts + self.disp_head(h), h

# %% [markdown]
# ## OmniObject3D model
#
# Same Pixel2Mesh architecture as `pix2mesh_dinov2.py` — three coarse-to-fine
# GCN deformation stages driven by DINOv2+FPN features.
#
# **One change from the Pix3D version:** the projector input dim grows because
# the FPN produces 512-ch maps × 4 levels = 2048 dims per vertex (vs 1024 in
# the ViT-S version).  Everything else is identical so the two files can share
# weights if you fine-tune from a Pix3D checkpoint.
#
# ```
# Stage 0  input: [img_feat(2048) + xyz(3)]            → project → hidden
# Stage 1  input: [img_feat(2048) + prev_h(hidden) + xyz(3)] → project → hidden
# Stage 2  input: [img_feat(2048) + prev_h(hidden) + xyz(3)] → project → hidden
# ```

# %%
class OmniMesh3D(nn.Module):
    """
    Coarse-to-fine mesh deformation network for OmniObject3D.

    Parameters
    ----------
    encoder    : DINOv2FPNEncoder
    hidden_dim : int   — GCN hidden dimension (default 512)
    n_stages   : int   — number of subdivision stages (default 3)
    n_gcn_blocks : int — GCNBlocks per DeformStage (default 5)
    """

    def __init__(self, encoder, hidden_dim=512, n_stages=3, n_gcn_blocks=5):
        super().__init__()
        self.n_stages   = n_stages
        self.hidden_dim = hidden_dim
        self.encoder    = encoder
        total_img = sum(encoder.feat_dims)  # 2048

        self.projectors    = nn.ModuleList()
        self.deform_stages = nn.ModuleList()
        for i in range(n_stages):
            in_dim = (total_img + 3) if i == 0 else (total_img + hidden_dim + 3)
            self.projectors.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            ))
            self.deform_stages.append(
                DeformStage(hidden_dim, hidden_dim, n_gcn_blocks))

        self._build_templates(n_stages)

    def _build_templates(self, n_stages):
        verts, faces = get_unit_cube()
        self.templates = []
        for i in range(n_stages):
            adj = build_adjacency(verts, faces)
            lap = build_laplacian(verts, faces)
            self.register_buffer(f"adj_{i}", adj)
            self.register_buffer(f"lap_{i}", lap)
            self.templates.append((verts.clone(), faces.clone()))
            if i < n_stages - 1:
                verts, faces = loop_subdivide(verts, faces)

    def forward(self, image, rot, trans, focal):
        """
        image : [B, 3, 224, 224]
        rot   : [B, 3, 3]   world-to-camera rotation
        trans : [B, 3]      world-to-camera translation
        focal : [B]         normalised focal length

        Returns dict with:
          vertices   — list of [B, V_i, 3] per stage
          faces      — list of [F_i, 3]    per stage
          laplacians — list of [V_i, V_i]  per stage
        """
        B, device = image.shape[0], image.device
        feat_maps = self.encoder(image)
        all_verts, all_faces, all_laps = [], [], []
        prev_h = None

        for i in range(self.n_stages):
            adj = getattr(self, f"adj_{i}").to(device)
            lap = getattr(self, f"lap_{i}").to(device)
            verts_tmpl, faces = self.templates[i]
            verts = verts_tmpl.to(device).unsqueeze(0).expand(B, -1, -1).clone()
            V = verts.shape[1]

            coords   = project_vertices(verts, rot, trans, focal)  # [B, V, 2]
            img_feat = sample_features(feat_maps, coords)           # [B, V, 2048]

            if prev_h is None:
                combined = torch.cat([img_feat, verts], dim=-1)
            else:
                if prev_h.shape[1] != V:
                    prev_h = F.interpolate(
                        prev_h.permute(0, 2, 1), size=V,
                        mode="linear", align_corners=False,
                    ).permute(0, 2, 1)
                combined = torch.cat([img_feat, prev_h, verts], dim=-1)

            vertex_feats = self.projectors[i](combined)
            new_verts, hidden = self.deform_stages[i](vertex_feats, verts, adj)

            all_verts.append(new_verts)
            all_faces.append(faces)
            all_laps.append(lap)
            prev_h = hidden

        return {"vertices": all_verts, "faces": all_faces, "laplacians": all_laps}


model = OmniMesh3D(encoder=encoder, hidden_dim=512, n_stages=3,
                   n_gcn_blocks=5).to(DEVICE)

total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params    : {total_p:>12,}")
print(f"Trainable       : {train_p:>12,}")
print(f"Frozen (encoder): {total_p - train_p:>12,}")

# %% [markdown]
# ### Forward pass sanity check

# %%
B = 2
img_test   = torch.randn(B, 3, 224, 224, device=DEVICE)
rot_test   = torch.eye(3, device=DEVICE).unsqueeze(0).expand(B, -1, -1).clone()
trans_test = torch.zeros(B, 3, device=DEVICE); trans_test[:, 2] = 2.0
focal_test = torch.ones(B, device=DEVICE) * 0.85

model.eval()
with torch.no_grad():
    out = model(img_test, rot_test, trans_test, focal_test)

print("\nForward pass output:")
for i, v in enumerate(out["vertices"]):
    f = out["faces"][i]
    print(f"  Stage {i+1}: verts {tuple(v.shape)}  faces {tuple(f.shape)}")

# %% [markdown]
# ## Mesh Discriminator
#
# Carried over from `pix2mesh_dinov2.py` unchanged — GCN-based binary
# classifier (real GT point cloud vs predicted mesh vertices).

# %%
class MeshDiscriminator(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.gc1  = GraphConv(3, hidden_dim)
        self.gc2  = GraphConv(hidden_dim, hidden_dim)
        self.gc3  = GraphConv(hidden_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, verts, adj):
        B = verts.shape[0]
        h = self.gc1(verts, adj)
        h = self.gc2(h, adj)
        h = self.gc3(h, adj)
        h = self.pool(h.permute(0, 2, 1)).squeeze(-1)
        return self.head(h)


discriminator = MeshDiscriminator(hidden_dim=256).to(DEVICE)
print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

# %% [markdown]
# ## Loss functions
#
# Same as `pix2mesh_dinov2.py` with one addition: **normal consistency loss**,
# which penalises flipped face normals on adjacent triangles.  This matters
# more for OmniObject3D because the GT scan meshes have clean, consistent
# normals — we want the predicted mesh to match that quality.

# %%
def chamfer_distance(pred, gt):
    """Bidirectional Chamfer distance.  pred, gt: [B, N, 3]"""
    d = ((pred.unsqueeze(2) - gt.unsqueeze(1))**2).sum(-1)
    return d.min(2).values.mean() + d.min(1).values.mean()


def edge_loss(vertices, faces):
    """Penalise long / uneven edges."""
    v0 = vertices[:, faces[:, 0]]
    v1 = vertices[:, faces[:, 1]]
    v2 = vertices[:, faces[:, 2]]
    return torch.stack([
        (v0 - v1).norm(dim=-1),
        (v1 - v2).norm(dim=-1),
        (v2 - v0).norm(dim=-1),
    ]).mean()


def laplacian_loss(vertices, lap):
    """Penalise non-smooth vertex positions via graph Laplacian."""
    return (torch.einsum("vw,bwc->bvc", lap, vertices)**2).mean()


def face_normals(vertices, faces):
    """
    Compute per-face unit normals.
    vertices : [B, V, 3]
    faces    : [F, 3]
    Returns  : [B, F, 3]
    """
    v0 = vertices[:, faces[:, 0]]
    v1 = vertices[:, faces[:, 1]]
    v2 = vertices[:, faces[:, 2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return F.normalize(n, dim=-1)


def normal_consistency_loss(vertices, faces):
    """
    Penalise adjacent faces whose normals point in opposite directions.
    Builds a face-adjacency list once per call (cheap at these mesh sizes).
    """
    normals = face_normals(vertices, faces)   # [B, F, 3]
    F_count = faces.shape[0]

    # Build edge → face mapping
    edge_to_faces = {}
    for fi, tri in enumerate(faces.tolist()):
        for i in range(3):
            e = tuple(sorted((tri[i], tri[(i+1) % 3])))
            edge_to_faces.setdefault(e, []).append(fi)

    adj_pairs = [pair for pair in edge_to_faces.values() if len(pair) == 2]
    if not adj_pairs:
        return torch.tensor(0.0, device=vertices.device)

    pairs = torch.tensor(adj_pairs, dtype=torch.long, device=vertices.device)
    n0 = normals[:, pairs[:, 0]]   # [B, E, 3]
    n1 = normals[:, pairs[:, 1]]   # [B, E, 3]
    # Loss = 1 - cos(θ): 0 when parallel, 2 when anti-parallel
    return (1.0 - (n0 * n1).sum(-1)).mean()


def compute_loss(pred_verts, gt_pts, faces, lap,
                 w_cd=1.0, w_edge=0.1, w_lap=0.5, w_norm=0.1):
    cd   = chamfer_distance(pred_verts, gt_pts)
    el   = edge_loss(pred_verts, faces)
    lpl  = laplacian_loss(pred_verts, lap)
    nc   = normal_consistency_loss(pred_verts, faces)
    total = w_cd*cd + w_edge*el + w_lap*lpl + w_norm*nc
    return total, {
        "chamfer": cd.item(), "edge": el.item(),
        "laplacian": lpl.item(), "normal": nc.item(),
    }


def compute_adversarial_loss(pred_logits, real_logits):
    fake = F.binary_cross_entropy_with_logits(
        pred_logits, torch.zeros_like(pred_logits))
    real = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits))
    return (fake + real) / 2.0


def fool_discriminator(pred_logits):
    return F.binary_cross_entropy_with_logits(
        pred_logits, torch.ones_like(pred_logits))


# Quick loss check
pred_v   = out["vertices"][2]
gt_pts   = torch.randn(2, 4096, 3, device=DEVICE) * 0.4
faces_s3 = out["faces"][2].to(DEVICE)
lap_s3   = out["laplacians"][2].to(DEVICE)
loss_val, breakdown = compute_loss(pred_v, gt_pts, faces_s3, lap_s3)
print(f"\nLoss sanity check:")
print(f"  Total : {loss_val.item():.4f}")
for k, v in breakdown.items():
    print(f"  {k:12s}: {v:.4f}")

# %% [markdown]
# ## Optimizer setup
#
# Three parameter groups with separate learning rates:
#
# | Group | LR | Rationale |
# |---|---|---|
# | DINOv2 backbone | 0 (frozen) | Pre-trained, no gradient |
# | FPN lateral + output convs | 1e-4 | Adapts routing to OmniObject3D data |
# | GCN projectors + deform stages | 3e-4 | Main learnable path |

# %%
def make_optimizer(model, lr_fpn=1e-4, lr_gcn=3e-4, weight_decay=1e-4):
    fpn_params = (
        list(model.encoder.lateral.parameters()) +
        list(model.encoder.output_convs.parameters())
    )
    gcn_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    return torch.optim.AdamW([
        {"params": fpn_params, "lr": lr_fpn},
        {"params": gcn_params, "lr": lr_gcn},
    ], weight_decay=weight_decay)

# %% [markdown]
# ## Checkpoint utilities

# %%
def save_ckpt(model, optimizer, disc, optimizer_disc, epoch, path, val_cd=None):
    torch.save({
        "epoch":          epoch,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "discriminator":  disc.state_dict(),
        "optimizer_disc": optimizer_disc.state_dict(),
        "val_cd":         val_cd,
    }, path)
    print(f"Saved → {path}")


def load_ckpt(model, path, optimizer=None, disc=None,
              optimizer_disc=None, device=None):
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer:      optimizer.load_state_dict(ckpt["optimizer"])
    if disc and "discriminator" in ckpt:
        disc.load_state_dict(ckpt["discriminator"])
    if optimizer_disc and "optimizer_disc" in ckpt:
        optimizer_disc.load_state_dict(ckpt["optimizer_disc"])
    print(f"Loaded epoch={ckpt['epoch']}  val_cd={ckpt.get('val_cd')}")
    return ckpt

# %% [markdown]
# ## Smoke test — end-to-end with real dataset batch

# %%
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omniobject3d_dinov2 import OmniObject3DDataset, collate_omni
from torch.utils.data import DataLoader

OMNI_ROOT = "data/OmniObject3D"

smoke_ds     = OmniObject3DDataset(OMNI_ROOT, split="train",
                                   categories=["chair", "sofa"])
smoke_loader = DataLoader(smoke_ds, batch_size=2, shuffle=True,
                          num_workers=0, collate_fn=collate_omni)

smoke_model = OmniMesh3D(encoder=encoder, hidden_dim=512,
                         n_stages=3, n_gcn_blocks=5).to(DEVICE)
smoke_disc  = MeshDiscriminator(hidden_dim=256).to(DEVICE)

smoke_opt      = make_optimizer(smoke_model)
smoke_opt_disc = torch.optim.AdamW(smoke_disc.parameters(),
                                   lr=3e-4, weight_decay=1e-4)

STAGE_WEIGHTS = [0.15, 0.25, 0.60]

smoke_model.train(); smoke_disc.train()
batch = next(iter(smoke_loader))

img   = batch["image"].to(DEVICE)
gt    = batch["gt_points"].to(DEVICE)
rot   = batch["rot"].to(DEVICE)
trans = batch["trans"].to(DEVICE)
focal = batch["focal"].to(DEVICE)

# Generator step
smoke_opt.zero_grad()
out = smoke_model(img, rot, trans, focal)
loss_g = torch.tensor(0.0, device=DEVICE)
for verts, faces, lap, w in zip(out["vertices"], out["faces"],
                                 out["laplacians"], STAGE_WEIGHTS):
    l, breakdown = compute_loss(verts, gt, faces.to(DEVICE), lap.to(DEVICE))
    loss_g = loss_g + w * l

adj_final  = smoke_model.adj_2.to(DEVICE)
fake_logits = smoke_disc(out["vertices"][-1], adj_final)
loss_adv    = fool_discriminator(fake_logits)
loss_g      = loss_g + 0.1 * loss_adv
loss_g.backward()
torch.nn.utils.clip_grad_norm_(smoke_model.parameters(), 1.0)
smoke_opt.step()

# Discriminator step
smoke_opt_disc.zero_grad()
with torch.no_grad():
    fake_logits = smoke_disc(out["vertices"][-1].detach(), adj_final)
real_logits = smoke_disc(gt[:, :out["vertices"][-1].shape[1]], adj_final)
loss_d = compute_adversarial_loss(fake_logits, real_logits)
loss_d.backward()
torch.nn.utils.clip_grad_norm_(smoke_disc.parameters(), 1.0)
smoke_opt_disc.step()

print(f"\nSmoke test:")
print(f"  G loss : {loss_g.item():.4f}")
print(f"  Adv    : {loss_adv.item():.4f}")
print(f"  D loss : {loss_d.item():.4f}")
print(f"  {breakdown}")
print("Smoke test PASSED ✓")

# ──────────────────────────────────────────────────────────────────────────────
# What's next:
#   omniobject3d_dinov2.py   — dataset ✓
#   omniobject3d_model.py    — model   ✓  (this file)
#   omniobject3d_train.py    — training loop with multi-view loss
# ──────────────────────────────────────────────────────────────────────────────
