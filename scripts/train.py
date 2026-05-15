import os
import json
import random
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import datetime
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Constants
PIX3D_CATS = ["bed", "bookcase", "chair", "desk", "misc", "sofa", "table", "tool", "wardrobe"]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]
STAGE_WEIGHTS = [0.15, 0.25, 0.3, 0.3]


def setup_distributed():
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"

    if "GLOO_SOCKET_IFNAME" in os.environ:
        print(f"GLOO_SOCKET_IFNAME: {os.environ.get('GLOO_SOCKET_IFNAME')}")

    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")

    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    print(f"Using backend: {backend}")

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=30)
    )


def cleanup_distributed():
    dist.destroy_process_group()


# Mesh utilities
def get_unit_cube():
    v = torch.tensor([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=torch.float32)
    f = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 7, 4], [0, 3, 7],
        [1, 5, 6], [1, 6, 2],
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
        m01, m12, m20 = midpoint(v0, v1), midpoint(v1, v2), midpoint(v2, v0)
        new_faces += [[v0, m01, m20], [v1, m12, m01], [v2, m20, m12], [m01, m12, m20]]

    new_v = torch.cat([vertices, torch.stack(mids).to(device)], 0) if mids else vertices
    new_f = torch.tensor(new_faces, dtype=torch.long, device=device)
    return new_v, new_f


def build_adjacency(vertices, faces):
    V = vertices.shape[0]
    adj = torch.zeros(V, V)
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            adj[a, b] = adj[b, a] = 1.0
    adj += torch.eye(V)
    return adj / adj.sum(1, keepdim=True).clamp(min=1)


def build_laplacian(vertices, faces):
    V = vertices.shape[0]
    adj = torch.zeros(V, V)
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            adj[a, b] = adj[b, a] = 1.0
    return torch.eye(V) - adj / adj.sum(1, keepdim=True).clamp(1)


# Feature extraction
def project_vertices(vertices, rot, trans, focal, return_raw=False):
    v_cam = torch.bmm(vertices, rot.transpose(1, 2)) + trans.unsqueeze(1)
    z = v_cam[..., 2:3].clamp(min=1e-6)
    f = focal.view(-1, 1, 1)
    xy = f * v_cam[..., :2] / z
    if return_raw:
        return xy.clamp(-1, 1), xy
    return xy.clamp(-1, 1)


def verify_projection(loader, device, n_batches=3, sat_threshold=0.5):
    """Sanity check: project the unit cube template with each batch's camera params
    and report how often coords saturate at ±1. High saturation => GT mesh
    normalisation and camera (rot/trans/focal) live in different frames, so the
    GCN sees near-identical features per vertex."""
    print("\n=== Projection verification ===")
    verts_tmpl, _ = get_unit_cube()
    any_bad = False
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        rot = batch["rot"].to(device)
        trans = batch["trans"].to(device)
        focal = batch["focal"].to(device)
        gt = batch["gt_points"].to(device)
        B = rot.shape[0]
        verts = verts_tmpl.to(device).unsqueeze(0).expand(B, -1, -1)

        _, raw = project_vertices(verts, rot, trans, focal, return_raw=True)
        sat_frac = ((raw.abs() >= 1.0).any(dim=-1)).float().mean().item()
        x_min, x_max = raw[..., 0].min().item(), raw[..., 0].max().item()
        y_min, y_max = raw[..., 1].min().item(), raw[..., 1].max().item()

        gt_min = gt.amin(dim=(0, 1)).tolist()
        gt_max = gt.amax(dim=(0, 1)).tolist()
        t_min = trans.amin(dim=0).tolist()
        t_max = trans.amax(dim=0).tolist()

        flag = "  <-- SATURATED" if sat_frac > sat_threshold else ""
        print(f"[batch {bi}] B={B}")
        print(f"  cube proj raw: x=[{x_min:+.3f}, {x_max:+.3f}]  y=[{y_min:+.3f}, {y_max:+.3f}]")
        print(f"  saturated vertices (|x| or |y| >= 1): {sat_frac*100:.1f}%{flag}")
        print(f"  gt point range : x=[{gt_min[0]:+.3f},{gt_max[0]:+.3f}] y=[{gt_min[1]:+.3f},{gt_max[1]:+.3f}] z=[{gt_min[2]:+.3f},{gt_max[2]:+.3f}]")
        print(f"  trans range    : x=[{t_min[0]:+.3f},{t_max[0]:+.3f}] y=[{t_min[1]:+.3f},{t_max[1]:+.3f}] z=[{t_min[2]:+.3f},{t_max[2]:+.3f}]")
        print(f"  focal range    : [{focal.min().item():.3f}, {focal.max().item():.3f}]")
        if sat_frac > sat_threshold:
            any_bad = True

    if any_bad:
        print("\n[WARN] High saturation detected. The unit cube template projects outside")
        print("       the image. Likely cause: GT mesh is normalised to unit-norm but")
        print("       trans/focal come from raw Pix3D coords. Fix: either skip")
        print("       normalise_mesh and predict in camera space, or scale trans by")
        print("       the same factor used to normalise the mesh.")
    else:
        print("\n[OK] Projection coords mostly inside [-1, 1].")
    print("=== End verification ===\n")


def sample_features(feature_maps, coords):
    grid = coords.unsqueeze(1)
    sampled = []
    for fm in feature_maps:
        s = F.grid_sample(fm, grid, align_corners=True, mode="bilinear", padding_mode="border")
        sampled.append(s.squeeze(2).permute(0, 2, 1))
    return torch.cat(sampled, dim=-1)


# DINOv2 + FPN Encoder
_DINOV2_MODEL = "dinov2_vitb14"
_FPN_CHANNELS = 256
_TAP_BLOCKS = [2, 5, 8, 11]


class DINOv2FPNEncoder(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", _DINOV2_MODEL, pretrained=True)
        embed_dim = self.dino.embed_dim

        self._feats = {}
        self._hooks = []
        for blk_idx in _TAP_BLOCKS:
            hook = self.dino.blocks[blk_idx].register_forward_hook(self._make_hook(blk_idx))
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

        self.feat_dims = [_FPN_CHANNELS] * 4

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
            spatial = self._patch_tokens_to_spatial(self._feats[blk_idx], h_p, w_p)
            laterals.append(self.lateral[i](spatial))

        fpn = [None] * 4
        fpn[3] = laterals[3]
        for i in range(2, -1, -1):
            upsampled = F.interpolate(fpn[i + 1], size=laterals[i].shape[-2:], mode="nearest")
            fpn[i] = laterals[i] + upsampled

        target_sizes = [128, 64, 32, 16]
        out = []
        for i in range(4):
            fm = F.interpolate(fpn[i], size=(target_sizes[i], target_sizes[i]), mode="bilinear", align_corners=False)
            out.append(self.output_convs[i](fm))

        return out


# GCN layers
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_self = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        B, V, _ = x.shape
        neigh = torch.einsum("vw,bwf->bvf", adj, x)
        out = self.fc_self(x) + self.fc_neigh(neigh) + self.bias
        out = self.bn(out.reshape(B * V, -1)).reshape(B, V, -1)
        return F.relu(out)


class GCNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim, bias=False) if in_dim != hidden_dim else nn.Identity()

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
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, feats, verts, adj):
        h = feats
        for blk in self.blocks:
            h = blk(h, adj)
        return verts + self.disp_head(h), h


# Main model
class Furniture3D(nn.Module):
    def __init__(self, encoder, hidden_dim=256, n_stages=3, n_gcn_blocks=3):
        super().__init__()
        self.n_stages = n_stages
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        total_img = sum(self.encoder.feat_dims)

        self.projectors = nn.ModuleList()
        self.deform_stages = nn.ModuleList()
        for i in range(n_stages):
            in_dim = (total_img + 3) if i == 0 else (total_img + hidden_dim + 3)
            self.projectors.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            ))
            self.deform_stages.append(DeformStage(hidden_dim, hidden_dim, n_gcn_blocks))

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

            coords = project_vertices(verts, rot, trans, focal)
            img_feat = sample_features(feat_maps, coords)

            if prev_h is None:
                combined = torch.cat([img_feat, verts], dim=-1)
            else:
                if prev_h.shape[1] != V:
                    prev_h = F.interpolate(
                        prev_h.permute(0, 2, 1), size=V, mode="linear", align_corners=False
                    ).permute(0, 2, 1)
                combined = torch.cat([img_feat, prev_h, verts], dim=-1)

            vertex_feats = self.projectors[i](combined)
            new_verts, hidden = self.deform_stages[i](vertex_feats, verts, adj)

            all_verts.append(new_verts)
            all_faces.append(faces)
            all_laps.append(lap)
            prev_h = hidden

        return {"vertices": all_verts, "faces": all_faces, "laplacians": all_laps}


# Dataset
def load_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            p = line.strip().split()
            if not p:
                continue
            if p[0] == "v":
                verts.append([float(x) for x in p[1:4]])
            elif p[0] == "f":
                idx = [int(x.split("/")[0]) - 1 for x in p[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def normalise_mesh(v):
    v = v - (v.max(0).values + v.min(0).values) / 2
    return v / v.norm(dim=-1).max().clamp(1e-8)


def sample_surface(verts, faces, n=2048):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)
    prob = (areas / areas.sum().clamp(1e-8)).numpy()
    fi = torch.from_numpy(np.random.choice(len(faces), n, p=prob))
    r1 = torch.rand(n).sqrt()
    u, v, w = 1 - r1, r1 * (1 - torch.rand(n)), r1 * torch.rand(n)
    return (u[:, None] * verts[faces[fi, 0]]
            + v[:, None] * verts[faces[fi, 1]]
            + w[:, None] * verts[faces[fi, 2]])


class Pix3DDataset(Dataset):
    def __init__(self, root, split="train", categories=None, img_size=224, n_pts=2048):
        self.root = root
        self.n_pts = n_pts
        with open(os.path.join(root, "pix3d.json")) as f:
            anns = json.load(f)
        cats = set(categories or PIX3D_CATS)
        anns = [a for a in anns if a["category"] in cats
                and not a.get("truncated") and not a.get("occluded")
                and a.get("trans_mat") and a.get("focal_length") and a.get("img_size")]
        random.seed(42)
        random.shuffle(anns)
        n = len(anns)
        splits = {
            "train": anns[:int(0.8 * n)],
            "val": anns[int(0.8 * n):int(0.9 * n)],
            "test": anns[int(0.9 * n):]
        }
        self.samples = splits[split]
        from torchvision import transforms
        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(DINO_MEAN, DINO_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a = self.samples[idx]
        image = self.tfm(Image.open(os.path.join(self.root, a["img"])).convert("RGB"))
        verts, faces = load_obj(os.path.join(self.root, a["model"]))
        verts = normalise_mesh(verts)
        gt_pts = sample_surface(verts, faces, self.n_pts)
        rot = torch.tensor(a["rot_mat"], dtype=torch.float32)
        trans = torch.tensor(a["trans_mat"], dtype=torch.float32)
        # Clamp depth so the projection never explodes near z=0
        trans[2] = trans[2].clamp(min=1.0)
        w, h = a["img_size"]
        focal = torch.tensor(a["focal_length"] / (max(w, h) / 2.), dtype=torch.float32)
        # Clamp focal to a sane NDC range (extreme telephoto/fisheye → border samples)
        focal = focal.clamp(0.3, 6.0)
        return {"image": image, "gt_points": gt_pts, "rot": rot, "trans": trans, "focal": focal}


# Loss functions
def chamfer_distance(pred, gt):
    d = ((pred.unsqueeze(2) - gt.unsqueeze(1)) ** 2).sum(-1)
    return d.min(2).values.mean() + d.min(1).values.mean()


def edge_loss(vertices, faces):
    v0, v1, v2 = vertices[:, faces[:, 0]], vertices[:, faces[:, 1]], vertices[:, faces[:, 2]]
    return torch.stack([
        (v0 - v1).norm(dim=-1), (v1 - v2).norm(dim=-1), (v2 - v0).norm(dim=-1)
    ]).mean()


def laplacian_loss(vertices, lap):
    return (torch.einsum("vw,bwc->bvc", lap, vertices) ** 2).mean()


def compute_loss(pred_verts, gt_pts, faces, lap, w_cd=1.0, w_edge=0.1, w_lap=0.5):
    cd = chamfer_distance(pred_verts, gt_pts)
    el = edge_loss(pred_verts, faces)
    lpl = laplacian_loss(pred_verts, lap)
    total = w_cd * cd + w_edge * el + w_lap * lpl
    return total, {"chamfer": cd.item(), "edge": el.item(), "laplacian": lpl.item()}


def get_stage_weights(n_stages, base_weights=None):
    if base_weights is not None and len(base_weights) == n_stages:
        return base_weights
    weights = np.linspace(1.0, float(n_stages), n_stages)
    weights = weights / weights.sum()
    return weights.tolist()


def _render_step(step, epoch, pred_verts_list, faces_list, gt_pts, losses, save_path=None):
    n_stages = len(pred_verts_list)
    # columns: loss | gt | stage_1 .. stage_N
    n_cols = 1 + 1 + n_stages
    fig = plt.figure(figsize=(4 * n_cols, 4))

    # --- Loss plot ---
    ax_loss = fig.add_subplot(1, n_cols, 1)
    xs = range(len(losses))
    ax_loss.plot(xs, losses, label="loss", linewidth=1.2)
    ax_loss.set_title(f"Loss (epoch {epoch}, step {step})", fontsize=9)
    ax_loss.set_xlabel("step", fontsize=8)
    ax_loss.legend(fontsize=7)
    ax_loss.grid(True, linewidth=0.4, alpha=0.5)
    ax_loss.tick_params(labelsize=7)

    # --- GT point cloud ---
    ax_gt = fig.add_subplot(1, n_cols, 2, projection="3d")
    gp = gt_pts[0].detach().cpu().numpy()
    ax_gt.scatter(gp[:, 0], gp[:, 1], gp[:, 2], s=1, c="#20B2AA", alpha=0.4)
    ax_gt.set_title("GT Point Cloud", fontsize=9)
    ax_gt.set_axis_off()

    # --- Predicted mesh per stage ---
    colors = plt.cm.tab10(np.linspace(0, 1, n_stages))
    for i, (verts, faces) in enumerate(zip(pred_verts_list, faces_list)):
        ax = fig.add_subplot(1, n_cols, 3 + i, projection="3d")
        pv = verts[0].detach().cpu().numpy()
        ff = faces.cpu().numpy()
        tris = [pv[f] for f in ff]
        poly = Poly3DCollection(tris, alpha=0.3, facecolor=colors[i], edgecolor="#555", linewidth=0.1)
        ax.add_collection3d(poly)
        lim = 0.6
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_title(f"Stage {i + 1}  ({pv.shape[0]}V)", fontsize=9)
        ax.set_axis_off()

    fig.suptitle(f"Epoch {epoch} | Step {step}", fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.pause(0.001)
    plt.close(fig)


def train(model, train_loader, val_loader, optimizer, scheduler, epochs, device, rank, ckpt_dir="./checkpoints", resume_ckpt=None, vis_every=5, vis_dir=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    if rank == 0:
        plt.ion()
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    model_inner = model.module if isinstance(model, DDP) else model
    model_n_stages = model_inner.n_stages
    stage_weights = get_stage_weights(model_n_stages, STAGE_WEIGHTS)

    losses = []
    global_step = 0

    start_epoch = 1
    if resume_ckpt and os.path.isfile(resume_ckpt):
        map_loc = {"cuda:0": f"cuda:{rank}"} if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(resume_ckpt, map_location=map_loc)
        model_inner.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        if rank == 0:
            print(f"Resumed from {resume_ckpt}  (epoch {ckpt['epoch']})")

    if rank == 0:
        print(f"Starting training from epoch {start_epoch} to {epochs}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0

        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, disable=(rank != 0), desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            img = batch["image"].to(device)
            gt = batch["gt_points"].to(device)
            rot = batch["rot"].to(device)
            trans = batch["trans"].to(device)
            focal = batch["focal"].to(device)

            optimizer.zero_grad()
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    out = model(img, rot, trans, focal)
                    loss = torch.tensor(0., device=device)
                    for verts, faces, lap, w in zip(out["vertices"], out["faces"], out["laplacians"], stage_weights):
                        l, _ = compute_loss(verts, gt, faces.to(device), lap.to(device))
                        loss = loss + w * l
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(img, rot, trans, focal)
                loss = torch.tensor(0., device=device)
                for verts, faces, lap, w in zip(out["vertices"], out["faces"], out["laplacians"], stage_weights):
                    l, _ = compute_loss(verts, gt, faces.to(device), lap.to(device))
                    loss = loss + w * l
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            global_step += 1
            if rank == 0:
                losses.append(loss.item())
                if global_step % vis_every == 0:
                    save_path = None
                    if vis_dir:
                        save_path = os.path.join(vis_dir, f"epoch{epoch:04d}_step{global_step:07d}.png")
                    _render_step(global_step, epoch, out["vertices"], out["faces"], gt, losses, save_path=save_path)

        avg_loss = epoch_loss / len(train_loader)

        val_loss = None
        if val_loader and epoch % 5 == 0:
            val_loss = evaluate(model, val_loader, device, stage_weights)
            if rank == 0:
                print(f"Epoch {epoch}/{epochs} | val_loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch}/{epochs} | train: {avg_loss:.4f}")
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "model": model_inner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"Checkpoint saved → {ckpt_path}")

    if rank == 0:
        save_quantized(model_inner, ckpt_dir)


def evaluate(model, val_loader, device, stage_weights):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            gt = batch["gt_points"].to(device)
            rot = batch["rot"].to(device)
            trans = batch["trans"].to(device)
            focal = batch["focal"].to(device)

            out = model(img, rot, trans, focal)
            loss = torch.tensor(0., device=device)
            for verts, faces, lap, w in zip(out["vertices"], out["faces"], out["laplacians"], stage_weights):
                l, _ = compute_loss(verts, gt, faces.to(device), lap.to(device))
                loss = loss + w * l
            total_loss += loss.item()

    model.train()
    return total_loss / len(val_loader)


def save_quantized(model, ckpt_dir):
    model.eval()
    q_model = torch.ao.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)  # type: ignore[attr-defined]
    path_model = os.path.join(ckpt_dir, "model_quantized.pth")
    torch.save(q_model.state_dict(), path_model)
    print(f"Quantized model saved → {path_model}")


def main():
    parser = argparse.ArgumentParser(description="Distributed training for Pixel2Mesh with DINOv2")
    parser.add_argument("--pix3d-root", type=str, default="data/pix3d", help="Path to Pix3D dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr-fpn", type=float, default=1e-4, help="Learning rate for FPN")
    parser.add_argument("--lr-gcn", type=float, default=4e-5, help="Learning rate for GCN")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n-stages", type=int, default=4, help="Number of deformation stages")
    parser.add_argument("--n-gcn-blocks", type=int, default=3, help="GCN blocks per stage")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1), help="Number of data loading workers")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume-ckpt", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--vis-every", type=int, default=1, help="Save mesh visualization every N steps")
    parser.add_argument("--vis-dir", type=str, default=None, help="If set, also save each visualization PNG to this directory")
    parser.add_argument("--verify-projection", action="store_true", help="Run projection sanity check before training and exit")

    args = parser.parse_args()

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Running on rank {rank}/{world_size} with device {device}")

    encoder = DINOv2FPNEncoder(freeze_backbone=True).to(device)
    model = Furniture3D(encoder=encoder, hidden_dim=args.hidden_dim, n_stages=args.n_stages, n_gcn_blocks=args.n_gcn_blocks).to(device)

    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

    fpn_params = list(model.module.encoder.lateral.parameters()) + list(model.module.encoder.output_convs.parameters())
    gcn_params = [p for n, p in model.module.named_parameters() if "encoder" not in n]

    optimizer = torch.optim.AdamW([
        {"params": fpn_params, "lr": args.lr_fpn},
        {"params": gcn_params, "lr": args.lr_gcn},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_ds = Pix3DDataset(args.pix3d_root, "train", categories=["chair"])
    val_ds = Pix3DDataset(args.pix3d_root, "val", categories=["chair"])

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers)

    if rank == 0:
        print(f"Training dataset: {len(train_ds)} samples")
        print(f"Validation dataset: {len(val_ds)} samples")
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model total params: {total_p:,}  trainable: {train_p:,}")

    if args.verify_projection:
        if rank == 0:
            verify_projection(train_loader, device)
        cleanup_distributed()
        return

    train(model, train_loader, val_loader, optimizer, scheduler,
          args.epochs, device, rank, ckpt_dir=args.ckpt_dir, resume_ckpt=args.resume_ckpt,
          vis_every=args.vis_every, vis_dir=args.vis_dir)

    cleanup_distributed()


if __name__ == "__main__":
    main()
