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
def project_vertices(vertices, rot, trans, focal):
    v_cam = torch.bmm(vertices, rot.transpose(1, 2)) + trans.unsqueeze(1)
    z = v_cam[..., 2:3].clamp(min=1e-6)
    f = focal.view(-1, 1, 1)
    xy = f * v_cam[..., :2] / z
    return xy.clamp(-1, 1)


def sample_features(feature_maps, coords):
    grid = coords.unsqueeze(1)
    sampled = []
    for fm in feature_maps:
        s = F.grid_sample(fm, grid, align_corners=True, mode="bilinear", padding_mode="border")
        sampled.append(s.squeeze(2).permute(0, 2, 1))
    return torch.cat(sampled, dim=-1)


# DINOv2 + FPN Encoder
_DINOV2_MODEL = "dinov2_vits14"
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


# Mesh Discriminator
class MeshDiscriminator(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.gc1 = GraphConv(3, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.gc3 = GraphConv(hidden_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, verts, adj):
        h = self.gc1(verts, adj)
        h = self.gc2(h, adj)
        h = self.gc3(h, adj)
        h = self.pool(h.permute(0, 2, 1)).squeeze(-1)
        return self.head(h)


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
                and not a.get("truncated") and not a.get("occluded")]
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
        w, h = a["img_size"]
        focal = torch.tensor(a["focal_length"] / (max(w, h) / 2.), dtype=torch.float32)
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


def compute_adversarial_loss(pred_logits, real_logits):
    fake_loss = F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred_logits))
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    return (fake_loss + real_loss) / 2.0


def fool_discriminator(pred_logits):
    return F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred_logits))


def get_stage_weights(n_stages, base_weights=None):
    if base_weights is not None and len(base_weights) == n_stages:
        return base_weights
    weights = np.linspace(1.0, float(n_stages), n_stages)
    weights = weights / weights.sum()
    return weights.tolist()


def train(model, disc, train_loader, val_loader, optimizer, optimizer_disc, scheduler, epochs, device, rank, w_adv=0.15, ckpt_dir="./checkpoints", resume_ckpt=None):
    os.makedirs(ckpt_dir, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
    scaler_disc = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    model_inner = model.module if isinstance(model, DDP) else model
    disc_inner = disc.module if isinstance(disc, DDP) else disc
    model_n_stages = model_inner.n_stages
    stage_weights = get_stage_weights(model_n_stages, STAGE_WEIGHTS)

    start_epoch = 1
    if resume_ckpt and os.path.isfile(resume_ckpt):
        map_loc = {"cuda:0": f"cuda:{rank}"} if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(resume_ckpt, map_location=map_loc)
        model_inner.load_state_dict(ckpt["model"])
        disc_inner.load_state_dict(ckpt["discriminator"])
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer_disc.load_state_dict(ckpt["optimizer_disc"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        if rank == 0:
            print(f"Resumed from {resume_ckpt}  (epoch {ckpt['epoch']})")

    if rank == 0:
        print(f"Starting training from epoch {start_epoch} to {epochs}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        disc.train()
        epoch_loss = epoch_adv = epoch_disc_loss = 0.0

        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, disable=(rank != 0), desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            img = batch["image"].to(device)
            gt = batch["gt_points"].to(device)
            rot = batch["rot"].to(device)
            trans = batch["trans"].to(device)
            focal = batch["focal"].to(device)

            adj_final = getattr(model_inner, f"adj_{model_n_stages - 1}").to(device)

            # === Generator step ===
            optimizer.zero_grad()
            optimizer_disc.zero_grad()
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    out = model(img, rot, trans, focal)
                    loss_g = torch.tensor(0., device=device)
                    for verts, faces, lap, w in zip(out["vertices"], out["faces"], out["laplacians"], stage_weights):
                        l, _ = compute_loss(verts, gt, faces.to(device), lap.to(device))
                        loss_g = loss_g + w * l
                    for p in disc.parameters():
                        p.requires_grad_(False)
                    fake_logits = disc(out["vertices"][-1], adj_final)
                    for p in disc.parameters():
                        p.requires_grad_(True)
                    loss_adv_g = fool_discriminator(fake_logits)
                    loss_g = loss_g + w_adv * loss_adv_g
                scaler.scale(loss_g).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(img, rot, trans, focal)
                loss_g = torch.tensor(0., device=device)
                for verts, faces, lap, w in zip(out["vertices"], out["faces"], out["laplacians"], stage_weights):
                    l, _ = compute_loss(verts, gt, faces.to(device), lap.to(device))
                    loss_g = loss_g + w * l
                for p in disc.parameters():
                    p.requires_grad_(False)
                fake_logits = disc(out["vertices"][-1], adj_final)
                for p in disc.parameters():
                    p.requires_grad_(True)
                loss_adv_g = fool_discriminator(fake_logits)
                loss_g = loss_g + w_adv * loss_adv_g
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # === Discriminator step ===
            pred_verts_final = out["vertices"][-1].detach()
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        fake_logits = disc(pred_verts_final, adj_final)
                    real_logits = disc(gt[:, :pred_verts_final.shape[1], :], adj_final)
                    loss_d = compute_adversarial_loss(fake_logits, real_logits)
                scaler_disc.scale(loss_d).backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()
            else:
                with torch.no_grad():
                    fake_logits = disc(pred_verts_final, adj_final)
                real_logits = disc(gt[:, :pred_verts_final.shape[1], :], adj_final)
                loss_d = compute_adversarial_loss(fake_logits, real_logits)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                optimizer_disc.step()

            epoch_loss += loss_g.item()
            epoch_adv += loss_adv_g.item()
            epoch_disc_loss += loss_d.item()
            pbar.set_postfix(loss=f"{loss_g.item():.4f}", adv=f"{loss_adv_g.item():.4f}", disc=f"{loss_d.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        avg_adv = epoch_adv / len(train_loader)
        avg_disc = epoch_disc_loss / len(train_loader)

        val_loss = None
        if val_loader and epoch % 5 == 0:
            val_loss = evaluate(model, val_loader, device, stage_weights)
            if rank == 0:
                print(f"Epoch {epoch}/{epochs} | val_loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch}/{epochs} | train: {avg_loss:.4f} | adv: {avg_adv:.4f} | disc: {avg_disc:.4f}")
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "model": model_inner.state_dict(),
                "discriminator": disc_inner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_disc": optimizer_disc.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"Checkpoint saved → {ckpt_path}")

    if rank == 0:
        save_quantized(model_inner, disc_inner, ckpt_dir)


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


def save_quantized(model, disc, ckpt_dir):
    model.eval()
    disc.eval()
    q_model = torch.ao.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)  # type: ignore[attr-defined]
    q_disc = torch.ao.quantization.quantize_dynamic(disc.cpu(), {nn.Linear}, dtype=torch.qint8)  # type: ignore[attr-defined]
    path_model = os.path.join(ckpt_dir, "model_quantized.pth")
    path_disc = os.path.join(ckpt_dir, "discriminator_quantized.pth")
    torch.save(q_model.state_dict(), path_model)
    torch.save(q_disc.state_dict(), path_disc)
    print(f"Quantized model saved → {path_model}")
    print(f"Quantized discriminator saved → {path_disc}")


def main():
    parser = argparse.ArgumentParser(description="Distributed training for Pixel2Mesh with DINOv2")
    parser.add_argument("--pix3d-root", type=str, default="data/pix3d", help="Path to Pix3D dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr-fpn", type=float, default=1e-4, help="Learning rate for FPN")
    parser.add_argument("--lr-gcn", type=float, default=5e-5, help="Learning rate for GCN")
    parser.add_argument("--lr-disc", type=float, default=3e-5, help="Learning rate for discriminator")
    parser.add_argument("--w-adv", type=float, default=0.15, help="Adversarial loss weight")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--disc-hidden-dim", type=int, default=256, help="Discriminator hidden dimension")
    parser.add_argument("--n-stages", type=int, default=4, help="Number of deformation stages")
    parser.add_argument("--n-gcn-blocks", type=int, default=3, help="GCN blocks per stage")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume-ckpt", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Running on rank {rank}/{world_size} with device {device}")

    encoder = DINOv2FPNEncoder(freeze_backbone=True).to(device)
    model = Furniture3D(encoder=encoder, hidden_dim=args.hidden_dim, n_stages=args.n_stages, n_gcn_blocks=args.n_gcn_blocks).to(device)
    disc = MeshDiscriminator(hidden_dim=args.disc_hidden_dim).to(device)

    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None, find_unused_parameters=False)
    disc = DDP(disc, device_ids=[rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

    fpn_params = list(model.module.encoder.lateral.parameters()) + list(model.module.encoder.output_convs.parameters())
    gcn_params = [p for n, p in model.module.named_parameters() if "encoder" not in n]

    optimizer = torch.optim.AdamW([
        {"params": fpn_params, "lr": args.lr_fpn},
        {"params": gcn_params, "lr": args.lr_gcn},
    ], weight_decay=1e-4)

    optimizer_disc = torch.optim.AdamW(disc.parameters(), lr=args.lr_disc, weight_decay=1e-4)
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
        disc_p = sum(p.numel() for p in disc.parameters())
        print(f"Model total params: {total_p:,}  trainable: {train_p:,}")
        print(f"Discriminator params: {disc_p:,}")

    train(model, disc, train_loader, val_loader, optimizer, optimizer_disc, scheduler,
          args.epochs, device, rank, w_adv=args.w_adv, ckpt_dir=args.ckpt_dir, resume_ckpt=args.resume_ckpt)

    cleanup_distributed()


if __name__ == "__main__":
    main()
