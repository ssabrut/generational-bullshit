"""
Pixel2Mesh with DINOv2 encoder — single-device training on Pix3D chairs.

Architecture:
  Encoder  : DINOv2 ViT-S/14 (frozen) + 4-level FPN → 4 × 256-ch feature maps
  Template : icosphere, 3 stages: 162 → 642 → 2562 vertices
  GCN      : 6 GraphConv layers per stage, W0/W1 style (no BN, LayerNorm instead)
  Projection: Pix3D perspective camera (R, T, focal_px)

Feature dims:
  Stage 1 input : 4×256 (img) + 3 (xyz)         = 1027
  Stage 2+ input: 4×256 (img) + 256 (hidden) + 3 = 1283
  Hidden dim    : 256

Usage:
  conda run -n AI python scripts/train_pix2mesh_chair.py \
      --data data/pix3d --epochs 100 --batch-size 4 --device cpu
"""

import argparse, json, math, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ── constants ─────────────────────────────────────────────────────────────────
IMG_MEAN   = [0.485, 0.456, 0.406]
IMG_STD    = [0.229, 0.224, 0.225]
FPN_CH     = 256
TAP_BLOCKS = [2, 5, 8, 11]       # DINOv2 ViT-S/14 blocks to tap
IMG_FEAT   = FPN_CH * 4          # 1024 per vertex
HIDDEN     = 256
GCN_LAYERS = 6                   # graph conv layers per stage

# ── icosphere template ────────────────────────────────────────────────────────

def _make_icosphere(subdivisions=2):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts_list = [
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ]
    verts_list = [torch.tensor(v, dtype=torch.float32) for v in verts_list]
    for i, v in enumerate(verts_list):
        verts_list[i] = v / v.norm()

    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]
    mid_cache = {}

    def _mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid_cache:
            m = (verts_list[a] + verts_list[b]) / 2.0
            mid_cache[key] = len(verts_list)
            verts_list.append(m / m.norm())
        return mid_cache[key]

    for _ in range(subdivisions):
        new_faces = []
        for v0, v1, v2 in faces:
            m01 = _mid(v0, v1); m12 = _mid(v1, v2); m20 = _mid(v2, v0)
            new_faces += [[v0,m01,m20],[v1,m12,m01],[v2,m20,m12],[m01,m12,m20]]
        faces = new_faces

    verts = torch.stack(verts_list)
    verts[:, 1] *= 0.8            # slight Y squash → ellipsoid
    return verts, torch.tensor(faces, dtype=torch.long)


def _subdivide(verts, faces):
    """Returns (new_verts, new_faces, unpool_matrix [V_out, V_in])."""
    V = verts.shape[0]
    mid_cache, mid_parents = {}, []

    def _mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid_cache:
            mid_cache[key] = V + len(mid_parents)
            mid_parents.append((a, b))
        return mid_cache[key]

    new_faces = []
    for tri in faces.tolist():
        v0, v1, v2 = tri
        m01 = _mid(v0, v1); m12 = _mid(v1, v2); m20 = _mid(v2, v0)
        new_faces += [[v0,m01,m20],[v1,m12,m01],[v2,m20,m12],[m01,m12,m20]]

    mid_v  = torch.stack([(verts[a] + verts[b]) / 2.0 for a, b in mid_parents])
    new_v  = torch.cat([verts, mid_v], 0)
    new_f  = torch.tensor(new_faces, dtype=torch.long)
    V_out  = new_v.shape[0]
    unpool = torch.zeros(V_out, V)
    for i in range(V):
        unpool[i, i] = 1.0
    for j, (a, b) in enumerate(mid_parents):
        unpool[V + j, a] = 0.5
        unpool[V + j, b] = 0.5
    return new_v, new_f, unpool


def _norm_adj(faces, V):
    adj = torch.zeros(V, V)
    for tri in faces.tolist():
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            adj[a, b] = adj[b, a] = 1.0
    adj += torch.eye(V)
    return adj / adj.sum(1, keepdim=True).clamp(1)


def _laplacian(faces, V):
    adj = torch.zeros(V, V)
    for tri in faces.tolist():
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            adj[a, b] = adj[b, a] = 1.0
    return torch.eye(V) - adj / adj.sum(1, keepdim=True).clamp(1)


def build_templates():
    v1, f1       = _make_icosphere(subdivisions=2)
    v2, f2, u12 = _subdivide(v1, f1)
    v3, f3, u23 = _subdivide(v2, f2)
    return (
        (v1, f1, _norm_adj(f1, v1.shape[0]), _laplacian(f1, v1.shape[0])),
        (v2, f2, _norm_adj(f2, v2.shape[0]), _laplacian(f2, v2.shape[0])),
        (v3, f3, _norm_adj(f3, v3.shape[0]), _laplacian(f3, v3.shape[0])),
        u12, u23,
    )


# ── DINOv2 + FPN encoder ──────────────────────────────────────────────────────

class DINOv2FPNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        for p in self.dino.parameters():
            p.requires_grad = False

        D = self.dino.embed_dim
        self._feats = {}
        for blk_idx in TAP_BLOCKS:
            self.dino.blocks[blk_idx].register_forward_hook(self._hook(blk_idx))

        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv2d(D, FPN_CH, 1, bias=False), nn.GroupNorm(16, FPN_CH), nn.ReLU(inplace=True))
            for _ in TAP_BLOCKS
        ])
        self.out_conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(FPN_CH, FPN_CH, 3, padding=1, bias=False), nn.GroupNorm(16, FPN_CH), nn.ReLU(inplace=True))
            for _ in TAP_BLOCKS
        ])

    def _hook(self, idx):
        def fn(_, __, out): self._feats[idx] = out[:, 1:]   # drop CLS
        return fn

    def forward(self, x):
        B, C, H, W = x.shape
        self._feats.clear()
        with torch.no_grad():
            self.dino(x)
        hp, wp = H // self.dino.patch_size, W // self.dino.patch_size

        lats = []
        for i, blk in enumerate(TAP_BLOCKS):
            t = self._feats[blk]                                       # [B, N, D]
            s = t.permute(0, 2, 1).reshape(B, -1, hp, wp)
            lats.append(self.lateral[i](s))

        # top-down FPN merge
        fpn = [None] * 4
        fpn[3] = lats[3]
        for i in range(2, -1, -1):
            fpn[i] = lats[i] + F.interpolate(fpn[i+1], size=lats[i].shape[-2:], mode="nearest")

        sizes = [128, 64, 32, 16]
        return [
            self.out_conv[i](F.interpolate(fpn[i], size=(sizes[i], sizes[i]), mode="bilinear", align_corners=False))
            for i in range(4)
        ]


# ── graph conv & decoder ──────────────────────────────────────────────────────

class GraphConv(nn.Module):
    """output = relu(A @ (X @ W0) + X @ W1 + bias) with LayerNorm."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w0   = nn.Linear(in_dim, out_dim, bias=False)
        self.w1   = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.ln   = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        agg = torch.einsum("vw,bwf->bvf", adj, x)
        return F.relu(self.ln(self.w0(agg) + self.w1(x) + self.bias))


class GCNStage(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_layers=GCN_LAYERS):
        super().__init__()
        self.proj  = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.convs = nn.ModuleList([GraphConv(hidden, hidden) for _ in range(n_layers)])
        self.head  = nn.Linear(hidden, 3)

    def forward(self, img_feat, prev_h, verts, adj):
        # img_feat: [B, V, IMG_FEAT], prev_h: [B, V, HIDDEN] or None, verts: [B, V, 3]
        parts = [img_feat, verts] if prev_h is None else [img_feat, prev_h, verts]
        h = self.proj(torch.cat(parts, dim=-1))
        for conv in self.convs:
            h = conv(h, adj)
        return verts + self.head(h), h     # (new_verts, hidden)


# ── full model ────────────────────────────────────────────────────────────────

class Pixel2MeshDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DINOv2FPNEncoder()
        self.stage1  = GCNStage(IMG_FEAT + 3,          HIDDEN)
        self.stage2  = GCNStage(IMG_FEAT + HIDDEN + 3, HIDDEN)
        self.stage3  = GCNStage(IMG_FEAT + HIDDEN + 3, HIDDEN)

        (v1, f1, a1, l1), (v2, f2, a2, l2), (v3, f3, a3, l3), u12, u23 = build_templates()
        print(f"Templates: {v1.shape[0]} / {v2.shape[0]} / {v3.shape[0]} verts")

        # store all as buffers so they move with .to(device)
        self.register_buffer("v1",  v1);  self.register_buffer("f1",  f1)
        self.register_buffer("v2",  v2);  self.register_buffer("f2",  f2)
        self.register_buffer("v3",  v3);  self.register_buffer("f3",  f3)
        self.register_buffer("a1",  a1);  self.register_buffer("l1",  l1)
        self.register_buffer("a2",  a2);  self.register_buffer("l2",  l2)
        self.register_buffer("a3",  a3);  self.register_buffer("l3",  l3)
        self.register_buffer("u12", u12)
        self.register_buffer("u23", u23)

    @staticmethod
    def _sample(feat_maps, coords):
        grid = coords.unsqueeze(1)
        parts = []
        for fm in feat_maps:
            s = F.grid_sample(fm, grid, mode="bilinear", align_corners=True, padding_mode="border")
            parts.append(s.squeeze(2).permute(0, 2, 1))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _project(verts, rot, trans, focal):
        """Pix3D perspective projection → normalised image coords [-1, 1]."""
        v_cam = torch.bmm(verts, rot.transpose(1, 2)) + trans.unsqueeze(1)
        z  = v_cam[..., 2:3].clamp(min=1e-4)
        f  = focal.view(-1, 1, 1)
        xy = f * v_cam[..., :2] / z
        return xy.clamp(-1, 1)

    @staticmethod
    def _unpool(x, mat):
        return torch.einsum("oi,bif->bof", mat, x)

    def forward(self, image, rot, trans, focal):
        B = image.shape[0]
        feat_maps = self.encoder(image)

        # ── stage 1 ──
        v1 = self.v1.unsqueeze(0).expand(B, -1, -1)
        xy1  = self._project(v1, rot, trans, focal)
        imf1 = self._sample(feat_maps, xy1)
        p1, h1 = self.stage1(imf1, None, v1, self.a1)

        # ── unpool 1→2 ──
        h1u = self._unpool(h1, self.u12)
        p1u = self._unpool(p1, self.u12)

        # ── stage 2 ──
        xy2  = self._project(p1u, rot, trans, focal)
        imf2 = self._sample(feat_maps, xy2)
        p2, h2 = self.stage2(imf2, h1u, p1u, self.a2)

        # ── unpool 2→3 ──
        h2u = self._unpool(h2, self.u23)
        p2u = self._unpool(p2, self.u23)

        # ── stage 3 ──
        xy3  = self._project(p2u, rot, trans, focal)
        imf3 = self._sample(feat_maps, xy3)
        p3, _ = self.stage3(imf3, h2u, p2u, self.a3)

        return (p1, self.f1, self.l1), (p2, self.f2, self.l2), (p3, self.f3, self.l3)


# ── losses ────────────────────────────────────────────────────────────────────

def chamfer(pred, gt):
    d = ((pred.unsqueeze(2) - gt.unsqueeze(1)) ** 2).sum(-1)
    return d.min(2).values.mean() + d.min(1).values.mean()


def edge_reg(verts, faces):
    v0, v1, v2 = verts[:, faces[:,0]], verts[:, faces[:,1]], verts[:, faces[:,2]]
    return ((v0-v1).norm(dim=-1).mean() + (v1-v2).norm(dim=-1).mean() + (v2-v0).norm(dim=-1).mean()) / 3


def lap_reg(verts, lap):
    return (torch.einsum("vw,bwc->bvc", lap, verts) ** 2).mean()


def total_loss(pred_verts, gt_pts, faces, lap, w_cd=1.0, w_e=0.1, w_l=0.3):
    cd  = chamfer(pred_verts, gt_pts)
    el  = edge_reg(pred_verts, faces)
    ll  = lap_reg(pred_verts, lap)
    return w_cd * cd + w_e * el + w_l * ll, {"cd": cd.item(), "edge": el.item(), "lap": ll.item()}


# ── dataset ───────────────────────────────────────────────────────────────────

def load_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            p = line.strip().split()
            if not p: continue
            if p[0] == "v":
                verts.append([float(x) for x in p[1:4]])
            elif p[0] == "f":
                idx = [int(x.split("/")[0]) - 1 for x in p[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i+1]])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def normalise_mesh(v):
    v = v - (v.max(0).values + v.min(0).values) / 2
    scale = v.norm(dim=-1).max().clamp(1e-8)
    return v / scale, scale.item()


def sample_surface(verts, faces, n=2048):
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    areas = 0.5 * torch.cross(v1-v0, v2-v0, dim=-1).norm(dim=-1)
    prob  = (areas / areas.sum().clamp(1e-8)).numpy()
    fi    = torch.from_numpy(np.random.choice(len(faces), n, p=prob))
    r1    = torch.rand(n).sqrt()
    u, v, w = 1-r1, r1*(1-torch.rand(n)), r1*torch.rand(n)
    return u[:,None]*verts[faces[fi,0]] + v[:,None]*verts[faces[fi,1]] + w[:,None]*verts[faces[fi,2]]


class Pix3DChairDataset(Dataset):
    def __init__(self, root, split="train", img_size=224, n_pts=2048):
        self.root  = root
        self.n_pts = n_pts
        with open(os.path.join(root, "pix3d.json")) as f:
            anns = json.load(f)
        anns = [
            a for a in anns
            if a["category"] == "chair"
            and not a.get("truncated")
            and not a.get("occluded")
        ]
        random.seed(42); random.shuffle(anns)
        n = len(anns)
        self.samples = {"train": anns[:int(.8*n)], "val": anns[int(.8*n):int(.9*n)], "test": anns[int(.9*n):]}[split]

        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        a = self.samples[idx]
        img = self.tfm(Image.open(os.path.join(self.root, a["img"])).convert("RGB"))

        verts, faces = load_obj(os.path.join(self.root, a["model"]))
        verts, scale = normalise_mesh(verts)
        gt_pts = sample_surface(verts, faces, self.n_pts)

        rot   = torch.tensor(a["rot_mat"],   dtype=torch.float32)
        trans = torch.tensor(a["trans_mat"], dtype=torch.float32) / scale
        w, h  = a["img_size"]
        focal = torch.tensor(a["focal_length"] / (max(w, h) / 2.0), dtype=torch.float32)

        return {"image": img, "gt": gt_pts, "rot": rot, "trans": trans, "focal": focal}


# ── training ──────────────────────────────────────────────────────────────────

STAGE_W = [0.2, 0.3, 0.5]


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            img   = batch["image"].to(device)
            gt    = batch["gt"].to(device)
            rot   = batch["rot"].to(device)
            trans = batch["trans"].to(device)
            focal = batch["focal"].to(device)

            (p1, f1, l1), (p2, f2, l2), (p3, f3, l3) = model(img, rot, trans, focal)

            loss = torch.tensor(0., device=device)
            for pv, pf, pl, sw in [(p1,f1,l1,STAGE_W[0]), (p2,f2,l2,STAGE_W[1]), (p3,f3,l3,STAGE_W[2])]:
                l, _ = total_loss(pv, gt, pf, pl)
                loss = loss + sw * l

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += loss.item(); n += 1

    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/pix3d")
    parser.add_argument("--out",        default="runs/pix2mesh_chair")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr-fpn",     type=float, default=1e-4)
    parser.add_argument("--lr-gcn",     type=float, default=5e-4)
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    root   = args.data if os.path.isabs(args.data) else os.path.join(os.path.dirname(os.path.dirname(__file__)), args.data)
    outdir = args.out  if os.path.isabs(args.out)  else os.path.join(os.path.dirname(os.path.dirname(__file__)), args.out)
    os.makedirs(outdir, exist_ok=True)

    device = torch.device(args.device)

    print("Building model …")
    model = Pixel2MeshDINO().to(device)

    fpn_params = list(model.encoder.lateral.parameters()) + list(model.encoder.out_conv.parameters())
    gcn_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    optimizer  = torch.optim.AdamW(
        [{"params": fpn_params, "lr": args.lr_fpn},
         {"params": gcn_params, "lr": args.lr_gcn}],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    print("Loading dataset …")
    train_ds = Pix3DChairDataset(root, "train")
    val_ds   = Pix3DChairDataset(root, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss  = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  train={tr_loss:.4f}  val={val_loss:.4f}")

        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(ckpt, os.path.join(outdir, "last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(outdir, "best.pt"))
            print(f"  ↑ new best val loss: {best_val:.4f}")

    print(f"Done. Best val: {best_val:.4f}. Saved to {outdir}/best.pt")


if __name__ == "__main__":
    main()
