"""
Pixel2Mesh with DINOv2 encoder — single-device training on Pix3D chairs.

Changes vs v2:
  - Normal-consistency loss (dihedral angle between adjacent faces)
  - OOB projection: off-image vertices get zeroed features instead of border clamp
  - Proj dropout removed (was double-dropout right at GCN input)
  - LR: FPN and GCN both at 5e-4 (FPN was under-trained at 1e-4)
  - LR schedule: 3-epoch linear warmup then cosine annealing
  - AMP (autocast + GradScaler) on CUDA
  - Best checkpoint saved by stage-3 Chamfer, not weighted total
  - Dataset: mask applied at native resolution (avoids bilinear bleed)
  - Dataset: isolated RNG so random.seed(42) doesn't leak into augmentation

Architecture:
  Encoder  : DINOv2 ViT-S/14 (frozen) + 4-level FPN → 4 × 256-ch feature maps
  Template : icosphere, 3 stages: 162 → 642 → 2562 vertices
  GCN      : 4 GraphConv layers per stage, hidden 128

Feature dims:
  Stage 1 input : 4×256 (img) + 3 (xyz)         = 1027
  Stage 2+ input: 4×256 (img) + 128 (hidden) + 3 = 1155

Usage:
  conda run -n AI python scripts/train_pix2mesh_chair.py \
      --data data/pix3d --epochs 100 --batch-size 4 --device cpu
"""

import argparse, json, math, os, random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ── constants ─────────────────────────────────────────────────────────────────
IMG_MEAN   = [0.485, 0.456, 0.406]
IMG_STD    = [0.229, 0.224, 0.225]
FPN_CH     = 256
TAP_BLOCKS = [2, 5, 8, 11]
IMG_FEAT   = FPN_CH * 4   # 1024
HIDDEN     = 128
GCN_LAYERS = 4
DROPOUT    = 0.1

# ── icosphere template ────────────────────────────────────────────────────────

def _make_icosphere(subdivisions=2):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts_list = [torch.tensor(v, dtype=torch.float32) for v in [
        [-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],
        [0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],
        [t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1],
    ]]
    verts_list = [v / v.norm() for v in verts_list]
    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]
    mid_cache = {}

    def _mid(a, b):
        key = (min(a,b), max(a,b))
        if key not in mid_cache:
            m = (verts_list[a] + verts_list[b]) / 2.0
            mid_cache[key] = len(verts_list)
            verts_list.append(m / m.norm())
        return mid_cache[key]

    for _ in range(subdivisions):
        new_faces = []
        for v0, v1, v2 in faces:
            m01=_mid(v0,v1); m12=_mid(v1,v2); m20=_mid(v2,v0)
            new_faces += [[v0,m01,m20],[v1,m12,m01],[v2,m20,m12],[m01,m12,m20]]
        faces = new_faces

    verts = torch.stack(verts_list)
    verts[:, 1] *= 0.8
    return verts, torch.tensor(faces, dtype=torch.long)


def _subdivide(verts, faces):
    V = verts.shape[0]
    mid_cache, mid_parents = {}, []

    def _mid(a, b):
        key = (min(a,b), max(a,b))
        if key not in mid_cache:
            mid_cache[key] = V + len(mid_parents)
            mid_parents.append((a, b))
        return mid_cache[key]

    new_faces = []
    for tri in faces.tolist():
        v0,v1,v2 = tri
        m01=_mid(v0,v1); m12=_mid(v1,v2); m20=_mid(v2,v0)
        new_faces += [[v0,m01,m20],[v1,m12,m01],[v2,m20,m12],[m01,m12,m20]]

    mid_v  = torch.stack([(verts[a]+verts[b])/2.0 for a,b in mid_parents])
    new_v  = torch.cat([verts, mid_v], 0)
    new_f  = torch.tensor(new_faces, dtype=torch.long)
    V_out  = new_v.shape[0]
    unpool = torch.zeros(V_out, V)
    for i in range(V):
        unpool[i, i] = 1.0
    for j, (a, b) in enumerate(mid_parents):
        unpool[V+j, a] = 0.5; unpool[V+j, b] = 0.5
    return new_v, new_f, unpool


def _norm_adj(faces, V):
    adj = torch.zeros(V, V)
    for tri in faces.tolist():
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            adj[a,b] = adj[b,a] = 1.0
    adj += torch.eye(V)
    return adj / adj.sum(1, keepdim=True).clamp(1)


def _laplacian(faces, V):
    adj = torch.zeros(V, V)
    for tri in faces.tolist():
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            adj[a,b] = adj[b,a] = 1.0
    return torch.eye(V) - adj / adj.sum(1, keepdim=True).clamp(1)


def _face_adjacency(faces):
    """Return [E, 2] face-index pairs sharing an edge, for normal-consistency loss."""
    edge_to_faces = defaultdict(list)
    for fi, tri in enumerate(faces.tolist()):
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1) % 3]]))
            edge_to_faces[e].append(fi)
    pairs = [f for f in edge_to_faces.values() if len(f) == 2]
    return torch.tensor(pairs, dtype=torch.long)


def build_templates():
    v1, f1       = _make_icosphere(subdivisions=2)
    v2, f2, u12 = _subdivide(v1, f1)
    v3, f3, u23 = _subdivide(v2, f2)
    return (
        (v1, f1, _norm_adj(f1,v1.shape[0]), _laplacian(f1,v1.shape[0]), _face_adjacency(f1)),
        (v2, f2, _norm_adj(f2,v2.shape[0]), _laplacian(f2,v2.shape[0]), _face_adjacency(f2)),
        (v3, f3, _norm_adj(f3,v3.shape[0]), _laplacian(f3,v3.shape[0]), _face_adjacency(f3)),
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
        def fn(_, __, out): self._feats[idx] = out[:, 1:]
        return fn

    def forward(self, x):
        B, C, H, W = x.shape
        self._feats.clear()
        with torch.no_grad():
            self.dino(x)
        hp, wp = H // self.dino.patch_size, W // self.dino.patch_size

        lats = []
        for i, blk in enumerate(TAP_BLOCKS):
            t = self._feats[blk]
            s = t.permute(0,2,1).reshape(B, -1, hp, wp)
            lats.append(self.lateral[i](s))

        fpn = [None] * 4
        fpn[3] = lats[3]
        for i in range(2, -1, -1):
            fpn[i] = lats[i] + F.interpolate(fpn[i+1], size=lats[i].shape[-2:], mode="nearest")

        sizes = [128, 64, 32, 16]
        return [
            self.out_conv[i](F.interpolate(fpn[i], size=(sizes[i],sizes[i]), mode="bilinear", align_corners=False))
            for i in range(4)
        ]


# ── graph conv & decoder ──────────────────────────────────────────────────────

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=DROPOUT):
        super().__init__()
        self.w0   = nn.Linear(in_dim, out_dim, bias=False)
        self.w1   = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.ln   = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        agg = torch.einsum("vw,bwf->bvf", adj, x)
        return self.drop(F.relu(self.ln(self.w0(agg) + self.w1(x) + self.bias)))


class GCNStage(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_layers=GCN_LAYERS):
        super().__init__()
        # No Dropout here — GraphConv already applies it, double-dropout at input hurts
        self.proj  = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.convs = nn.ModuleList([GraphConv(hidden, hidden) for _ in range(n_layers)])
        self.head  = nn.Linear(hidden, 3)

    def forward(self, img_feat, prev_h, verts, adj):
        parts = [img_feat, verts] if prev_h is None else [img_feat, prev_h, verts]
        h = self.proj(torch.cat(parts, dim=-1))
        for conv in self.convs:
            h = conv(h, adj)
        return verts + self.head(h), h


# ── full model ────────────────────────────────────────────────────────────────

class Pixel2MeshDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DINOv2FPNEncoder()
        self.stage1  = GCNStage(IMG_FEAT + 3,          HIDDEN)
        self.stage2  = GCNStage(IMG_FEAT + HIDDEN + 3, HIDDEN)
        self.stage3  = GCNStage(IMG_FEAT + HIDDEN + 3, HIDDEN)

        (v1,f1,a1,l1,ep1),(v2,f2,a2,l2,ep2),(v3,f3,a3,l3,ep3),u12,u23 = build_templates()
        print(f"Templates: {v1.shape[0]} / {v2.shape[0]} / {v3.shape[0]} verts")

        self.register_buffer("v1",v1);  self.register_buffer("f1",f1)
        self.register_buffer("v2",v2);  self.register_buffer("f2",f2)
        self.register_buffer("v3",v3);  self.register_buffer("f3",f3)
        self.register_buffer("a1",a1);  self.register_buffer("l1",l1)
        self.register_buffer("a2",a2);  self.register_buffer("l2",l2)
        self.register_buffer("a3",a3);  self.register_buffer("l3",l3)
        self.register_buffer("ep1",ep1)
        self.register_buffer("ep2",ep2)
        self.register_buffer("ep3",ep3)
        self.register_buffer("u12",u12)
        self.register_buffer("u23",u23)

    @staticmethod
    def _sample(feat_maps, coords):
        # Zero out features for vertices that project outside the image
        in_bounds = (coords.abs() <= 1.0).all(dim=-1, keepdim=True).float()  # [B, V, 1]
        grid = coords.clamp(-1, 1).unsqueeze(1)
        parts = []
        for fm in feat_maps:
            s = F.grid_sample(fm, grid, mode="bilinear", align_corners=True, padding_mode="border")
            parts.append(s.squeeze(2).permute(0,2,1))
        return torch.cat(parts, dim=-1) * in_bounds

    @staticmethod
    def _project(verts, rot, trans, focal):
        v_cam = torch.bmm(verts, rot.transpose(1,2)) + trans.unsqueeze(1)
        z  = v_cam[..., 2:3].clamp(min=1e-4)
        f  = focal.view(-1,1,1)
        return f * v_cam[..., :2] / z   # no clamp — OOB handled in _sample

    @staticmethod
    def _unpool(x, mat):
        return torch.einsum("oi,bif->bof", mat, x)

    def forward(self, image, rot, trans, focal):
        B = image.shape[0]
        feat_maps = self.encoder(image)

        v1 = self.v1.unsqueeze(0).expand(B,-1,-1)
        xy1  = self._project(v1, rot, trans, focal)
        imf1 = self._sample(feat_maps, xy1)
        p1, h1 = self.stage1(imf1, None, v1, self.a1)

        h1u = self._unpool(h1, self.u12)
        p1u = self._unpool(p1, self.u12)
        xy2  = self._project(p1u, rot, trans, focal)
        imf2 = self._sample(feat_maps, xy2)
        p2, h2 = self.stage2(imf2, h1u, p1u, self.a2)

        h2u = self._unpool(h2, self.u23)
        p2u = self._unpool(p2, self.u23)
        xy3  = self._project(p2u, rot, trans, focal)
        imf3 = self._sample(feat_maps, xy3)
        p3, _ = self.stage3(imf3, h2u, p2u, self.a3)

        return (
            (p1, self.f1, self.l1, self.ep1),
            (p2, self.f2, self.l2, self.ep2),
            (p3, self.f3, self.l3, self.ep3),
        )


# ── losses ────────────────────────────────────────────────────────────────────

def chamfer(pred, gt):
    d = ((pred.unsqueeze(2) - gt.unsqueeze(1)) ** 2).sum(-1)
    return d.min(2).values.mean() + d.min(1).values.mean()


def edge_reg(verts, faces):
    v0,v1,v2 = verts[:,faces[:,0]], verts[:,faces[:,1]], verts[:,faces[:,2]]
    return ((v0-v1).norm(dim=-1).mean()+(v1-v2).norm(dim=-1).mean()+(v2-v0).norm(dim=-1).mean())/3


def lap_reg(verts, lap):
    return (torch.einsum("vw,bwc->bvc", lap, verts)**2).mean()


def normal_consistency(verts, faces, edge_pairs):
    """Penalise dihedral angle between adjacent faces (1 - cos θ)."""
    v0 = verts[:, faces[:, 0]]
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]
    normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)  # [B, F, 3]
    n1 = normals[:, edge_pairs[:, 0]]   # [B, E, 3]
    n2 = normals[:, edge_pairs[:, 1]]
    return (1.0 - (n1 * n2).sum(-1)).mean()


def total_loss(pred_verts, gt_pts, faces, lap, edge_pairs,
               w_cd=1.0, w_e=0.1, w_l=0.3, w_n=0.1):
    cd = chamfer(pred_verts, gt_pts)
    el = edge_reg(pred_verts, faces)
    ll = lap_reg(pred_verts, lap)
    nl = normal_consistency(pred_verts, faces, edge_pairs)
    return (w_cd*cd + w_e*el + w_l*ll + w_n*nl,
            {"cd": cd.item(), "edge": el.item(), "lap": ll.item(), "nc": nl.item()})


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
                idx = [int(x.split("/")[0])-1 for x in p[1:]]
                for i in range(1, len(idx)-1):
                    faces.append([idx[0], idx[i], idx[i+1]])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def normalise_mesh(v):
    v = v - (v.max(0).values + v.min(0).values) / 2
    scale = v.norm(dim=-1).max().clamp(1e-8)
    return v / scale, scale.item()


def sample_surface(verts, faces, n=2048):
    v0,v1,v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    areas = 0.5 * torch.cross(v1-v0, v2-v0, dim=-1).norm(dim=-1)
    prob  = (areas / areas.sum().clamp(1e-8)).numpy()
    fi    = torch.from_numpy(np.random.choice(len(faces), n, p=prob))
    r1    = torch.rand(n).sqrt()
    u, v, w = 1-r1, r1*(1-torch.rand(n)), r1*torch.rand(n)
    return u[:,None]*verts[faces[fi,0]] + v[:,None]*verts[faces[fi,1]] + w[:,None]*verts[faces[fi,2]]


_colour_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05)
_to_tensor     = transforms.ToTensor()
_normalise     = transforms.Normalize(IMG_MEAN, IMG_STD)


class Pix3DChairDataset(Dataset):
    def __init__(self, root, split="train", img_size=224, n_pts=2048):
        self.root     = root
        self.n_pts    = n_pts
        self.img_size = img_size
        self.augment  = (split == "train")

        with open(os.path.join(root, "pix3d.json")) as f:
            anns = json.load(f)
        anns = [a for a in anns
                if a["category"] == "chair"
                and not a.get("truncated")
                and not a.get("occluded")]

        # Isolated RNG so the fixed seed doesn't bleed into augmentation RNG state
        _rng = random.Random(42)
        _rng.shuffle(anns)

        n = len(anns)
        self.samples = {
            "train": anns[:int(.8*n)],
            "val":   anns[int(.8*n):int(.9*n)],
            "test":  anns[int(.9*n):],
        }[split]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        a = self.samples[idx]
        s = self.img_size

        # ── image + mask at native resolution ────────────────────────────────
        img  = Image.open(os.path.join(self.root, a["img"])).convert("RGB")
        mask = Image.open(os.path.join(self.root, a["mask"])).convert("L")

        # Mask before resize to avoid bilinear bleed of background into foreground edges
        mask_arr = np.array(mask) > 128
        img_arr  = np.array(img)
        img_arr[~mask_arr] = 0
        img = Image.fromarray(img_arr)

        img  = img.resize((s, s), Image.BILINEAR)
        mask = mask.resize((s, s), Image.NEAREST)

        # ── augmentation: horizontal flip ─────────────────────────────────────
        do_flip = self.augment and random.random() < 0.5
        if do_flip:
            img  = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)

        # ── colour jitter (train only) ────────────────────────────────────────
        if self.augment:
            img = _colour_jitter(img)

        # ── tensor + normalise ────────────────────────────────────────────────
        img_t  = _normalise(_to_tensor(img))

        # ── GT mesh ───────────────────────────────────────────────────────────
        verts, faces = load_obj(os.path.join(self.root, a["model"]))
        verts, scale = normalise_mesh(verts)
        gt_pts = sample_surface(verts, faces, self.n_pts)

        # ── camera ────────────────────────────────────────────────────────────
        rot   = torch.tensor(a["rot_mat"],   dtype=torch.float32)
        trans = torch.tensor(a["trans_mat"], dtype=torch.float32) / scale
        w, h  = a["img_size"]
        focal = torch.tensor(a["focal_length"] / (max(w,h) / 2.0), dtype=torch.float32)

        # Flip: negate camera x-row and x-trans; GT mesh mirrors accordingly
        if do_flip:
            rot[0, :]   = -rot[0, :]
            trans[0]    = -trans[0]
            gt_pts[:,0] = -gt_pts[:,0]

        return {"image": img_t, "gt": gt_pts, "rot": rot, "trans": trans, "focal": focal}


# ── median camera (used by inference script) ──────────────────────────────────

def compute_median_camera(root):
    """Return (rot_identity, median_trans, median_focal) from training split."""
    with open(os.path.join(root, "pix3d.json")) as f:
        anns = json.load(f)
    anns = [a for a in anns
            if a["category"] == "chair"
            and not a.get("truncated") and not a.get("occluded")]
    _rng = random.Random(42)
    _rng.shuffle(anns)
    train = anns[:int(.8*len(anns))]

    focals, tz = [], []
    for a in train:
        w, h = a["img_size"]
        focals.append(a["focal_length"] / (max(w,h) / 2.0))
        tz.append(a["trans_mat"][2])

    med_focal = float(np.median(focals))
    med_tz    = float(np.median(tz))
    print(f"Median focal (normalised): {med_focal:.3f}")
    print(f"Median T_z: {med_tz:.3f}")
    return torch.eye(3), torch.tensor([0.0, 0.0, med_tz]), torch.tensor(med_focal)


# ── training ──────────────────────────────────────────────────────────────────

STAGE_W = [0.2, 0.3, 0.5]


def run_epoch(model, loader, optimizer, device, train=True, scaler=None):
    model.train(train)
    total, cd3_total, n = 0.0, 0.0, 0
    ctx     = torch.enable_grad() if train else torch.no_grad()
    use_amp = (scaler is not None) and (device.type == "cuda")

    with ctx:
        for batch in tqdm(loader, leave=False):
            img   = batch["image"].to(device)
            gt    = batch["gt"].to(device)
            rot   = batch["rot"].to(device)
            trans = batch["trans"].to(device)
            focal = batch["focal"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                stages = model(img, rot, trans, focal)
                loss   = torch.tensor(0., device=device)
                cd3    = 0.0
                for i, ((pv, pf, pl, ep), sw) in enumerate(zip(stages, STAGE_W)):
                    l, info = total_loss(pv, gt, pf, pl, ep)
                    loss = loss + sw * l
                    if i == 2:
                        cd3 = info["cd"]

            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total     += loss.item()
            cd3_total += cd3
            n         += 1

    return total / max(n, 1), cd3_total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/pix3d")
    parser.add_argument("--out",        default="runs/pix2mesh_chair_v3")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--warmup",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr-fpn",     type=float, default=5e-4)   # FPN is randomly init
    parser.add_argument("--lr-gcn",     type=float, default=5e-4)
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    root   = args.data if os.path.isabs(args.data) else os.path.join(os.path.dirname(os.path.dirname(__file__)), args.data)
    outdir = args.out  if os.path.isabs(args.out)  else os.path.join(os.path.dirname(os.path.dirname(__file__)), args.out)
    os.makedirs(outdir, exist_ok=True)

    device = torch.device(args.device)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print("Building model …")
    model = Pixel2MeshDINO().to(device)

    fpn_params = list(model.encoder.lateral.parameters()) + list(model.encoder.out_conv.parameters())
    gcn_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    optimizer  = torch.optim.AdamW(
        [{"params": fpn_params, "lr": args.lr_fpn},
         {"params": gcn_params, "lr": args.lr_gcn}],
        weight_decay=5e-4,
    )

    # Linear warmup for `--warmup` epochs, then cosine annealing
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup]
    )

    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
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

    best_cd3 = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_cd3   = run_epoch(model, train_loader, optimizer, device, train=True,  scaler=scaler)
        val_loss, val_cd3 = run_epoch(model, val_loader,   None,      device, train=False, scaler=None)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={tr_loss:.4f} (cd3={tr_cd3:.4f})  "
              f"val={val_loss:.4f} (cd3={val_cd3:.4f})")

        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(ckpt, os.path.join(outdir, "last.pt"))
        if val_cd3 < best_cd3:
            best_cd3 = val_cd3
            torch.save(ckpt, os.path.join(outdir, "best.pt"))
            print(f"  ↑ new best stage-3 CD: {best_cd3:.4f}")

    print(f"Done. Best val stage-3 CD: {best_cd3:.4f}  →  {outdir}/best.pt")


if __name__ == "__main__":
    main()
