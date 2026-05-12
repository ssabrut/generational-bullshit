import time
import os
import datetime
import random
import torch
import json
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from matplotlib import pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet50, ResNet50_Weights

PIX3D_CATS = ["bed","bookcase","chair","desk","misc","sofa","table","tool","wardrobe"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
STAGE_WEIGHTS = [0.2, 0.3, 0.5]  # coarse → fine supervision

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def get_device():
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    print(f"[rank {rank}] init_process_group: world_size={world_size} master={master_addr}:{master_port}", flush=True)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    print(f"[rank {rank}] process group ready", flush=True)

def cleanup():
    dist.destroy_process_group()

def get_unit_cube():
    """Returns (vertices [8,3], faces [12,3]) — unit cube centered at origin."""
    v = torch.tensor([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=torch.float32)
    f = torch.tensor([
        [0,1,2],[0,2,3],  # back
        [4,6,5],[4,7,6],  # front
        [0,4,5],[0,5,1],  # bottom
        [2,6,7],[2,7,3],  # top
        [0,7,4],[0,3,7],  # left
        [1,5,6],[1,6,2],  # right
    ], dtype=torch.long)
    return v, f


def loop_subdivide(vertices, faces):
    """One step of Loop subdivision — splits every triangle into 4."""
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
    """Row-normalised dense adjacency [V,V] with self-loops."""
    V = vertices.shape[0]
    adj = torch.zeros(V, V)
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1)%3])
            adj[a, b] = adj[b, a] = 1.0
    adj += torch.eye(V)
    return adj / adj.sum(1, keepdim=True).clamp(min=1)


def build_laplacian(vertices, faces):
    """Uniform Laplacian L = I - D^{-1}A."""
    V = vertices.shape[0]
    adj = torch.zeros(V, V)
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1)%3])
            adj[a, b] = adj[b, a] = 1.0
    return torch.eye(V) - adj / adj.sum(1, keepdim=True).clamp(1)

def project_vertices(vertices, rot, trans, focal):
    """
    Pix3D camera model: v_cam = R @ v + T, then perspective divide.
    vertices : [B, V, 3]
    rot      : [B, 3, 3]
    trans    : [B, 3]
    focal    : [B]   normalised focal length (focal_px / (img_size/2))
    Returns  : [B, V, 2]  in [-1, 1]  (for grid_sample)
    """
    v_cam = torch.bmm(vertices, rot.transpose(1, 2)) + trans.unsqueeze(1)
    z = v_cam[..., 2:3].clamp(min=1e-6)
    f = focal.view(-1, 1, 1)
    xy = f * v_cam[..., :2] / z
    return xy.clamp(-1, 1)


def sample_features(feature_maps, coords):
    """
    Bilinearly sample feature maps at projected vertex coords.
    feature_maps : list of [B, C_i, H_i, W_i]
    coords       : [B, V, 2]
    Returns      : [B, V, sum_C_i]
    """
    grid = coords.unsqueeze(1)  # [B, 1, V, 2] for grid_sample
    sampled = []
    for fm in feature_maps:
        s = F.grid_sample(fm, grid, align_corners=True, mode="bilinear", padding_mode="border")  # [B,C,1,V]
        sampled.append(s.squeeze(2).permute(0, 2, 1))              # [B,V,C]
    return torch.cat(sampled, dim=-1)

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze_stages=2):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        bb = resnet50(weights=weights)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1  # [B, 256, H/4,  W/4]
        self.layer2 = bb.layer2  # [B, 512, H/8,  W/8]
        self.layer3 = bb.layer3  # [B,1024, H/16, W/16]
        self.layer4 = bb.layer4  # [B,2048, H/32, W/32]
        self.feat_dims = [256, 512, 1024, 2048]
        for stage in [self.stem, self.layer1, self.layer2][:freeze_stages]:
            for p in stage.parameters(): 
                p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

class GraphConv(nn.Module):
    """Single graph convolution layer with neighbour aggregation."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_self = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        # x: [B,V,F]   adj: [V,V] pre-normalised
        B, V, _ = x.shape
        neigh = torch.einsum("vw,bwf->bvf", adj, x)  # aggregate neighbours
        out = self.fc_self(x) + self.fc_neigh(neigh) + self.bias
        out = self.bn(out.reshape(B*V, -1)).reshape(B, V, -1)
        return F.relu(out)


class GCNBlock(nn.Module):
    """Two GraphConv layers with residual skip."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim, bias=False) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x, adj):
        return self.gc2(self.gc1(x, adj), adj) + self.skip(x)


class DeformStage(nn.Module):
    """N GCNBlocks → displacement head → v_new = v_old + Δv"""
    def __init__(self, feat_dim, hidden_dim=128, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            GCNBlock(feat_dim if i==0 else hidden_dim, hidden_dim)
            for i in range(n_blocks)
        ])
        self.disp_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, feats, verts, adj):
        h = feats
        for blk in self.blocks: h = blk(h, adj)
        new_verts = verts + self.disp_head(h)  # residual update
        return new_verts, h
    
class Furniture3D(nn.Module):
    def __init__(self, encoder, hidden_dim=128, n_stages=3, n_gcn_blocks=2):
        super().__init__()
        self.n_stages = n_stages
        self.hidden_dim = hidden_dim

        # Pretrained ResNet-50 encoder
        self.encoder = encoder
        total_img = sum(self.encoder.feat_dims)  # 3840

        # Per-stage: projector MLP + deformation GCN
        self.projectors = nn.ModuleList()
        self.deform_stages = nn.ModuleList()
        for i in range(n_stages):
            # Stage 0: img(3840) + xyz(3) | Stage 1+: img + prev_hidden + xyz
            in_dim = (total_img + 3) if i == 0 else (total_img + hidden_dim + 3)
            self.projectors.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            ))
            self.deform_stages.append(DeformStage(hidden_dim, hidden_dim, n_gcn_blocks))

        # Pre-compute fixed mesh topology for each stage
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

        feat_maps = self.encoder(image)  # multi-scale features
        all_verts, all_faces, all_laps = [], [], []
        prev_h = None

        for i in range(self.n_stages):
            adj = getattr(self, f"adj_{i}").to(device)
            lap = getattr(self, f"lap_{i}").to(device)
            verts_tmpl, faces = self.templates[i]
            verts = verts_tmpl.to(device).unsqueeze(0).expand(B,-1,-1).clone()
            V = verts.shape[1]

            # Project + sample image features at vertex locations
            coords = project_vertices(verts, rot, trans, focal)   # [B,V,2]
            img_feat = sample_features(feat_maps, coords)            # [B,V,3840]

            # Build input for projector MLP
            if prev_h is None:
                combined = torch.cat([img_feat, verts], dim=-1)
            else:
                if prev_h.shape[1] != V:  # interpolate if vertex count changed
                    prev_h = F.interpolate(
                        prev_h.permute(0,2,1), size=V, mode="linear", align_corners=False
                    ).permute(0,2,1)
                combined = torch.cat([img_feat, prev_h, verts], dim=-1)

            vertex_feats = self.projectors[i](combined)              # [B,V,hidden]
            new_verts, hidden = self.deform_stages[i](vertex_feats, verts, adj)

            all_verts.append(new_verts)
            all_faces.append(faces)
            all_laps.append(lap)
            prev_h = hidden

        return {"vertices": all_verts, "faces": all_faces, "laplacians": all_laps}
    
def chamfer_distance(pred, gt):
    """Bidirectional Chamfer. pred,gt: [B,N,3]"""
    d = ((pred.unsqueeze(2) - gt.unsqueeze(1))**2).sum(-1)  # [B,N,M]
    return d.min(2).values.mean() + d.min(1).values.mean()


def edge_loss(vertices, faces):
    """Mean edge length — keeps tessellation uniform."""
    v0, v1, v2 = vertices[:,faces[:,0]], vertices[:,faces[:,1]], vertices[:,faces[:,2]]
    return torch.stack([
        (v0-v1).norm(dim=-1), (v1-v2).norm(dim=-1), (v2-v0).norm(dim=-1)
    ]).mean()


def laplacian_loss(vertices, lap):
    """Smoothing loss: ||L @ V||^2"""
    return (torch.einsum("vw,bwc->bvc", lap, vertices)**2).mean()


def compute_loss(pred_verts, gt_pts, faces, lap, w_cd=1.0, w_edge=0.1, w_lap=0.5):
    cd  = chamfer_distance(pred_verts, gt_pts)
    el  = edge_loss(pred_verts, faces)
    lpl = laplacian_loss(pred_verts, lap)
    total = w_cd*cd + w_edge*el + w_lap*lpl
    return total, {"chamfer": cd.item(), "edge": el.item(), "laplacian": lpl.item()}

def load_obj(path):
    """Minimal OBJ loader → (verts [V,3], faces [F,3])."""
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            p = line.strip().split()
            if not p: continue
            if p[0] == "v": verts.append([float(x) for x in p[1:4]])
            elif p[0] == "f":
                idx = [int(x.split("/")[0])-1 for x in p[1:]]
                for i in range(1, len(idx)-1):
                    faces.append([idx[0], idx[i], idx[i+1]])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def normalise_mesh(v):
    """Centre and scale to unit sphere."""
    v = v - (v.max(0).values + v.min(0).values) / 2
    return v / v.norm(dim=-1).max().clamp(1e-8)


def sample_surface(verts, faces, n=2048):
    """Sample n points uniformly on triangle surfaces."""
    v0,v1,v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    areas = 0.5 * torch.cross(v1-v0, v2-v0, dim=-1).norm(dim=-1)
    prob = (areas / areas.sum().clamp(1e-8)).numpy()
    fi = torch.from_numpy(np.random.choice(len(faces), n, p=prob))
    r1 = torch.rand(n).sqrt()
    u, v, w = 1-r1, r1*(1-torch.rand(n)), r1*torch.rand(n)
    return (u[:,None]*verts[faces[fi,0]]
          + v[:,None]*verts[faces[fi,1]]
          + w[:,None]*verts[faces[fi,2]])


class Pix3DDataset(Dataset):
    def __init__(self, root, split="train", categories=None, img_size=224, n_pts=2048):
        self.root = root
        self.n_pts = n_pts
        with open(os.path.join(root, "pix3d.json")) as f:
            anns = json.load(f)

        cats = set(categories or PIX3D_CATS)
        anns = [a for a in anns if a["category"] in cats and not a.get("truncated") and not a.get("occluded")]
        random.seed(42)
        random.shuffle(anns)
        n = len(anns)
        splits = {"train": anns[:int(0.8*n)], "val": anns[int(0.8*n):int(0.9*n)], "test": anns[int(0.9*n):]}
        self.samples = splits[split]
        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        a = self.samples[idx]
        image = self.tfm(Image.open(os.path.join(self.root, a["img"])).convert("RGB"))
        verts, faces = load_obj(os.path.join(self.root, a["model"]))
        verts = normalise_mesh(verts)
        gt_pts = sample_surface(verts, faces, self.n_pts)
        rot = torch.tensor(a["rot_mat"],  dtype=torch.float32)  # [3,3]
        trans = torch.tensor(a["trans_mat"],dtype=torch.float32)  # [3]
        w, h = a["img_size"]
        focal = torch.tensor(a["focal_length"] / (max(w,h)/2.), dtype=torch.float32)
        return {"image": image, "gt_points": gt_pts, "rot": rot, "trans": trans, "focal": focal}
    
def train(model, loader, optimizer, n_epochs=5, val_loader=None, vis_every=20, device=DEVICE, train_sampler=None):
    history = {'train': [], 'val': []}
    step_losses, val_losses = [], []
    global_step = 0

    epoch_bar = tqdm(range(1, n_epochs + 1), desc='Epochs', unit='epoch')
    for epoch in epoch_bar:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss, t0 = 0.0, time.time()

        step_bar = tqdm(loader, desc=f'Train {epoch}/{n_epochs}', leave=False, unit='batch')
        for batch in step_bar:
            img = batch['image'].to(device)
            gt = batch['gt_points'].to(device)
            rot = batch['rot'].to(device)
            trans = batch['trans'].to(device)
            focal = batch['focal'].to(device)

            optimizer.zero_grad()
            out = model(img, rot, trans, focal)

            loss = torch.tensor(0., device=device)
            for verts, faces, lap, w in zip(
                out['vertices'], out['faces'], out['laplacians'], STAGE_WEIGHTS
            ):
                l, _ = compute_loss(verts, gt, faces.to(device), lap.to(device))
                loss = loss + w * l

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step_losses.append(loss.item())
            global_step += 1

            step_bar.set_postfix(loss=f'{loss.item():.4f}')

        train_avg = epoch_loss / len(loader)
        history['train'].append(train_avg)

        val_avg = None
        if val_loader:
            model.eval()
            v_loss = 0.0
            val_bar = tqdm(val_loader, desc=f'Val   {epoch}/{n_epochs}', leave=False, unit='batch')
            with torch.no_grad():
                for batch in val_bar:
                    img = batch['image'].to(device)
                    gt = batch['gt_points'].to(device)
                    rot = batch['rot'].to(device)
                    trans = batch['trans'].to(device)
                    focal = batch['focal'].to(device)
                    out = model(img, rot, trans, focal)
                    for verts, faces, lap, w in zip(
                        out['vertices'], out['faces'], out['laplacians'], STAGE_WEIGHTS
                    ):
                        l, _ = compute_loss(verts, gt, faces.to(device), lap.to(device))
                        v_loss += w * l.item()
                    val_bar.set_postfix(loss=f'{v_loss / (val_bar.n or 1):.4f}')
            val_avg = v_loss / len(val_loader)
            history['val'].append(val_avg)
            val_losses.append(val_avg)

        t = time.time() - t0
        postfix = {'train': f'{train_avg:.4f}', 'time': f'{t:.1f}s'}
        if val_avg is not None:
            postfix['val'] = f'{val_avg:.4f}'
        epoch_bar.set_postfix(**postfix)

    return history

def save_ckpt(model, optimizer, epoch, path, val_cd=None):
    torch.save({
        "epoch": epoch, 
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_cd": val_cd
    }, path)
    print(f"Saved → {path}")

def plot_history(history):
    has_val = bool(history.get("val"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, yscale in zip(axes, ["linear", "log"]):
        ax.plot(history["train"], label="Train", c="#4A90D9", lw=2)
        if has_val: ax.plot(history["val"], label="Val", c="#E8672A", lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_yscale(yscale)
        ax.set_title(f"Loss ({yscale} scale)")
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Must be called before any distributed ops or DDP construction
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = get_device()

    PIX3D_ROOT = "data/pix3d"
    batch_size = 16
    num_workers = 0

    train_ds = Pix3DDataset(PIX3D_ROOT, "train", categories=["chair"])
    val_ds   = Pix3DDataset(PIX3D_ROOT, "val",   categories=["chair"])

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    )

    # shuffle must be False when a sampler is provided
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=train_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)

    encoder = ResNetEncoder(pretrained=True, freeze_stages=2).to(device)
    model   = DDP(Furniture3D(encoder=encoder, hidden_dim=256, n_stages=3, n_gcn_blocks=3).to(device))

    optimizer = torch.optim.AdamW([
        {"params": model.module.encoder.parameters(), "lr": 3e-5},
        {"params": [p for n, p in model.module.named_parameters() if "encoder" not in n], "lr": 3e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    history = train(
        model, train_loader, optimizer,
        n_epochs=10, val_loader=val_loader,
        device=device, train_sampler=train_sampler,
    )

    # Only rank 0 saves checkpoints and plots
    if rank == 0:
        save_ckpt(model.module, optimizer, epoch=10, path="./furniture3d.pth")
        demo_hist = {
            "train": [5.0*np.exp(-0.06*e) + 0.1 + 0.04*np.random.randn() for e in range(60)],
            "val":   [5.5*np.exp(-0.05*e) + 0.12+ 0.06*np.random.randn() for e in range(60)],
        }
        plot_history(demo_hist)

    cleanup()