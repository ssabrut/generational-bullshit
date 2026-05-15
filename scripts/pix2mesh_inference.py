"""
Pixel2Mesh inference pipeline using the pretrained TF checkpoint (converted to PT).

Architecture recovered from checkpoint variable shapes:
  Encoder  : VGG-like CNN, 18 conv layers, features tapped at conv7/10/13/17
             → 64 + 128 + 256 + 512 = 960 ch (matches stage-1 GCN input 963 = 960+3)
  Stage 1  : GCN layers 1-14,  input 963,  output 3D coords
  Stage 2  : GCN layers 15-28, input 1219 (960+256+3), output 3D coords
  Stage 3  : GCN layers 29-43, input 1219,              output 3D coords

Graph conv : output = (A @ X) @ W0 + X @ W1 + bias  (original TF formulation)

Usage:
    conda run -n AI python scripts/pix2mesh_inference.py \
        --image GAN/exports/chair.png \
        --out    GAN/exports/pix2mesh_out.obj
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# WEIGHTS = os.path.join(ROOT, "Models", "pre-trained", "pix2mesh_pretrained.pt")
WEIGHTS = os.path.join(ROOT, "runs", "pix2mesh_chair", "best.pt")

# ── VGG-like encoder ──────────────────────────────────────────────────────────
#   Blocks 1-4 end with 2×2 maxpool.
#   conv11 and conv14 use stride-2 (5×5 kernel) instead of pool.
#   Feature taps: after conv7 (64ch), conv10 (128ch), conv13 (256ch), conv17 (512ch).

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        def _conv(cin, cout, k=3, s=1):
            return nn.Sequential(nn.Conv2d(cin, cout, k, stride=s, padding=k//2), nn.ReLU(inplace=True))

        # block 1: 224→112
        self.block1 = nn.Sequential(_conv(3, 16), _conv(16, 16), nn.MaxPool2d(2))
        # block 2: 112→56
        self.block2 = nn.Sequential(_conv(16, 32), _conv(32, 32), _conv(32, 32), nn.MaxPool2d(2))
        # block 3: 56→28, tap at 64ch
        self.block3 = nn.Sequential(_conv(32, 64), _conv(64, 64), _conv(64, 64), nn.MaxPool2d(2))
        # block 4: 28→14, tap at 128ch
        self.block4 = nn.Sequential(_conv(64, 128), _conv(128, 128), _conv(128, 128), nn.MaxPool2d(2))
        # block 5: 14→7 via stride-2 5×5, tap at 256ch
        self.block5 = nn.Sequential(
            nn.Sequential(nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.ReLU(inplace=True)),
            _conv(256, 256), _conv(256, 256),
        )
        # block 6: 7→4 via stride-2 5×5, tap at 512ch
        self.block6 = nn.Sequential(
            nn.Sequential(nn.Conv2d(256, 512, 5, stride=2, padding=2), nn.ReLU(inplace=True)),
            _conv(512, 512), _conv(512, 512), _conv(512, 512),
        )

    def forward(self, x):
        f1 = self.block3(self.block2(self.block1(x)))        # [B, 64,  28, 28]
        f2 = self.block4(f1)                                 # [B, 128, 14, 14]
        f3 = self.block5(f2)                                 # [B, 256,  7,  7]
        f4 = self.block6(f3)                                 # [B, 512,  4,  4]
        return [f1, f2, f3, f4]

    def load_from_state_dict(self, sd: dict):
        """Map encoder.conv{i}.weight/bias from converted checkpoint."""
        conv_map = {
            # block1
            0: ("block1", 0, 0), 1: ("block1", 0, 1),
            # block2
            2: ("block2", 0, 0), 3: ("block2", 0, 1), 4: ("block2", 0, 2),
            # block3
            5: ("block3", 0, 0), 6: ("block3", 0, 1), 7: ("block3", 0, 2),
            # block4
            8: ("block4", 0, 0), 9: ("block4", 0, 1), 10: ("block4", 0, 2),
            # block5
            11: ("block5", 0, 0), 12: ("block5", 0, 1), 13: ("block5", 0, 2),
            # block6
            14: ("block6", 0, 0), 15: ("block6", 0, 1), 16: ("block6", 0, 2), 17: ("block6", 0, 3),
        }
        for conv_i, (block, _, seq_i) in conv_map.items():
            block_module = getattr(self, block)
            # block_module[0] is the Sequential of conv-relu pairs
            # For block5/6 the first entry is already a Sequential(conv,relu)
            if block in ("block5", "block6"):
                if seq_i == 0:
                    layer = block_module[0][0]          # stride-2 conv
                else:
                    layer = block_module[seq_i][0]      # remaining _conv → nn.Sequential(conv, relu)[0]
            else:
                # block1-4: block_module[0] is _conv(a, b) = Sequential(conv, relu)
                # indices: seq_i in [0..n_convs-1], last entry is MaxPool
                layer = block_module[seq_i][0]

            w = sd[f"encoder.conv{conv_i}.weight"]
            b = sd[f"encoder.conv{conv_i}.bias"]
            assert layer.weight.shape == w.shape, \
                f"Shape mismatch conv{conv_i}: model {layer.weight.shape} vs ckpt {w.shape}"
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)
        print("  Encoder weights loaded.")


# ── Graph convolution (original P2M formulation) ──────────────────────────────
# output = (A @ X) @ W0 + X @ W1 + bias
# A is the row-normalised adjacency (including self-loops), passed externally.

class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w0   = nn.Parameter(torch.empty(in_dim, out_dim))
        self.w1   = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   [B, V, in_dim]
        # adj: [V, V]
        agg = torch.einsum("vw, bwf -> bvf", adj, x)
        return F.relu(agg @ self.w0 + x @ self.w1 + self.bias)


# ── GCN stage ─────────────────────────────────────────────────────────────────

class GCNStage(nn.Module):
    def __init__(self, layer_ids: list[int], in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layer_ids = layer_ids
        dims = [in_dim] + [hidden_dim] * (len(layer_ids) - 2) + [out_dim]
        self.convs = nn.ModuleList([
            GraphConv(dims[i], dims[i + 1]) for i in range(len(layer_ids) - 1)
        ])
        # last conv predicts displacement (no relu applied after)
        self.disp = GraphConv(hidden_dim, 3) if out_dim == hidden_dim else None

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = x
        for conv in self.convs:
            h = conv(h, adj)
        return h


# ── Full Pixel2Mesh model ─────────────────────────────────────────────────────

class Pixel2Mesh(nn.Module):
    # GCN layer ranges (inclusive) per stage
    STAGE1 = list(range(1, 15))    # layers 1-14
    STAGE2 = list(range(15, 29))   # layers 15-28
    STAGE3 = list(range(29, 44))   # layers 29-43

    def __init__(self):
        super().__init__()
        self.encoder = VGGEncoder()

        # Stage 1: input=963 (960+3), hidden=256, output at layer 14 → 3
        self.gcn1 = nn.ModuleList()
        for i, lid in enumerate(self.STAGE1):
            in_d  = 963 if i == 0 else 256
            out_d = 3   if lid == 14 else 256
            self.gcn1.append(GraphConv(in_d, out_d))

        # Stage 2: input=1219 (960+256+3), hidden=256, output at layer 28 → 3
        self.gcn2 = nn.ModuleList()
        for i, lid in enumerate(self.STAGE2):
            in_d  = 1219 if i == 0 else 256
            out_d = 3    if lid == 28 else 256
            self.gcn2.append(GraphConv(in_d, out_d))

        # Stage 3: input=1219, hidden=256→128, output at layer 43 → 3
        self.gcn3 = nn.ModuleList()
        for i, lid in enumerate(self.STAGE3):
            in_d  = 1219 if i == 0 else (256 if lid <= 42 else 128)
            out_d = 128  if lid == 42 else (3 if lid == 43 else 256)
            if lid in (42, 43):
                in_d = 256 if lid == 42 else 128
            self.gcn3.append(GraphConv(in_d, out_d))

    def load_pretrained(self, path: str, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        sd   = ckpt["state_dict"]

        self.encoder.load_from_state_dict(sd)

        def load_gcn(module_list, layer_ids):
            for idx, lid in enumerate(layer_ids):
                gc = module_list[idx]
                prefix = f"gcn.{lid}"
                gc.w0.data.copy_(sd[f"{prefix}.w0"])
                gc.w1.data.copy_(sd[f"{prefix}.w1"])
                gc.bias.data.copy_(sd[f"{prefix}.bias"])

        load_gcn(self.gcn1, self.STAGE1)
        load_gcn(self.gcn2, self.STAGE2)
        load_gcn(self.gcn3, self.STAGE3)
        print("  GCN weights loaded.")

    @staticmethod
    def _run_gcn(layers, x, adj):
        h = x
        for layer in layers:
            h = layer(h, adj)
        return h

    @staticmethod
    def _sample_features(feat_maps, coords_2d):
        """
        feat_maps : list of [B, C, H, W]
        coords_2d : [B, V, 2]  in [-1, 1]
        returns   : [B, V, sum_C]
        """
        grid = coords_2d.unsqueeze(1)           # [B, 1, V, 2]
        parts = []
        for fm in feat_maps:
            s = F.grid_sample(fm, grid, mode="bilinear", align_corners=True, padding_mode="border")
            parts.append(s.squeeze(2).permute(0, 2, 1))   # [B, V, C]
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _project(verts, scale=1.0):
        """Weak perspective: just take XY, normalise to [-1,1]."""
        xy = verts[..., :2] * scale
        return xy.clamp(-1.0, 1.0)

    @staticmethod
    def _unpool(features, unpool_mat):
        """
        features   : [B, V_in, C]
        unpool_mat : [V_out, V_in]  (sparse or dense)
        returns    : [B, V_out, C]
        """
        return torch.einsum("oi, bic -> boc", unpool_mat, features)

    def forward(self, image, adj1, adj2, adj3, verts1,
                unpool12, unpool23):
        """
        image      : [B, 3, 224, 224]
        adj{i}     : [V_i, V_i] normalised adjacency
        verts1     : [B, V1, 3] initial ellipsoid vertices
        unpool12   : [V2, V1]  maps stage-1 features/verts → V2
        unpool23   : [V3, V2]  maps stage-2 features/verts → V3
        """
        device = image.device
        feat_maps = self.encoder(image)

        # ── stage 1 ────────────────────────────────────────────────────────
        xy1    = self._project(verts1)
        img_f1 = self._sample_features(feat_maps, xy1)       # [B, V1, 960]
        x1     = torch.cat([img_f1, verts1], dim=-1)         # [B, V1, 963]
        h1     = self._run_gcn(self.gcn1[:-1], x1, adj1)     # [B, V1, 256]
        d1     = self.gcn1[-1](h1, adj1)                     # [B, V1, 3]
        pred1  = verts1 + d1

        # ── unpool 1→2 ─────────────────────────────────────────────────────
        up12   = unpool12.to(device)
        h1_up  = self._unpool(h1,    up12)                   # [B, V2, 256]
        v2_up  = self._unpool(pred1, up12)                   # [B, V2, 3]

        # ── stage 2 ────────────────────────────────────────────────────────
        xy2    = self._project(v2_up)
        img_f2 = self._sample_features(feat_maps, xy2)       # [B, V2, 960]
        x2     = torch.cat([img_f2, h1_up, v2_up], dim=-1)   # [B, V2, 1219]
        h2     = self._run_gcn(self.gcn2[:-1], x2, adj2)     # [B, V2, 256]
        d2     = self.gcn2[-1](h2, adj2)                     # [B, V2, 3]
        pred2  = v2_up + d2

        # ── unpool 2→3 ─────────────────────────────────────────────────────
        up23   = unpool23.to(device)
        h2_up  = self._unpool(h2,    up23)                   # [B, V3, 256]
        v3_up  = self._unpool(pred2, up23)                   # [B, V3, 3]

        # ── stage 3 ────────────────────────────────────────────────────────
        xy3    = self._project(v3_up)
        img_f3 = self._sample_features(feat_maps, xy3)       # [B, V3, 960]
        x3     = torch.cat([img_f3, h2_up, v3_up], dim=-1)   # [B, V3, 1219]
        h3     = self._run_gcn(self.gcn3[:-1], x3, adj3)     # [B, V3, 256]
        d3     = self.gcn3[-1](h3, adj3)                     # [B, V3, 3]
        pred3  = v3_up + d3

        return pred1, pred2, pred3


# ── Ellipsoid template (icosphere with 2 subdivisions) ───────────────────────

def _make_icosphere(subdivisions=2):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ]
    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]
    verts = torch.tensor(verts, dtype=torch.float32)
    norms = verts.norm(dim=1, keepdim=True)
    verts = verts / norms

    verts_list = list(verts)
    mid_cache = {}

    def _mid(a, b):
        key = (min(a,b), max(a,b))
        if key not in mid_cache:
            m = (verts_list[a] + verts_list[b]) / 2.0
            m = m / m.norm()
            mid_cache[key] = len(verts_list)
            verts_list.append(m)
        return mid_cache[key]
    for _ in range(subdivisions):
        new_faces = []
        for tri in faces:
            v0, v1, v2 = tri
            m01 = _mid(v0, v1)
            m12 = _mid(v1, v2)
            m20 = _mid(v2, v0)
            new_faces += [[v0,m01,m20],[v1,m12,m01],[v2,m20,m12],[m01,m12,m20]]
        faces = new_faces

    verts_t = torch.stack(verts_list)
    faces_t = torch.tensor(faces, dtype=torch.long)
    return verts_t, faces_t


def _normalised_adj(verts, faces):
    V = verts.shape[0]
    adj = torch.zeros(V, V)
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1)%3])
            adj[a, b] = adj[b, a] = 1.0
    adj += torch.eye(V)
    deg = adj.sum(1, keepdim=True).clamp(min=1.0)
    return adj / deg


def _subdivide(verts, faces):
    """Subdivide mesh and return (new_verts, new_faces, unpool_matrix).

    unpool_matrix [V_out, V_in]: maps old vertex features to the finer mesh.
    Original vertices are kept (identity rows); midpoint rows average parents.
    """
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
        v0, v1, v2 = tri
        m01 = _mid(v0, v1); m12 = _mid(v1, v2); m20 = _mid(v2, v0)
        new_faces += [[v0,m01,m20],[v1,m12,m01],[v2,m20,m12],[m01,m12,m20]]

    mid_verts = torch.stack([(verts[a] + verts[b]) / 2.0 for a, b in mid_parents])
    new_v = torch.cat([verts, mid_verts], 0)
    new_f = torch.tensor(new_faces, dtype=torch.long)

    V_out = new_v.shape[0]
    unpool = torch.zeros(V_out, V)
    for i in range(V):                        # original vertices: identity
        unpool[i, i] = 1.0
    for j, (a, b) in enumerate(mid_parents):  # midpoints: average parents
        unpool[V + j, a] = 0.5
        unpool[V + j, b] = 0.5

    return new_v, new_f, unpool


def build_templates():
    """Return per-stage (verts, faces, adj) and unpool matrices (up12, up23)."""
    v1, f1 = _make_icosphere(subdivisions=2)
    v2, f2, up12 = _subdivide(v1, f1)
    v3, f3, up23 = _subdivide(v2, f2)

    # Squash to a unit ellipsoid (match original P2M initialisation)
    for v in [v1, v2, v3]:
        v[:, 1] *= 0.8

    a1 = _normalised_adj(v1, f1)
    a2 = _normalised_adj(v2, f2)
    a3 = _normalised_adj(v3, f3)

    print(f"  Template mesh: {v1.shape[0]} / {v2.shape[0]} / {v3.shape[0]} vertices across 3 stages")
    return (v1, f1, a1), (v2, f2, a2), (v3, f3, a3), up12, up23


# ── I/O helpers ───────────────────────────────────────────────────────────────

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

def load_image(path: str, size: int = 224) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)   # [1, 3, H, W]


def save_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Pixel2Mesh output\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"  Saved: {path}  ({len(verts)} verts, {len(faces)} faces)")


# ── Main ──────────────────────────────────────────────────────────────────────

def _is_dinov2_ckpt(path: str) -> bool:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return "model" in ckpt and any("stage1" in k for k in ckpt["model"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   default="GAN/exports/chair.png")
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--out",     default="GAN/exports/pix2mesh_out.obj")
    parser.add_argument("--device",  default="cpu")
    # Camera overrides (optional; sane defaults work for centred object images)
    parser.add_argument("--focal",   type=float, default=2.5,
                        help="Normalised focal length (focal_px / (max(H,W)/2))")
    parser.add_argument("--tz",      type=float, default=2.5,
                        help="Camera Z translation (depth of object centre)")
    args = parser.parse_args()

    image_path = os.path.join(ROOT, args.image) if not os.path.isabs(args.image) else args.image
    out_path   = os.path.join(ROOT, args.out)   if not os.path.isabs(args.out)   else args.out
    device     = torch.device(args.device)

    print(f"\n[1/4] Loading weights from {args.weights}")
    use_dinov2 = _is_dinov2_ckpt(args.weights)
    print(f"  Model type: {'DINOv2-GCN (fine-tuned)' if use_dinov2 else 'VGG-GCN (pretrained)'}")

    if use_dinov2:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from train_pix2mesh_chair import Pixel2MeshDINO, compute_median_camera

        model = Pixel2MeshDINO()
        ckpt  = torch.load(args.weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded epoch {ckpt.get('epoch', '?')}")
        model.to(device).eval()

        # use dataset median camera if pix3d root is available, else fall back to CLI args
        pix3d_root = os.path.join(ROOT, "data", "pix3d")
        if os.path.exists(os.path.join(pix3d_root, "pix3d.json")):
            print(f"\n[2/4] Computing median camera from training split …")
            _rot, _trans, _focal = compute_median_camera(pix3d_root)
            rot   = _rot.unsqueeze(0).to(device)
            trans = _trans.unsqueeze(0).to(device)
            focal = _focal.unsqueeze(0).to(device)
        else:
            print(f"\n[2/4] Using CLI camera (identity R, T=[0,0,{args.tz}], f={args.focal})")
            rot   = torch.eye(3, device=device).unsqueeze(0)
            trans = torch.tensor([[0.0, 0.0, args.tz]], device=device)
            focal = torch.tensor([args.focal], device=device)

        print(f"\n[3/4] Running inference on {image_path}")
        image = load_image(image_path).to(device)

        with torch.no_grad():
            (p1, f1, _), (p2, f2, _), (p3, f3, _) = model(image, rot, trans, focal)

        stages = [(p1, f1), (p2, f2), (p3, f3)]

    else:
        model = Pixel2Mesh()
        model.load_pretrained(args.weights, device=device)
        model.to(device).eval()

        print(f"\n[2/4] Building template meshes")
        (v1, f1, a1), (_, f2, a2), (_, f3, a3), up12, up23 = build_templates()

        print(f"\n[3/4] Running inference on {image_path}")
        image  = load_image(image_path).to(device)
        verts1 = v1.unsqueeze(0).to(device)
        adj1, adj2, adj3 = a1.to(device), a2.to(device), a3.to(device)

        with torch.no_grad():
            pred1, pred2, pred3 = model(image, adj1, adj2, adj3, verts1, up12, up23)

        stages = [(pred1, f1), (pred2, f2), (pred3, f3)]

    print(f"\n[4/4] Saving outputs")
    base, ext = os.path.splitext(out_path)
    for i, (pv, pf) in enumerate(stages, 1):
        save_obj(f"{base}_stage{i}{ext}", pv[0].cpu().numpy(), pf.cpu().numpy())

    print("\nDone. Final mesh:", f"{base}_stage3{ext}")


if __name__ == "__main__":
    main()
