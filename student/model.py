"""Student: DinoV2 encoder (frozen) + per-vertex cross-attention + GCN + offset head.

Architecture (smoke-test version, single deformation stage):
  image (3, 224, 224)
    → DinoV2-S/14 patch tokens  (B, 256, 384)
  template verts (V=642, 3)
    → embedded to hidden state h_v  (B, V, D)
  for each stage:
    cross-attn(h_v, patches) → context  (B, V, D)
    GCN(h_v + context, edges)  → new h_v
    offset = MLP(h_v)          → (B, V, 3)
    verts ← verts + offset
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .template import Template

DINO_DIM = 384  # dinov2_vits14 patch token dim


# ── DinoV2 encoder ────────────────────────────────────────────────────────────

class DinoV2Encoder(nn.Module):
    """Frozen DinoV2-S/14 backbone; returns patch tokens (B, N_patches, 384)."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224). 224 / 14 = 16 → 256 patch tokens.
        with torch.no_grad():
            out = self.backbone.forward_features(x)
        return out["x_norm_patchtokens"]  # (B, 256, 384)


# ── Cross-attention from vertices to patch tokens ─────────────────────────────

class VertexPatchCrossAttn(nn.Module):
    def __init__(self, d_vert: int, d_patch: int = DINO_DIM, d_attn: int = 128, n_heads: int = 4):
        super().__init__()
        assert d_attn % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_attn // n_heads
        self.q = nn.Linear(d_vert,  d_attn)
        self.k = nn.Linear(d_patch, d_attn)
        self.v = nn.Linear(d_patch, d_attn)
        self.o = nn.Linear(d_attn,  d_vert)

    def forward(self, h_v: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        # h_v: (B, V, d_vert), patches: (B, P, d_patch)
        B, V, _ = h_v.shape
        P = patches.shape[1]
        q = self.q(h_v   ).view(B, V, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,V,Dh)
        k = self.k(patches).view(B, P, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,P,Dh)
        v = self.v(patches).view(B, P, self.n_heads, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = attn.softmax(dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(B, V, -1)  # (B, V, d_attn)
        return self.o(ctx)


# ── GCN block ─────────────────────────────────────────────────────────────────

class GCNBlock(nn.Module):
    """Mean-aggregation GCN. h_v ← MLP(h_v ‖ mean(h_neighbors))."""

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, h_v: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # h_v: (B, V, D); edge_index: (2, E) — src, dst
        B, V, D = h_v.shape
        src, dst = edge_index[0], edge_index[1]
        # Aggregate: sum messages from src → dst, then normalize by degree.
        msgs = h_v[:, src, :]                                       # (B, E, D)
        agg  = torch.zeros(B, V, D, device=h_v.device, dtype=h_v.dtype)
        agg  = agg.index_add(1, dst, msgs)
        deg  = torch.zeros(V, device=h_v.device, dtype=h_v.dtype).index_add(
            0, dst, torch.ones_like(dst, dtype=h_v.dtype)
        ).clamp(min=1).view(1, V, 1)
        agg  = agg / deg
        return h_v + self.mlp(torch.cat([h_v, agg], dim=-1))


# ── Deformation stage = cross-attn → GCN → offset ─────────────────────────────

class DeformStage(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.cross = VertexPatchCrossAttn(d_vert=d)
        self.gcn1  = GCNBlock(d)
        self.gcn2  = GCNBlock(d)
        self.offset_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3),
        )

    def forward(
        self,
        h_v: torch.Tensor,
        verts: torch.Tensor,
        patches: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_v = h_v + self.cross(h_v, patches)
        h_v = self.gcn1(h_v, edge_index)
        h_v = self.gcn2(h_v, edge_index)
        delta = self.offset_head(h_v)         # (B, V, 3)
        return h_v, verts + delta


# ── Full student ──────────────────────────────────────────────────────────────

class Student(nn.Module):
    def __init__(self, template: Template, hidden: int = 128, n_stages: int = 1):
        super().__init__()
        self.encoder = DinoV2Encoder()
        self.embed   = nn.Linear(3, hidden)
        self.stages  = nn.ModuleList([DeformStage(hidden) for _ in range(n_stages)])

        # Register template tensors as buffers so .to(device) moves them.
        self.register_buffer("template_verts", template.verts)              # (V, 3)
        self.register_buffer("template_faces", template.faces)              # (F, 3)
        self.register_buffer("template_edges", template.edge_index)         # (2, E)

    def forward(self, image: torch.Tensor) -> dict:
        # image: (B, 3, 224, 224)
        patches = self.encoder(image)                                   # (B, P, 384)
        B = image.shape[0]
        verts = self.template_verts.unsqueeze(0).expand(B, -1, -1)      # (B, V, 3)
        h_v   = self.embed(verts)                                       # (B, V, D)
        all_verts = [verts]
        for stage in self.stages:
            h_v, verts = stage(h_v, verts, patches, self.template_edges)
            all_verts.append(verts)
        return {
            "verts":     verts,           # final deformed verts (B, V, 3)
            "all_verts": all_verts,       # list over stages (for multi-stage Chamfer)
            "faces":     self.template_faces,
            "edge_index": self.template_edges,
        }
