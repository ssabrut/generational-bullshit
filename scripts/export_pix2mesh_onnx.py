"""
Export a trained Pixel2MeshDINO checkpoint to ONNX.

Produces:
  <out>.onnx       – ONNX model (opset 17, batch=1, fixed 224×224 input)
  <out>_faces.npz  – face arrays for all 3 stages (topology is input-invariant)

ONNX inputs  (names / shapes):
  image  [1, 3, 224, 224]   normalised RGB, background zeroed
  rot    [1, 3, 3]           rotation matrix
  trans  [1, 3]              translation vector
  focal  [1]                 normalised focal length

ONNX outputs:
  verts1 [1, 162,  3]        stage-1 predicted vertices
  verts2 [1, 642,  3]        stage-2 predicted vertices
  verts3 [1, 2562, 3]        stage-3 predicted vertices

Usage:
  conda run -n AI python scripts/export_pix2mesh_onnx.py \\
      --weights runs/pix2mesh_chair_v3/best.pt \\
      --out     runs/pix2mesh_chair_v3/model
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from train_pix2mesh_chair import Pixel2MeshDINO, TAP_BLOCKS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Hook-free encoder ─────────────────────────────────────────────────────────
# The training encoder uses forward hooks to tap intermediate DINO blocks.
# Hooks are Python side-effects and make ONNX tracing unreliable; we replicate
# the same data flow by iterating over dino.blocks directly instead.

class _HookFreeEncoder(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.dino     = enc.dino           # full DINOv2 ViT kept as a submodule
        self.lateral  = enc.lateral
        self.out_conv = enc.out_conv

    def forward(self, x):
        B, _, H, W = x.shape
        hp = H // self.dino.patch_size     # patch grid height (Python int at trace time)

        # Replicate DINOv2 VisionTransformer.prepare_tokens_with_masks(x, None)
        tok = self.dino.patch_embed(x)
        cls = self.dino.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        tok = tok + self.dino.interpolate_pos_encoding(tok, W, H)

        # Iterate blocks; tap outputs at fixed indices (resolved at trace time)
        tap2 = tap5 = tap8 = tap11 = tok  # initialised to satisfy type checker
        for i, blk in enumerate(self.dino.blocks):
            tok = blk(tok)
            if i == 2:  tap2  = tok[:, 1:]
            if i == 5:  tap5  = tok[:, 1:]
            if i == 8:  tap8  = tok[:, 1:]
            if i == 11: tap11 = tok[:, 1:]

        # FPN — same logic as the training encoder
        raw = [tap2, tap5, tap8, tap11]
        lats = [
            self.lateral[j](raw[j].permute(0, 2, 1).reshape(B, -1, hp, hp))
            for j in range(4)
        ]

        fpn = [None] * 4
        fpn[3] = lats[3]
        for i in range(2, -1, -1):
            fpn[i] = lats[i] + F.interpolate(fpn[i + 1], size=lats[i].shape[-2:], mode="nearest")

        sizes = [128, 64, 32, 16]
        return [
            self.out_conv[i](F.interpolate(fpn[i], size=(sizes[i], sizes[i]),
                                           mode="bilinear", align_corners=False))
            for i in range(4)
        ]


# ── Exportable wrapper ────────────────────────────────────────────────────────
# Replaces the hooked encoder, drops Laplacian / edge_pairs from the forward
# return (unused at inference time), and outputs plain (verts1, verts2, verts3).

class _ExportableP2M(nn.Module):
    def __init__(self, model: Pixel2MeshDINO):
        super().__init__()
        self.encoder = _HookFreeEncoder(model.encoder)
        self.stage1  = model.stage1
        self.stage2  = model.stage2
        self.stage3  = model.stage3

        for name in ["v1", "a1", "v2", "a2", "v3", "a3", "u12", "u23"]:
            self.register_buffer(name, getattr(model, name))

    @staticmethod
    def _sample(feat_maps, coords):
        in_bounds = (coords.abs() <= 1.0).all(dim=-1, keepdim=True).float()
        grid = coords.clamp(-1, 1).unsqueeze(1)
        parts = []
        for fm in feat_maps:
            s = F.grid_sample(fm, grid, mode="bilinear", align_corners=True, padding_mode="border")
            parts.append(s.squeeze(2).permute(0, 2, 1))
        return torch.cat(parts, dim=-1) * in_bounds

    @staticmethod
    def _project(verts, rot, trans, focal):
        v_cam = torch.bmm(verts, rot.transpose(1, 2)) + trans.unsqueeze(1)
        z = v_cam[..., 2:3].clamp(min=1e-4)
        return focal.view(-1, 1, 1) * v_cam[..., :2] / z

    @staticmethod
    def _unpool(x, mat):
        return torch.einsum("oi,bif->bof", mat, x)

    def forward(self, image, rot, trans, focal):
        B = image.shape[0]
        feat_maps = self.encoder(image)

        v1  = self.v1.unsqueeze(0).expand(B, -1, -1)
        p1, h1 = self.stage1(self._sample(feat_maps, self._project(v1, rot, trans, focal)),
                              None, v1, self.a1)

        p1u = self._unpool(p1, self.u12)
        h1u = self._unpool(h1, self.u12)
        p2, h2 = self.stage2(self._sample(feat_maps, self._project(p1u, rot, trans, focal)),
                              h1u, p1u, self.a2)

        p2u = self._unpool(p2, self.u23)
        h2u = self._unpool(h2, self.u23)
        p3, _ = self.stage3(self._sample(feat_maps, self._project(p2u, rot, trans, focal)),
                             h2u, p2u, self.a3)

        return p1, p2, p3


# ── Utilities ─────────────────────────────────────────────────────────────────

def _disable_xformers(model):
    """Set use_xformers=False on every attention module that has it."""
    disabled = 0
    for m in model.modules():
        if hasattr(m, "use_xformers") and m.use_xformers:
            m.use_xformers = False
            disabled += 1
    if disabled:
        print(f"  Disabled xformers on {disabled} attention module(s)")


def _faces_npz(model: Pixel2MeshDINO):
    return {
        "faces1": model.f1.cpu().numpy(),
        "faces2": model.f2.cpu().numpy(),
        "faces3": model.f3.cpu().numpy(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join(ROOT, "runs", "pix2mesh_chair_v3", "best.pt"))
    parser.add_argument("--out",     default=os.path.join(ROOT, "runs", "pix2mesh_chair_v3", "model"),
                        help="Output path prefix; .onnx and _faces.npz are appended")
    parser.add_argument("--opset",   type=int, default=18)
    parser.add_argument("--verify",  action="store_true", default=True,
                        help="Run onnxruntime after export to check numerical parity")
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    onnx_path  = args.out if args.out.endswith(".onnx") else args.out + ".onnx"
    faces_path = onnx_path.replace(".onnx", "_faces.npz")
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    device = torch.device(args.device)

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    print(f"\n[1/4] Loading checkpoint from {args.weights}")
    base_model = Pixel2MeshDINO()
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["model"])
    base_model.eval()
    print(f"      epoch {ckpt.get('epoch', '?')}")

    # ── 2. Build exportable wrapper ───────────────────────────────────────────
    print("\n[2/4] Wrapping model …")
    _disable_xformers(base_model)
    export_model = _ExportableP2M(base_model).to(device)
    export_model.eval()

    # Use randn dummy inputs — zeros can cause constant-folding to collapse the
    # encoder (e.g. the in_bounds mask becomes all-ones, giving the optimizer
    # enough static information to eliminate branches / large subgraphs).
    torch.manual_seed(0)
    dummy_image = torch.randn(1, 3, 224, 224, device=device)
    dummy_rot   = torch.eye(3, device=device).unsqueeze(0)
    dummy_trans = torch.tensor([[0.0, 0.0, 2.5]], device=device)
    dummy_focal = torch.tensor([2.5], device=device)

    # Smoke-test the wrapper before export
    with torch.no_grad():
        pt_p1, pt_p2, pt_p3 = export_model(dummy_image, dummy_rot, dummy_trans, dummy_focal)
    print(f"      PyTorch output shapes: {tuple(pt_p1.shape)} / {tuple(pt_p2.shape)} / {tuple(pt_p3.shape)}")

    # ── 3. Export to ONNX ─────────────────────────────────────────────────────
    print(f"\n[3/4] Exporting to {onnx_path}  (opset {args.opset}) …")
    torch.onnx.export(
        export_model,
        (dummy_image, dummy_rot, dummy_trans, dummy_focal),
        onnx_path,
        input_names  = ["image", "rot", "trans", "focal"],
        output_names = ["verts1", "verts2", "verts3"],
        opset_version = args.opset,
        do_constant_folding = True,
        export_params = True,
    )
    # PyTorch's dynamo exporter splits large models into <name>.onnx (graph) +
    # <name>.onnx.data (weights).  Sum both when checking total size.
    data_path = onnx_path + ".data"
    total_mb  = (os.path.getsize(onnx_path) +
                 (os.path.getsize(data_path) if os.path.exists(data_path) else 0)) / 1e6
    if os.path.exists(data_path):
        print(f"      Graph: {os.path.getsize(onnx_path)/1e6:.1f} MB  "
              f"Weights (external): {os.path.getsize(data_path)/1e6:.1f} MB  "
              f"Total: {total_mb:.1f} MB")
        print(f"      NOTE: distribute model.onnx + model.onnx.data together")
    else:
        print(f"      Written {total_mb:.1f} MB")
    if total_mb < 50:
        print("  *** WARNING: total size is suspiciously small — DINOv2 backbone may not have been captured.")
        print("  *** Check for export errors above.")

    # Save face topology separately
    np.savez_compressed(faces_path, **_faces_npz(base_model))
    print(f"      Faces saved to {faces_path}")

    # ── 4. Verify with onnxruntime ────────────────────────────────────────────
    if args.verify:
        print("\n[4/4] Verifying with onnxruntime …")
        try:
            import onnxruntime as ort
            import onnx
        except ImportError:
            print("      onnxruntime / onnx not installed — skipping verification")
            print("      Install with:  pip install onnx onnxruntime")
            return

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("      ONNX graph check passed")

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        feeds = {
            "image": dummy_image.numpy(),
            "rot":   dummy_rot.numpy(),
            "trans": dummy_trans.numpy(),
            "focal": dummy_focal.numpy(),
        }
        ort_p1, ort_p2, ort_p3 = sess.run(None, feeds)

        for name, pt, ort_out in [("verts1", pt_p1, ort_p1),
                                   ("verts2", pt_p2, ort_p2),
                                   ("verts3", pt_p3, ort_p3)]:
            diff = abs(pt.cpu().numpy() - ort_out).max()
            print(f"      {name}  max|PT − ORT| = {diff:.2e}  {'✓' if diff < 1e-4 else '✗ HIGH'}")

    print(f"\nDone.  ONNX model: {onnx_path}")
    print(f"       Faces file: {faces_path}")


if __name__ == "__main__":
    main()
