"""Export TripoSR's triplane generator to ONNX (image → scene_codes).

The NeRF renderer and marching-cubes are intentionally excluded — they run on
the server (see server/app.py). Only the feedforward half is exported:

    image [1, 3, 512, 512] (float32, ImageNet-normalised)
        → scene_codes [1, 3, 40, 64, 64]  (float32)

Pipeline replicated from TSR.forward() in teacher/TripoSR/tsr/system.py:
  1. DINOSingleImageTokenizer  (ViT-B/16)
  2. Triplane1DTokenizer.forward()  (learned triplane embeddings)
  3. Transformer1D backbone  (16-layer cross-attn transformer)
  4. Transformer1D detokenize + TriplaneUpsampleNetwork

ONNX tracing notes
------------------
- AttnProcessor2_0 uses F.scaled_dot_product_attention → opset 17+ only
- interpolate_pos_encoding in ViTModel uses dynamic shapes; we fix input size
  to 512×512 so the patch grid (32×32) is constant at trace time
- einops.rearrange is resolved to plain tensor ops during tracing

Usage:
    conda run -n AI python scripts/export_triposr_onnx.py
    conda run -n AI python scripts/export_triposr_onnx.py --out runs/triposr_onnx/triplane_gen --verify
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "teacher" / "TripoSR"))

from tsr.system import TSR


# ── Exportable wrapper ────────────────────────────────────────────────────────

class _TriplaneGenerator(nn.Module):
    """Wraps the feedforward image→scene_codes path of TSR.

    Accepts a pre-normalised image tensor so the caller controls
    normalisation (important for on-device preprocessing on the server).
    The ViT's interpolate_pos_encoding is called with fixed H=W=512 so
    torch.onnx.export can resolve the patch grid size at trace time.
    """

    def __init__(self, tsr: TSR):
        super().__init__()
        self.image_tokenizer = tsr.image_tokenizer
        self.tokenizer       = tsr.tokenizer
        self.backbone        = tsr.backbone
        self.post_processor  = tsr.post_processor

        # Freeze the learned triplane embeddings as a buffer so they end up
        # in the ONNX graph without needing an input slot.
        self.register_buffer("triplane_embeddings", tsr.tokenizer.embeddings.clone())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [1, 3, 512, 512]  float32, ImageNet-normalised

        Returns:
            scene_codes: [1, 3, 40, 64, 64]  float32
        """
        B = image.shape[0]

        # --- image tokenizer (DINOv2 ViT-B/16) ---
        # mirror of DINOSingleImageTokenizer.forward with packed=False
        image_5d = image.unsqueeze(1)           # [B, 1, C, H, W]
        image_5d = (image_5d - self.image_tokenizer.image_mean) / self.image_tokenizer.image_std
        vit_out   = self.image_tokenizer.model(
            image_5d.squeeze(1),                # [B, C, H, W]
            interpolate_pos_encoding=True,
        )
        local_features = vit_out.last_hidden_state.permute(0, 2, 1)  # [B, Ct, Nt]
        local_features = local_features.unsqueeze(1)                  # [B, 1, Ct, Nt]
        # flatten Nv=1 → input_image_tokens [B, Nt, Ct]
        input_image_tokens = rearrange(local_features, "B Nv Ct Nt -> B (Nv Nt) Ct", Nv=1)

        # --- triplane tokenizer (learned embeddings, broadcast to batch) ---
        tokens = rearrange(
            self.triplane_embeddings.unsqueeze(0).expand(B, -1, -1, -1, -1),
            "B Np Ct Hp Wp -> B Ct (Np Hp Wp)",
        )

        # --- transformer backbone (16-layer cross-attention) ---
        tokens = self.backbone(tokens, encoder_hidden_states=input_image_tokens)

        # --- detokenize + upsample ---
        batch_size, Ct, Nt = tokens.shape
        # Triplane1DTokenizer.detokenize: [B, Ct, Nt] → [B, 3, Ct, Hp, Wp]
        triplanes = rearrange(
            tokens,
            "B Ct (Np Hp Wp) -> B Np Ct Hp Wp",
            Np=3,
            Hp=self.tokenizer.cfg.plane_size,
            Wp=self.tokenizer.cfg.plane_size,
        )
        scene_codes = self.post_processor(triplanes)  # [B, 3, 40, 64, 64]
        return scene_codes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "models" / "pre-trained" / "triposr"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "runs" / "triposr_onnx" / "triplane_gen"),
        help="Output path prefix; .onnx is appended",
    )
    parser.add_argument("--opset",  type=int, default=18)
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    onnx_path = args.out if args.out.endswith(".onnx") else args.out + ".onnx"
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    device = torch.device(args.device)

    # 1. Load checkpoint
    print(f"\n[1/4] Loading TripoSR from {args.model_dir}")
    tsr = TSR.from_pretrained(
        args.model_dir, config_name="config.yaml", weight_name="model.ckpt"
    )
    tsr.eval().to(device)

    # 2. Build exportable wrapper
    print("\n[2/4] Building exportable wrapper …")
    gen = _TriplaneGenerator(tsr).to(device)
    gen.eval()

    # Disable gradient checkpointing (incompatible with tracing)
    for m in gen.modules():
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = False

    torch.manual_seed(0)
    dummy_image = torch.randn(1, 3, 512, 512, device=device)

    with torch.no_grad():
        pt_out = gen(dummy_image)
    print(f"      PyTorch output shape: {tuple(pt_out.shape)}")
    assert tuple(pt_out.shape) == (1, 3, 40, 64, 64), "Unexpected output shape"

    # 3. Export to ONNX
    # dynamo=False forces the legacy TorchScript-based exporter, bypassing
    # onnxscript's version converter which has a broken Resize adapter for
    # opset < 18 on torch >= 2.x.  Opset 18 is the minimum where Resize is
    # natively supported without a downconversion step.
    print(f"\n[3/4] Exporting to {onnx_path}  (opset {args.opset}) …")
    torch.onnx.export(
        gen,
        dummy_image,
        onnx_path,
        input_names=["image"],
        output_names=["scene_codes"],
        opset_version=args.opset,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes={"image": {0: "batch"}, "scene_codes": {0: "batch"}},
        dynamo=False,
    )

    data_path = onnx_path + ".data"
    total_mb = (
        os.path.getsize(onnx_path)
        + (os.path.getsize(data_path) if os.path.exists(data_path) else 0)
    ) / 1e6
    if os.path.exists(data_path):
        print(
            f"      Graph: {os.path.getsize(onnx_path)/1e6:.1f} MB  "
            f"Weights: {os.path.getsize(data_path)/1e6:.1f} MB  "
            f"Total: {total_mb:.1f} MB"
        )
        print("      NOTE: distribute .onnx + .onnx.data together")
    else:
        print(f"      Written {total_mb:.1f} MB")

    if total_mb < 200:
        print(
            "  *** WARNING: model is suspiciously small — "
            "DINOv2 backbone may not have been captured."
        )

    # 4. Verify with onnxruntime
    if args.verify:
        print("\n[4/4] Verifying with onnxruntime …")
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            print("      onnx / onnxruntime not installed — skipping")
            print("      pip install onnx onnxruntime")
            return

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("      ONNX graph check passed")

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"image": dummy_image.cpu().numpy()})[0]

        diff = abs(pt_out.cpu().numpy() - ort_out).max()
        print(f"      scene_codes  max|PT − ORT| = {diff:.2e}  {'✓' if diff < 1e-4 else '✗ HIGH'}")

    print(f"\nDone. ONNX model: {onnx_path}")


if __name__ == "__main__":
    main()
