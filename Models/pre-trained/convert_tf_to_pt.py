"""
Convert the original Pixel2Mesh TF1 checkpoint (gcn.ckpt) to a PyTorch state dict.

TF Conv2D weight layout : [H, W, C_in, C_out]
PyTorch Conv2d weight layout: [C_out, C_in, H, W]
=> transpose axes (3, 2, 0, 1)

GCN dense weights are [in, out] in both frameworks — no transpose needed.
Biases are 1-D in both — no transform needed.
"""

import os
import numpy as np
import torch
import tensorflow as tf  # noqa: TF2 compat layer reads TF1 ckpts fine

CKPT = os.path.join(os.path.dirname(__file__), "checkpoint", "gcn.ckpt")
OUT  = os.path.join(os.path.dirname(__file__), "pix2mesh_pretrained.pt")


def load_tf_weights(ckpt_path: str) -> dict[str, np.ndarray]:
    reader = tf.train.load_checkpoint(ckpt_path)
    shapes = reader.get_variable_to_shape_map()
    return {name: reader.get_tensor(name) for name in shapes}


def convert(tf_weights: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}

    # ── VGG encoder ──────────────────────────────────────────────────────────
    # TF names: gcn/Conv2D_N/W:0  and  gcn/Conv2D_N/b:0
    # We expose them as: encoder.conv{N}.weight / .bias  (0-indexed)
    conv_indices = sorted(
        {int(k.split("/")[1].replace("Conv2D", "").replace("_", "") or "0")
         for k in tf_weights if "Conv2D" in k and "/W:" in k}
    )
    for i in conv_indices:
        suffix = "" if i == 0 else f"_{i}"
        w = tf_weights[f"gcn/Conv2D{suffix}/W:0"]   # [H, W, C_in, C_out]
        b = tf_weights[f"gcn/Conv2D{suffix}/b:0"]
        state_dict[f"encoder.conv{i}.weight"] = torch.from_numpy(
            w.transpose(3, 2, 0, 1)
        )
        state_dict[f"encoder.conv{i}.bias"] = torch.from_numpy(b)

    # ── GCN blocks ───────────────────────────────────────────────────────────
    # TF names: gcn/graphconvolution_{N}_vars/weights_0:0
    #                                         weights_1:0
    #                                         bias:0
    # We expose them as: gcn.{N}.w0 / .w1 / .bias
    gcn_indices = sorted(
        {int(k.split("/")[1].replace("graphconvolution_", "").replace("_vars", ""))
         for k in tf_weights if "graphconvolution" in k}
    )
    for i in gcn_indices:
        prefix = f"gcn/graphconvolution_{i}_vars"
        w0 = tf_weights[f"{prefix}/weights_0:0"]
        w1 = tf_weights[f"{prefix}/weights_1:0"]
        b  = tf_weights[f"{prefix}/bias:0"]
        state_dict[f"gcn.{i}.w0"]   = torch.from_numpy(w0)
        state_dict[f"gcn.{i}.w1"]   = torch.from_numpy(w1)
        state_dict[f"gcn.{i}.bias"] = torch.from_numpy(b)

    return state_dict


def main():
    print(f"Reading checkpoint: {CKPT}")
    tf_weights = load_tf_weights(CKPT)
    print(f"  {len(tf_weights)} variables found")

    state_dict = convert(tf_weights)
    print(f"  {len(state_dict)} tensors in converted state dict")

    torch.save({"state_dict": state_dict}, OUT)
    print(f"Saved: {OUT}")

    # quick sanity check
    print("\nSample entries:")
    for k, v in list(state_dict.items())[:6]:
        print(f"  {k:40s} {tuple(v.shape)}  {v.dtype}")


if __name__ == "__main__":
    main()
