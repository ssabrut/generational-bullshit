"""End-to-end smoke test for the student.

- Loads ONE cache entry
- Builds Student (DinoV2-S/14 encoder + 1 deformation stage)
- Overfits a small number of steps on this single sample to verify:
    * forward pass works
    * gradients flow through cross-attn + GCN
    * Chamfer loss decreases monotonically
    * the deformed template visibly moves toward the target

Saves a side-by-side preview: input | target points | template | deformed mesh.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.dataset import TeacherCacheDataset  # noqa: E402
from student.loss import compute_loss  # noqa: E402
from student.model import Student  # noqa: E402
from student.template import build_template  # noqa: E402

# Reuse the back-face-culled shader from the teacher previews
sys.path.insert(0, str(ROOT / "teacher"))
from render_previews import _cull_and_shade  # noqa: E402

_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def render_mesh(verts: np.ndarray, faces: np.ndarray, ax, azim=30, elev=15) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v = verts @ _YUP_TO_ZUP.T
    front, shaded = _cull_and_shade(v, faces, azim, elev)
    ax.add_collection3d(
        Poly3DCollection(v[front], facecolors=shaded, edgecolors="none")
    )
    b = np.stack([v.min(0), v.max(0)])
    ax.set_xlim(b[0, 0], b[1, 0])
    ax.set_ylim(b[0, 1], b[1, 1])
    ax.set_zlim(b[0, 2], b[1, 2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.set_axis_off()


def render_points(pts: np.ndarray, ax, azim=30, elev=15) -> None:
    p = pts @ _YUP_TO_ZUP.T
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.5, c="#3355aa", alpha=0.6)
    b = np.stack([p.min(0), p.max(0)])
    ax.set_xlim(b[0, 0], b[1, 0])
    ax.set_ylim(b[0, 1], b[1, 1])
    ax.set_zlim(b[0, 2], b[1, 2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.set_axis_off()


def main() -> None:
    torch.manual_seed(0)
    device = (
        "cpu"  # MPS is hit-or-miss with DinoV2 ops; CPU is safer for the smoke test
    )

    print(f"Loading dataset…  device={device}")
    ds = TeacherCacheDataset(
        cache_root=ROOT / "data/teacher_cache", categories=("chair",)
    )
    print(f"  {len(ds)} entries available")
    sample = ds[0]
    image = sample["image"].unsqueeze(0).to(device)  # (1, 3, 224, 224)
    target = (
        sample["points"].unsqueeze(0).to(device)
    )  # (1, N, 3) — unit-sphere-normalized

    # Template lives in Pix3D units; normalize to unit sphere to match target.
    print("Building template…")
    tpl = build_template()
    tpl_verts_np_raw = tpl.verts.numpy().copy()
    # Normalize template to unit sphere too
    tpl.verts = tpl.verts - tpl.verts.mean(dim=0, keepdim=True)
    tpl.verts = tpl.verts / tpl.verts.norm(dim=1).max()
    print(
        f"  template: {tpl.verts.shape[0]} verts, {tpl.faces.shape[0]} faces, "
        f"{tpl.edge_index.shape[1]} bidirectional edges"
    )

    print("Building student (this downloads DinoV2-S/14 on first run)…")
    t0 = time.time()
    model = Student(template=tpl, hidden=128, n_stages=1).to(device)
    laplacian = tpl.laplacian.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"  built in {time.time() - t0:.1f}s | trainable {n_trainable / 1e6:.2f}M / "
        f"total {n_total / 1e6:.2f}M params"
    )

    # Capture the initial (un-deformed) prediction
    with torch.no_grad():
        out0 = model(image)
        loss0, comp0 = compute_loss(
            out0["verts"], target, model.template_edges, laplacian
        )
    print(f"\nstep   0 | loss {float(loss0):.4f}  {comp0}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
    )

    n_steps = 50
    losses = [float(loss0)]
    for step in range(1, n_steps + 1):
        out = model(image)
        loss, comp = compute_loss(out["verts"], target, model.template_edges, laplacian)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % 10 == 0 or step == 1:
            print(f"step {step:3d} | loss {float(loss):.4f}  {comp}")

    # Final prediction
    with torch.no_grad():
        out_final = model(image)
    pred_verts = out_final["verts"][0].cpu().numpy()
    faces = model.template_faces.cpu().numpy()
    target_np = target[0].cpu().numpy()
    tpl_norm = tpl.verts.cpu().numpy()

    # ── visualization ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), subplot_kw={"projection": "3d"})
    # Manually replace axes[0] with a 2D axis for the image
    fig.delaxes(axes[0])
    ax_img = fig.add_subplot(1, 5, 1)
    img_show = sample["image"].permute(1, 2, 0).numpy()
    img_show = img_show * np.array([0.229, 0.224, 0.225]) + np.array(
        [0.485, 0.456, 0.406]
    )
    ax_img.imshow(np.clip(img_show, 0, 1))
    ax_img.set_title("input")
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    render_points(target_np, axes[1])
    axes[1].set_title("target points")
    render_mesh(tpl_norm, faces, axes[2])
    axes[2].set_title("template (before)")
    render_mesh(pred_verts, faces, axes[3])
    axes[3].set_title(f"deformed (after {n_steps} steps)")

    fig.delaxes(axes[4])
    ax_loss = fig.add_subplot(1, 5, 5)
    ax_loss.plot(losses, color="#aa3355")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("total loss")
    ax_loss.set_title(f"loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    ax_loss.grid(alpha=0.3)

    out_path = ROOT / "data/student_smoke.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nsaved → {out_path}")
    print(
        f"loss dropped: {losses[0]:.4f} → {losses[-1]:.4f}  "
        f"(Δ = {losses[0] - losses[-1]:.4f})"
    )
    # Unused: keep tpl_verts_np_raw silence
    _ = tpl_verts_np_raw


if __name__ == "__main__":
    main()
