"""TripoSR mesh server — Option A deployment backend.

Architecture:
  iPad → POST /mesh (JPEG/PNG) → server
  server:
    1. preprocess image (rembg + gray composite + crop + resize)
    2. run triplane generator ONNX → scene_codes  [1, 3, 40, 64, 64]
    3. load scene_codes into the TripoSR decoder + renderer (PyTorch, CPU/GPU)
    4. run marching cubes → trimesh.Trimesh
    5. return .obj or .glb

The ONNX model handles the heavy DINOv2+Transformer inference.
The PyTorch side only runs the small NeRF MLP + marching cubes, so the
server can serve the ONNX path on any CPU while keeping marching cubes
on the same machine.

Usage:
    # Export ONNX first (one-time):
    python scripts/export_triposr_onnx.py

    # Start server:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # From iPad (or curl):
    curl -X POST http://<server>:8000/mesh \
         -F "file=@photo.jpg" -F "format=glb" --output mesh.glb
"""

import io
import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
import torch
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "teacher" / "TripoSR"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tsr.system import TSR
from run_triposr import prepare_image  # reuse the tested preprocessing

logger = logging.getLogger("triposr_server")

# ── Config ────────────────────────────────────────────────────────────────────

ONNX_PATH    = os.environ.get(
    "TRIPOSR_ONNX",
    str(REPO_ROOT / "runs" / "triposr_onnx" / "triplane_gen.onnx"),
)
MODEL_DIR    = os.environ.get(
    "TRIPOSR_MODEL_DIR",
    str(REPO_ROOT / "models" / "pre-trained" / "triposr"),
)
DEVICE       = os.environ.get("TRIPOSR_DEVICE", "cpu")
MC_RES       = int(os.environ.get("TRIPOSR_MC_RES", "256"))
MC_THRESHOLD = float(os.environ.get("TRIPOSR_MC_THRESHOLD", "25.0"))
CHUNK_SIZE   = int(os.environ.get("TRIPOSR_CHUNK_SIZE", "8192"))
REMBG_MODEL  = os.environ.get("TRIPOSR_REMBG_MODEL", "isnet-general-use")
IMAGE_SIZE   = 512
FG_RATIO     = 0.85

# ImageNet normalisation constants (matches DINOSingleImageTokenizer)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Global state (loaded once at startup) ────────────────────────────────────

class _State:
    ort_session: ort.InferenceSession = None
    tsr: TSR = None
    rembg_session = None


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )

    logger.info("Loading ONNX triplane generator from %s", ONNX_PATH)
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if DEVICE != "cpu"
        else ["CPUExecutionProvider"]
    )
    state.ort_session = ort.InferenceSession(ONNX_PATH, providers=providers)
    logger.info("ONNX session ready  (providers: %s)", state.ort_session.get_providers())

    logger.info("Loading TripoSR decoder/renderer from %s", MODEL_DIR)
    state.tsr = TSR.from_pretrained(
        MODEL_DIR, config_name="config.yaml", weight_name="model.ckpt"
    )
    state.tsr.renderer.set_chunk_size(CHUNK_SIZE)
    # We only use decoder + renderer — the tokenizer/backbone/post_processor
    # are in the ONNX graph, so we can discard them from memory.
    del state.tsr.image_tokenizer
    del state.tsr.tokenizer
    del state.tsr.backbone
    del state.tsr.post_processor
    state.tsr.eval().to(DEVICE)
    logger.info("TripoSR decoder ready on %s", DEVICE)

    import rembg
    state.rembg_session = rembg.new_session(REMBG_MODEL)
    logger.info("rembg session ready  (model: %s)", REMBG_MODEL)

    yield

    # cleanup (nothing to do for onnxruntime / trimesh)


app = FastAPI(title="TripoSR Mesh Server", lifespan=lifespan)


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess(file_bytes: bytes) -> np.ndarray:
    """Return float32 [1, 3, 512, 512] ImageNet-normalised array."""
    from rembg import remove

    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
    cutout  = remove(pil_img, session=state.rembg_session)
    arr     = np.array(cutout, dtype=np.float32) / 255.0
    alpha   = arr[..., 3]
    rgb     = arr[..., :3]

    prepped = prepare_image(rgb, alpha, size=IMAGE_SIZE, fg_ratio=FG_RATIO)
    if prepped is None:
        raise HTTPException(status_code=422, detail="Could not detect a foreground object in the image.")

    img_np = np.array(prepped, dtype=np.float32) / 255.0           # [H, W, 3]
    img_np = (img_np - _MEAN) / _STD                               # ImageNet norm
    img_np = img_np.transpose(2, 0, 1)[None]                       # [1, 3, H, W]
    return img_np.astype(np.float32)


# ── Mesh extraction (PyTorch side) ────────────────────────────────────────────

def _scene_codes_to_mesh(scene_codes_np: np.ndarray) -> trimesh.Trimesh:
    """Run NeRF MLP + marching cubes on scene_codes from ONNX."""
    scene_codes = torch.from_numpy(scene_codes_np).to(DEVICE)  # [1, 3, 40, 64, 64]
    meshes = state.tsr.extract_mesh(
        scene_codes,
        has_vertex_color=True,
        resolution=MC_RES,
        threshold=MC_THRESHOLD,
    )
    return meshes[0]


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/mesh")
async def generate_mesh(
    file: UploadFile = File(..., description="Input image (JPEG or PNG)"),
    format: Literal["obj", "glb"] = Form("glb", description="Output mesh format"),
):
    """Accept a single product/object image, return a 3-D mesh."""
    file_bytes = await file.read()

    # 1. Preprocess
    try:
        image_np = _preprocess(file_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {exc}") from exc

    # 2. Triplane ONNX inference
    try:
        (scene_codes_np,) = state.ort_session.run(None, {"image": image_np})
    except Exception as exc:
        logger.exception("ONNX inference failed")
        raise HTTPException(status_code=500, detail=f"ONNX inference error: {exc}") from exc

    # 3. Marching cubes
    try:
        mesh = _scene_codes_to_mesh(scene_codes_np)
    except Exception as exc:
        logger.exception("Mesh extraction failed")
        raise HTTPException(status_code=500, detail=f"Mesh extraction error: {exc}") from exc

    # 4. Serialise
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, f"mesh.{format}")
        mesh.export(out_path)
        with open(out_path, "rb") as f:
            mesh_bytes = f.read()

    media_type = "model/gltf-binary" if format == "glb" else "text/plain"
    return Response(
        content=mesh_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=mesh.{format}"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "onnx_providers": state.ort_session.get_providers() if state.ort_session else None,
        "device": DEVICE,
        "mc_resolution": MC_RES,
    }
