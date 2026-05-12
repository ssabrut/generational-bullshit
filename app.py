from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from models import ProcessFloorplanResponse
from pipeline import run_pipeline

# NOTE: Keep your existing notebook processing logic in pipeline.py.
# app.py is a thin HTTP wrapper that reads the uploaded image, saves it
# temporarily, and forwards the image into the established pipeline.
app = FastAPI(
    title="Floorplan Processing API",
    description="Upload a floor plan image and return walls/openings JSON.",
    version="1.0",
)


@app.get("/", response_model=dict)
async def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "Floorplan API is running."}


async def _save_upload_to_temp_file(upload: UploadFile) -> Path:
    contents = await upload.read()
    if not contents:
        raise ValueError("Uploaded file is empty.")

    suffix = Path(upload.filename).suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(contents)
        tmp_path = Path(tmp_file.name)

    return tmp_path


@app.post(
    "/process-floorplan",
    response_model=ProcessFloorplanResponse,
    summary="Process a floor plan image and return walls/openings JSON",
)
async def process_floorplan(file: UploadFile = File(...)) -> ProcessFloorplanResponse:
    """Upload an image file and return the final floor plan JSON structure."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded file.")

    temp_path = await _save_upload_to_temp_file(file)
    try:
        image = cv2.imread(str(temp_path))
        if image is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        # This is a CPU-bound pipeline, so run it on a threadpool.
        output = await run_in_threadpool(run_pipeline, image)
        return output.json
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass
