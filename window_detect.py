"""
window_detect.py — wall-gap gated template classifier for the floor plan pipeline.

Detection flow
--------------
1. Wall-filter (Otsu + morphology + distance transform) → connected-component
   gap labelling to find wall openings sized _GAP_MIN_PX–_GAP_MAX_PX.
2. For each gap component, crop the image (gap bbox + padding) and run
   cv2.matchTemplate against every entry in the template bank
   (Models/Windows/ + Models/Doors/, 4 rotations × N scales).
3. The template with the highest TM_CCOEFF_NORMED score determines the label
   ("window" or "door").  Gaps where no template scores above _MATCH_THR
   are discarded.
4. Any result overlapping an existing YOLO bbox is suppressed — YOLO wins.

Returns dicts in the same format as detect_openings() in pipeline.py:
  {"label": "window"|"door", "cx": float, "cy": float,
   "bbox": (x1, y1, x2, y2), "conf": float}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────

_MODELS_DIR = Path(__file__).parent / "Models"

# ── tunable constants ─────────────────────────────────────────────────────────

# wall-gap detection
_MORPH_KERNEL      = (3, 3)
_MORPH_ITERS       = 2
_DILATE_ITERS      = 3
_DIST_THRESH_LO    = 0.5
_DIST_THRESH_HI    = 0.2
_NOISE_AREA        = 50
_HARRIS_BLOCK      = 2
_HARRIS_K_SIZE     = 3
_HARRIS_K          = 0.04
_HARRIS_ERODE_ITER = 10
_CORNERS_THR       = 0.01
_CLOSING_MAX_LEN   = 130
_GAP_MIN_PX        = 10
_GAP_MAX_PX        = 5_000
_GAP_PAD           = 40    # px added around gap bbox before cropping

# template matching
_MATCH_THR   = 0.60        # minimum TM_CCOEFF_NORMED score to accept a classification
_SCALES      = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
_ROTATIONS   = [0, 1, 2, 3]   # k for np.rot90 → 0°, 90°, 180°, 270°

# output
_WIN_RESCALE      = 1.05
_IOU_SUPPRESS_THR = 0.05

# ── template bank (built once, cached at module level) ────────────────────────

_BANK: list[dict] | None = None


def _binarize(img: np.ndarray) -> np.ndarray:
    """Otsu + invert so line pixels are 255, background 0."""
    _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return out


def _build_bank() -> list[dict]:
    global _BANK
    if _BANK is not None:
        return _BANK
    entries: list[dict] = []
    for label, folder in [("window", "Windows"), ("door", "Doors")]:
        for p in sorted((_MODELS_DIR / folder).glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            binary = _binarize(img)
            for k in _ROTATIONS:
                rotated = np.rot90(binary, k)
                for s in _SCALES:
                    h, w = rotated.shape
                    nh, nw = max(8, int(h * s)), max(8, int(w * s))
                    scaled = cv2.resize(rotated, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    _, scaled = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)
                    entries.append({"img": scaled, "label": label})
    _BANK = entries
    return _BANK


# ── wall-gap detection ────────────────────────────────────────────────────────

def _wall_filter(gray: np.ndarray) -> np.ndarray:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel  = np.ones(_MORPH_KERNEL, np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=_MORPH_ITERS)
    sure_bg = cv2.dilate(opening, kernel, iterations=_DILATE_ITERS)
    dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        _DIST_THRESH_LO * dist, _DIST_THRESH_HI * dist.max(), 255, 0)
    return cv2.subtract(sure_bg, np.uint8(sure_fg))


def _remove_noise(img: np.ndarray) -> np.ndarray:
    work = img.copy()
    work[work < 128] = 0
    work[work > 128] = 255
    contours, _ = cv2.findContours(~work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(work)
    for c in contours:
        if cv2.contourArea(c) > _NOISE_AREA:
            cv2.fillPoly(mask, [c], 255)
    return mask


def _mark_outside_black(img: np.ndarray, mask: np.ndarray):
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, mask
    biggest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest], 255)
    img[mask == 0] = 0
    return img, mask


def _close_corners(img: np.ndarray) -> np.ndarray:
    dst    = cv2.cornerHarris(img, _HARRIS_BLOCK, _HARRIS_K_SIZE, _HARRIS_K)
    kernel = np.ones((1, 1), np.uint8)
    dst    = cv2.erode(dst, kernel, iterations=_HARRIS_ERODE_ITER)
    active = dst > _CORNERS_THR * dst.max()
    for y, row in enumerate(active):
        xs = np.argwhere(row)
        for x1, x2 in zip(xs[:-1], xs[1:]):
            if x2[0] - x1[0] < _CLOSING_MAX_LEN:
                cv2.line(img, (x1[0], y), (x2[0], y), 0, 1)
    for x, col in enumerate(active.T):
        ys = np.argwhere(col)
        for y1, y2 in zip(ys[:-1], ys[1:]):
            if y2[0] - y1[0] < _CLOSING_MAX_LEN:
                cv2.line(img, (x, y1[0]), (x, y2[0]), 0, 1)
    return img


def _find_gap_components(img: np.ndarray) -> list[np.ndarray]:
    mask = _remove_noise(img)
    work = ~mask
    _close_corners(work)
    work, _ = _mark_outside_black(work, mask)
    _, labels = cv2.connectedComponents(work)
    components = []
    for label in np.unique(labels):
        if label == 0:
            continue
        comp  = labels == label
        count = int(np.count_nonzero(comp))
        if _GAP_MIN_PX <= count <= _GAP_MAX_PX:
            components.append(comp)
    return components


# ── geometry helpers ──────────────────────────────────────────────────────────

def _rescale_bbox(x, y, w, h, factor):
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * factor, h * factor
    return cx - nw / 2, cy - nh / 2, nw, nh


def _iou(b1: tuple, b2: tuple) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


# ── public API ────────────────────────────────────────────────────────────────

def detect_windows_fm(
    gray_no_text: np.ndarray,
    existing_bboxes: list[tuple] | None = None,
    iou_threshold: float = _IOU_SUPPRESS_THR,
) -> list[dict[str, Any]]:
    """
    Detect windows and doors by classifying wall-gap components via template matching.

    For each gap found by the wall-filter, a padded crop is matched against the
    full template bank. The highest-scoring template determines the label.
    Gaps where no template scores above _MATCH_THR are discarded.

    Parameters
    ----------
    gray_no_text    : uint8 grayscale of the text-removed floor plan.
    existing_bboxes : (x1,y1,x2,y2) YOLO detections — FM results overlapping
                      these by more than iou_threshold are suppressed.
    iou_threshold   : overlap fraction above which an FM result is dropped.

    Returns
    -------
    list of dicts: {label, cx, cy, bbox:(x1,y1,x2,y2), conf}
    """
    existing_bboxes = existing_bboxes or []

    binary     = _binarize(gray_no_text)
    img_h, img_w = binary.shape
    bank       = _build_bank()

    gap_img    = ~_wall_filter(gray_no_text)
    components = _find_gap_components(gap_img.copy())

    results: list[dict[str, Any]] = []

    for comp in components:
        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1

        # Crop with padding so the full symbol is visible
        cx1 = max(0,     x - _GAP_PAD)
        cy1 = max(0,     y - _GAP_PAD)
        cx2 = min(img_w, x + w + _GAP_PAD)
        cy2 = min(img_h, y + h + _GAP_PAD)
        crop = binary[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape

        # Match every template against this crop; keep the global best score
        best_conf  = 0.0
        best_label = None

        for entry in bank:
            tmpl = entry["img"]
            th, tw = tmpl.shape
            if th >= ch or tw >= cw:
                continue
            result = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_conf:
                best_conf  = max_val
                best_label = entry["label"]

        if best_conf < _MATCH_THR or best_label is None:
            continue

        rx, ry, rw, rh = _rescale_bbox(x, y, w, h, _WIN_RESCALE)
        bbox   = (rx, ry, rx + rw, ry + rh)
        cx_gap = rx + rw / 2
        cy_gap = ry + rh / 2

        if any(_iou(bbox, eb) > iou_threshold for eb in existing_bboxes):
            continue

        results.append(dict(
            label=best_label,
            cx=cx_gap,
            cy=cy_gap,
            bbox=bbox,
            conf=round(best_conf, 3),
        ))

    return results
