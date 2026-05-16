"""
window_detect.py — compact window detector for the floor plan pipeline.

Ports the FloorplanToBlenderLib window-detection approach as a standalone
module with no dependency on that library.

Detection flow
--------------
1. Wall-filter the grayscale image (Otsu + morphology + distance transform)
   to isolate uncertain gap regions.
2. Harris-corner gap-closing (room-closing heuristic) then connected-
   component labelling to find wall gaps sized _GAP_MIN_PX–_GAP_MAX_PX.
3. Pixel-density bandpass on the ORIGINAL grayscale at each gap location:
     density = count(px > 0) / sum(px values)
   Windows show sparse near-white regions (thin parallel lines) → density
   falls in (_WIN_LOW, _WIN_HIGH).  Denser or more complex regions (doors,
   noise) fall outside the band.
4. Any candidate overlapping an existing YOLO bbox is suppressed — YOLO
   door detections win unconditionally.

Returns dicts in the same format as detect_openings() in pipeline.py:
  {"label": "window", "cx": float, "cy": float,
   "bbox": (x1, y1, x2, y2), "conf": float}

The synthetic conf is 1.0 at the bandpass centre, 0.0 at its edges, so
YOLO door/window detections (real conf scores) should always be preferred
when merging results.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# ── tunable constants (from FloorplanToBlenderLib/const.py) ──────────────────

_MORPH_KERNEL = (3, 3)
_MORPH_ITERS = 2
_DILATE_ITERS = 3
_DIST_THRESH_LO = 0.5  # fraction of dist-transform for sure-fg
_DIST_THRESH_HI = 0.2  # fraction of dist-transform.max() for sure-fg
_NOISE_AREA = 50  # min contour area to keep during noise removal
_HARRIS_BLOCK = 2
_HARRIS_K_SIZE = 3
_HARRIS_K = 0.04
_HARRIS_ERODE_ITER = 10
_CORNERS_THR = 0.01  # fraction of Harris max
_CLOSING_MAX_LEN = 130  # max pixel gap to close between Harris corners
_GAP_MIN_PX = 10  # min component size to consider as opening
_GAP_MAX_PX = 5_000  # max component size (larger → room, not gap)
_WIN_LOW = 0.001
_WIN_HIGH = 0.00459
_WIN_RESCALE = 1.05  # expand detected window bbox slightly
_BOX_ACCURACY = 0.001  # polygon approximation for contours
_IOU_SUPPRESS_THR = 0.05  # IoU above which a FM window is suppressed by YOLO


# ── image helpers ─────────────────────────────────────────────────────────────


def _wall_filter(gray: np.ndarray) -> np.ndarray:
    """
    Reproduce FloorplanToBlenderLib.detect.wall_filter.
    Returns the 'unknown' band between sure-background and sure-foreground.
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones(_MORPH_KERNEL, np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=_MORPH_ITERS)
    sure_bg = cv2.dilate(opening, kernel, iterations=_DILATE_ITERS)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        _DIST_THRESH_LO * dist,
        _DIST_THRESH_HI * dist.max(),
        255,
        0,
    )
    return cv2.subtract(sure_bg, np.uint8(sure_fg))


def _remove_noise(img: np.ndarray) -> np.ndarray:
    """Return mask of large-enough contours (clone of image.remove_noise)."""
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
    """Zero-out the exterior of the largest contour (image.mark_outside_black)."""
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, mask
    biggest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest], 255)
    img[mask == 0] = 0
    return img, mask


def _close_corners(img: np.ndarray) -> np.ndarray:
    """
    Harris-corner gap-closing: draws short lines between nearby corners on
    the same row/column (FloorplanToBlenderLib.__corners_and_draw_lines).
    Mutates img in-place and returns it.
    """
    dst = cv2.cornerHarris(img, _HARRIS_BLOCK, _HARRIS_K_SIZE, _HARRIS_K)
    kernel = np.ones((1, 1), np.uint8)
    dst = cv2.erode(dst, kernel, iterations=_HARRIS_ERODE_ITER)
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
    """
    Find connected components in wall-gap space sized _GAP_MIN_PX–_GAP_MAX_PX.
    Returns a list of boolean masks, one per candidate gap.
    """
    mask = _remove_noise(img)
    work = ~mask
    _close_corners(work)
    work, _ = _mark_outside_black(work, mask)

    _, labels = cv2.connectedComponents(work)
    components = []
    for label in np.unique(labels):
        if label == 0:
            continue
        comp = labels == label
        count = int(np.count_nonzero(comp))
        if _GAP_MIN_PX <= count <= _GAP_MAX_PX:
            components.append(comp)
    return components


# ── geometry helpers ──────────────────────────────────────────────────────────


def _rescale_bbox(
    x: float, y: float, w: float, h: float, factor: float
) -> tuple[float, float, float, float]:
    """Expand a bbox by factor around its centre (transform.rescale_rect)."""
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
    Detect windows via the FloorplanToBlenderLib pixel-density bandpass method.

    Parameters
    ----------
    gray_no_text   : uint8 grayscale of the text-removed image (same spatial
                     space as the YOLO input passed to detect_openings).
    existing_bboxes: (x1,y1,x2,y2) tuples from YOLO — window candidates that
                     overlap these by more than iou_threshold are suppressed.
                     Door detections should always be in this list so they win.
    iou_threshold  : overlap fraction above which a candidate is dropped.

    Returns
    -------
    list of dicts: {label, cx, cy, bbox:(x1,y1,x2,y2), conf}
    conf is synthetic (1.0 at bandpass centre → 0.0 at edges), intentionally
    lower than typical YOLO scores so door confidence always dominates.
    """
    existing_bboxes = existing_bboxes or []

    # Step 1 — isolate gap regions
    gap_img = ~_wall_filter(gray_no_text)

    # Step 2 — find small wall-gap components
    components = _find_gap_components(gap_img.copy())

    windows: list[dict[str, Any]] = []

    for comp in components:
        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1

        # Step 3 — pixel-density bandpass on original grayscale
        patch = gray_no_text[y : y + h, x : x + w]
        total = float(np.sum(patch))
        if total == 0:
            continue
        density = float(np.sum(patch > 0)) / total

        if not (_WIN_LOW < density < _WIN_HIGH):
            continue

        # Rescale bbox to better fit the window symbol outline
        rx, ry, rw, rh = _rescale_bbox(x, y, w, h, _WIN_RESCALE)
        x1, y1, x2, y2 = rx, ry, rx + rw, ry + rh
        cx_w, cy_w = (x1 + x2) / 2, (y1 + y2) / 2
        bbox = (x1, y1, x2, y2)

        # Step 4 — suppress if overlaps any higher-priority YOLO detection
        if any(_iou(bbox, eb) > iou_threshold for eb in existing_bboxes):
            continue

        # Synthetic conf: 1.0 at bandpass midpoint, 0.0 at edges
        mid = (_WIN_LOW + _WIN_HIGH) / 2.0
        half = (_WIN_HIGH - _WIN_LOW) / 2.0
        conf = round(1.0 - abs(density - mid) / half, 3)

        windows.append(dict(label="window", cx=cx_w, cy=cy_w, bbox=bbox, conf=conf))

    return windows
