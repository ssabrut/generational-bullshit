"""Content-aware cropping with configurable padding."""

from __future__ import annotations

import cv2
import numpy as np

from app.types import BBox


class EmptyImageError(ValueError):
    """Raised when a binary image contains no foreground pixels to crop to."""


def crop_to_content(binary: np.ndarray, padding: int) -> tuple[np.ndarray, BBox]:
    """Crop a binary image to its foreground bounding box, with padding.

    Parameters
    ----------
    binary:
        ``uint8`` binary image where non-zero pixels are foreground.
    padding:
        Pixels of margin retained on each side of the bounding box. The crop
        is clamped to the image bounds so the result is always in-frame.

    Returns
    -------
    tuple[np.ndarray, BBox]
        ``(cropped_image, bbox)`` where ``bbox`` is ``(x1, y1, x2, y2)`` in
        the original image's coordinate frame.

    Raises
    ------
    EmptyImageError
        If ``binary`` has no foreground pixels.
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        raise EmptyImageError("No foreground pixels found — cannot crop.")

    H, W = binary.shape[:2]
    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    return binary[y1:y2, x1:x2], (x1, y1, x2, y2)
