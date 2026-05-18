"""Rotation helpers that preserve full image content (no corner clipping)."""

from __future__ import annotations

import cv2
import numpy as np


def rotate_expand(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an image clockwise by ``angle`` degrees, expanding the canvas.

    The output canvas is sized so the rotated image fits without clipping.
    Newly exposed pixels are filled with white (255) to match the typical
    floor-plan background.

    Parameters
    ----------
    image:
        Input image. Any number of channels supported by ``cv2.warpAffine``.
    angle:
        Clockwise rotation in degrees.

    Returns
    -------
    np.ndarray
        Rotated image with shape ``(new_h, new_w, ...)``.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    return cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def unrotate_mask(
    mask: np.ndarray, original_shape: tuple[int, ...], angle: int
) -> np.ndarray:
    """Inverse of :func:`rotate_expand` for a binary mask.

    Projects ``mask`` (computed on the rotated/expanded canvas) back into the
    coordinate frame of an image with shape ``original_shape``. Uses nearest-
    neighbour interpolation to keep mask values strictly in ``{0, 255}``.
    """
    h_rot, w_rot = mask.shape[:2]
    h_orig, w_orig = original_shape[:2]
    cx, cy = w_rot / 2, h_rot / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += (w_orig / 2) - cx
    M[1, 2] += (h_orig / 2) - cy

    return cv2.warpAffine(
        mask,
        M,
        (w_orig, h_orig),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
