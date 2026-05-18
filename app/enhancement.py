"""Grayscale conversion, denoising, and binarisation."""

from __future__ import annotations

import cv2
import numpy as np


def binarize(image: np.ndarray, gaussian_ksize: int = 5) -> np.ndarray:
    """Convert a BGR image to an inverted Otsu-thresholded binary image.

    Pipeline: BGR → grayscale → Gaussian blur → inverted Otsu threshold.
    The result has foreground (walls, fixtures) as 255 and background as 0,
    which is the convention expected by ``cv2.findNonZero`` downstream.

    Parameters
    ----------
    image:
        BGR input image.
    gaussian_ksize:
        Gaussian kernel side. Even values are bumped to the next odd value
        as required by OpenCV.

    Returns
    -------
    np.ndarray
        ``uint8`` binary image of shape ``(H, W)``.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = gaussian_ksize | 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary
