"""Multi-angle OCR-based text detection and erasure.

Floor plans frequently contain labels rotated at 0/90/180/270°. A single OCR
pass misses most of them, so this module rotates the image through a
configurable sweep, accumulates detections into one mask, and paints them out.
"""

from __future__ import annotations

from typing import TypedDict

import cv2
import easyocr
import numpy as np

from app.config import PreprocessorConfig
from app.rotation import rotate_expand, unrotate_mask


class Detection(TypedDict):
    """One OCR hit: polygon bounding box, decoded text, confidence."""

    box: np.ndarray
    text: str
    conf: float


class TextRemover:
    """Detects and erases text from a floor-plan image using EasyOCR.

    The reader is constructed once and reused across calls to amortise the
    expensive model load.
    """

    def __init__(self, config: PreprocessorConfig) -> None:
        self._cfg = config
        self._reader = easyocr.Reader(list(config.languages), gpu=config.gpu)

    def remove(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        """Erase all detected text from ``image``.

        Parameters
        ----------
        image:
            BGR image to clean.

        Returns
        -------
        tuple[np.ndarray, int]
            ``(cleaned_image, best_angle)`` where ``best_angle`` is the
            rotation (degrees, clockwise) that produced the most detections —
            a useful upright-orientation hint for downstream stages.
        """
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        angle_counts: dict[int, int] = {}

        for angle in range(0, 360, self._cfg.angle_step):
            rotated = rotate_expand(image, angle)
            detections = self._detect(rotated)
            angle_counts[angle] = len(detections)
            if not detections:
                continue
            rot_mask = self._build_mask(rotated.shape, detections)
            combined_mask = cv2.bitwise_or(
                combined_mask, unrotate_mask(rot_mask, image.shape, angle)
            )

        best_angle = max(angle_counts, key=angle_counts.get)
        return self._erase(image, combined_mask), best_angle

    def _detect(self, image: np.ndarray) -> list[Detection]:
        return [
            Detection(box=np.asarray(box, dtype=np.int32), text=text, conf=conf)
            for box, text, conf in self._reader.readtext(image)
            if conf >= self._cfg.confidence_threshold
        ]

    def _build_mask(
        self, shape: tuple[int, ...], detections: list[Detection]
    ) -> np.ndarray:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for d in detections:
            cv2.fillPoly(mask, [d["box"]], 255)
        if self._cfg.dilation_px > 0:
            side = self._cfg.dilation_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (side, side))
            mask = cv2.dilate(mask, kernel)
        return mask

    @staticmethod
    def _erase(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = image.copy()
        out[mask == 255] = 255
        return out
