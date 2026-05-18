"""Shared data structures returned by the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

BBox = tuple[int, int, int, int]
"""Axis-aligned bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates."""


@dataclass(frozen=True)
class PipelineResult:
    """Structured output of :meth:`FloorPlanPreprocessor.process`.

    Attributes
    ----------
    raw_no_text:
        BGR image with detected text painted out. Cropped to ``crop_bbox``
        so it shares the spatial frame of ``preprocessed``.
    preprocessed:
        Binary (uint8) image after Gaussian blur and Otsu thresholding,
        cropped and padded to the content bounding box.
    rotation_angle:
        Clockwise rotation (degrees) at which the OCR sweep found the most
        text. Useful as a hint for downstream upright-orientation logic.
    crop_bbox:
        ``(x1, y1, x2, y2)`` of the crop applied to both images, expressed
        in the **original** (uncropped) image coordinate frame.

    Notes
    -----
    The dataclass supports tuple unpacking for the two most common outputs::

        raw_no_text, preprocessed = preprocessor.process(image)
    """

    raw_no_text: np.ndarray
    preprocessed: np.ndarray
    rotation_angle: int
    crop_bbox: BBox

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.raw_no_text
        yield self.preprocessed
