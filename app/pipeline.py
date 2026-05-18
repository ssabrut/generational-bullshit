"""Top-level orchestration for the floor-plan preprocessing pipeline."""

from __future__ import annotations

import logging

import numpy as np

from app.config import PreprocessorConfig
from app.cropping import crop_to_content
from app.enhancement import binarize
from app.text_removal import TextRemover
from app.types import PipelineResult

logger = logging.getLogger(__name__)


class FloorPlanPreprocessor:
    """End-to-end preprocessing pipeline for floor-plan images.

    Stages
    ------
    1. **Text removal** — multi-angle EasyOCR sweep produces a union mask of
       text regions, which is painted white on the source image.
    2. **Enhancement** — grayscale → Gaussian blur → inverted Otsu.
    3. **Crop & pad** — bound to foreground content with configurable margin;
       the same crop is applied to the text-free BGR image so callers can
       align colour and binary views.

    Example
    -------
    >>> import cv2
    >>> from app import FloorPlanPreprocessor, PreprocessorConfig
    >>> pre = FloorPlanPreprocessor(PreprocessorConfig(gpu=False))
    >>> result = pre.process(cv2.imread("floor.png"))
    >>> result.preprocessed.shape  # cropped binary
    """

    def __init__(self, config: PreprocessorConfig | None = None) -> None:
        self._cfg = config or PreprocessorConfig()
        self._text_remover = TextRemover(self._cfg)

    @property
    def config(self) -> PreprocessorConfig:
        """The active configuration (immutable)."""
        return self._cfg

    def process(self, image: np.ndarray) -> PipelineResult:
        """Run the full pipeline on a single BGR image.

        Parameters
        ----------
        image:
            BGR image as a ``uint8`` numpy array of shape ``(H, W, 3)``.

        Returns
        -------
        PipelineResult
            Structured output containing the text-free colour crop, the
            binarised crop, the best detected rotation angle, and the
            bounding box used for cropping.

        Raises
        ------
        TypeError
            If ``image`` is not a numpy array.
        ValueError
            If ``image`` is not a 3-channel BGR array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be HxWx3 BGR, got shape {image.shape}")

        logger.debug("Stage 1: text removal")
        raw_no_text, best_angle = self._text_remover.remove(image)

        logger.debug("Stage 2: binarisation")
        binary = binarize(raw_no_text, self._cfg.gaussian_ksize)

        logger.debug("Stage 3: crop & pad")
        preprocessed, bbox = crop_to_content(binary, self._cfg.crop_padding)
        x1, y1, x2, y2 = bbox
        raw_no_text_cropped = raw_no_text[y1:y2, x1:x2]

        return PipelineResult(
            raw_no_text=raw_no_text_cropped,
            preprocessed=preprocessed,
            rotation_angle=best_angle,
            crop_bbox=bbox,
        )
