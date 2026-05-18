"""Configuration objects for the preprocessing pipeline.

Encapsulates every tunable knob in a single immutable dataclass so callers
(scripts, notebooks, services) can serialise, log, and override settings
without touching positional arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreprocessorConfig:
    """Tunable parameters for :class:`app.pipeline.FloorPlanPreprocessor`.

    Attributes
    ----------
    confidence_threshold:
        Minimum EasyOCR detection confidence (0–1) for a region to be erased.
    dilation_px:
        Half-side of the square structuring element used to enlarge the text
        mask before erasure. ``0`` disables dilation.
    angle_step:
        Step (degrees) of the multi-angle OCR sweep. Must divide 360.
    crop_padding:
        Pixels of whitespace kept around the binary content bounding box.
    gaussian_ksize:
        Kernel side for the Gaussian blur preceding Otsu. Coerced to odd.
    languages:
        EasyOCR language codes. Defaults to English, French, Italian.
    gpu:
        Whether EasyOCR should use a CUDA device when available.
    """

    confidence_threshold: float = 0.1
    dilation_px: int = 15
    angle_step: int = 90
    crop_padding: int = 100
    gaussian_ksize: int = 5
    languages: tuple[str, ...] = field(default_factory=lambda: ("en", "fr", "it"))
    gpu: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.dilation_px < 0:
            raise ValueError("dilation_px must be >= 0")
        if self.angle_step <= 0 or 360 % self.angle_step != 0:
            raise ValueError("angle_step must be a positive divisor of 360")
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be >= 0")
        if self.gaussian_ksize < 1:
            raise ValueError("gaussian_ksize must be >= 1")
        if not self.languages:
            raise ValueError("languages must contain at least one code")
