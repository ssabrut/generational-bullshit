"""Floor plan preprocessing package.

Public API
----------
- :class:`FloorPlanPreprocessor` — end-to-end preprocessing pipeline.
- :class:`PipelineResult`        — structured pipeline output.
- :class:`PreprocessorConfig`    — typed configuration object.
"""

from app.config import PreprocessorConfig
from app.pipeline import FloorPlanPreprocessor
from app.types import PipelineResult

__all__ = [
    "FloorPlanPreprocessor",
    "PipelineResult",
    "PreprocessorConfig",
]
