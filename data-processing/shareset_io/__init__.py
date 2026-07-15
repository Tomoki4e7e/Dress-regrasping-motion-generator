"""Thin I/O for ShareSet episode mask / depth outputs (Step 1).

ShareSet generation logic is never modified. Downstream coverage and
keypoint code should read frames only through this package.
"""

from .paths import (
    CROP_HEIGHT,
    CROP_WIDTH,
    MASK_FOREGROUND,
    EpisodePaths,
)
from .reader import EpisodeReader, FrameBundle, apply_mask_to_depth, list_frame_files

__all__ = [
    "CROP_HEIGHT",
    "CROP_WIDTH",
    "MASK_FOREGROUND",
    "EpisodePaths",
    "EpisodeReader",
    "FrameBundle",
    "apply_mask_to_depth",
    "list_frame_files",
]
