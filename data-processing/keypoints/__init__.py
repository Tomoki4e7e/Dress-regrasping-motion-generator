"""Keypoints from sock/leg masks (Step 3).

Opening rim (6–8 points), heel/toe along sock∪leg PCA axis, depth from
camera_depth, and toe-origin / foot-length normalization.
"""

from .compute import (
    DEFAULT_N_OPENING,
    FrameKeypoints,
    KeypointsResult,
    compute_episode_keypoints,
    estimate_frame_keypoints,
    save_episode_keypoints,
)

__all__ = [
    "DEFAULT_N_OPENING",
    "FrameKeypoints",
    "KeypointsResult",
    "compute_episode_keypoints",
    "estimate_frame_keypoints",
    "save_episode_keypoints",
]
