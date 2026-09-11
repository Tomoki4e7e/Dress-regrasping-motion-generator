"""Strict data contracts shared by residual-flow training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

ACTION_DIM = 18
EXTERNAL_TORQUE_DIM = 18
LINK_TORQUE_DIM = 18
IMAGE_SIZE = 64

Split = Literal["train", "test"]

# [left arm 7, left gripper finger+mimic, right arm 7, right gripper finger+mimic]
LEFT_GRIPPER = (7, 8)
RIGHT_GRIPPER = (16, 17)
GRIPPER_INDICES = LEFT_GRIPPER + RIGHT_GRIPPER


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode and its YAML-style half-open frame window."""

    name: str
    split: Split
    directory: Path
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class ResidualSample:
    """One training window.

    Histories are oldest-to-newest. Chunks start at the current timestep and
    contain future residual actions / external torques.
    """

    visual: np.ndarray  # (history, 2, H, W): sock depth, leg depth
    angles: np.ndarray  # (history, 18)
    link_torque: np.ndarray  # (history, 18)
    external_torque: np.ndarray  # (history, 18)
    features: np.ndarray  # (history, feature_dim)
    base_action: np.ndarray  # (history, 18)
    residual_chunk: np.ndarray  # (horizon, 18)
    future_external_torque: np.ndarray  # (horizon, 18)
    valid: bool
    episode: str
    frame_index: int

    def validate(self) -> None:
        """Raise on a schema mismatch before tensors reach a model."""
        history = self.angles.shape[0]
        if self.visual.ndim != 4 or self.visual.shape[:2] != (history, 2):
            raise ValueError(f"visual must be (history,2,H,W), got {self.visual.shape}")
        for name, value, dim in (
            ("angles", self.angles, ACTION_DIM),
            ("link_torque", self.link_torque, LINK_TORQUE_DIM),
            ("external_torque", self.external_torque, EXTERNAL_TORQUE_DIM),
            ("base_action", self.base_action, ACTION_DIM),
        ):
            if value.shape != (history, dim):
                raise ValueError(f"{name} must be {(history, dim)}, got {value.shape}")
        if self.features.ndim != 2 or self.features.shape[0] != history:
            raise ValueError("features must share the history axis")
        horizon = self.residual_chunk.shape[0]
        if self.residual_chunk.shape != (horizon, ACTION_DIM):
            raise ValueError("invalid residual chunk")
        if self.future_external_torque.shape != (horizon, EXTERNAL_TORQUE_DIM):
            raise ValueError("future force chunk must match residual horizon")
