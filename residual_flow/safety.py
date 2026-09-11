"""Model-independent residual limits, candidate selection and anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .contracts import ACTION_DIM, LEFT_GRIPPER, RIGHT_GRIPPER


@dataclass(frozen=True)
class ResidualLimits:
    """Physical per-cycle limits for the 5 Hz joint-position interface."""

    arm_delta_rad: float = 0.035
    gripper_delta_m: float = 0.0015
    max_delta_change_rad: float = 0.025
    max_gripper_change_m: float = 0.001

    def vector(self) -> np.ndarray:
        result = np.full(ACTION_DIM, self.arm_delta_rad, dtype=np.float32)
        result[list(LEFT_GRIPPER + RIGHT_GRIPPER)] = self.gripper_delta_m
        return result

    def change_vector(self) -> np.ndarray:
        result = np.full(ACTION_DIM, self.max_delta_change_rad, dtype=np.float32)
        result[list(LEFT_GRIPPER + RIGHT_GRIPPER)] = self.max_gripper_change_m
        return result


class ResidualSafetyFilter:
    """Bound residual magnitude/rate and preserve mimic-joint kinematics."""

    def __init__(self, limits: ResidualLimits = ResidualLimits()):
        self.limits = limits
        self.previous = np.zeros(ACTION_DIM, dtype=np.float32)

    @staticmethod
    def _enforce_mimic(delta: np.ndarray) -> np.ndarray:
        result = delta.copy()
        # Existing joint limits show mimic travel is half the finger travel.
        result[LEFT_GRIPPER[1]] = 0.5 * result[LEFT_GRIPPER[0]]
        result[RIGHT_GRIPPER[1]] = 0.5 * result[RIGHT_GRIPPER[0]]
        return result

    def apply(self, residual: np.ndarray, *, enabled: bool = True) -> np.ndarray:
        residual = np.asarray(residual, dtype=np.float32)
        if residual.shape != (ACTION_DIM,):
            raise ValueError(f"residual must be ({ACTION_DIM},), got {residual.shape}")
        if not enabled or not np.isfinite(residual).all():
            safe = np.zeros_like(residual)
        else:
            safe = np.clip(residual, -self.limits.vector(), self.limits.vector())
            change = np.clip(
                safe - self.previous,
                -self.limits.change_vector(),
                self.limits.change_vector(),
            )
            safe = self._enforce_mimic(self.previous + change)
        self.previous = safe
        return safe.copy()

    def reset(self) -> None:
        self.previous.fill(0.0)


@dataclass(frozen=True)
class CandidateWeights:
    force: float = 1.0
    base_deviation: float = 0.25
    jerk: float = 0.1
    progress: float = 1.0
    support: float = 2.0


def rank_candidates(
    residuals: np.ndarray,
    predicted_force: np.ndarray,
    predicted_progress: np.ndarray,
    support_distance: np.ndarray,
    *,
    previous_residual: Optional[np.ndarray] = None,
    weights: CandidateWeights = CandidateWeights(),
) -> tuple[int, np.ndarray]:
    """Rank generated chunks while penalizing the trivial stop solution."""
    residuals = np.asarray(residuals)
    predicted_force = np.asarray(predicted_force)
    if residuals.ndim != 3 or residuals.shape[2] != ACTION_DIM:
        raise ValueError("residuals must be (candidates,horizon,18)")
    if predicted_force.shape[:2] != residuals.shape[:2]:
        raise ValueError("force and residual chunks must share candidate/horizon axes")
    n = residuals.shape[0]
    for name, value in (
        ("predicted_progress", predicted_progress),
        ("support_distance", support_distance),
    ):
        if np.asarray(value).shape != (n,):
            raise ValueError(f"{name} must be ({n},)")

    force_cost = np.mean(np.square(predicted_force), axis=tuple(range(1, predicted_force.ndim)))
    deviation_cost = np.mean(np.square(residuals), axis=(1, 2))
    diffs = np.diff(residuals, axis=1)
    jerk_cost = np.mean(np.square(diffs), axis=(1, 2)) if diffs.shape[1] else np.zeros(n)
    if previous_residual is not None:
        jerk_cost += np.mean(
            np.square(residuals[:, 0] - np.asarray(previous_residual)[None, :]), axis=1
        )
    scores = (
        weights.force * force_cost
        + weights.base_deviation * deviation_cost
        + weights.jerk * jerk_cost
        - weights.progress * np.asarray(predicted_progress)
        + weights.support * np.asarray(support_distance)
    )
    return int(np.argmin(scores)), scores.astype(np.float32)


class ForceAnomalyDetector:
    """Calibrated one-class detector based on future-force prediction errors."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, threshold: float):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.maximum(np.asarray(scale, dtype=np.float32), 1e-6)
        self.threshold = float(threshold)

    @classmethod
    def fit(
        cls,
        predicted: np.ndarray,
        observed: np.ndarray,
        *,
        quantile: float = 0.995,
    ) -> "ForceAnomalyDetector":
        if not 0.5 < quantile < 1.0:
            raise ValueError("quantile must be between 0.5 and 1")
        errors = np.asarray(observed) - np.asarray(predicted)
        if errors.ndim != 2:
            raise ValueError("predicted and observed must be (samples,channels)")
        mean = np.mean(errors, axis=0)
        scale = np.std(errors, axis=0)
        scores = np.sqrt(np.mean(np.square((errors - mean) / np.maximum(scale, 1e-6)), axis=1))
        return cls(mean, scale, float(np.quantile(scores, quantile)))

    def score(self, predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
        errors = np.asarray(observed) - np.asarray(predicted)
        standardized = (errors - self.mean) / self.scale
        return np.sqrt(np.mean(np.square(standardized), axis=-1))

    def detect(self, predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
        return self.score(predicted, observed) > self.threshold
