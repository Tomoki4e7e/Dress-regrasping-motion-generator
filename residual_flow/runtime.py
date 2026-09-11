"""Pure-Python runtime adapter placed between the frozen base and ROS commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch

from .evaluation import predict
from .models import ConditionalFlowPolicy, DeterministicResidualPolicy
from .normalization import ResidualNormalizer
from .safety import (
    ForceAnomalyDetector,
    ResidualSafetyFilter,
    rank_candidates,
)


@dataclass
class ResidualDecision:
    combined_action: np.ndarray
    applied_residual: np.ndarray
    predicted_external_torque: Optional[np.ndarray]
    anomaly_score: Optional[float]
    anomaly_detected: bool
    residual_enabled: bool
    candidate_scores: np.ndarray


class ResidualController:
    """Generate bounded corrections without modifying the base policy itself."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalizer: ResidualNormalizer,
        *,
        device: torch.device = torch.device("cpu"),
        flow_steps: int = 8,
        candidates: int = 4,
        anomaly_detector: Optional[ForceAnomalyDetector] = None,
        safety_filter: Optional[ResidualSafetyFilter] = None,
        external_torque_norm_limit: Optional[float] = None,
    ):
        if candidates < 1:
            raise ValueError("candidates must be positive")
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.device = device
        self.flow_steps = flow_steps
        self.candidates = candidates if isinstance(model, ConditionalFlowPolicy) else 1
        self.anomaly_detector = anomaly_detector
        self.safety_filter = safety_filter or ResidualSafetyFilter()
        self.external_torque_norm_limit = external_torque_norm_limit
        self._previous_force_prediction: Optional[np.ndarray] = None

    def _tensor_batch(self, observation: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
        normalized = self.normalizer.transform_observation(observation)
        result = {}
        for name in (
            "visual",
            "angles",
            "link_torque",
            "external_torque",
            "features",
            "base_action",
        ):
            value = torch.as_tensor(normalized[name], dtype=torch.float32)
            if value.ndim < 2:
                raise ValueError(f"{name} must include a history axis")
            value = value.unsqueeze(0)
            repeats = (self.candidates,) + (1,) * (value.ndim - 1)
            result[name] = value.repeat(repeats).to(self.device)
        return result

    def step(
        self,
        observation: Mapping[str, np.ndarray],
        base_action: np.ndarray,
    ) -> ResidualDecision:
        base_action = np.asarray(base_action, dtype=np.float32)
        if base_action.shape != (18,):
            raise ValueError("base_action must be the physical 18-D command")
        current_external = np.asarray(observation["external_torque"])[-1]
        anomaly_score: Optional[float] = None
        anomaly = False
        if self.anomaly_detector is not None and self._previous_force_prediction is not None:
            anomaly_score = float(
                self.anomaly_detector.score(
                    self._previous_force_prediction[None, :],
                    current_external[None, :],
                )[0]
            )
            anomaly = anomaly_score > self.anomaly_detector.threshold

        hard_force = (
            self.external_torque_norm_limit is not None
            and float(np.linalg.norm(current_external)) > self.external_torque_norm_limit
        )
        tensor_batch = self._tensor_batch(observation)
        with torch.no_grad():
            output = predict(self.model, tensor_batch, flow_steps=self.flow_steps)
        residuals = self.normalizer.residual.denormalize(
            output.residual.detach().cpu().numpy()
        )
        if output.future_external_torque is None:
            forces = np.zeros_like(residuals)
        else:
            forces = self.normalizer.external_torque.denormalize(
                output.future_external_torque.detach().cpu().numpy()
            )
        selected, scores = rank_candidates(
            residuals,
            forces,
            predicted_progress=np.zeros(self.candidates, dtype=np.float32),
            support_distance=np.mean(np.square(residuals), axis=(1, 2)),
            previous_residual=self.safety_filter.previous,
        )
        enabled = not hard_force
        safe_residual = self.safety_filter.apply(residuals[selected, 0], enabled=enabled)
        predicted_force = None
        if output.future_external_torque is not None:
            predicted_force = forces[selected, 0].copy()
            self._previous_force_prediction = predicted_force
        return ResidualDecision(
            combined_action=base_action + safe_residual,
            applied_residual=safe_residual,
            predicted_external_torque=predicted_force,
            anomaly_score=anomaly_score,
            anomaly_detected=anomaly,
            residual_enabled=enabled,
            candidate_scores=scores,
        )

    def reset(self) -> None:
        self.safety_filter.reset()
        self._previous_force_prediction = None
