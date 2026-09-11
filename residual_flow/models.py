"""Compact residual-action models for small dressing datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import torch
from torch import Tensor, nn

from .contracts import ACTION_DIM, EXTERNAL_TORQUE_DIM


def _require_sequence(name: str, value: Tensor, width: int) -> None:
    if value.ndim != 3 or value.shape[-1] != width:
        raise ValueError(
            "{} must have shape (batch,history,{}), got {}".format(
                name, width, tuple(value.shape)
            )
        )


class VisualEncoder(nn.Module):
    """Small per-frame CNN that accepts arbitrary practical image sizes."""

    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(24, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, visual: Tensor) -> Tensor:
        if visual.ndim != 5 or visual.shape[2] != 2:
            raise ValueError(
                "visual must have shape (batch,history,2,H,W), got {}".format(
                    tuple(visual.shape)
                )
            )
        batch, history = visual.shape[:2]
        encoded = self.network(visual.reshape(batch * history, *visual.shape[2:]))
        return encoded.reshape(batch, history, -1)


class TemporalConditionEncoder(nn.Module):
    """Encode visual, proprioceptive, force and base-policy histories."""

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int = 64,
        visual_dim: int = 32,
        frame_dim: int = 64,
    ) -> None:
        super().__init__()
        if feature_dim < 0:
            raise ValueError("feature_dim must be non-negative")
        self.feature_dim = feature_dim
        self.visual_encoder = VisualEncoder(visual_dim)
        scalar_width = ACTION_DIM * 4 + feature_dim
        self.frame_encoder = nn.Sequential(
            nn.Linear(visual_dim + scalar_width, frame_dim),
            nn.SiLU(),
            nn.LayerNorm(frame_dim),
        )
        self.temporal = nn.GRU(frame_dim, condition_dim, batch_first=True)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        visual = batch["visual"].float()
        histories = []
        for name, width in (
            ("angles", ACTION_DIM),
            ("link_torque", ACTION_DIM),
            ("external_torque", EXTERNAL_TORQUE_DIM),
            ("base_action", ACTION_DIM),
        ):
            value = batch[name].float()
            _require_sequence(name, value, width)
            histories.append(value)
        features = batch["features"].float()
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                "features must have shape (batch,history,{}), got {}".format(
                    self.feature_dim, tuple(features.shape)
                )
            )
        encoded_visual = self.visual_encoder(visual)
        history = encoded_visual.shape[1]
        if any(value.shape[:2] != encoded_visual.shape[:2] for value in histories):
            raise ValueError("all histories must share batch and history dimensions")
        if features.shape[:2] != encoded_visual.shape[:2]:
            raise ValueError("features must share batch and history dimensions")
        if history < 1:
            raise ValueError("history must be non-empty")
        frames = self.frame_encoder(
            torch.cat((encoded_visual, *histories, features), dim=-1)
        )
        _, hidden = self.temporal(frames)
        return hidden[-1]


class DeterministicResidualPolicy(nn.Module):
    """Direct regression baseline for a fixed residual-action chunk."""

    def __init__(
        self,
        feature_dim: int,
        horizon: int,
        condition_dim: int = 64,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.condition_encoder = TemporalConditionEncoder(feature_dim, condition_dim)
        self.head = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * ACTION_DIM),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        result = self.head(self.condition_encoder(batch))
        return result.reshape(result.shape[0], self.horizon, ACTION_DIM)


class ConditionalVelocityNetwork(nn.Module):
    """Time-conditioned vector field over a flattened output chunk."""

    def __init__(
        self,
        sample_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.sample_dim = sample_dim
        self.network = nn.Sequential(
            nn.Linear(sample_dim + condition_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, sample_dim),
        )

    def forward(self, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.sample_dim:
            raise ValueError("state has the wrong flattened sample dimension")
        if time.ndim == 1:
            time = time[:, None]
        if time.shape != (state.shape[0], 1):
            raise ValueError("time must have shape (batch,) or (batch,1)")
        if condition.ndim != 2 or condition.shape[0] != state.shape[0]:
            raise ValueError("condition must have shape (batch,condition_dim)")
        return self.network(torch.cat((state, time.to(state), condition), dim=-1))


class ConditionalFlowPolicy(nn.Module):
    """Conditional flow-matching model for residual action chunks."""

    target_dim = ACTION_DIM

    def __init__(
        self,
        feature_dim: int,
        horizon: int,
        condition_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.sample_width = self.target_dim
        self.sample_dim = horizon * self.sample_width
        self.condition_encoder = TemporalConditionEncoder(feature_dim, condition_dim)
        self.velocity = ConditionalVelocityNetwork(
            self.sample_dim, condition_dim, hidden_dim
        )

    def condition(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self.condition_encoder(batch)

    def forward(
        self, state: Tensor, time: Tensor, condition: Tensor
    ) -> Tensor:
        flat = state.reshape(state.shape[0], -1)
        velocity = self.velocity(flat, time, condition)
        return velocity.reshape(state.shape[0], self.horizon, self.sample_width)

    def training_target(self, batch: Mapping[str, Tensor]) -> Tensor:
        target = batch["residual_chunk"].float()
        expected = (self.horizon, ACTION_DIM)
        if target.ndim != 3 or tuple(target.shape[1:]) != expected:
            raise ValueError("residual_chunk must end in {}".format(expected))
        return target

    def split_sample(self, sample: Tensor) -> Tuple[Tensor, Tensor]:
        """Return residuals and an empty force tensor for a common API."""
        empty = sample.new_empty(sample.shape[0], self.horizon, 0)
        return sample, empty


class ForceAwareConditionalFlowPolicy(ConditionalFlowPolicy):
    """Joint CFM over residual actions and future external torque."""

    target_dim = ACTION_DIM + EXTERNAL_TORQUE_DIM

    def training_target(self, batch: Mapping[str, Tensor]) -> Tensor:
        residual = batch["residual_chunk"].float()
        force = batch["future_external_torque"].float()
        expected_residual = (self.horizon, ACTION_DIM)
        expected_force = (self.horizon, EXTERNAL_TORQUE_DIM)
        if residual.ndim != 3 or tuple(residual.shape[1:]) != expected_residual:
            raise ValueError("residual_chunk must end in {}".format(expected_residual))
        if force.ndim != 3 or tuple(force.shape[1:]) != expected_force:
            raise ValueError(
                "future_external_torque must end in {}".format(expected_force)
            )
        return torch.cat((residual, force), dim=-1)

    def split_sample(self, sample: Tensor) -> Tuple[Tensor, Tensor]:
        if sample.shape[-1] != self.sample_width:
            raise ValueError("joint sample has the wrong final dimension")
        return sample[..., :ACTION_DIM], sample[..., ACTION_DIM:]


@dataclass(frozen=True)
class ModelConfig:
    """Minimal dimensions needed to construct the three comparison models."""

    feature_dim: int
    horizon: int
    condition_dim: int = 64
    hidden_dim: int = 128


def build_comparison_models(config: ModelConfig) -> Mapping[str, nn.Module]:
    """Build baseline, action CFM and force-aware CFM with matched encoders."""
    return {
        "deterministic": DeterministicResidualPolicy(
            config.feature_dim,
            config.horizon,
            config.condition_dim,
            config.hidden_dim,
        ),
        "action_cfm": ConditionalFlowPolicy(
            config.feature_dim,
            config.horizon,
            config.condition_dim,
            config.hidden_dim,
        ),
        "force_aware_cfm": ForceAwareConditionalFlowPolicy(
            config.feature_dim,
            config.horizon,
            config.condition_dim,
            config.hidden_dim,
        ),
    }
