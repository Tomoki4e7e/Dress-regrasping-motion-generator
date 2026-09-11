"""Comparable prediction and evaluation utilities for residual policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn

from .models import ConditionalFlowPolicy, DeterministicResidualPolicy
from .training import move_batch, sample_euler, valid_mask


@dataclass
class ResidualPrediction:
    residual: Tensor
    future_external_torque: Optional[Tensor] = None


@dataclass
class EvaluationMetrics:
    residual_mse: float
    residual_mae: float
    first_action_mse: float
    force_mse: Optional[float]
    valid_samples: int


@torch.no_grad()
def predict(
    model: nn.Module,
    batch: Mapping[str, Tensor],
    *,
    flow_steps: int = 16,
    initial_noise: Optional[Tensor] = None,
) -> ResidualPrediction:
    """Predict through a deterministic policy or sample through a flow policy."""
    model.eval()
    if isinstance(model, DeterministicResidualPolicy):
        return ResidualPrediction(model(batch))
    if isinstance(model, ConditionalFlowPolicy):
        sample = sample_euler(
            model, batch, steps=flow_steps, initial_noise=initial_noise
        )
        residual, force = model.split_sample(sample)
        return ResidualPrediction(
            residual,
            force if force.shape[-1] else None,
        )
    raise TypeError("unsupported residual model type")


def prediction_metrics(
    prediction: ResidualPrediction,
    batch: Mapping[str, Tensor],
) -> EvaluationMetrics:
    """Compute action and optional force errors while excluding invalid windows."""
    target = batch["residual_chunk"].float()
    if prediction.residual.shape != target.shape:
        raise ValueError("predicted and target residual shapes differ")
    mask = valid_mask(batch, target.shape[0], target.device)
    count = int(mask.sum().item())
    if count == 0:
        return EvaluationMetrics(
            float("nan"), float("nan"), float("nan"), None, 0
        )
    error = prediction.residual[mask] - target[mask]
    force_mse: Optional[float] = None
    if prediction.future_external_torque is not None:
        force_target = batch["future_external_torque"].float()
        if prediction.future_external_torque.shape != force_target.shape:
            raise ValueError("predicted and target force shapes differ")
        force_error = prediction.future_external_torque[mask] - force_target[mask]
        force_mse = float(force_error.square().mean())
    return EvaluationMetrics(
        residual_mse=float(error.square().mean()),
        residual_mae=float(error.abs().mean()),
        first_action_mse=float(error[:, 0].square().mean()),
        force_mse=force_mse,
        valid_samples=count,
    )


@dataclass
class _MetricTotals:
    residual_squared: float = 0.0
    residual_absolute: float = 0.0
    residual_values: int = 0
    first_squared: float = 0.0
    first_values: int = 0
    force_squared: float = 0.0
    force_values: int = 0
    samples: int = 0


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    flow_steps: int = 16,
) -> EvaluationMetrics:
    """Evaluate a model with element-weighted aggregate metrics."""
    model.eval()
    totals = _MetricTotals()
    predicts_force = False
    for raw_batch in loader:
        moved = move_batch(raw_batch, device)
        batch = moved  # type: ignore[assignment]
        target = batch["residual_chunk"]
        if not isinstance(target, Tensor):
            raise TypeError("residual_chunk must be a tensor")
        output = predict(model, batch, flow_steps=flow_steps)
        mask = valid_mask(batch, target.shape[0], target.device)
        if not bool(mask.any()):
            continue
        error = output.residual[mask] - target.float()[mask]
        first_error = error[:, 0]
        totals.residual_squared += float(error.square().sum())
        totals.residual_absolute += float(error.abs().sum())
        totals.residual_values += error.numel()
        totals.first_squared += float(first_error.square().sum())
        totals.first_values += first_error.numel()
        totals.samples += int(mask.sum().item())
        if output.future_external_torque is not None:
            force_target = batch["future_external_torque"]
            if not isinstance(force_target, Tensor):
                raise TypeError("future_external_torque must be a tensor")
            force_error = output.future_external_torque[mask] - force_target.float()[mask]
            totals.force_squared += float(force_error.square().sum())
            totals.force_values += force_error.numel()
            predicts_force = True
    if totals.samples == 0:
        return EvaluationMetrics(
            float("nan"), float("nan"), float("nan"), None, 0
        )
    force_mse = (
        totals.force_squared / totals.force_values
        if predicts_force and totals.force_values
        else None
    )
    return EvaluationMetrics(
        totals.residual_squared / totals.residual_values,
        totals.residual_absolute / totals.residual_values,
        totals.first_squared / totals.first_values,
        force_mse,
        totals.samples,
    )


def evaluate_comparison(
    models: Mapping[str, nn.Module],
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    flow_steps: int = 16,
) -> Dict[str, EvaluationMetrics]:
    """Evaluate named models with identical loader and integration settings."""
    return {
        name: evaluate_model(
            model, loader, device=device, flow_steps=flow_steps
        )
        for name, model in models.items()
    }
