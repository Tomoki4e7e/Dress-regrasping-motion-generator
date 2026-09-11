"""Losses and lightweight training utilities for residual-flow policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn

from .models import ConditionalFlowPolicy, DeterministicResidualPolicy


TensorBatch = Mapping[str, Tensor]


def valid_mask(batch: TensorBatch, batch_size: int, device: torch.device) -> Tensor:
    """Return a boolean sample mask, defaulting to all valid."""
    value = batch.get("valid")
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    mask = torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)
    if mask.numel() != batch_size:
        raise ValueError("valid must contain one value per batch sample")
    return mask


def _masked_mean(per_sample: Tensor, mask: Tensor) -> Tensor:
    if per_sample.ndim != 1 or per_sample.shape != mask.shape:
        raise ValueError("per-sample loss and valid mask must be one-dimensional")
    if not bool(mask.any()):
        return per_sample.sum() * 0.0
    return per_sample[mask].mean()


def deterministic_loss(
    model: DeterministicResidualPolicy,
    batch: TensorBatch,
) -> Tensor:
    """Mean per-sample MSE on valid residual chunks."""
    target = batch["residual_chunk"].float()
    prediction = model(batch)
    if prediction.shape != target.shape:
        raise ValueError("prediction and residual_chunk shapes differ")
    per_sample = (prediction - target).square().flatten(1).mean(1)
    return _masked_mean(
        per_sample, valid_mask(batch, prediction.shape[0], prediction.device)
    )


def flow_matching_loss(
    model: ConditionalFlowPolicy,
    batch: TensorBatch,
    *,
    time: Optional[Tensor] = None,
    noise: Optional[Tensor] = None,
) -> Tensor:
    """Conditional flow-matching loss on a straight Gaussian-to-data path."""
    target = model.training_target(batch)
    batch_size = target.shape[0]
    if noise is None:
        noise = torch.randn_like(target)
    if noise.shape != target.shape:
        raise ValueError("noise and flow target shapes differ")
    if time is None:
        time = torch.rand(batch_size, device=target.device, dtype=target.dtype)
    time = time.to(device=target.device, dtype=target.dtype).reshape(-1)
    if time.shape != (batch_size,):
        raise ValueError("time must contain one value per batch sample")
    interpolation = time.reshape(batch_size, 1, 1)
    state = (1.0 - interpolation) * noise + interpolation * target
    desired_velocity = target - noise
    predicted_velocity = model(state, time, model.condition(batch))
    per_sample = (
        (predicted_velocity - desired_velocity).square().flatten(1).mean(1)
    )
    return _masked_mean(
        per_sample, valid_mask(batch, batch_size, target.device)
    )


@torch.no_grad()
def sample_euler(
    model: ConditionalFlowPolicy,
    batch: TensorBatch,
    *,
    steps: int = 16,
    initial_noise: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Integrate the learned ODE from Gaussian noise to a sample with Euler steps."""
    if steps < 1:
        raise ValueError("steps must be positive")
    condition = model.condition(batch)
    batch_size = condition.shape[0]
    shape = (batch_size, model.horizon, model.sample_width)
    if initial_noise is None:
        state = torch.randn(
            shape,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError("initial_noise has the wrong shape")
        state = initial_noise.to(device=condition.device, dtype=condition.dtype)
    dt = 1.0 / float(steps)
    for index in range(steps):
        time = torch.full(
            (batch_size,),
            index * dt,
            device=state.device,
            dtype=state.dtype,
        )
        state = state + dt * model(state, time, condition)
    return state


def move_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    """Move tensor values while preserving metadata strings and integers."""
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


@dataclass
class EpochResult:
    loss: float
    valid_samples: int
    batches: int


def train_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_grad_norm: Optional[float] = 1.0,
) -> EpochResult:
    """Train either supported model type for one loader pass."""
    model.train()
    weighted_loss = 0.0
    valid_samples = 0
    batches = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        tensor_batch = batch  # type: ignore[assignment]
        target = tensor_batch["residual_chunk"]
        if not isinstance(target, Tensor):
            raise TypeError("residual_chunk must be a tensor")
        mask = valid_mask(tensor_batch, target.shape[0], target.device)
        count = int(mask.sum().item())
        if count == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, DeterministicResidualPolicy):
            loss = deterministic_loss(model, tensor_batch)
        elif isinstance(model, ConditionalFlowPolicy):
            loss = flow_matching_loss(model, tensor_batch)
        else:
            raise TypeError("unsupported residual model type")
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        weighted_loss += float(loss.detach()) * count
        valid_samples += count
        batches += 1
    mean_loss = weighted_loss / valid_samples if valid_samples else float("nan")
    return EpochResult(mean_loss, valid_samples, batches)


def train_comparison_epoch(
    models: Mapping[str, nn.Module],
    loader: Iterable[Mapping[str, object]],
    optimizers: Mapping[str, torch.optim.Optimizer],
    *,
    device: torch.device,
    max_grad_norm: Optional[float] = 1.0,
) -> Mapping[str, EpochResult]:
    """Train named comparison models over the same re-iterable loader."""
    if set(models) != set(optimizers):
        raise ValueError("models and optimizers must have identical names")
    return {
        name: train_epoch(
            model,
            loader,
            optimizers[name],
            device=device,
            max_grad_norm=max_grad_norm,
        )
        for name, model in models.items()
    }
