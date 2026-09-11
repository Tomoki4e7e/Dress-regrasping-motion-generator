from __future__ import annotations

import math
from typing import Dict

import pytest
import torch
from torch import Tensor

from residual_flow.contracts import ACTION_DIM, EXTERNAL_TORQUE_DIM
from residual_flow.evaluation import (
    ResidualPrediction,
    evaluate_comparison,
    prediction_metrics,
    predict,
)
from residual_flow.models import (
    ConditionalFlowPolicy,
    DeterministicResidualPolicy,
    ForceAwareConditionalFlowPolicy,
    ModelConfig,
    TemporalConditionEncoder,
    build_comparison_models,
)
from residual_flow.training import (
    deterministic_loss,
    flow_matching_loss,
    sample_euler,
    train_comparison_epoch,
)


FEATURE_DIM = 7
HISTORY = 3
HORIZON = 2


def synthetic_batch(batch_size: int = 3) -> Dict[str, Tensor]:
    torch.manual_seed(7)
    return {
        "visual": torch.rand(batch_size, HISTORY, 2, 16, 16),
        "angles": torch.randn(batch_size, HISTORY, ACTION_DIM),
        "link_torque": torch.randn(batch_size, HISTORY, ACTION_DIM),
        "external_torque": torch.randn(
            batch_size, HISTORY, EXTERNAL_TORQUE_DIM
        ),
        "features": torch.randn(batch_size, HISTORY, FEATURE_DIM),
        "base_action": torch.randn(batch_size, HISTORY, ACTION_DIM),
        "residual_chunk": torch.randn(batch_size, HORIZON, ACTION_DIM),
        "future_external_torque": torch.randn(
            batch_size, HORIZON, EXTERNAL_TORQUE_DIM
        ),
        "valid": torch.tensor([True] * batch_size),
    }


def test_shared_encoder_and_deterministic_shapes_and_gradients() -> None:
    batch = synthetic_batch()
    encoder = TemporalConditionEncoder(
        FEATURE_DIM, condition_dim=12, visual_dim=8, frame_dim=12
    )
    assert encoder(batch).shape == (3, 12)

    model = DeterministicResidualPolicy(
        FEATURE_DIM, HORIZON, condition_dim=12, hidden_dim=16
    )
    output = model(batch)
    assert output.shape == (3, HORIZON, ACTION_DIM)
    loss = deterministic_loss(model, batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    "model_type,sample_width",
    [
        (ConditionalFlowPolicy, ACTION_DIM),
        (
            ForceAwareConditionalFlowPolicy,
            ACTION_DIM + EXTERNAL_TORQUE_DIM,
        ),
    ],
)
def test_flow_targets_shapes_and_backward(model_type: type, sample_width: int) -> None:
    batch = synthetic_batch()
    model = model_type(
        FEATURE_DIM, HORIZON, condition_dim=12, hidden_dim=20
    )
    target = model.training_target(batch)
    assert target.shape == (3, HORIZON, sample_width)
    state = torch.randn_like(target)
    velocity = model(state, torch.tensor([0.0, 0.5, 1.0]), model.condition(batch))
    assert velocity.shape == target.shape

    loss = flow_matching_loss(
        model,
        batch,
        time=torch.tensor([0.2, 0.5, 0.8]),
        noise=torch.zeros_like(target),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.velocity.network[-1].weight.grad is not None


def test_force_aware_sample_split_is_lossless() -> None:
    model = ForceAwareConditionalFlowPolicy(
        FEATURE_DIM, HORIZON, condition_dim=8, hidden_dim=12
    )
    sample = torch.randn(2, HORIZON, ACTION_DIM + EXTERNAL_TORQUE_DIM)
    residual, force = model.split_sample(sample)
    assert residual.shape == (2, HORIZON, ACTION_DIM)
    assert force.shape == (2, HORIZON, EXTERNAL_TORQUE_DIM)
    assert torch.equal(torch.cat((residual, force), dim=-1), sample)


def test_euler_sampler_integrates_constant_velocity() -> None:
    batch = synthetic_batch(batch_size=2)
    model = ConditionalFlowPolicy(
        FEATURE_DIM, HORIZON, condition_dim=8, hidden_dim=12
    )
    constant = 0.25

    def constant_velocity(state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        del time, condition
        return torch.full_like(state, constant)

    model.forward = constant_velocity  # type: ignore[assignment]
    initial = torch.zeros(2, HORIZON, ACTION_DIM)
    sampled = sample_euler(model, batch, steps=5, initial_noise=initial)
    assert torch.allclose(sampled, torch.full_like(sampled, constant))


def test_losses_ignore_invalid_samples() -> None:
    batch = synthetic_batch()
    batch["valid"] = torch.tensor([True, False, False])
    model = DeterministicResidualPolicy(
        FEATURE_DIM, HORIZON, condition_dim=8, hidden_dim=12
    )
    prediction = model(batch).detach()
    expected = (
        prediction[0] - batch["residual_chunk"][0]
    ).square().mean()
    assert torch.allclose(deterministic_loss(model, batch), expected)

    batch["valid"] = torch.zeros(3, dtype=torch.bool)
    loss = deterministic_loss(model, batch)
    assert loss.item() == 0.0
    loss.backward()


def test_metrics_use_valid_rows_and_report_force_only_when_present() -> None:
    batch = synthetic_batch()
    batch["valid"] = torch.tensor([True, False, True])
    residual = batch["residual_chunk"].clone()
    residual[[0, 2]] += 2.0
    force = batch["future_external_torque"].clone()
    force[[0, 2]] += 3.0
    metrics = prediction_metrics(ResidualPrediction(residual, force), batch)
    assert metrics.residual_mse == pytest.approx(4.0)
    assert metrics.residual_mae == pytest.approx(2.0)
    assert metrics.first_action_mse == pytest.approx(4.0)
    assert metrics.force_mse == pytest.approx(9.0)
    assert metrics.valid_samples == 2

    no_force = prediction_metrics(ResidualPrediction(residual), batch)
    assert no_force.force_mse is None


def test_build_train_predict_and_evaluate_comparison() -> None:
    batch = synthetic_batch(batch_size=2)
    models = build_comparison_models(
        ModelConfig(
            FEATURE_DIM, HORIZON, condition_dim=8, hidden_dim=12
        )
    )
    assert set(models) == {"deterministic", "action_cfm", "force_aware_cfm"}
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=1e-3)
        for name, model in models.items()
    }
    loader = [batch]
    train_results = train_comparison_epoch(
        models, loader, optimizers, device=torch.device("cpu")
    )
    assert all(result.valid_samples == 2 for result in train_results.values())
    assert all(math.isfinite(result.loss) for result in train_results.values())

    evaluation = evaluate_comparison(
        models, loader, device=torch.device("cpu"), flow_steps=2
    )
    assert all(metrics.valid_samples == 2 for metrics in evaluation.values())
    assert evaluation["deterministic"].force_mse is None
    assert evaluation["action_cfm"].force_mse is None
    assert evaluation["force_aware_cfm"].force_mse is not None
    assert predict(models["deterministic"], batch).residual.shape == (
        2,
        HORIZON,
        ACTION_DIM,
    )


def test_shape_validation_is_actionable() -> None:
    batch = synthetic_batch()
    batch["features"] = torch.randn(3, HISTORY, FEATURE_DIM + 1)
    model = DeterministicResidualPolicy(FEATURE_DIM, HORIZON)
    with pytest.raises(ValueError, match="features must have shape"):
        model(batch)
