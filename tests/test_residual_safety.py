import numpy as np
import torch

from residual_flow.models import DeterministicResidualPolicy
from residual_flow.normalization import ChannelStats, ResidualNormalizer
from residual_flow.runtime import ResidualController
from residual_flow.safety import (
    ForceAnomalyDetector,
    ResidualLimits,
    ResidualSafetyFilter,
    rank_candidates,
)
from residual_flow.validate_anomaly import evaluate


def test_safety_filter_bounds_rate_and_mimic():
    limits = ResidualLimits(
        arm_delta_rad=0.03,
        gripper_delta_m=0.002,
        max_delta_change_rad=0.01,
        max_gripper_change_m=0.001,
    )
    filt = ResidualSafetyFilter(limits)
    result = filt.apply(np.ones(18, dtype=np.float32))
    assert np.max(result[:7]) <= 0.01
    assert result[8] == 0.5 * result[7]
    assert result[17] == 0.5 * result[16]
    assert np.all(filt.apply(np.full(18, np.nan)) == 0)


def test_candidate_ranking_does_not_only_minimize_force():
    residuals = np.zeros((2, 3, 18), np.float32)
    force = np.zeros((2, 3, 18), np.float32)
    force[1] = 0.1
    progress = np.array([0.0, 2.0], np.float32)
    support = np.zeros(2, np.float32)
    selected, scores = rank_candidates(residuals, force, progress, support)
    assert selected == 1
    assert scores[1] < scores[0]


def test_force_anomaly_detector_and_replay_metrics():
    rng = np.random.default_rng(3)
    predicted = rng.normal(size=(400, 18)).astype(np.float32)
    observed = predicted + rng.normal(scale=0.02, size=predicted.shape)
    detector = ForceAnomalyDetector.fit(predicted[:200], observed[:200], quantile=0.99)
    nominal = detector.detect(predicted[200:], observed[200:])
    anomalous = observed[200:].copy()
    anomalous[50] += 2.0
    assert detector.detect(predicted[200:], anomalous)[50]
    assert np.mean(nominal) < 0.1

    metrics = evaluate(predicted, observed, magnitude=8.0, seed=4)
    assert metrics["synthetic_event"]["recall"] > 0.9
    assert "limitation" in metrics


def test_runtime_adapter_applies_hard_force_fallback():
    stats18 = ChannelStats(np.zeros(18, np.float32), np.ones(18, np.float32))
    normalizer = ResidualNormalizer(
        angles=stats18,
        link_torque=stats18,
        external_torque=stats18,
        features=ChannelStats(np.zeros(3, np.float32), np.ones(3, np.float32)),
        base_action=stats18,
        residual=stats18,
    )
    model = DeterministicResidualPolicy(3, horizon=2, condition_dim=8, hidden_dim=8)
    controller = ResidualController(
        model,
        normalizer,
        device=torch.device("cpu"),
        external_torque_norm_limit=1.0,
    )
    observation = {
        "visual": np.zeros((2, 2, 8, 8), np.float32),
        "angles": np.zeros((2, 18), np.float32),
        "link_torque": np.zeros((2, 18), np.float32),
        "external_torque": np.ones((2, 18), np.float32),
        "features": np.zeros((2, 3), np.float32),
        "base_action": np.zeros((2, 18), np.float32),
    }
    decision = controller.step(observation, np.zeros(18, np.float32))
    assert not decision.residual_enabled
    assert np.all(decision.applied_residual == 0)
    assert np.all(decision.combined_action == 0)
