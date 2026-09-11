"""Replay validation for nominal-force anomaly detection.

This validates detector mechanics with held-out nominal traces and controlled
signal perturbations. It is not a substitute for labelled physical snag/slip
trials or for closed-loop force-reduction tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .safety import ForceAnomalyDetector


def inject_force_events(
    observed: np.ndarray,
    *,
    fraction: float = 0.1,
    magnitude: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject channel-coherent spikes and drops for a detector smoke test."""
    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must be between 0 and 0.5")
    rng = np.random.default_rng(seed)
    changed = np.asarray(observed, dtype=np.float32).copy()
    labels = np.zeros(changed.shape[0], dtype=bool)
    count = max(1, int(round(changed.shape[0] * fraction)))
    indices = rng.choice(changed.shape[0], size=count, replace=False)
    scale = np.std(changed, axis=0, keepdims=True)
    signs = rng.choice((-1.0, 1.0), size=(count, 1))
    changed[indices] += signs * magnitude * np.maximum(scale, 1e-3)
    labels[indices] = True
    return changed, labels


def binary_metrics(labels: np.ndarray, detected: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    detected = np.asarray(detected, dtype=bool)
    tp = int(np.sum(labels & detected))
    fp = int(np.sum(~labels & detected))
    tn = int(np.sum(~labels & ~detected))
    fn = int(np.sum(labels & ~detected))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def evaluate(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    calibration_fraction: float = 0.5,
    event_fraction: float = 0.1,
    magnitude: float = 5.0,
    seed: int = 0,
) -> dict[str, object]:
    n_calibration = int(len(predicted) * calibration_fraction)
    if n_calibration < 10 or len(predicted) - n_calibration < 10:
        raise ValueError("need at least 10 calibration and 10 evaluation samples")
    detector = ForceAnomalyDetector.fit(
        predicted[:n_calibration], observed[:n_calibration]
    )
    perturbed, labels = inject_force_events(
        observed[n_calibration:],
        fraction=event_fraction,
        magnitude=magnitude,
        seed=seed,
    )
    detected = detector.detect(predicted[n_calibration:], perturbed)
    nominal_detected = detector.detect(
        predicted[n_calibration:], observed[n_calibration:]
    )
    return {
        "threshold": detector.threshold,
        "synthetic_event": binary_metrics(labels, detected),
        "heldout_nominal_false_positive_rate": float(np.mean(nominal_detected)),
        "n_calibration": n_calibration,
        "n_evaluation": len(predicted) - n_calibration,
        "limitation": (
            "Synthetic signal injection validates the detector implementation only; "
            "labelled robot trials are required for snag/slip claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with predicted and observed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-fraction", type=float, default=0.1)
    parser.add_argument("--magnitude", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = np.load(args.input)
    result = evaluate(
        payload["predicted"],
        payload["observed"],
        event_fraction=args.event_fraction,
        magnitude=args.magnitude,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
