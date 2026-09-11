"""Train and compare deterministic, action-CFM and force-aware CFM residuals."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ResidualWindowDataset, iter_loadable_episodes, load_manifest
from .evaluation import evaluate_comparison
from .models import ModelConfig, build_comparison_models
from .normalization import NormalizedDataset, ResidualNormalizer
from .training import train_epoch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", default="data_center_general_depth_n5_sock")
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _zero_residual_metrics(loader: DataLoader) -> dict[str, float | int]:
    squared = absolute = first_squared = 0.0
    values = first_values = samples = 0
    for batch in loader:
        target = batch["residual_chunk"].float()
        mask = batch["valid"].bool()
        if not bool(mask.any()):
            continue
        selected = target[mask]
        squared += float(selected.square().sum())
        absolute += float(selected.abs().sum())
        first_squared += float(selected[:, 0].square().sum())
        values += selected.numel()
        first_values += selected[:, 0].numel()
        samples += int(mask.sum())
    return {
        "residual_mse": squared / max(values, 1),
        "residual_mae": absolute / max(values, 1),
        "first_action_mse": first_squared / max(first_values, 1),
        "force_mse": None,
        "valid_samples": samples,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    specs = load_manifest(args.data_root, args.manifest, args.dataset_name)
    train_episodes = list(
        iter_loadable_episodes(
            specs, args.prediction_root, split="train", image_size=args.image_size
        )
    )
    test_episodes = list(
        iter_loadable_episodes(
            specs, args.prediction_root, split="test", image_size=args.image_size
        )
    )
    normalizer = ResidualNormalizer.fit(train_episodes)
    train = NormalizedDataset(
        ResidualWindowDataset(
            train_episodes, history=args.history, horizon=args.horizon
        ),
        normalizer,
    )
    test = NormalizedDataset(
        ResidualWindowDataset(
            test_episodes, history=args.history, horizon=args.horizon
        ),
        normalizer,
    )
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    feature_dim = train_episodes[0].features.shape[1]
    models = build_comparison_models(
        ModelConfig(
            feature_dim=feature_dim,
            horizon=args.horizon,
            condition_dim=args.condition_dim,
            hidden_dim=args.hidden_dim,
        )
    )
    device = torch.device(args.device)
    models = {name: model.to(device) for name, model in models.items()}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        for name, model in models.items()
    }
    history: dict[str, list[dict[str, float | int]]] = {
        name: [] for name in models
    }
    for epoch in range(args.epochs):
        for name, model in models.items():
            result = train_epoch(
                model, train_loader, optimizers[name], device=device
            )
            history[name].append(asdict(result))
        print(
            "epoch={} {}".format(
                epoch + 1,
                " ".join(
                    f"{name}={history[name][-1]['loss']:.6f}" for name in models
                ),
            )
        )

    metrics = {
        name: asdict(result)
        for name, result in evaluate_comparison(
            models, test_loader, device=device, flow_steps=args.flow_steps
        ).items()
    }
    metrics["zero_residual"] = _zero_residual_metrics(test_loader)
    args.output.mkdir(parents=True, exist_ok=True)
    normalizer.save(args.output / "normalization.json")
    for name, model in models.items():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(
                    ModelConfig(
                        feature_dim=feature_dim,
                        horizon=args.horizon,
                        condition_dim=args.condition_dim,
                        hidden_dim=args.hidden_dim,
                    )
                ),
                "model_name": name,
            },
            args.output / f"{name}.pth",
        )
    report = {
        "metrics_normalized": metrics,
        "history": history,
        "train_episodes": [episode.spec.name for episode in train_episodes],
        "test_episodes": [episode.spec.name for episode in test_episodes],
        "parameters": {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        },
        "limitations": [
            "Metrics are on demonstrated state support and do not prove closed-loop force reduction.",
            "Successful-only data supports one-class departure detection, not snag/slip classification.",
        ],
    }
    # Python 3.8 cannot serialize Paths or use dict union at runtime.
    report["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output / "comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
