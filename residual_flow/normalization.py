"""Train-split-only normalization for heterogeneous residual policy signals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from torch.utils.data import Dataset

from .data import EpisodeArrays


@dataclass(frozen=True)
class ChannelStats:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, epsilon: float = 1e-6) -> "ChannelStats":
        flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
        mean = np.mean(flat, axis=0)
        scale = np.std(flat, axis=0)
        return cls(mean.astype(np.float32), np.maximum(scale, epsilon).astype(np.float32))

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.mean) / self.scale).astype(np.float32)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        return (value * self.scale + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, list[float]]) -> "ChannelStats":
        return cls(
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["scale"], dtype=np.float32),
        )


@dataclass(frozen=True)
class ResidualNormalizer:
    """Separate scales prevent force channels from dominating action CFM."""

    angles: ChannelStats
    link_torque: ChannelStats
    external_torque: ChannelStats
    features: ChannelStats
    base_action: ChannelStats
    residual: ChannelStats

    @classmethod
    def fit(cls, episodes: list[EpisodeArrays]) -> "ResidualNormalizer":
        if not episodes:
            raise ValueError("cannot fit normalization without train episodes")

        def concatenate(name: str) -> np.ndarray:
            return np.concatenate([getattr(episode, name) for episode in episodes], axis=0)

        return cls(
            angles=ChannelStats.fit(concatenate("angles")),
            link_torque=ChannelStats.fit(concatenate("link_torque")),
            external_torque=ChannelStats.fit(concatenate("external_torque")),
            features=ChannelStats.fit(concatenate("features")),
            base_action=ChannelStats.fit(concatenate("base_actions")),
            residual=ChannelStats.fit(concatenate("residual")),
        )

    def transform_observation(
        self, sample: Mapping[str, object]
    ) -> dict[str, object]:
        result = dict(sample)
        result["angles"] = self.angles.normalize(np.asarray(sample["angles"]))
        result["link_torque"] = self.link_torque.normalize(
            np.asarray(sample["link_torque"])
        )
        result["external_torque"] = self.external_torque.normalize(
            np.asarray(sample["external_torque"])
        )
        result["features"] = self.features.normalize(np.asarray(sample["features"]))
        result["base_action"] = self.base_action.normalize(
            np.asarray(sample["base_action"])
        )
        return result

    def transform(self, sample: Mapping[str, object]) -> dict[str, object]:
        result = self.transform_observation(sample)
        result["residual_chunk"] = self.residual.normalize(
            np.asarray(sample["residual_chunk"])
        )
        result["future_external_torque"] = self.external_torque.normalize(
            np.asarray(sample["future_external_torque"])
        )
        return result

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: getattr(self, name).to_dict()
            for name in (
                "angles",
                "link_torque",
                "external_torque",
                "features",
                "base_action",
                "residual",
            )
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ResidualNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            **{name: ChannelStats.from_dict(stats) for name, stats in payload.items()}
        )


class NormalizedDataset(Dataset):
    def __init__(self, dataset: Dataset, normalizer: ResidualNormalizer):
        self.dataset = dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.normalizer.transform(self.dataset[index])
