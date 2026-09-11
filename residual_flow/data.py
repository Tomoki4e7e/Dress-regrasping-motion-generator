"""Read-only ShareSet audit and residual-window dataset construction."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Optional

import numpy as np
import yaml
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .contracts import (
    ACTION_DIM,
    EXTERNAL_TORQUE_DIM,
    IMAGE_SIZE,
    EpisodeSpec,
    ResidualSample,
    Split,
)

CSV_WIDTHS = {
    "angle.csv": ACTION_DIM,
    "torque.csv": ACTION_DIM,
    "external_torque.csv": EXTERNAL_TORQUE_DIM,
}
VISUAL_DIRS = {
    "rgb": Path("camera_right"),
    "sock_mask": Path("camera_right_mask/sock_mask"),
    "leg_mask": Path("camera_right_mask/leg_mask"),
    "camera_depth": Path("camera_depth"),
    "sock_depth": Path("depth_mask/sock_depth"),
    "leg_depth": Path("depth_mask/leg_depth"),
}
FEATURE_NAMES = (
    "coverage.npy",
    "opening_norm.npy",
    "heel_norm.npy",
    "toe_norm.npy",
    "foot_length.npy",
    "features_ok.npy",
)


@dataclass
class EpisodeAudit:
    episode: str
    split: str
    requested_window: tuple[int, int]
    expected_frames: int
    csv_rows: dict[str, Optional[int]] = field(default_factory=dict)
    csv_widths: dict[str, list[int]] = field(default_factory=dict)
    visual_frames: dict[str, int] = field(default_factory=dict)
    features_present: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable_for_residual(self) -> bool:
        required_visual = ("sock_depth", "leg_depth")
        return (
            not self.errors
            and all(self.visual_frames.get(k, 0) >= self.requested_window[1] for k in required_visual)
            and self.features_present
        )


def load_manifest(
    data_root: Path | str,
    manifest_path: Path | str,
    dataset_name: str,
) -> list[EpisodeSpec]:
    """Load the explicit sock manifest; never infer train/test from directories."""
    data_root = Path(data_root).expanduser().resolve()
    with Path(manifest_path).expanduser().open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if dataset_name not in payload:
        raise KeyError(f"{dataset_name!r} missing from {manifest_path}")

    result: list[EpisodeSpec] = []
    for split in ("train", "test"):
        for name, bounds in (payload[dataset_name].get(split) or {}).items():
            start, end = int(bounds["start"]), int(bounds["end"])
            if start < 0 or end <= start:
                raise ValueError(f"invalid frame window for {name}: {start}:{end}")
            result.append(
                EpisodeSpec(
                    name=name,
                    split=split,  # type: ignore[arg-type]
                    directory=data_root / dataset_name / split / name,
                    start=start,
                    end=end,
                )
            )
    return result


def _png_count(directory: Path) -> int:
    return sum(1 for path in directory.glob("*.png") if path.stem.isdigit())


def _csv_shape(path: Path) -> tuple[int, list[int]]:
    rows = 0
    widths: set[int] = set()
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            rows += 1
            widths.add(len(row))
    return rows, sorted(widths)


def audit_episode(spec: EpisodeSpec) -> EpisodeAudit:
    """Check alignment without reading or rewriting bulk image data."""
    report = EpisodeAudit(
        episode=spec.name,
        split=spec.split,
        requested_window=(spec.start, spec.end),
        expected_frames=spec.length,
    )
    if not spec.directory.is_dir():
        report.errors.append(f"episode directory missing: {spec.directory}")
        return report

    for filename, expected_width in CSV_WIDTHS.items():
        path = spec.directory / filename
        if not path.is_file():
            report.csv_rows[filename] = None
            report.csv_widths[filename] = []
            report.errors.append(f"missing {filename}")
            continue
        rows, widths = _csv_shape(path)
        report.csv_rows[filename] = rows
        report.csv_widths[filename] = widths
        if widths != [expected_width]:
            report.errors.append(
                f"{filename}: expected width {expected_width}, observed {widths}"
            )
        if rows < spec.end:
            report.errors.append(
                f"{filename}: {rows} rows do not cover requested end={spec.end}"
            )

    for name, relative in VISUAL_DIRS.items():
        count = _png_count(spec.directory / relative)
        report.visual_frames[name] = count
        if name in ("sock_depth", "leg_depth") and count < spec.end:
            report.errors.append(
                f"{relative}: {count} frames do not cover requested end={spec.end}"
            )

    feature_dir = spec.directory / "features"
    missing_features = [name for name in FEATURE_NAMES if not (feature_dir / name).is_file()]
    report.features_present = not missing_features
    if missing_features:
        report.warnings.append(f"features incomplete: {missing_features}")
    if not (spec.directory / "touch.csv").is_file():
        report.warnings.append("touch.csv absent; external_torque.csv is the tactile proxy")
    return report


def audit_dataset(specs: list[EpisodeSpec]) -> list[EpisodeAudit]:
    return [audit_episode(spec) for spec in specs]


def save_audit(reports: list[EpisodeAudit], output: Path | str) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    episodes = []
    for report in reports:
        payload = asdict(report)
        payload["usable_for_residual"] = report.usable_for_residual
        episodes.append(payload)
    summary = {
        "episodes": episodes,
        "n_episodes": len(reports),
        "n_usable": sum(report.usable_for_residual for report in reports),
        "n_errors": sum(len(report.errors) for report in reports),
        "n_warnings": sum(len(report.warnings) for report in reports),
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def load_csv_window(path: Path, start: int, end: int, width: int) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != width:
        raise ValueError(f"{path}: expected {width} columns, got {data.shape[1]}")
    if data.shape[0] < end:
        raise ValueError(f"{path}: expected at least {end} rows, got {data.shape[0]}")
    result = data[start:end]
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: non-finite values in requested window")
    return result


def _load_depth_sequence(directory: Path, start: int, end: int, size: int) -> np.ndarray:
    images: list[np.ndarray] = []
    for frame in range(start, end):
        path = directory / f"{frame}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = image.convert("L").filter(ImageFilter.GaussianBlur(radius=15))
            image = image.crop((0, 0, 1280, 960)).resize((size, size))
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(images)


def _load_feature_sequence(feature_dir: Path, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = {name: np.load(feature_dir / name) for name in FEATURE_NAMES}
    length = arrays["coverage.npy"].shape[0]
    # Feature packages may have been generated over the YAML slice or the full episode.
    sl = slice(0, end - start) if length == end - start else slice(start, end)
    coverage = arrays["coverage.npy"][sl, None]
    opening = arrays["opening_norm.npy"][sl].reshape(coverage.shape[0], -1)
    heel = arrays["heel_norm.npy"][sl].reshape(coverage.shape[0], -1)
    toe = arrays["toe_norm.npy"][sl].reshape(coverage.shape[0], -1)
    foot_length = arrays["foot_length.npy"][sl, None]
    valid = arrays["features_ok.npy"][sl].astype(bool)
    features = np.concatenate((coverage, opening, heel, toe, foot_length), axis=1).astype(np.float32)
    valid &= np.isfinite(features).all(axis=1)
    return np.nan_to_num(features, nan=0.0), valid


@dataclass
class EpisodeArrays:
    spec: EpisodeSpec
    visual: np.ndarray
    angles: np.ndarray
    link_torque: np.ndarray
    external_torque: np.ndarray
    features: np.ndarray
    features_ok: np.ndarray
    base_actions: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        """One-step teacher residual: demo action at t+1 minus base prediction at t."""
        return self.angles[1:] - self.base_actions[:-1]


def load_episode_arrays(
    spec: EpisodeSpec,
    base_prediction: Path | str,
    *,
    image_size: int = IMAGE_SIZE,
) -> EpisodeArrays:
    """Load one strict, aligned episode and externally generated base predictions."""
    angle = load_csv_window(spec.directory / "angle.csv", spec.start, spec.end, ACTION_DIM)
    link = load_csv_window(spec.directory / "torque.csv", spec.start, spec.end, ACTION_DIM)
    external = load_csv_window(
        spec.directory / "external_torque.csv", spec.start, spec.end, EXTERNAL_TORQUE_DIM
    )
    sock = _load_depth_sequence(
        spec.directory / VISUAL_DIRS["sock_depth"], spec.start, spec.end, image_size
    )
    leg = _load_depth_sequence(
        spec.directory / VISUAL_DIRS["leg_depth"], spec.start, spec.end, image_size
    )
    features, features_ok = _load_feature_sequence(
        spec.directory / "features", spec.start, spec.end
    )
    base = np.load(base_prediction).astype(np.float32)
    if base.shape != angle.shape:
        raise ValueError(
            f"{base_prediction}: base predictions {base.shape} != angles {angle.shape}"
        )
    visual = np.stack((sock, leg), axis=1)
    return EpisodeArrays(spec, visual, angle, link, external, features, features_ok, base)


class ResidualWindowDataset(Dataset):
    """Window episodes without crossing episode boundaries."""

    def __init__(self, episodes: list[EpisodeArrays], *, history: int = 10, horizon: int = 4):
        if history < 1 or horizon < 1:
            raise ValueError("history and horizon must be positive")
        self.episodes = episodes
        self.history = history
        self.horizon = horizon
        self._index: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(episodes):
            # Current t needs history ending at t and targets t..t+horizon-1;
            # residual[t] corresponds to demo[t+1] - base[t].
            last_t = len(episode.angles) - horizon - 1
            for t in range(history - 1, last_t + 1):
                self._index.append((episode_index, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Mapping[str, np.ndarray | bool | str | int]:
        episode_index, t = self._index[index]
        episode = self.episodes[episode_index]
        history_slice = slice(t - self.history + 1, t + 1)
        target_slice = slice(t, t + self.horizon)
        valid = bool(
            episode.features_ok[history_slice].all()
            and episode.features_ok[t + 1 : t + self.horizon + 1].all()
        )
        sample = ResidualSample(
            visual=episode.visual[history_slice],
            angles=episode.angles[history_slice],
            link_torque=episode.link_torque[history_slice],
            external_torque=episode.external_torque[history_slice],
            features=episode.features[history_slice],
            base_action=episode.base_actions[history_slice],
            residual_chunk=episode.residual[target_slice],
            future_external_torque=episode.external_torque[
                t + 1 : t + self.horizon + 1
            ],
            valid=valid,
            episode=episode.spec.name,
            frame_index=t + episode.spec.start,
        )
        sample.validate()
        return asdict(sample)


def prediction_path(root: Path | str, spec: EpisodeSpec) -> Path:
    return Path(root) / spec.split / f"{spec.name}.npy"


def iter_loadable_episodes(
    specs: list[EpisodeSpec],
    prediction_root: Path | str,
    *,
    split: Optional[Split] = None,
    image_size: int = IMAGE_SIZE,
) -> Iterator[EpisodeArrays]:
    for spec in specs:
        if split is not None and spec.split != split:
            continue
        yield load_episode_arrays(
            spec, prediction_path(prediction_root, spec), image_size=image_size
        )
