from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from residual_flow.data import (
    ResidualWindowDataset,
    audit_dataset,
    load_episode_arrays,
    load_manifest,
)
from residual_flow.normalization import NormalizedDataset, ResidualNormalizer


def _write_episode(root: Path, length: int = 8) -> None:
    for relative in (
        "camera_right",
        "camera_right_mask/sock_mask",
        "camera_right_mask/leg_mask",
        "camera_depth",
        "depth_mask/sock_depth",
        "depth_mask/leg_depth",
    ):
        directory = root / relative
        directory.mkdir(parents=True)
        for frame in range(length):
            channels = 3 if relative == "camera_right" else 1
            shape = (12, 16, channels) if channels == 3 else (12, 16)
            Image.fromarray(np.full(shape, frame, dtype=np.uint8)).save(
                directory / f"{frame}.png"
            )
    values = np.arange(length * 18, dtype=np.float32).reshape(length, 18) / 100
    for filename, offset in (
        ("angle.csv", 0.0),
        ("torque.csv", 1.0),
        ("external_torque.csv", 2.0),
    ):
        np.savetxt(root / filename, values + offset, delimiter=",")
    features = root / "features"
    features.mkdir()
    np.save(features / "coverage.npy", np.linspace(0, 1, length, dtype=np.float32))
    np.save(features / "opening_norm.npy", np.zeros((length, 8, 2), np.float32))
    np.save(features / "heel_norm.npy", np.zeros((length, 2), np.float32))
    np.save(features / "toe_norm.npy", np.zeros((length, 2), np.float32))
    np.save(features / "foot_length.npy", np.ones(length, np.float32))
    np.save(features / "features_ok.npy", np.ones(length, bool))


def _manifest(tmp_path: Path, episode: Path, length: int = 8):
    data_root = tmp_path / "data"
    target = data_root / "sock" / "train" / episode.name
    target.parent.mkdir(parents=True)
    episode.rename(target)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {"sock": {"train": {target.name: {"start": 0, "end": length}}, "test": {}}}
        )
    )
    return data_root, manifest, target


def test_audit_and_window_dataset(tmp_path):
    episode = tmp_path / "episode"
    _write_episode(episode)
    data_root, manifest, target = _manifest(tmp_path, episode)
    specs = load_manifest(data_root, manifest, "sock")
    reports = audit_dataset(specs)
    assert len(reports) == 1
    assert reports[0].usable_for_residual

    prediction = tmp_path / "base.npy"
    base = np.loadtxt(target / "angle.csv", delimiter=",").astype(np.float32)
    np.save(prediction, base - 0.25)
    arrays = load_episode_arrays(specs[0], prediction, image_size=8)
    dataset = ResidualWindowDataset([arrays], history=3, horizon=2)
    assert len(dataset) == 4
    item = dataset[0]
    assert item["visual"].shape == (3, 2, 8, 8)
    assert item["residual_chunk"].shape == (2, 18)
    assert np.allclose(item["residual_chunk"], 0.25 + (18 / 100))
    assert item["valid"]


def test_audit_reports_missing_leg_depth(tmp_path):
    episode = tmp_path / "episode"
    _write_episode(episode)
    data_root, manifest, target = _manifest(tmp_path, episode)
    for path in (target / "depth_mask/leg_depth").glob("*.png"):
        path.unlink()
    report = audit_dataset(load_manifest(data_root, manifest, "sock"))[0]
    assert not report.usable_for_residual
    assert any("leg_depth" in error for error in report.errors)


def test_base_prediction_shape_is_strict(tmp_path):
    episode = tmp_path / "episode"
    _write_episode(episode)
    data_root, manifest, _ = _manifest(tmp_path, episode)
    prediction = tmp_path / "bad.npy"
    np.save(prediction, np.zeros((8, 17), np.float32))
    with pytest.raises(ValueError, match="base predictions"):
        load_episode_arrays(load_manifest(data_root, manifest, "sock")[0], prediction)


def test_normalizer_uses_separate_force_and_residual_scales(tmp_path):
    episode = tmp_path / "episode"
    _write_episode(episode)
    data_root, manifest, target = _manifest(tmp_path, episode)
    prediction = tmp_path / "base.npy"
    angles = np.loadtxt(target / "angle.csv", delimiter=",").astype(np.float32)
    np.save(prediction, angles - 0.1)
    arrays = load_episode_arrays(
        load_manifest(data_root, manifest, "sock")[0], prediction, image_size=8
    )
    normalizer = ResidualNormalizer.fit([arrays])
    raw = ResidualWindowDataset([arrays], history=2, horizon=2)
    normalized = NormalizedDataset(raw, normalizer)[0]
    assert np.max(np.abs(normalized["future_external_torque"])) < 3
    assert np.max(np.abs(normalized["residual_chunk"])) < 3
    path = normalizer.save(tmp_path / "normalization.json")
    loaded = ResidualNormalizer.load(path)
    assert np.allclose(loaded.residual.mean, normalizer.residual.mean)
