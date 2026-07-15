"""Step 4: package coverage + keypoints for learning / reward."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from coverage.compute import (
    BaselineMode,
    CoverageResult,
    compute_episode_coverage,
    save_episode_coverage,
)
from keypoints.compute import (
    DEFAULT_N_OPENING,
    KeypointsResult,
    compute_episode_keypoints,
    save_episode_keypoints,
)
from shareset_io import EpisodeReader

# Arrays expected under features/ after a full package write.
REQUIRED_NPY = (
    "coverage.npy",
    "leg_area.npy",
    "coverage_ok.npy",
    "opening_xy.npy",
    "opening_depth.npy",
    "opening_norm.npy",
    "heel_xy.npy",
    "toe_xy.npy",
    "heel_depth.npy",
    "toe_depth.npy",
    "heel_norm.npy",
    "toe_norm.npy",
    "foot_length.npy",
    "keypoints_ok.npy",
    "features_ok.npy",
)


@dataclass
class FeatureBundle:
    """In-memory package: coverage + keypoints + combined ok flags."""

    coverage: CoverageResult
    keypoints: KeypointsResult
    coverage_ok: np.ndarray  # (T,) bool
    keypoints_ok: np.ndarray  # (T,) bool
    features_ok: np.ndarray  # (T,) bool = coverage_ok & keypoints_ok
    baseline_window: int = 30

    def __post_init__(self) -> None:
        if self.coverage.frame_ids != self.keypoints.frame_ids:
            raise ValueError("coverage and keypoints frame_ids differ")
        t = len(self.coverage.frame_ids)
        for name, arr in (
            ("coverage_ok", self.coverage_ok),
            ("keypoints_ok", self.keypoints_ok),
            ("features_ok", self.features_ok),
        ):
            if arr.shape != (t,):
                raise ValueError(f"{name} shape mismatch for T={t}")


@dataclass
class LoadedFeatures:
    """Features loaded from an episode ``features/`` directory."""

    features_dir: Path
    frame_ids: list[int]
    coverage: np.ndarray
    leg_area: np.ndarray
    coverage_ok: np.ndarray
    opening_xy: np.ndarray
    opening_depth: np.ndarray
    opening_norm: np.ndarray
    heel_xy: np.ndarray
    toe_xy: np.ndarray
    heel_depth: np.ndarray
    toe_depth: np.ndarray
    heel_norm: np.ndarray
    toe_norm: np.ndarray
    foot_length: np.ndarray
    keypoints_ok: np.ndarray
    features_ok: np.ndarray
    n_opening: int
    meta: dict = field(default_factory=dict)


def compute_episode_features(
    reader: EpisodeReader,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    n_opening: int = DEFAULT_N_OPENING,
    baseline_mode: BaselineMode = "early_max",
    baseline_window: int = 30,
    fixed_baseline: Optional[float] = None,
) -> FeatureBundle:
    """Compute coverage and keypoints with the same frame slice."""
    coverage = compute_episode_coverage(
        reader,
        start=start,
        end=end,
        baseline_mode=baseline_mode,
        baseline_window=baseline_window,
        fixed_baseline=fixed_baseline,
    )
    keypoints = compute_episode_keypoints(
        reader,
        start=start,
        end=end,
        n_opening=n_opening,
    )
    coverage_ok = coverage.ok.astype(bool)
    keypoints_ok = keypoints.ok.astype(bool)
    features_ok = coverage_ok & keypoints_ok
    return FeatureBundle(
        coverage=coverage,
        keypoints=keypoints,
        coverage_ok=coverage_ok,
        keypoints_ok=keypoints_ok,
        features_ok=features_ok,
        baseline_window=baseline_window,
    )


def package_summary_dict(bundle: FeatureBundle, episode_dir: Path | str) -> dict:
    """Build the ``package`` block for ``features.json``."""
    episode_dir = Path(episode_dir)
    missing = sorted(
        set(bundle.coverage.missing_frames) | set(bundle.keypoints.missing_frames)
    )
    fail_reasons: dict[str, list[int]] = {
        k: list(v) for k, v in sorted(bundle.keypoints.fail_reasons.items())
    }
    if bundle.coverage.missing_frames:
        fail_reasons["coverage_missing"] = list(bundle.coverage.missing_frames)

    return {
        "n_opening": bundle.keypoints.n_opening,
        "n_frames": len(bundle.coverage.frame_ids),
        "n_ok": int(bundle.features_ok.sum()),
        "n_missing": len(missing),
        "missing_frames": missing,
        "fail_reasons": fail_reasons,
        "frame_ids": list(bundle.coverage.frame_ids),
        "ok": bundle.features_ok.astype(bool).tolist(),
        "episode": str(episode_dir.resolve()),
    }


def save_episode_features(
    bundle: FeatureBundle,
    out_dir: Path | str,
    *,
    episode_dir: Optional[Path | str] = None,
) -> Path:
    """Write Step 2/3 npys plus package ok flags; merge ``package`` into features.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = episode_dir or out_dir

    save_episode_coverage(
        bundle.coverage,
        out_dir,
        episode_dir=ep,
        baseline_window=bundle.baseline_window,
    )
    save_episode_keypoints(bundle.keypoints, out_dir, episode_dir=ep)

    np.save(out_dir / "coverage_ok.npy", bundle.coverage_ok.astype(bool))
    np.save(out_dir / "features_ok.npy", bundle.features_ok.astype(bool))

    summary = package_summary_dict(bundle, ep)
    features_path = out_dir / "features.json"
    payload: dict = {}
    if features_path.is_file():
        with features_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    if "episode" not in payload:
        payload["episode"] = summary["episode"]
    payload["package"] = {k: v for k, v in summary.items() if k != "episode"}

    with features_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return out_dir


def load_episode_features(features_dir: Path | str) -> LoadedFeatures:
    """Load a packaged ``features/`` directory written by ``save_episode_features``."""
    features_dir = Path(features_dir)
    missing = [name for name in REQUIRED_NPY if not (features_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete features package under {features_dir}: missing {missing}"
        )

    meta: dict = {}
    features_json = features_dir / "features.json"
    if features_json.is_file():
        with features_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    package = meta.get("package") or {}
    keypoints_meta = meta.get("keypoints") or {}
    frame_ids = package.get("frame_ids") or keypoints_meta.get("frame_ids") or meta.get(
        "frame_ids"
    )
    if frame_ids is None:
        # Fall back to index range from coverage length.
        cov = np.load(features_dir / "coverage.npy")
        frame_ids = list(range(int(cov.shape[0])))

    opening_xy = np.load(features_dir / "opening_xy.npy")
    n_opening = int(
        package.get("n_opening")
        or keypoints_meta.get("n_opening")
        or opening_xy.shape[1]
    )

    return LoadedFeatures(
        features_dir=features_dir.resolve(),
        frame_ids=[int(x) for x in frame_ids],
        coverage=np.load(features_dir / "coverage.npy"),
        leg_area=np.load(features_dir / "leg_area.npy"),
        coverage_ok=np.load(features_dir / "coverage_ok.npy").astype(bool),
        opening_xy=opening_xy,
        opening_depth=np.load(features_dir / "opening_depth.npy"),
        opening_norm=np.load(features_dir / "opening_norm.npy"),
        heel_xy=np.load(features_dir / "heel_xy.npy"),
        toe_xy=np.load(features_dir / "toe_xy.npy"),
        heel_depth=np.load(features_dir / "heel_depth.npy"),
        toe_depth=np.load(features_dir / "toe_depth.npy"),
        heel_norm=np.load(features_dir / "heel_norm.npy"),
        toe_norm=np.load(features_dir / "toe_norm.npy"),
        foot_length=np.load(features_dir / "foot_length.npy"),
        keypoints_ok=np.load(features_dir / "keypoints_ok.npy").astype(bool),
        features_ok=np.load(features_dir / "features_ok.npy").astype(bool),
        n_opening=n_opening,
        meta=meta,
    )
