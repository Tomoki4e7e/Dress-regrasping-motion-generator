"""Coverage = clipped decrease of visible leg-mask area vs baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from shareset_io import MASK_FOREGROUND, EpisodeReader

BaselineMode = Literal["early_max", "episode_max", "fixed"]


@dataclass
class CoverageResult:
    """Per-frame coverage for one episode."""

    frame_ids: list[int]
    leg_area: np.ndarray  # float64, nan if leg_mask missing
    coverage: np.ndarray  # float64 in [0, 1], nan if missing / baseline invalid
    ok: np.ndarray  # bool: valid coverage value
    baseline_area: float
    baseline_mode: str
    missing_frames: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.frame_ids)
        if self.leg_area.shape != (n,) or self.coverage.shape != (n,) or self.ok.shape != (n,):
            raise ValueError("frame_ids / leg_area / coverage / ok length mismatch")


@dataclass
class EpisodeCoverage:
    """Serializable episode summary (features.json)."""

    episode: str
    baseline_area: float
    baseline_mode: str
    baseline_window: int
    n_frames: int
    n_ok: int
    n_missing: int
    coverage_min: Optional[float]
    coverage_max: Optional[float]
    coverage_mean: Optional[float]
    missing_frames: list[int]
    frame_ids: list[int]
    coverage: list[Optional[float]]
    leg_area: list[Optional[float]]
    ok: list[bool]


def foreground_area(mask: np.ndarray, foreground: int = MASK_FOREGROUND) -> int:
    """Count foreground pixels in a ShareSet mask (``255``)."""
    return int(np.count_nonzero(mask == foreground))


def coverage_from_areas(
    leg_areas: np.ndarray,
    baseline_area: float,
) -> np.ndarray:
    """``clip((baseline - area_t) / baseline, 0, 1)``; nan stays nan.

    Uses leg-area *decrease* so sock slack does not inflate coverage.
    """
    areas = np.asarray(leg_areas, dtype=np.float64)
    out = np.full(areas.shape, np.nan, dtype=np.float64)
    if not np.isfinite(baseline_area) or baseline_area <= 0:
        return out
    valid = np.isfinite(areas)
    raw = (baseline_area - areas[valid]) / baseline_area
    out[valid] = np.clip(raw, 0.0, 1.0)
    return out


def _resolve_baseline(
    leg_areas: np.ndarray,
    *,
    mode: BaselineMode,
    baseline_window: int,
    fixed_baseline: Optional[float],
) -> float:
    valid = leg_areas[np.isfinite(leg_areas)]
    if mode == "fixed":
        if fixed_baseline is None or not np.isfinite(fixed_baseline) or fixed_baseline <= 0:
            raise ValueError("fixed baseline_mode requires a positive fixed_baseline")
        return float(fixed_baseline)
    if valid.size == 0:
        return float("nan")
    if mode == "episode_max":
        return float(np.max(valid))
    if mode == "early_max":
        if baseline_window < 1:
            raise ValueError("baseline_window must be >= 1")
        early = leg_areas[:baseline_window]
        early_valid = early[np.isfinite(early)]
        if early_valid.size == 0:
            return float(np.max(valid))
        return float(np.max(early_valid))
    raise ValueError(f"unknown baseline_mode: {mode!r}")


def compute_episode_coverage(
    reader: EpisodeReader,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    baseline_mode: BaselineMode = "early_max",
    baseline_window: int = 30,
    fixed_baseline: Optional[float] = None,
) -> CoverageResult:
    """Compute coverage for frames in ``reader`` via Step 1 ``EpisodeReader``.

    ``start`` / ``end`` are list indices (yaml-style), same as ``iter_frames``.
    """
    files = reader.frame_files
    if start is None:
        start = 0
    if end is None:
        end = len(files)
    names = files[start:end]

    frame_ids: list[int] = []
    areas: list[float] = []
    missing: list[int] = []

    for name in names:
        frame_id = int(Path(name).stem)
        frame_ids.append(frame_id)
        frame = reader.load_frame(frame_id, require=("leg_mask",))
        if frame.leg_mask is None or "leg_mask" in frame.missing:
            areas.append(float("nan"))
            missing.append(frame_id)
            continue
        areas.append(float(foreground_area(frame.leg_mask)))

    leg_area = np.asarray(areas, dtype=np.float64)
    baseline = _resolve_baseline(
        leg_area,
        mode=baseline_mode,
        baseline_window=baseline_window,
        fixed_baseline=fixed_baseline,
    )
    coverage = coverage_from_areas(leg_area, baseline)
    ok = np.isfinite(coverage)

    return CoverageResult(
        frame_ids=frame_ids,
        leg_area=leg_area,
        coverage=coverage,
        ok=ok,
        baseline_area=baseline,
        baseline_mode=baseline_mode,
        missing_frames=missing,
    )


def _optional_float_list(arr: np.ndarray) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for value in arr.tolist():
        out.append(None if value is None or not np.isfinite(value) else float(value))
    return out


def to_episode_coverage(
    result: CoverageResult,
    episode_dir: Path | str,
    *,
    baseline_window: int = 30,
) -> EpisodeCoverage:
    episode_dir = Path(episode_dir)
    ok_cov = result.coverage[result.ok]
    baseline = (
        float(result.baseline_area) if np.isfinite(result.baseline_area) else float("nan")
    )
    return EpisodeCoverage(
        episode=str(episode_dir.resolve()),
        baseline_area=baseline,
        baseline_mode=result.baseline_mode,
        baseline_window=baseline_window,
        n_frames=len(result.frame_ids),
        n_ok=int(result.ok.sum()),
        n_missing=len(result.missing_frames),
        coverage_min=float(np.min(ok_cov)) if ok_cov.size else None,
        coverage_max=float(np.max(ok_cov)) if ok_cov.size else None,
        coverage_mean=float(np.mean(ok_cov)) if ok_cov.size else None,
        missing_frames=list(result.missing_frames),
        frame_ids=list(result.frame_ids),
        coverage=_optional_float_list(result.coverage),
        leg_area=_optional_float_list(result.leg_area),
        ok=result.ok.astype(bool).tolist(),
    )


def save_episode_coverage(
    result: CoverageResult,
    out_dir: Path | str,
    *,
    episode_dir: Optional[Path | str] = None,
    baseline_window: int = 30,
) -> Path:
    """Write ``coverage.npy``, ``leg_area.npy``, and merge coverage into ``features.json``.

    Missing / invalid frames are stored as NaN in npys and ``null`` in JSON.
    Existing non-coverage keys (e.g. ``keypoints``, ``package``) are preserved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "coverage.npy", result.coverage.astype(np.float32))
    np.save(out_dir / "leg_area.npy", result.leg_area.astype(np.float64))

    summary = to_episode_coverage(
        result,
        episode_dir or out_dir,
        baseline_window=baseline_window,
    )
    cov_payload = asdict(summary)
    if cov_payload["baseline_area"] is not None and not np.isfinite(cov_payload["baseline_area"]):
        cov_payload["baseline_area"] = None

    features_path = out_dir / "features.json"
    payload: dict = {}
    if features_path.is_file():
        with features_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    # Merge coverage keys at top level without wiping keypoints / package blocks.
    payload.update(cov_payload)

    with features_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return out_dir
