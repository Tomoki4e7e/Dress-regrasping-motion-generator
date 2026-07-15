"""Keypoints from ShareSet sock/leg masks: opening rim, heel, toe (+ depth, norm)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from shareset_io import MASK_FOREGROUND, EpisodeReader

DEFAULT_N_OPENING = 8
MIN_N_OPENING = 6
MAX_N_OPENING = 8
OPENING_DIST_PERCENTILE = 20.0
MIN_FOOT_LENGTH = 1.0
MIN_CONTOUR_AREA = 50.0
MIN_MASK_PIXELS = 30


@dataclass
class FrameKeypoints:
    """Keypoints for one frame (NaN arrays if failed)."""

    frame_id: int
    opening_xy: np.ndarray  # (N, 2)
    opening_depth: np.ndarray  # (N,)
    opening_norm: np.ndarray  # (N, 2)
    heel_xy: np.ndarray  # (2,)
    toe_xy: np.ndarray  # (2,)
    heel_depth: float
    toe_depth: float
    heel_norm: np.ndarray  # (2,)
    toe_norm: np.ndarray  # (2,)
    foot_length: float
    ok: bool
    fail_reason: Optional[str] = None


@dataclass
class KeypointsResult:
    """Per-frame keypoints for one episode."""

    frame_ids: list[int]
    n_opening: int
    opening_xy: np.ndarray  # (T, N, 2)
    opening_depth: np.ndarray  # (T, N)
    opening_norm: np.ndarray  # (T, N, 2)
    heel_xy: np.ndarray  # (T, 2)
    toe_xy: np.ndarray  # (T, 2)
    heel_depth: np.ndarray  # (T,)
    toe_depth: np.ndarray  # (T,)
    heel_norm: np.ndarray  # (T, 2)
    toe_norm: np.ndarray  # (T, 2)
    foot_length: np.ndarray  # (T,)
    ok: np.ndarray  # (T,) bool
    missing_frames: list[int] = field(default_factory=list)
    fail_reasons: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = len(self.frame_ids)
        n = self.n_opening
        checks = (
            (self.opening_xy.shape == (t, n, 2), "opening_xy"),
            (self.opening_depth.shape == (t, n), "opening_depth"),
            (self.opening_norm.shape == (t, n, 2), "opening_norm"),
            (self.heel_xy.shape == (t, 2), "heel_xy"),
            (self.toe_xy.shape == (t, 2), "toe_xy"),
            (self.heel_depth.shape == (t,), "heel_depth"),
            (self.toe_depth.shape == (t,), "toe_depth"),
            (self.heel_norm.shape == (t, 2), "heel_norm"),
            (self.toe_norm.shape == (t, 2), "toe_norm"),
            (self.foot_length.shape == (t,), "foot_length"),
            (self.ok.shape == (t,), "ok"),
        )
        for good, name in checks:
            if not good:
                raise ValueError(f"{name} shape mismatch for T={t}, N={n}")


def _empty_frame(frame_id: int, n_opening: int, reason: str) -> FrameKeypoints:
    nan2 = np.full((2,), np.nan, dtype=np.float64)
    return FrameKeypoints(
        frame_id=frame_id,
        opening_xy=np.full((n_opening, 2), np.nan, dtype=np.float64),
        opening_depth=np.full((n_opening,), np.nan, dtype=np.float64),
        opening_norm=np.full((n_opening, 2), np.nan, dtype=np.float64),
        heel_xy=nan2.copy(),
        toe_xy=nan2.copy(),
        heel_depth=float("nan"),
        toe_depth=float("nan"),
        heel_norm=nan2.copy(),
        toe_norm=nan2.copy(),
        foot_length=float("nan"),
        ok=False,
        fail_reason=reason,
    )


def _binarize(mask: np.ndarray) -> np.ndarray:
    return (mask == MASK_FOREGROUND).astype(np.uint8)


def largest_external_contour(mask_bin: np.ndarray):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def sample_polyline(pts_xy: np.ndarray, n: int) -> np.ndarray:
    """Equally arc-length sample an open polyline to ``n`` points."""
    pts = np.asarray(pts_xy, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("polyline needs at least 2 points")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        if seg[j] < 1e-9:
            out[i] = pts[j]
        else:
            a = (t - cum[j]) / seg[j]
            out[i] = pts[j] * (1.0 - a) + pts[j + 1] * a
    return out


def opening_rim_from_masks(
    sock_bin: np.ndarray,
    leg_bin: np.ndarray,
    n_opening: int,
    *,
    percentile: float = OPENING_DIST_PERCENTILE,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Sample ``n_opening`` points on sock contour closest to leg (opening rim)."""
    contour = largest_external_contour(sock_bin)
    if contour is None or cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        return None, "empty_sock"
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return None, "empty_sock"

    leg_inv = np.where(leg_bin > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(leg_inv, cv2.DIST_L2, 5)
    xs = np.clip(pts[:, 0].astype(np.int32), 0, dist.shape[1] - 1)
    ys = np.clip(pts[:, 1].astype(np.int32), 0, dist.shape[0] - 1)
    dvals = dist[ys, xs]
    thr = float(np.percentile(dvals, percentile))
    near = dvals <= max(thr, 2.0)

    # Longest contiguous True run on circular contour.
    near_ext = np.concatenate([near, near])
    best_len, best_start, cur, start = 0, 0, 0, 0
    for i, flag in enumerate(near_ext):
        if flag:
            if cur == 0:
                start = i
            cur += 1
            if cur > best_len:
                best_len, best_start = cur, start
        else:
            cur = 0
    if best_len < 3:
        return None, "opening_too_short"

    idxs = [(best_start + k) % len(pts) for k in range(best_len)]
    arc = pts[idxs]
    return sample_polyline(arc, n_opening), None


def heel_toe_from_masks(
    sock_bin: np.ndarray,
    leg_bin: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Toe = distal sock extreme; heel = proximal leg extreme along union PCA axis."""
    if int(sock_bin.sum()) < MIN_MASK_PIXELS:
        return None, None, "empty_sock"
    if int(leg_bin.sum()) < MIN_MASK_PIXELS:
        return None, None, "empty_leg"

    union = ((sock_bin > 0) | (leg_bin > 0)).astype(np.uint8)
    uy, ux = np.nonzero(union)
    if len(ux) < MIN_MASK_PIXELS:
        return None, None, "empty_union"

    pts = np.column_stack([ux, uy]).astype(np.float64)
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]

    sy, sx = np.nonzero(sock_bin)
    ly, lx = np.nonzero(leg_bin)
    sock_c = np.array([sx.mean(), sy.mean()], dtype=np.float64)
    leg_c = np.array([lx.mean(), ly.mean()], dtype=np.float64)
    if (sock_c - mean) @ axis < (leg_c - mean) @ axis:
        axis = -axis

    def extreme(mask: np.ndarray, distal: bool) -> Optional[np.ndarray]:
        my, mx = np.nonzero(mask)
        if len(mx) == 0:
            return None
        mp = np.column_stack([mx, my]).astype(np.float64)
        proj = (mp - mean) @ axis
        idx = int(np.argmax(proj) if distal else np.argmin(proj))
        return mp[idx]

    toe = extreme(sock_bin, distal=True)
    heel = extreme(leg_bin, distal=False)
    if toe is None or heel is None:
        return None, None, "extreme_fail"
    return toe, heel, None


def sample_depth(depth: Optional[np.ndarray], xy: np.ndarray) -> float:
    """Look up camera_depth at rounded pixel; NaN if missing / OOB / zero background."""
    if depth is None or not np.all(np.isfinite(xy)):
        return float("nan")
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    h, w = depth.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return float("nan")
    return float(depth[y, x])


def sample_depth_batch(depth: Optional[np.ndarray], xys: np.ndarray) -> np.ndarray:
    out = np.empty((len(xys),), dtype=np.float64)
    for i, xy in enumerate(xys):
        out[i] = sample_depth(depth, xy)
    return out


def normalize_about_toe(
    points: np.ndarray,
    toe: np.ndarray,
    heel: np.ndarray,
) -> tuple[np.ndarray, float]:
    """``p' = (p - toe) / ||heel - toe||``; foot_length is NaN if too small."""
    foot_length = float(np.linalg.norm(heel - toe))
    if not np.isfinite(foot_length) or foot_length < MIN_FOOT_LENGTH:
        return np.full_like(points, np.nan, dtype=np.float64), float("nan")
    return (points - toe) / foot_length, foot_length


def estimate_frame_keypoints(
    frame_id: int,
    sock_mask: Optional[np.ndarray],
    leg_mask: Optional[np.ndarray],
    camera_depth: Optional[np.ndarray],
    *,
    n_opening: int = DEFAULT_N_OPENING,
    opening_percentile: float = OPENING_DIST_PERCENTILE,
) -> FrameKeypoints:
    if n_opening < MIN_N_OPENING or n_opening > MAX_N_OPENING:
        raise ValueError(f"n_opening must be in [{MIN_N_OPENING}, {MAX_N_OPENING}]")

    if sock_mask is None:
        return _empty_frame(frame_id, n_opening, "missing_sock_mask")
    if leg_mask is None:
        return _empty_frame(frame_id, n_opening, "missing_leg_mask")

    sock_bin = _binarize(sock_mask)
    leg_bin = _binarize(leg_mask)

    opening, open_err = opening_rim_from_masks(
        sock_bin, leg_bin, n_opening, percentile=opening_percentile
    )
    if opening is None:
        return _empty_frame(frame_id, n_opening, open_err or "opening_fail")

    toe, heel, ht_err = heel_toe_from_masks(sock_bin, leg_bin)
    if toe is None or heel is None:
        return _empty_frame(frame_id, n_opening, ht_err or "heel_toe_fail")

    stacked = np.vstack([opening, heel.reshape(1, 2), toe.reshape(1, 2)])
    stacked_norm, foot_length = normalize_about_toe(stacked, toe, heel)
    if not np.isfinite(foot_length):
        return _empty_frame(frame_id, n_opening, "foot_length_zero")

    opening_norm = stacked_norm[:n_opening]
    heel_norm = stacked_norm[n_opening]
    toe_norm = stacked_norm[n_opening + 1]

    opening_depth = sample_depth_batch(camera_depth, opening)
    heel_depth = sample_depth(camera_depth, heel)
    toe_depth = sample_depth(camera_depth, toe)

    return FrameKeypoints(
        frame_id=frame_id,
        opening_xy=opening.astype(np.float64),
        opening_depth=opening_depth,
        opening_norm=opening_norm.astype(np.float64),
        heel_xy=heel.astype(np.float64),
        toe_xy=toe.astype(np.float64),
        heel_depth=heel_depth,
        toe_depth=toe_depth,
        heel_norm=heel_norm.astype(np.float64),
        toe_norm=toe_norm.astype(np.float64),
        foot_length=foot_length,
        ok=True,
        fail_reason=None,
    )


def compute_episode_keypoints(
    reader: EpisodeReader,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    n_opening: int = DEFAULT_N_OPENING,
    opening_percentile: float = OPENING_DIST_PERCENTILE,
) -> KeypointsResult:
    """Compute keypoints for frames in ``reader`` via Step 1 ``EpisodeReader``.

    ``start`` / ``end`` are list indices (yaml-style), same as ``iter_frames``.
    """
    if n_opening < MIN_N_OPENING or n_opening > MAX_N_OPENING:
        raise ValueError(f"n_opening must be in [{MIN_N_OPENING}, {MAX_N_OPENING}]")

    files = reader.frame_files
    if start is None:
        start = 0
    if end is None:
        end = len(files)
    names = files[start:end]
    t = len(names)

    opening_xy = np.full((t, n_opening, 2), np.nan, dtype=np.float64)
    opening_depth = np.full((t, n_opening), np.nan, dtype=np.float64)
    opening_norm = np.full((t, n_opening, 2), np.nan, dtype=np.float64)
    heel_xy = np.full((t, 2), np.nan, dtype=np.float64)
    toe_xy = np.full((t, 2), np.nan, dtype=np.float64)
    heel_depth = np.full((t,), np.nan, dtype=np.float64)
    toe_depth = np.full((t,), np.nan, dtype=np.float64)
    heel_norm = np.full((t, 2), np.nan, dtype=np.float64)
    toe_norm = np.full((t, 2), np.nan, dtype=np.float64)
    foot_length = np.full((t,), np.nan, dtype=np.float64)
    ok = np.zeros((t,), dtype=bool)
    frame_ids: list[int] = []
    missing_frames: list[int] = []
    fail_reasons: dict[str, list[int]] = {}

    for i, name in enumerate(names):
        frame_id = int(Path(name).stem)
        frame_ids.append(frame_id)
        frame = reader.load_frame(
            frame_id, require=("sock_mask", "leg_mask", "camera_depth")
        )
        fk = estimate_frame_keypoints(
            frame_id,
            frame.sock_mask,
            frame.leg_mask,
            frame.camera_depth,
            n_opening=n_opening,
            opening_percentile=opening_percentile,
        )

        opening_xy[i] = fk.opening_xy
        opening_depth[i] = fk.opening_depth
        opening_norm[i] = fk.opening_norm
        heel_xy[i] = fk.heel_xy
        toe_xy[i] = fk.toe_xy
        heel_depth[i] = fk.heel_depth
        toe_depth[i] = fk.toe_depth
        heel_norm[i] = fk.heel_norm
        toe_norm[i] = fk.toe_norm
        foot_length[i] = fk.foot_length
        ok[i] = fk.ok
        if not fk.ok:
            missing_frames.append(frame_id)
            reason = fk.fail_reason or "unknown"
            fail_reasons.setdefault(reason, []).append(frame_id)

    return KeypointsResult(
        frame_ids=frame_ids,
        n_opening=n_opening,
        opening_xy=opening_xy,
        opening_depth=opening_depth,
        opening_norm=opening_norm,
        heel_xy=heel_xy,
        toe_xy=toe_xy,
        heel_depth=heel_depth,
        toe_depth=toe_depth,
        heel_norm=heel_norm,
        toe_norm=toe_norm,
        foot_length=foot_length,
        ok=ok,
        missing_frames=missing_frames,
        fail_reasons=fail_reasons,
    )


def _optional_float(value: float) -> Optional[float]:
    return None if value is None or not np.isfinite(value) else float(value)


def _optional_xy_list(arr: np.ndarray) -> list[Optional[list[Optional[float]]]]:
    """Serialize (T, 2) or (T, N, 2) with nulls for non-finite."""
    out: list = []
    if arr.ndim == 2:
        for row in arr:
            if not np.all(np.isfinite(row)):
                out.append(None)
            else:
                out.append([float(row[0]), float(row[1])])
        return out
    if arr.ndim == 3:
        for frame in arr:
            if not np.all(np.isfinite(frame)):
                out.append(None)
            else:
                out.append([[float(x), float(y)] for x, y in frame])
        return out
    raise ValueError(f"unexpected xy array ndim={arr.ndim}")


def keypoints_summary_dict(
    result: KeypointsResult,
    episode_dir: Path | str,
) -> dict:
    episode_dir = Path(episode_dir)
    ok_len = result.foot_length[result.ok]
    return {
        "keypoints": {
            "n_opening": result.n_opening,
            "n_frames": len(result.frame_ids),
            "n_ok": int(result.ok.sum()),
            "n_missing": len(result.missing_frames),
            "missing_frames": list(result.missing_frames),
            "fail_reasons": {k: list(v) for k, v in sorted(result.fail_reasons.items())},
            "foot_length_mean": _optional_float(float(np.mean(ok_len))) if ok_len.size else None,
            "foot_length_min": _optional_float(float(np.min(ok_len))) if ok_len.size else None,
            "foot_length_max": _optional_float(float(np.max(ok_len))) if ok_len.size else None,
            "frame_ids": list(result.frame_ids),
            "ok": result.ok.astype(bool).tolist(),
            "toe_xy": _optional_xy_list(result.toe_xy),
            "heel_xy": _optional_xy_list(result.heel_xy),
            "foot_length": [_optional_float(float(v)) for v in result.foot_length.tolist()],
        },
        "episode": str(episode_dir.resolve()),
    }


def save_episode_keypoints(
    result: KeypointsResult,
    out_dir: Path | str,
    *,
    episode_dir: Optional[Path | str] = None,
) -> Path:
    """Write keypoint npys and merge a ``keypoints`` block into ``features.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "opening_xy.npy", result.opening_xy.astype(np.float32))
    np.save(out_dir / "opening_depth.npy", result.opening_depth.astype(np.float32))
    np.save(out_dir / "opening_norm.npy", result.opening_norm.astype(np.float32))
    np.save(out_dir / "heel_xy.npy", result.heel_xy.astype(np.float32))
    np.save(out_dir / "toe_xy.npy", result.toe_xy.astype(np.float32))
    np.save(out_dir / "heel_depth.npy", result.heel_depth.astype(np.float32))
    np.save(out_dir / "toe_depth.npy", result.toe_depth.astype(np.float32))
    np.save(out_dir / "heel_norm.npy", result.heel_norm.astype(np.float32))
    np.save(out_dir / "toe_norm.npy", result.toe_norm.astype(np.float32))
    np.save(out_dir / "foot_length.npy", result.foot_length.astype(np.float32))
    np.save(out_dir / "keypoints_ok.npy", result.ok.astype(bool))

    summary = keypoints_summary_dict(result, episode_dir or out_dir)
    features_path = out_dir / "features.json"
    payload: dict = {}
    if features_path.is_file():
        with features_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    # Merge without wiping coverage keys; keep episode path consistent.
    if "episode" not in payload:
        payload["episode"] = summary["episode"]
    payload["keypoints"] = summary["keypoints"]

    with features_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return out_dir
