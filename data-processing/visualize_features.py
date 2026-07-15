#!/usr/bin/env python3
"""Visualize and validate features/*.npy against data_processing contracts.

Checks shapes / ranges / normalization invariants, overlays keypoints on
RGB+masks, and writes summary plots under ``features/viz/``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from feature_package.compute import REQUIRED_NPY, LoadedFeatures
from shareset_io.paths import CROP_HEIGHT, CROP_WIDTH, MASK_FOREGROUND, EpisodePaths
from shareset_io.reader import load_gray_u8

# Heuristic: toe ≈ (0,0), heel unit direction (|h|≈1), opening on unit foot.
TOE_NORM_TOL = 1e-4
HEEL_NORM_LEN_TOL = 5e-2
FOOT_LENGTH_MIN = 1.0

# Step 2+3 core arrays (Step 4 package flags may be derived if missing).
CORE_NPY = (
    "coverage.npy",
    "leg_area.npy",
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
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class EpisodeReport:
    episode: Path
    checks: list[CheckResult] = field(default_factory=list)
    viz_dir: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(CheckResult(name, ok, detail))


def load_features_flexible(features_dir: Path | str) -> LoadedFeatures:
    """Load features/; derive coverage_ok / features_ok when Step 4 flags are absent."""
    features_dir = Path(features_dir)
    missing_core = [name for name in CORE_NPY if not (features_dir / name).is_file()]
    if missing_core:
        raise FileNotFoundError(
            f"incomplete core features under {features_dir}: missing {missing_core}"
        )

    meta: dict = {}
    features_json = features_dir / "features.json"
    if features_json.is_file():
        with features_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    package = meta.get("package") or {}
    keypoints_meta = meta.get("keypoints") or {}
    coverage = np.load(features_dir / "coverage.npy")
    frame_ids = package.get("frame_ids") or keypoints_meta.get("frame_ids") or meta.get(
        "frame_ids"
    )
    if frame_ids is None:
        frame_ids = list(range(int(coverage.shape[0])))

    opening_xy = np.load(features_dir / "opening_xy.npy")
    n_opening = int(
        package.get("n_opening")
        or keypoints_meta.get("n_opening")
        or opening_xy.shape[1]
    )
    keypoints_ok = np.load(features_dir / "keypoints_ok.npy").astype(bool)

    cov_ok_path = features_dir / "coverage_ok.npy"
    if cov_ok_path.is_file():
        coverage_ok = np.load(cov_ok_path).astype(bool)
    else:
        coverage_ok = np.isfinite(coverage)

    feat_ok_path = features_dir / "features_ok.npy"
    if feat_ok_path.is_file():
        features_ok = np.load(feat_ok_path).astype(bool)
    else:
        features_ok = coverage_ok & keypoints_ok

    return LoadedFeatures(
        features_dir=features_dir.resolve(),
        frame_ids=[int(x) for x in frame_ids],
        coverage=coverage,
        leg_area=np.load(features_dir / "leg_area.npy"),
        coverage_ok=coverage_ok,
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
        keypoints_ok=keypoints_ok,
        features_ok=features_ok,
        n_opening=n_opening,
        meta=meta,
    )

def _load_rgb(episode_dir: Path, frame_id: int) -> Optional[np.ndarray]:
    paths = EpisodePaths.from_root(episode_dir)
    path = paths.rgb / f"{frame_id}.png"
    if not path.is_file():
        return None
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return arr[:CROP_HEIGHT, :CROP_WIDTH]


def _mask_overlay(
    rgb: np.ndarray,
    sock: Optional[np.ndarray],
    leg: Optional[np.ndarray],
    alpha: float = 0.35,
) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    if leg is not None:
        m = leg == MASK_FOREGROUND
        out[m] = (1.0 - alpha) * out[m] + alpha * np.array([80.0, 180.0, 255.0])
    if sock is not None:
        m = sock == MASK_FOREGROUND
        out[m] = (1.0 - alpha) * out[m] + alpha * np.array([255.0, 160.0, 40.0])
    return np.clip(out, 0, 255).astype(np.uint8)


def validate_features(feat, episode_dir: Path) -> EpisodeReport:
    """Validate npy shapes and contracts from Steps 2–4."""
    report = EpisodeReport(episode=episode_dir)
    t = len(feat.frame_ids)
    n = feat.n_opening

    has_core = all((feat.features_dir / name).is_file() for name in CORE_NPY)
    has_pkg = all((feat.features_dir / name).is_file() for name in REQUIRED_NPY)
    report.add(
        "required_files",
        has_core,
        (
            f"full Step4 package under {feat.features_dir}"
            if has_pkg
            else f"core Step2+3 present (derived coverage_ok/features_ok); "
            f"missing Step4={[n for n in REQUIRED_NPY if not (feat.features_dir / n).is_file()]}"
        ),
    )
    report.add(
        "opening_n",
        n == 8 or (6 <= n <= 8),
        f"n_opening={n} (spec 8, allowed 6–8)",
    )

    expected = {
        "coverage": (t,),
        "leg_area": (t,),
        "coverage_ok": (t,),
        "opening_xy": (t, n, 2),
        "opening_depth": (t, n),
        "opening_norm": (t, n, 2),
        "heel_xy": (t, 2),
        "toe_xy": (t, 2),
        "heel_depth": (t,),
        "toe_depth": (t,),
        "heel_norm": (t, 2),
        "toe_norm": (t, 2),
        "foot_length": (t,),
        "keypoints_ok": (t,),
        "features_ok": (t,),
    }
    shape_ok = True
    details = []
    for key, exp in expected.items():
        arr = getattr(feat, key)
        if arr.shape != exp:
            shape_ok = False
            details.append(f"{key} {arr.shape} != {exp}")
    report.add("shapes", shape_ok, "; ".join(details) if details else "all shapes match")

    cov = feat.coverage
    valid_cov = feat.coverage_ok
    if valid_cov.any():
        cvals = cov[valid_cov]
        in_range = bool(np.all((cvals >= 0.0 - 1e-6) & (cvals <= 1.0 + 1e-6)))
        report.add(
            "coverage_range",
            in_range,
            f"ok frames [{float(cvals.min()):.4f}, {float(cvals.max()):.4f}]",
        )
    else:
        report.add("coverage_range", False, "no coverage_ok frames")

    # coverage_ok implies finite coverage in [0,1]; missing stored as NaN
    invalid = ~feat.coverage_ok
    nan_ok = True if not invalid.any() else bool(np.all(np.isnan(cov[invalid])))
    report.add(
        "coverage_nan_on_fail",
        nan_ok,
        f"n_fail={int(invalid.sum())}, NaN on fail={nan_ok}",
    )

    kp_ok = feat.keypoints_ok
    if kp_ok.any():
        toe_n = feat.toe_norm[kp_ok]
        heel_n = feat.heel_norm[kp_ok]
        fl = feat.foot_length[kp_ok]
        toe_near_zero = bool(np.all(np.linalg.norm(toe_n, axis=-1) <= TOE_NORM_TOL))
        heel_len = np.linalg.norm(heel_n, axis=-1)
        heel_unit = bool(np.all(np.abs(heel_len - 1.0) <= HEEL_NORM_LEN_TOL))
        fl_ok = bool(np.all(fl >= FOOT_LENGTH_MIN))

        # Recompute heel_norm from xy to confirm toe-origin / foot-length formula
        heel_xy = feat.heel_xy[kp_ok]
        toe_xy = feat.toe_xy[kp_ok]
        L = np.linalg.norm(heel_xy - toe_xy, axis=-1)
        recon = (heel_xy - toe_xy) / L[:, None]
        recon_match = bool(np.allclose(heel_n, recon, atol=1e-4, equal_nan=True))

        opening_n = feat.opening_norm[kp_ok]
        opening_finite = bool(np.all(np.isfinite(opening_n)))

        report.add("toe_norm_origin", toe_near_zero, f"max |toe_norm|={float(np.linalg.norm(toe_n, axis=-1).max()):.2e}")
        report.add(
            "heel_norm_unit",
            heel_unit,
            f"|heel_norm| in [{float(heel_len.min()):.4f}, {float(heel_len.max()):.4f}]",
        )
        report.add("foot_length_positive", fl_ok, f"foot_length min={float(fl.min()):.2f}")
        report.add("heel_norm_recompute", recon_match, "heel_norm == (heel-toe)/L")
        report.add("opening_norm_finite", opening_finite, f"opening_norm finite on {int(kp_ok.sum())} ok frames")
    else:
        report.add("toe_norm_origin", False, "no keypoints_ok frames")
        report.add("heel_norm_unit", False, "no keypoints_ok frames")
        report.add("foot_length_positive", False, "no keypoints_ok frames")
        report.add("heel_norm_recompute", False, "no keypoints_ok frames")
        report.add("opening_norm_finite", False, "no keypoints_ok frames")

    combined = feat.coverage_ok & feat.keypoints_ok
    report.add(
        "features_ok_and",
        bool(np.array_equal(feat.features_ok, combined)),
        f"features_ok == coverage_ok & keypoints_ok "
        f"(n_ok={int(feat.features_ok.sum())}/{t})",
    )

    # failed keypoints → NaN in xy/norm
    fail = ~feat.keypoints_ok
    if fail.any():
        nan_xy = bool(np.all(np.isnan(feat.toe_xy[fail]))) and bool(
            np.all(np.isnan(feat.heel_xy[fail]))
        )
        report.add(
            "keypoints_nan_on_fail",
            nan_xy,
            f"n_fail={int(fail.sum())}, toe/heel xy NaN on fail={nan_xy}",
        )
    else:
        report.add("keypoints_nan_on_fail", True, "no failed keypoints frames")

    return report


def plot_timeseries(feat, out_path: Path) -> None:
    t = np.arange(len(feat.frame_ids))
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, feat.coverage, color="C0", label="coverage")
    ax.fill_between(t, 0, 1, where=~feat.coverage_ok, color="red", alpha=0.2, label="coverage fail")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("coverage")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Coverage / leg area / foot length / ok flags")

    ax = axes[1]
    ax.plot(t, feat.leg_area, color="C1", label="leg_area")
    ax.set_ylabel("leg_area [px]")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(t, feat.foot_length, color="C2", label="foot_length")
    ax.set_ylabel("foot_length [px]")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    ax.plot(t, feat.coverage_ok.astype(float), label="coverage_ok", drawstyle="steps-mid")
    ax.plot(t, feat.keypoints_ok.astype(float), label="keypoints_ok", drawstyle="steps-mid")
    ax.plot(t, feat.features_ok.astype(float), label="features_ok", drawstyle="steps-mid")
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("ok")
    ax.set_xlabel("frame index")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_normalized(feat, out_path: Path, max_frames: int = 40) -> None:
    """Scatter toe-origin normalized opening + heel for a subset of ok frames."""
    ok_idx = np.where(feat.features_ok)[0]
    if ok_idx.size == 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "no features_ok frames", ha="center", va="center")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    step = max(1, ok_idx.size // max_frames)
    sel = ok_idx[::step][:max_frames]

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, ti in enumerate(sel):
        op = feat.opening_norm[ti]
        heel = feat.heel_norm[ti]
        toe = feat.toe_norm[ti]
        alpha = 0.25 + 0.6 * (i / max(1, len(sel) - 1))
        ax.scatter(op[:, 0], op[:, 1], s=12, c="C0", alpha=alpha)
        ax.scatter([heel[0]], [heel[1]], s=40, c="C3", marker="s", alpha=alpha)
        ax.scatter([toe[0]], [toe[1]], s=40, c="C2", marker="*", alpha=alpha)
        ax.plot(
            [toe[0], heel[0]],
            [toe[1], heel[1]],
            color="gray",
            alpha=alpha * 0.5,
            linewidth=0.8,
        )

    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", color="0.5", linewidth=1)
    ax.add_patch(circle)
    ax.axhline(0, color="0.7", linewidth=0.5)
    ax.axvline(0, color="0.7", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x' (toe origin / foot length)")
    ax.set_ylabel("y'")
    ax.set_title(f"Normalized keypoints ({len(sel)} frames)\n* toe  ■ heel  · opening")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_overlay_frames(
    feat,
    episode_dir: Path,
    out_dir: Path,
    frame_indices: Optional[list[int]] = None,
    n_samples: int = 6,
) -> list[Path]:
    """Overlay opening / heel / toe on RGB+mask for selected frames."""
    t = len(feat.frame_ids)
    if frame_indices is None:
        ok = np.where(feat.features_ok)[0]
        if ok.size == 0:
            frame_indices = list(np.linspace(0, max(0, t - 1), num=min(n_samples, t), dtype=int))
        else:
            frame_indices = list(
                ok[np.linspace(0, ok.size - 1, num=min(n_samples, ok.size), dtype=int)]
            )

    paths = EpisodePaths.from_root(episode_dir)
    saved: list[Path] = []
    n = len(frame_indices)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, ti in zip(axes_flat, frame_indices):
        frame_id = feat.frame_ids[ti]
        rgb = _load_rgb(episode_dir, frame_id)
        if rgb is None:
            ax.set_title(f"frame {frame_id}: RGB missing")
            ax.axis("off")
            continue
        sock = leg = None
        sock_p = paths.sock_mask / f"{frame_id}.png"
        leg_p = paths.leg_mask / f"{frame_id}.png"
        if sock_p.is_file():
            sock = load_gray_u8(sock_p)
        if leg_p.is_file():
            leg = load_gray_u8(leg_p)
        canvas = _mask_overlay(rgb, sock, leg)

        ax.imshow(canvas)
        op = feat.opening_xy[ti]
        heel = feat.heel_xy[ti]
        toe = feat.toe_xy[ti]
        if np.isfinite(op).all():
            ax.scatter(op[:, 0], op[:, 1], c="yellow", s=28, edgecolors="k", linewidths=0.4, zorder=5)
            ax.plot(op[:, 0], op[:, 1], color="yellow", linewidth=1.0, alpha=0.8, zorder=4)
        if np.isfinite(heel).all():
            ax.scatter([heel[0]], [heel[1]], c="red", s=60, marker="s", edgecolors="k", zorder=6, label="heel")
        if np.isfinite(toe).all():
            ax.scatter([toe[0]], [toe[1]], c="lime", s=80, marker="*", edgecolors="k", zorder=6, label="toe")
        status = "OK" if feat.features_ok[ti] else "FAIL"
        cov_v = feat.coverage[ti]
        cov_s = f"{cov_v:.2f}" if np.isfinite(cov_v) else "nan"
        ax.set_title(f"t={ti} id={frame_id} [{status}] cov={cov_s}", fontsize=9)
        ax.axis("off")

    for ax in axes_flat[len(frame_indices) :]:
        ax.axis("off")

    fig.suptitle(f"Keypoints overlay — {episode_dir.name}", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "overlay_frames.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    saved.append(out_path)
    return saved


def visualize_episode(
    episode_dir: Path,
    *,
    features_subdir: str = "features",
    n_overlay: int = 6,
) -> EpisodeReport:
    episode_dir = Path(episode_dir).expanduser().resolve()
    features_dir = episode_dir / features_subdir
    feat = load_features_flexible(features_dir)
    report = validate_features(feat, episode_dir)

    viz_dir = features_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    report.viz_dir = viz_dir

    plot_timeseries(feat, viz_dir / "timeseries.png")
    plot_normalized(feat, viz_dir / "normalized_scatter.png")
    plot_overlay_frames(feat, episode_dir, viz_dir, n_samples=n_overlay)

    # Persist check summary
    summary = {
        "episode": str(episode_dir),
        "n_frames": len(feat.frame_ids),
        "n_opening": feat.n_opening,
        "n_coverage_ok": int(feat.coverage_ok.sum()),
        "n_keypoints_ok": int(feat.keypoints_ok.sum()),
        "n_features_ok": int(feat.features_ok.sum()),
        "coverage_min": float(np.nanmin(feat.coverage)) if np.isfinite(feat.coverage).any() else None,
        "coverage_max": float(np.nanmax(feat.coverage)) if np.isfinite(feat.coverage).any() else None,
        "foot_length_mean": float(np.nanmean(feat.foot_length)) if np.isfinite(feat.foot_length).any() else None,
        "all_checks_ok": report.ok,
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in report.checks],
    }
    with (viz_dir / "validation.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return report


def discover_episodes(dataset_split: Path) -> list[Path]:
    root = Path(dataset_split).expanduser().resolve()
    return sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "features").is_dir()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--episode", type=Path, help="Single episode directory")
    g.add_argument(
        "--dataset-split",
        type=Path,
        help="Split dir containing episode folders with features/",
    )
    parser.add_argument("--n-overlay", type=int, default=6, help="Overlay sample count")
    args = parser.parse_args()

    episodes = (
        [Path(args.episode)]
        if args.episode is not None
        else discover_episodes(args.dataset_split)
    )
    if not episodes:
        print("No episodes with features/ found.")
        return 1

    n_pass = 0
    for ep in episodes:
        report = visualize_episode(ep, n_overlay=args.n_overlay)
        status = "PASS" if report.ok else "FAIL"
        if report.ok:
            n_pass += 1
        print(f"[{status}] {ep.name}")
        for c in report.checks:
            mark = "✓" if c.ok else "✗"
            print(f"  {mark} {c.name}: {c.detail}")
        if report.viz_dir:
            print(f"  viz → {report.viz_dir}")

    print(f"\n{n_pass}/{len(episodes)} episodes passed validation.")
    return 0 if n_pass == len(episodes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
