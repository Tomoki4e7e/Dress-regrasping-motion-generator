"""CLI: compute coverage for ShareSet episodes (Step 2).

Examples:
  PYTHONPATH=. python3 -m coverage \\
    --episode .../data_center_general_depth_n5_sample/train/2025-01-06-18-25-46

  PYTHONPATH=. python3 -m coverage \\
    --dataset-split .../data_center_general_depth_n5_sample/train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shareset_io import EpisodeReader

from .compute import compute_episode_coverage, save_episode_coverage


def _is_episode_dir(path: Path) -> bool:
    return (path / "camera_right_mask" / "leg_mask").is_dir()


def _discover_episodes(split_dir: Path) -> list[Path]:
    return sorted(p for p in split_dir.iterdir() if p.is_dir() and _is_episode_dir(p))


def _default_out_dir(episode_dir: Path, out_root: Path | None) -> Path:
    if out_root is None:
        return episode_dir / "features"
    return out_root / episode_dir.name / "features"


def process_episode(
    episode_dir: Path,
    *,
    out_root: Path | None,
    baseline_mode: str,
    baseline_window: int,
    fixed_baseline: float | None,
    start: int | None,
    end: int | None,
) -> None:
    reader = EpisodeReader(episode_dir)
    result = compute_episode_coverage(
        reader,
        start=start,
        end=end,
        baseline_mode=baseline_mode,  # type: ignore[arg-type]
        baseline_window=baseline_window,
        fixed_baseline=fixed_baseline,
    )
    out_dir = _default_out_dir(episode_dir, out_root)
    save_episode_coverage(
        result,
        out_dir,
        episode_dir=episode_dir,
        baseline_window=baseline_window,
    )
    ok_cov = result.coverage[result.ok]
    cov_mean = float(ok_cov.mean()) if ok_cov.size else float("nan")
    print(
        f"{episode_dir.name}: n={result.ok.size} ok={int(result.ok.sum())} "
        f"missing={len(result.missing_frames)} "
        f"baseline={result.baseline_area:.1f} cov_mean={cov_mean:.4f} "
        f"-> {out_dir}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 2: leg-mask coverage from area decrease")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode", type=Path, help="One ShareSet episode directory")
    group.add_argument(
        "--dataset-split",
        type=Path,
        help="Split dir containing episode subdirs (e.g. .../train)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Write under OUT_ROOT/<episode>/features (default: <episode>/features)",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("early_max", "episode_max", "fixed"),
        default="early_max",
        help="How to choose baseline leg area A0 (default: early_max)",
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=30,
        help="For early_max: use max leg area in first N list frames (default: 30)",
    )
    parser.add_argument(
        "--fixed-baseline",
        type=float,
        default=None,
        help="Required when --baseline-mode fixed",
    )
    parser.add_argument("--start", type=int, default=None, help="List index start (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="List index end (exclusive)")
    args = parser.parse_args(argv)

    if args.baseline_mode == "fixed" and (args.fixed_baseline is None or args.fixed_baseline <= 0):
        parser.error("--fixed-baseline > 0 is required with --baseline-mode fixed")

    episodes: list[Path]
    if args.episode is not None:
        episode = args.episode.expanduser().resolve()
        if not _is_episode_dir(episode):
            print(f"not an episode with leg_mask: {episode}", file=sys.stderr)
            return 1
        episodes = [episode]
    else:
        split = args.dataset_split.expanduser().resolve()
        episodes = _discover_episodes(split)
        if not episodes:
            print(f"no episodes under {split}", file=sys.stderr)
            return 1

    out_root = args.out_root.expanduser().resolve() if args.out_root else None
    for ep in episodes:
        process_episode(
            ep,
            out_root=out_root,
            baseline_mode=args.baseline_mode,
            baseline_window=args.baseline_window,
            fixed_baseline=args.fixed_baseline,
            start=args.start,
            end=args.end,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
