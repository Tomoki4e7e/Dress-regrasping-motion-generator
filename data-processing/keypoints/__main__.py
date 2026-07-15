"""CLI: compute keypoints for ShareSet episodes (Step 3).

Examples:
  PYTHONPATH=. python3 -m keypoints \\
    --episode .../data_center_general_depth_n5_sample/train/2025-01-06-18-25-46

  PYTHONPATH=. python3 -m keypoints \\
    --dataset-split .../data_center_general_depth_n5_sample/train \\
    --n-opening 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shareset_io import EpisodeReader

from .compute import (
    DEFAULT_N_OPENING,
    MAX_N_OPENING,
    MIN_N_OPENING,
    compute_episode_keypoints,
    save_episode_keypoints,
)


def _is_episode_dir(path: Path) -> bool:
    return (path / "camera_right_mask" / "leg_mask").is_dir() and (
        path / "camera_right_mask" / "sock_mask"
    ).is_dir()


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
    n_opening: int,
    start: int | None,
    end: int | None,
) -> None:
    reader = EpisodeReader(episode_dir)
    result = compute_episode_keypoints(
        reader,
        start=start,
        end=end,
        n_opening=n_opening,
    )
    out_dir = _default_out_dir(episode_dir, out_root)
    save_episode_keypoints(result, out_dir, episode_dir=episode_dir)
    ok_len = result.foot_length[result.ok]
    fl_mean = float(ok_len.mean()) if ok_len.size else float("nan")
    print(
        f"{episode_dir.name}: n={result.ok.size} ok={int(result.ok.sum())} "
        f"missing={len(result.missing_frames)} "
        f"n_opening={result.n_opening} foot_len_mean={fl_mean:.1f} "
        f"-> {out_dir}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Step 3: opening / heel / toe keypoints from sock+leg masks"
    )
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
        "--n-opening",
        type=int,
        default=DEFAULT_N_OPENING,
        help=f"Opening rim point count ({MIN_N_OPENING}-{MAX_N_OPENING}, default: {DEFAULT_N_OPENING})",
    )
    parser.add_argument("--start", type=int, default=None, help="List index start (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="List index end (exclusive)")
    args = parser.parse_args(argv)

    if args.n_opening < MIN_N_OPENING or args.n_opening > MAX_N_OPENING:
        parser.error(f"--n-opening must be in [{MIN_N_OPENING}, {MAX_N_OPENING}]")

    episodes: list[Path]
    if args.episode is not None:
        episode = args.episode.expanduser().resolve()
        if not _is_episode_dir(episode):
            print(f"not an episode with sock_mask/leg_mask: {episode}", file=sys.stderr)
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
            n_opening=args.n_opening,
            start=args.start,
            end=args.end,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
