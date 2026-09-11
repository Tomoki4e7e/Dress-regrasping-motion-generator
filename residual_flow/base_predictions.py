"""Export teacher-forced SAMDAMSARNN actions for residual supervision.

For honest residual labels, pass ``--checkpoint-map`` with a checkpoint trained
without each mapped episode. A single checkpoint is supported for debugging but
is rejected when ``--require-crossfit`` is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter

from .contracts import ACTION_DIM, IMAGE_SIZE, EpisodeSpec
from .data import VISUAL_DIRS, load_csv_window, load_manifest, prediction_path


def normalize(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    scale = high - low
    if np.any(scale == 0):
        raise ValueError("normalization range contains a zero-width channel")
    return (value - low) / scale


def denormalize(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return value * (high - low) + low


def _rgb_sequence(directory: Path, start: int, end: int, size: int) -> np.ndarray:
    result = []
    for frame in range(start, end):
        with Image.open(directory / f"{frame}.png") as image:
            image = image.convert("RGB").crop((0, 0, 1280, 960)).resize((size, size))
            result.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return np.stack(result)


def _depth_sequence(directory: Path, start: int, end: int, size: int) -> np.ndarray:
    result = []
    for frame in range(start, end):
        with Image.open(directory / f"{frame}.png") as image:
            image = image.convert("L").filter(ImageFilter.GaussianBlur(radius=15))
            image = image.crop((0, 0, 1280, 960)).resize((size, size))
            result.append(np.asarray(image, dtype=np.float32)[None, ...] / 255.0)
    return np.stack(result)


def _load_model(
    shareset_src: Path,
    checkpoint: Path,
    *,
    rec_dim: int,
    k_dim: int,
    image_size: int,
    device: torch.device,
) -> torch.nn.Module:
    sys.path.insert(0, str(shareset_src))
    try:
        from models.SAMDAMSARNN import SAMDAMSARNN
    finally:
        sys.path.pop(0)
    model = SAMDAMSARNN(
        rec_dim=rec_dim,
        k_dim=k_dim,
        joint_dim=36,
        im_size=[image_size, image_size],
    )
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    return model.to(device).eval()


def export_episode(
    spec: EpisodeSpec,
    model: torch.nn.Module,
    stats_path: Path,
    output: Path,
    *,
    device: torch.device,
    image_size: int = IMAGE_SIZE,
) -> Path:
    """Run teacher-forced observations while carrying the recurrent state."""
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    low = np.asarray(stats["joint_min"], dtype=np.float32)
    high = np.asarray(stats["joint_max"], dtype=np.float32)
    if low.shape != (36,) or high.shape != (36,):
        raise ValueError("SAMDAMSARNN statistics must contain 36 channels")

    angles = load_csv_window(spec.directory / "angle.csv", spec.start, spec.end, 18)
    torque = load_csv_window(spec.directory / "torque.csv", spec.start, spec.end, 18)
    joints = np.concatenate((angles, torque), axis=1)
    normalized_joints = normalize(joints, low, high)
    rgb = _rgb_sequence(spec.directory / VISUAL_DIRS["rgb"], spec.start, spec.end, image_size)
    sock = _depth_sequence(
        spec.directory / VISUAL_DIRS["sock_depth"], spec.start, spec.end, image_size
    )
    leg = _depth_sequence(
        spec.directory / VISUAL_DIRS["leg_depth"], spec.start, spec.end, image_size
    )

    predictions = np.empty((spec.length, ACTION_DIM), dtype=np.float32)
    state = None
    with torch.no_grad():
        for t in range(spec.length):
            tensors = [
                torch.from_numpy(value[t : t + 1]).to(device)
                for value in (rgb, normalized_joints, sock, leg)
            ]
            _, predicted, _, _, state, _ = model(*tensors, state)
            predicted_np = predicted[0].detach().cpu().numpy()
            predictions[t] = denormalize(predicted_np, low, high)[:ACTION_DIM]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, predictions)
    return output


def _checkpoint_map(path: Optional[Path]) -> dict[str, Path]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(name): Path(checkpoint) for name, checkpoint in payload.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shareset-src", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-map", type=Path)
    parser.add_argument("--require-crossfit", action="store_true")
    parser.add_argument("--dataset-name", default="data_center_general_depth_n5_sock")
    parser.add_argument("--rec-dim", type=int, default=50)
    parser.add_argument("--k-dim", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specs = load_manifest(args.data_root, args.manifest, args.dataset_name)
    mapping = _checkpoint_map(args.checkpoint_map)
    if args.require_crossfit:
        missing = [spec.name for spec in specs if spec.split == "train" and spec.name not in mapping]
        distinct = {str(path.resolve()) for path in mapping.values()}
        if missing or len(distinct) < 2:
            raise ValueError(
                "cross-fitting requires every train episode mapped and at least two checkpoints; "
                f"missing={missing}, checkpoints={len(distinct)}"
            )
    if not mapping and args.checkpoint is None:
        raise ValueError("provide --checkpoint or --checkpoint-map")

    device = torch.device(args.device)
    cache: dict[Path, torch.nn.Module] = {}
    metadata: dict[str, dict[str, str]] = {}
    for spec in specs:
        checkpoint = mapping.get(spec.name, args.checkpoint)
        if checkpoint is None:
            raise ValueError(f"no checkpoint assigned to {spec.name}")
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint not in cache:
            cache[checkpoint] = _load_model(
                args.shareset_src.resolve(),
                checkpoint,
                rec_dim=args.rec_dim,
                k_dim=args.k_dim,
                image_size=IMAGE_SIZE,
                device=device,
            )
        destination = prediction_path(args.output_root, spec)
        export_episode(
            spec, cache[checkpoint], args.stats, destination, device=device
        )
        metadata[f"{spec.split}/{spec.name}"] = {
            "checkpoint": str(checkpoint),
            "prediction": str(destination.resolve()),
        }
        print(destination)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
