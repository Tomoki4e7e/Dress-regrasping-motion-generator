"""Read ShareSet mask / depth frames without modifying ShareSet code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
from PIL import Image

from .paths import (
    CROP_HEIGHT,
    CROP_WIDTH,
    MASK_FOREGROUND,
    EpisodePaths,
)


def list_frame_files(directory: Path | str) -> list[str]:
    """Return ``*.png`` basenames sorted by integer stem (``0.png``, ``1.png``, ...).

    Matches ``MaskDepthDataLoader`` sorting: ``key=lambda x: int(x[:-4])``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    names = [
        name
        for name in directory.iterdir()
        if name.is_file() and name.suffix.lower() == ".png"
    ]
    return sorted((p.name for p in names), key=lambda x: int(Path(x).stem))


def apply_mask_to_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    foreground: int = MASK_FOREGROUND,
) -> np.ndarray:
    """Core dual used offline and online: ``where(mask == foreground, depth, 0)``."""
    if mask.shape != depth.shape:
        raise ValueError(f"mask/depth shape mismatch: {mask.shape} vs {depth.shape}")
    return np.where(mask == foreground, depth, 0).astype(np.uint8)


def crop_to_contract(image: np.ndarray) -> np.ndarray:
    """Apply ``[:960, :1280]`` crop used by the online camera callback."""
    return image[:CROP_HEIGHT, :CROP_WIDTH]


def load_gray_u8(path: Path | str, *, crop: bool = True) -> np.ndarray:
    """Load a grayscale PNG as ``(H, W)`` uint8.

    Raises ``FileNotFoundError`` if missing, ``ValueError`` if unreadable.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as img:
        arr = np.array(img.convert("L"), dtype=np.uint8)
    if crop:
        arr = crop_to_contract(arr)
    return arr


@dataclass
class FrameBundle:
    """One frame's ShareSet visual outputs for downstream feature extraction."""

    frame_id: int
    filename: str
    sock_mask: Optional[np.ndarray] = None
    leg_mask: Optional[np.ndarray] = None
    camera_depth: Optional[np.ndarray] = None
    sock_depth: Optional[np.ndarray] = None
    leg_depth: Optional[np.ndarray] = None
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.missing) == 0

    @property
    def foot_mask(self) -> Optional[np.ndarray]:
        """Alias: online code names leg as foot."""
        return self.leg_mask

    @property
    def foot_depth(self) -> Optional[np.ndarray]:
        return self.leg_depth


class EpisodeReader:
    """Thin reader over one episode directory (Step 0 path / shape contract).

    Does not run GaussianBlur / resize — those belong to the training loader.
    Coverage / keypoints should use the arrays returned here.
    """

    def __init__(self, episode_dir: Path | str, *, crop: bool = True):
        self.paths = EpisodePaths.from_root(episode_dir)
        self.crop = crop
        self._frame_files = self._discover_frame_files()

    def _discover_frame_files(self) -> list[str]:
        # Prefer RGB listing (same as MaskDepthDataLoader), fall back to masks.
        for directory in (
            self.paths.rgb,
            self.paths.sock_mask,
            self.paths.leg_mask,
            self.paths.camera_depth,
        ):
            files = list_frame_files(directory)
            if files:
                return files
        return []

    @property
    def episode_dir(self) -> Path:
        return self.paths.root

    @property
    def frame_files(self) -> list[str]:
        return list(self._frame_files)

    @property
    def frame_ids(self) -> list[int]:
        return [int(Path(name).stem) for name in self._frame_files]

    def __len__(self) -> int:
        return len(self._frame_files)

    def filename_for(self, frame_id: int) -> str:
        return f"{frame_id}.png"

    def _load_optional(self, path: Path) -> tuple[Optional[np.ndarray], bool]:
        if not path.is_file():
            return None, False
        return load_gray_u8(path, crop=self.crop), True

    def load_mask(self, object_name: str, frame_id: int) -> np.ndarray:
        path = self.paths.mask_dir(object_name) / self.filename_for(frame_id)
        return load_gray_u8(path, crop=self.crop)

    def load_camera_depth(self, frame_id: int) -> np.ndarray:
        path = self.paths.camera_depth / self.filename_for(frame_id)
        return load_gray_u8(path, crop=self.crop)

    def load_depth_mask(self, object_name: str, frame_id: int) -> np.ndarray:
        path = self.paths.depth_mask_dir(object_name) / self.filename_for(frame_id)
        return load_gray_u8(path, crop=self.crop)

    def compute_depth_mask(self, object_name: str, frame_id: int) -> np.ndarray:
        """Recompute ``where(mask==255, depth, 0)`` from raw mask + camera_depth."""
        mask = self.load_mask(object_name, frame_id)
        depth = self.load_camera_depth(frame_id)
        return apply_mask_to_depth(mask, depth)

    def load_frame(
        self,
        frame_id: int,
        *,
        require: Sequence[str] = ("sock_mask", "leg_mask", "camera_depth", "sock_depth", "leg_depth"),
    ) -> FrameBundle:
        """Load available modalities for ``frame_id``; record missing keys in ``missing``."""
        filename = self.filename_for(frame_id)
        missing: list[str] = []

        sock_mask, sock_mask_ok = self._load_optional(self.paths.sock_mask / filename)
        leg_mask, leg_mask_ok = self._load_optional(self.paths.leg_mask / filename)
        camera_depth, depth_ok = self._load_optional(self.paths.camera_depth / filename)
        sock_depth, sock_depth_ok = self._load_optional(self.paths.sock_depth / filename)
        leg_depth, leg_depth_ok = self._load_optional(self.paths.leg_depth / filename)

        status = {
            "sock_mask": sock_mask_ok,
            "leg_mask": leg_mask_ok,
            "camera_depth": depth_ok,
            "sock_depth": sock_depth_ok,
            "leg_depth": leg_depth_ok,
        }
        for key in require:
            if key not in status:
                raise ValueError(f"unknown require key: {key!r}")
            if not status[key]:
                missing.append(key)

        return FrameBundle(
            frame_id=frame_id,
            filename=filename,
            sock_mask=sock_mask,
            leg_mask=leg_mask,
            camera_depth=camera_depth,
            sock_depth=sock_depth,
            leg_depth=leg_depth,
            missing=missing,
        )

    def iter_frames(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        *,
        require: Sequence[str] = ("sock_mask", "leg_mask", "camera_depth", "sock_depth", "leg_depth"),
    ) -> Iterator[FrameBundle]:
        """Iterate frames.

        ``start`` / ``end`` are list indices into the sorted frame list (yaml-style),
        not frame_id values. ``end`` is exclusive, same as yaml end usage with slicing.
        """
        files = self._frame_files
        if start is None:
            start = 0
        if end is None:
            end = len(files)
        for name in files[start:end]:
            frame_id = int(Path(name).stem)
            yield self.load_frame(frame_id, require=require)
