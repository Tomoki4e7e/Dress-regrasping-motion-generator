"""Episode directory layout matching Step 0 I/O contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Online / training crop contract (W x H), ndarray indexing is [:H, :W].
CROP_WIDTH = 1280
CROP_HEIGHT = 960

MASK_FOREGROUND = 255

# Relative paths under an episode root.
REL_RGB = Path("camera_right")
REL_SOCK_MASK = Path("camera_right_mask") / "sock_mask"
REL_LEG_MASK = Path("camera_right_mask") / "leg_mask"
REL_CAMERA_DEPTH = Path("camera_depth")
REL_SOCK_DEPTH = Path("depth_mask") / "sock_depth"
REL_LEG_DEPTH = Path("depth_mask") / "leg_depth"


@dataclass(frozen=True)
class EpisodePaths:
    """Resolved paths for one ShareSet episode directory."""

    root: Path
    rgb: Path
    sock_mask: Path
    leg_mask: Path
    camera_depth: Path
    sock_depth: Path
    leg_depth: Path

    @classmethod
    def from_root(cls, episode_dir: Path | str) -> "EpisodePaths":
        root = Path(episode_dir).expanduser().resolve()
        return cls(
            root=root,
            rgb=root / REL_RGB,
            sock_mask=root / REL_SOCK_MASK,
            leg_mask=root / REL_LEG_MASK,
            camera_depth=root / REL_CAMERA_DEPTH,
            sock_depth=root / REL_SOCK_DEPTH,
            leg_depth=root / REL_LEG_DEPTH,
        )

    def mask_dir(self, object_name: str) -> Path:
        if object_name == "sock":
            return self.sock_mask
        if object_name in ("leg", "foot"):
            return self.leg_mask
        raise ValueError(f"unknown mask object: {object_name!r} (expected sock|leg|foot)")

    def depth_mask_dir(self, object_name: str) -> Path:
        if object_name == "sock":
            return self.sock_depth
        if object_name in ("leg", "foot"):
            return self.leg_depth
        raise ValueError(f"unknown depth object: {object_name!r} (expected sock|leg|foot)")
