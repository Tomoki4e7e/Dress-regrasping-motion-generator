"""Coverage from leg-mask area decrease (Step 2).

Sock slack can inflate sock-mask area, so coverage is derived from how much
visible leg area shrinks relative to an episode baseline.
"""

from .compute import (
    CoverageResult,
    EpisodeCoverage,
    compute_episode_coverage,
    coverage_from_areas,
    foreground_area,
    save_episode_coverage,
)

__all__ = [
    "CoverageResult",
    "EpisodeCoverage",
    "compute_episode_coverage",
    "coverage_from_areas",
    "foreground_area",
    "save_episode_coverage",
]
