"""Force-aware residual flow policies for sock dressing.

The package is intentionally independent from ShareSet: ShareSet remains a
read-only source of demonstrations and frozen base-policy components.
"""

from .contracts import ACTION_DIM, EXTERNAL_TORQUE_DIM, ResidualSample

__all__ = ["ACTION_DIM", "EXTERNAL_TORQUE_DIM", "ResidualSample"]
