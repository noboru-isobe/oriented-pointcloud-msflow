"""Angle utilities for oriented point cloud varifolds."""

import math
import torch


def wrap_angles(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-π, π).

    Uses torch.remainder which always returns non-negative values,
    ensuring the result is in [-π, π).

    Args:
        angles: Tensor of angles in radians.

    Returns:
        Tensor of angles wrapped to [-π, π).
    """
    return torch.remainder(angles + math.pi, 2 * math.pi) - math.pi
