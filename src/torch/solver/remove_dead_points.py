"""Remove dead points from oriented point cloud varifold.

Dead points are identified by their effective mass m_i * q_i,
which corresponds to their contribution to the perimeter P = Σ m_i q_i.
Points with negligible effective mass are physically irrelevant and
can be removed to allow curve topology changes (e.g., fusion).
"""

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold


def remove_dead_points(
    varifold: OrientedPointCloudVarifold,
    masses: torch.Tensor,
    coherence: torch.Tensor,
    threshold: float = 1e-4,
) -> tuple[OrientedPointCloudVarifold, torch.Tensor]:
    """Remove points with negligible effective mass m_i * q_i.

    Args:
        varifold: Current oriented point cloud.
        masses: (N,) mass values from KDE.
        coherence: (N,) coherence values.
        threshold: Remove points where m_i*q_i < threshold * max(m*q).

    Returns:
        (new_varifold, keep_mask) where keep_mask is (N,) bool tensor.
    """
    effective_mass = masses * coherence
    max_eff = effective_mass.max()

    if max_eff <= 0:
        # All points are dead — keep them all to avoid empty varifold
        keep = torch.ones(varifold.n_points, dtype=torch.bool,
                          device=varifold.positions.device)
        return varifold, keep

    keep = effective_mass >= threshold * max_eff

    if keep.all():
        return varifold, keep

    new_varifold = OrientedPointCloudVarifold(
        positions=varifold.positions[keep],
        angles=varifold.angles[keep],
    )
    return new_varifold, keep
