"""Perimeter computation module.

Implements coherence-based perimeter estimation for oriented point cloud varifolds:

    P̂_σ(μ) = Σ_i m_i q_i

where q_i = |V_σ(x_i)| / U_σ(x_i) is the local coherence.

Key properties:
- Hidden boundaries (opposing normals) have q ≈ 0 → not counted
- Normal boundaries have q ≈ 1 → fully counted
- No 2D domain integration required (boundary points only)
- AD-compatible (smooth polynomial kernels)
"""

from .coherence_perimeter import (
    mollifier_2d,
    compute_scalar_density,
    compute_vector_field,
    compute_coherence,
    compute_recommended_sigma,
    compute_perimeter_coherence,
    compute_perimeter_coherence_with_details,
    MOLLIFIER_2D_CONSTANTS,
)

__all__ = [
    "mollifier_2d",
    "compute_scalar_density",
    "compute_vector_field",
    "compute_coherence",
    "compute_recommended_sigma",
    "compute_perimeter_coherence",
    "compute_perimeter_coherence_with_details",
    "MOLLIFIER_2D_CONSTANTS",
]
