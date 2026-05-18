"""Coherence-based perimeter estimation for oriented point cloud varifolds.

Implements the boundary-point-only perimeter estimation using local normal coherence:

    P̂_σ(μ) = Σ_i m_i q_i

where:
    U_σ(x_i) = Σ_j m_j ρ_σ(x_i - x_j)     (scalar density)
    V_σ(x_i) = Σ_j m_j ρ_σ(x_i - x_j) n_j  (vector field)
    q_i = |V_σ(x_i)| / U_σ(x_i)            (coherence ∈ [0,1])

Key properties:
- Hidden boundaries (opposing normals) have q ≈ 0 → not counted
- Normal boundaries have q ≈ 1 → fully counted
- No 2D domain integration required (boundary points only)
- AD-compatible (smooth polynomial kernels)
"""

import math
import torch
from typing import Literal

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import _SafeKernelEval


# =============================================================================
# 2D Mollifier normalization constants
# =============================================================================

# For 2D mollifier: ρ_σ(z) = (C_2D / σ²) η(|z|/σ)
# where η is the 1D kernel from mass.py
# C_2D is chosen so that ∫_{ℝ²} ρ_σ = 1
MOLLIFIER_2D_CONSTANTS = {
    "wendland_c2": 7.0 / math.pi,
    "biweight": 3.0 / math.pi,
    "epanechnikov": 2.0 / math.pi,
}


def mollifier_2d(
    z: torch.Tensor,
    sigma: float,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> torch.Tensor:
    """Compute 2D mollifier values.

    ρ_σ(z) = (C_2D / σ²) η(|z|/σ)

    where η is the 1D kernel and C_2D ensures ∫ρ_σ = 1.

    Args:
        z: (..., 2) displacement vectors
        sigma: bandwidth parameter
        kernel: kernel function name

    Returns:
        (...) mollifier values
    """
    C_2D = MOLLIFIER_2D_CONSTANTS[kernel]
    kernel_vals = _SafeKernelEval.apply(z, sigma, kernel)
    return (C_2D / (sigma * sigma)) * kernel_vals


# =============================================================================
# Core computation functions
# =============================================================================

def compute_scalar_density(
    positions: torch.Tensor,
    masses: torch.Tensor,
    sigma: float,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
    backend: Literal["naive", "keops"] = "naive",
) -> torch.Tensor:
    """Compute scalar density at each boundary point.

    U_σ(x_i) = Σ_j m_j ρ_σ(x_i - x_j)

    Args:
        positions: (N, 2) point positions
        masses: (N,) point masses
        sigma: mollifier bandwidth
        kernel: kernel function name
        backend: ``"naive"`` (default; full N×N via mollifier_2d /
            _SafeKernelEval) or ``"keops"`` (lazy pairwise, O(N)
            memory). See ``src/torch/math_utils/keops_pairwise.py``.

    Returns:
        (N,) scalar density at each point
    """
    if backend == "keops":
        from src.torch.math_utils.keops_pairwise import scalar_density_keops
        return scalar_density_keops(positions, masses, sigma, kernel)

    # Pairwise displacements: z_ij = x_i - x_j
    z = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)

    # Mollifier values
    K = mollifier_2d(z, sigma, kernel)  # (N, N)

    # Weighted sum: U_i = Σ_j m_j K_ij
    U = (K * masses.unsqueeze(0)).sum(dim=1)  # (N,)

    return U


def compute_vector_field(
    positions: torch.Tensor,
    normals: torch.Tensor,
    masses: torch.Tensor,
    sigma: float,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
    backend: Literal["naive", "keops"] = "naive",
) -> torch.Tensor:
    """Compute vector field at each boundary point.

    V_σ(x_i) = Σ_j m_j ρ_σ(x_i - x_j) n_j

    Args:
        positions: (N, 2) point positions
        normals: (N, 2) unit normal vectors
        masses: (N,) point masses
        sigma: mollifier bandwidth
        kernel: kernel function name
        backend: ``"naive"`` (default) or ``"keops"`` — see
            ``compute_scalar_density`` for details.

    Returns:
        (N, 2) vector field at each point
    """
    if backend == "keops":
        from src.torch.math_utils.keops_pairwise import vector_field_keops
        return vector_field_keops(positions, normals, masses, sigma, kernel)

    # Pairwise displacements: z_ij = x_i - x_j
    z = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)

    # Mollifier values
    K = mollifier_2d(z, sigma, kernel)  # (N, N)

    # Weighted sum: V_i = Σ_j m_j K_ij n_j
    weights = K * masses.unsqueeze(0)  # (N, N)
    V = torch.einsum("ij,jd->id", weights, normals)  # (N, 2)

    return V


def compute_coherence(
    U: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Compute local coherence (visibility) at each point.

    q_i = |V_i| / U_i

    - q ≈ 1 for normal boundaries (aligned normals)
    - q ≈ 0 for hidden boundaries (opposing normals cancel)
    - q = 0 for isolated points (U_i = 0)

    When U_i → 0, we have |V_i| → 0 as well (no nearby points contribute).
    We define the limit |V|/U → 0, meaning isolated points don't contribute
    to the perimeter.

    Args:
        U: (N,) scalar density
        V: (N, 2) vector field

    Returns:
        (N,) coherence values in [0, 1]
    """
    V_norm = V.norm(dim=-1)  # (N,)
    # U_safe prevents 0/0 in backward: torch.where evaluates both branches'
    # gradients, so V_norm / U with U=0 produces NaN even though the forward
    # value is masked to 0.
    U_safe = torch.where(U > 0, U, torch.ones_like(U))
    return torch.where(U > 0, V_norm / U_safe, torch.zeros_like(U))


# =============================================================================
# Parameter selection
# =============================================================================

def compute_recommended_sigma(
    positions: torch.Tensor,
    c_sigma: float = 3.0,
) -> float:
    """Compute recommended σ based on nearest neighbor distances.

    σ = c_σ × median(nearest neighbor distance)

    Args:
        positions: (N, 2) point positions
        c_sigma: multiplier (default 3.0, recommended range [2, 4])

    Returns:
        Recommended sigma value
    """
    # Pairwise distances
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)
    dist = diff.norm(dim=-1)  # (N, N)

    # Set diagonal to inf to exclude self-distance
    N = positions.shape[0]
    dist = dist + torch.eye(N, device=positions.device, dtype=positions.dtype) * 1e10

    # Nearest neighbor distance for each point
    nn_dist, _ = dist.min(dim=1)  # (N,)

    # Median NN distance
    median_nn = nn_dist.median().item()

    return c_sigma * median_nn


# =============================================================================
# Main perimeter estimation
# =============================================================================

def compute_perimeter_coherence(
    varifold: OrientedPointCloudVarifold,
    masses: torch.Tensor,
    sigma: float | None = None,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
    c_sigma: float = 3.0,
    backend: Literal["naive", "keops"] = "naive",
) -> torch.Tensor:
    """Compute perimeter using coherence-based estimation.

    P̂_σ(μ) = Σ_i m_i q_i

    where q_i = |V_σ(x_i)| / U_σ(x_i) is the local coherence.

    Hidden boundaries (opposing normals) automatically have q ≈ 0
    and don't contribute to the perimeter.

    Args:
        varifold: Oriented point cloud varifold
        masses: (N,) point masses
        sigma: mollifier bandwidth (None for auto-compute)
        kernel: kernel function name
        c_sigma: multiplier for auto σ computation
        backend: ``"naive"`` (default) or ``"keops"``. Propagated to
            ``compute_scalar_density`` and ``compute_vector_field``.

    Returns:
        Scalar perimeter estimate (differentiable)
    """
    positions = varifold.positions
    normals = varifold.normals

    # Auto-compute sigma if not provided
    if sigma is None:
        sigma = compute_recommended_sigma(positions, c_sigma)

    # Compute scalar density and vector field
    U = compute_scalar_density(positions, masses, sigma, kernel, backend=backend)
    V = compute_vector_field(positions, normals, masses, sigma, kernel, backend=backend)

    # Compute coherence
    q = compute_coherence(U, V)  # (N,)

    # Perimeter = Σ m_i q_i
    perimeter = (masses * q).sum()

    return perimeter


def compute_perimeter_coherence_with_details(
    varifold: OrientedPointCloudVarifold,
    masses: torch.Tensor,
    sigma: float | None = None,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
    c_sigma: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Compute perimeter with intermediate values for analysis.

    Returns:
        Tuple of (perimeter, U, V, q, sigma_used)
    """
    positions = varifold.positions
    normals = varifold.normals

    if sigma is None:
        sigma = compute_recommended_sigma(positions, c_sigma)

    U = compute_scalar_density(positions, masses, sigma, kernel)
    V = compute_vector_field(positions, normals, masses, sigma, kernel)
    q = compute_coherence(U, V)
    perimeter = (masses * q).sum()

    return perimeter, U, V, q, sigma
