"""Arc length parametrization for uniform point sampling on curves."""

import torch
import math
from typing import Tuple, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .curves import ParametricCurve
    from src.torch.oriented_varifold import OrientedPointCloudVarifold


def compute_arc_length_cumulative(
    speed_fn,
    n_integration_points: int = 2048,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute cumulative arc length using trapezoidal rule.

    Args:
        speed_fn: function |γ'(t)| returning speed at parameter t
        n_integration_points: number of integration grid points
        device: torch device
        dtype: torch dtype

    Returns:
        t_grid: parameter values [0, 2π]
        s_cumulative: cumulative arc length at each t
        total_length: total arc length L
    """
    t_grid = torch.linspace(0, 2 * math.pi, n_integration_points, device=device, dtype=dtype)
    speed = speed_fn(t_grid)
    dt = t_grid[1] - t_grid[0]

    # Trapezoidal rule: ds = (speed[i] + speed[i+1]) / 2 * dt
    ds = (speed[:-1] + speed[1:]) / 2 * dt
    s_cumulative = torch.cat([torch.zeros(1, device=device, dtype=dtype), torch.cumsum(ds, dim=0)])

    return t_grid, s_cumulative, s_cumulative[-1].item()


def verify_arc_length_with_torchquad(
    speed_fn,
    n_points: int = 1001,
    device: str = "cpu",
) -> float:
    """
    Verify total arc length using torchquad Simpson rule.

    Args:
        speed_fn: function |γ'(t)| returning speed at parameter t
        n_points: number of integration points for Simpson rule
        device: torch device

    Returns:
        total_length: total arc length computed by Simpson rule
    """
    from torchquad import Simpson

    # Wrapper to ensure correct shape for torchquad
    def integrand(t):
        # torchquad passes t as shape (N, 1), we need to squeeze and unsqueeze
        t_squeezed = t.squeeze(-1)
        result = speed_fn(t_squeezed)
        return result.unsqueeze(-1)

    simpson = Simpson()
    integration_domain = torch.tensor([[0.0, 2 * math.pi]], device=device)
    result = simpson.integrate(integrand, dim=1, N=n_points, integration_domain=integration_domain)
    return result.item()


def find_parameters_for_arc_lengths(
    t_grid: torch.Tensor,
    s_cumulative: torch.Tensor,
    target_arc_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Find parameter values t for given target arc lengths s.

    Uses searchsorted + linear interpolation to invert s(t).

    Args:
        t_grid: parameter grid [0, 2π]
        s_cumulative: cumulative arc length at each t
        target_arc_lengths: desired arc lengths to find parameters for

    Returns:
        t_values: parameter values corresponding to target arc lengths
    """
    # searchsorted finds the index where target would be inserted
    # side='right' means we find the first index where s_cumulative > target
    indices = torch.searchsorted(s_cumulative, target_arc_lengths, side='right')

    # Clamp indices to valid range for interpolation
    indices = torch.clamp(indices, 1, len(s_cumulative) - 1)

    # Get bracketing values for interpolation
    s_left = s_cumulative[indices - 1]
    s_right = s_cumulative[indices]
    t_left = t_grid[indices - 1]
    t_right = t_grid[indices]

    # Linear interpolation: t = t_left + (t_right - t_left) * (s - s_left) / (s_right - s_left)
    # Avoid division by zero
    ds = s_right - s_left
    ds = torch.where(ds > 0, ds, torch.ones_like(ds))
    alpha = (target_arc_lengths - s_left) / ds
    alpha = torch.clamp(alpha, 0.0, 1.0)

    t_values = t_left + alpha * (t_right - t_left)
    return t_values


def sample_curve_arc_length(
    curve: "ParametricCurve",
    n_points: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    n_integration_points: int = 2048,
) -> "OrientedPointCloudVarifold":
    """
    Sample points from a parametric curve using arc length parametrization.

    Points are distributed uniformly along the curve's arc length,
    ensuring equal spacing regardless of the curve's parametric speed.

    Args:
        curve: ParametricCurve with speed_fn defined
        n_points: number of points to sample
        device: torch device
        dtype: torch dtype
        n_integration_points: grid size for arc length computation

    Returns:
        OrientedPointCloudVarifold with uniformly-spaced points
    """
    from src.torch.oriented_varifold import OrientedPointCloudVarifold

    if curve.speed_fn is None:
        raise ValueError(f"Curve '{curve.name}' does not have speed_fn defined for arc length sampling")

    # Compute cumulative arc length
    t_grid, s_cumulative, total_length = compute_arc_length_cumulative(
        curve.speed_fn, n_integration_points, device, dtype
    )

    # Target arc lengths: equally spaced from 0 to L (excluding endpoint for closed curve)
    target_s = torch.linspace(0, total_length, n_points + 1, device=device, dtype=dtype)[:-1]

    # Find parameter values for target arc lengths
    t_values = find_parameters_for_arc_lengths(t_grid, s_cumulative, target_s)

    # Sample curve at these parameters
    x, y = curve.position_fn(t_values)
    positions = torch.stack([x, y], dim=1)
    angles = curve.normal_angle_fn(t_values)

    return OrientedPointCloudVarifold(positions=positions, angles=angles)


def sample_curve_mass_uniform(
    curve: "ParametricCurve",
    n_points: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    n_integration_points: int = 2048,
    max_iter: int = 20,
    alpha: float = 0.5,
    tol: float = 0.05,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> "OrientedPointCloudVarifold":
    """
    Sample points from a parametric curve with mass-uniform distribution.

    Points are distributed such that the KDE-based mass m_i is approximately
    uniform across all points. This prevents points with small mass from
    being "left behind" during mean curvature flow.

    Algorithm (Fixed-Point Iteration):
    1. Start with arc-length uniform distribution
    2. Compute w_i = χ_τ(θ_i) / θ_i at each point
    3. Check convergence: max(|w_i - w̄|) / w̄ < tol
    4. Update arc-length gaps:
       Δs_i^new = Δs_i^old × (w̄ / w_i)^α
       - w_i < w̄ (dense, θ large) → spread out
       - w_i > w̄ (sparse, θ small) → cluster
    5. Normalize: Σ Δs_i = L (total perimeter)
    6. Recompute point positions
    7. Repeat until convergence

    Args:
        curve: ParametricCurve with speed_fn defined
        n_points: number of points to sample
        device: torch device
        dtype: torch dtype
        n_integration_points: grid size for arc length computation
        max_iter: maximum number of fixed-point iterations
        alpha: relaxation coefficient (0 < alpha <= 1)
        tol: convergence tolerance for w_i uniformity
        kernel: kernel function for KDE

    Returns:
        OrientedPointCloudVarifold with mass-uniform points
    """
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.oriented_varifold.mass import (
        compute_kde_density,
        compute_recommended_params,
        chi_tau,
    )

    if curve.speed_fn is None:
        raise ValueError(
            f"Curve '{curve.name}' does not have speed_fn defined for mass-uniform sampling"
        )

    # Compute cumulative arc length table for inversion
    t_grid, s_cumulative, total_length = compute_arc_length_cumulative(
        curve.speed_fn, n_integration_points, device, dtype
    )

    # Initialize with arc-length uniform distribution
    # delta_s[i] = arc length from point i to point i+1
    delta_s = torch.full(
        (n_points,), total_length / n_points, device=device, dtype=dtype
    )

    for iteration in range(max_iter):
        # Compute cumulative arc lengths for current gaps
        cumsum_s = torch.cat([torch.zeros(1, device=device, dtype=dtype), delta_s.cumsum(dim=0)])
        target_s = cumsum_s[:-1]  # (n_points,) starting positions

        # Find parameter values and sample curve
        t_values = find_parameters_for_arc_lengths(t_grid, s_cumulative, target_s)
        x, y = curve.position_fn(t_values)
        positions = torch.stack([x, y], dim=1)

        # Compute recommended delta and tau
        delta, tau = compute_recommended_params(positions, kernel=kernel)

        # Compute KDE density and w_i = χ_τ(θ_i) / θ_i
        theta = compute_kde_density(positions, delta, kernel=kernel)
        chi = chi_tau(theta, tau)

        # w_i = χ_τ(θ_i) / θ_i (proportional to mass)
        # Safe division: when chi > 0, theta >= tau/2 > 0
        w = torch.where(chi > 0, chi / theta, torch.zeros_like(theta))

        # Check for convergence
        w_mean = w.mean()
        if w_mean <= 0:
            # All points have zero weight, fallback to arc-length
            break

        relative_variation = (w - w_mean).abs().max() / w_mean

        if relative_variation < tol:
            break

        # Update arc-length gaps: Δs_i^new = Δs_i × (w̄ / w_i)^α
        # w_i < w̄ → ratio > 1 → increase gap (spread out)
        # w_i > w̄ → ratio < 1 → decrease gap (cluster)
        ratio = torch.where(w > 0, w_mean / w, torch.ones_like(w))
        delta_s = delta_s * (ratio ** alpha)

        # Normalize to preserve total length
        delta_s = delta_s * (total_length / delta_s.sum())

    # Final sampling
    cumsum_s = torch.cat([torch.zeros(1, device=device, dtype=dtype), delta_s.cumsum(dim=0)])
    target_s = cumsum_s[:-1]
    t_values = find_parameters_for_arc_lengths(t_grid, s_cumulative, target_s)

    x, y = curve.position_fn(t_values)
    positions = torch.stack([x, y], dim=1)
    angles = curve.normal_angle_fn(t_values)

    return OrientedPointCloudVarifold(positions=positions, angles=angles)
