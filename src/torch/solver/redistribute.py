"""Tangential velocity redistribution via ∇log θ.

After each MM step, redistribute points along the curve to equalize
KDE densities θ_i using the tangential velocity:

    w_i = -λ Π_i^⊤ ∇log θ_i

where Π_i^⊤ is the tangent projection at point i.

In the continuous limit this gives ρ_t = Δ_Γ ρ (heat equation on the
surface), smoothing the density toward a constant.

Reference:
    Deckelnick (1997): tangential velocity for parametrization control.
    Pan, Dong, Guo, Shi (arXiv:2508.02676): Fokker-Planck tangential velocity.
"""

import torch
from typing import Literal

from ..oriented_varifold.mass import (
    compute_kde_density,
    compute_masses,
    KERNEL_CONSTANTS,
    KERNEL_FUNCTIONS,
)
from ..math_utils.pairwise import compute_pairwise_kernel
from ..math_utils.curvature import compute_regularized_curvature
from ..math_utils.angles import wrap_angles


def _compute_density_log_gradient(
    positions: torch.Tensor,
    delta: float,
    kernel: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ∇log θ_i for tangential velocity.

    ∇θ_i = -(1/(N C_η δ²)) Σ_j ρ'(s_ij) (x_j - x_i) / |x_j - x_i|
    Self-interaction contributes 0 (ρ'(s) = 0 for s > 1).

    θ_i is computed WITHOUT the self-term to avoid artificially reducing
    CV(θ) and weakening the tangential force.

    All points contribute equally to the density regardless of coherence.
    Hidden boundary suppression is handled by the caller (update *= q_i).

    Args:
        positions: (N, n) point positions.
        delta: KDE bandwidth.
        kernel: Kernel name.

    Returns:
        grad_log_theta: (N, n) — ∇log θ_i = ∇θ_i / θ_i
        theta: (N,) — KDE density (self-term excluded)
    """
    N = positions.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]
    eta_fn = KERNEL_FUNCTIONS[kernel]

    pw = compute_pairwise_kernel(positions, delta, kernel)

    rho_prime = pw.rho_prime  # (N, N)
    eta_vals = eta_fn(pw.s)   # (N, N)

    # ∇θ_i = -(1/(N C_η δ²)) Σ_j ρ'(s_ij) · (x_j - x_i) / |x_j - x_i|
    grad_theta = -(rho_prime.unsqueeze(-1) * pw.unit_diff).sum(dim=1) / (
        N * C_eta * delta * delta
    )

    # θ_i (self-term excluded: diagonal has s > 1 so η(s) = 0)
    theta = eta_vals.sum(dim=1) / (N * C_eta * delta)

    # ∇log θ_i = ∇θ_i / θ_i
    grad_log_theta = grad_theta / theta.clamp_min(1e-30).unsqueeze(-1)

    return grad_log_theta, theta


def redistribute_points(
    positions: torch.Tensor,
    angles: torch.Tensor,
    delta: float,
    kernel: Literal[
        "wendland_c2", "biweight", "epanechnikov", "buet_rumpf_exp",
    ] = "wendland_c2",
    n_iters: int = 10,
    step_size: float = 0.01,
    tol: float = 1e-4,
    max_disp_ratio: float = 0.05,
    mass_tau: float = 0.0,
    delta_redist: float | None = None,
    coherence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Redistribute points to equalize KDE densities θ_i.

    Uses tangential velocity w = -λ Π^⊤ ∇log θ (density diffusion).
    Angles are corrected using BLM regularized mean curvature
    (Buet, Leonardi, Masnou 2017).

    Args:
        positions: (N, 2) point positions on a curve.
        angles: (N,) normal angles (tangent = (-sin θ, cos θ)).
        delta: KDE bandwidth for mass/curvature computation.
        kernel: Kernel name.
        n_iters: Max gradient descent iterations.
        step_size: Learning rate λ for tangential velocity.
        tol: Convergence threshold on CV(θ) = std(θ)/mean(θ).
        max_disp_ratio: Max displacement per step as fraction of δ_redist.
        mass_tau: Cutoff threshold for mass computation.
        delta_redist: Bandwidth for redistribution density (None = use delta).
        coherence: (N,) coherence values. If provided, tangential velocity is
            scaled by q_i so that hidden boundary points (q ≈ 0) don't move.

    Returns:
        (positions (N,2), angles (N,), info dict with diagnostics)
    """
    N = positions.shape[0]
    tangents = torch.stack(
        [-torch.sin(angles), torch.cos(angles)], dim=1
    )  # (N, 2)
    pos = positions.clone()

    d_redist = delta_redist if delta_redist is not None else delta
    max_disp = max_disp_ratio * d_redist

    # Diagnostics
    cv_history = []

    actual_iters = 0
    for it in range(n_iters):
        grad_log_theta, theta = _compute_density_log_gradient(
            pos, d_redist, kernel,
        )

        # Convergence check: CV(θ) = std(θ) / mean(θ)
        cv = theta.std().item() / (theta.mean().item() + 1e-30)
        cv_history.append(cv)
        if cv < tol:
            break

        # Tangential velocity: w = -(∇log θ · τ) τ
        w_tan = -(grad_log_theta * tangents).sum(
            dim=1, keepdim=True
        ) * tangents  # (N, 2)

        # Update (λ = step_size)
        update = step_size * w_tan

        # Suppress hidden boundary points (q ≈ 0 → no redistribution)
        if coherence is not None:
            update = update * coherence.unsqueeze(-1)

        # Displacement clipping (safety valve)
        disp_max = update.norm(dim=1).max()
        if disp_max.item() > max_disp:
            update = update * (max_disp / disp_max)

        pos = pos + update
        actual_iters = it + 1

    # Final CV
    _, theta_final = _compute_density_log_gradient(pos, d_redist, kernel)
    cv_final = theta_final.std().item() / (theta_final.mean().item() + 1e-30)
    cv_history.append(cv_final)

    info = {
        'cv_history': cv_history,
        'n_iters': actual_iters,
        'converged': actual_iters < n_iters,
    }

    # --- Curvature-based angle correction (BLM regularized curvature) ---
    normals = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    masses = compute_masses(positions, delta, mass_tau, kernel)

    kappa, _ = compute_regularized_curvature(
        positions, normals, masses, epsilon=delta, kernel=kernel,
    )

    # Tangential displacement from redistribution
    u_tan = ((pos - positions) * tangents).sum(dim=1)  # (N,)

    # Angle correction: Δθ = κ · u_tan  (Frenet relation)
    new_angles = wrap_angles(angles + kappa * u_tan)

    return pos.detach(), new_angles.detach(), info
