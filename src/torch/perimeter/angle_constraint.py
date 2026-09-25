"""Weak form angle constraint for oriented point cloud varifolds.

Implements the weak form constraint for Δθ = -∂_τ s:

    ∫ Δθ φ_ℓ dλ = ∫ s (t · ∇φ_ℓ) dλ

Discretized with test functions φ_ℓ(y) = ψ_σ(y - x_ℓ):
    A @ Δθ = B @ s

where:
    A_{ℓi} = m_eff_i ψ_σ(x_i - x_ℓ)
    B_{ℓi} = m_eff_i (t_i · ∇ψ_σ(x_i - x_ℓ))

and m_eff = m * q (effective mass with coherence weighting).
"""

import torch
from typing import Literal

from src.torch.perimeter.coherence_perimeter import (
    MOLLIFIER_2D_CONSTANTS,
    mollifier_2d,
)


# Kernel derivatives η'(u) for gradient computation
KERNEL_ETA_PRIME = {
    "wendland_c2": lambda u: -20.0 * u * (1.0 - u).clamp(min=0.0) ** 3,
    "biweight": lambda u: -4.0 * u * (1.0 - u * u).clamp(min=0.0),
    "epanechnikov": lambda u: -2.0 * u,
}


def compute_kernel_gradient(
    diff: torch.Tensor,
    sigma: float,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> torch.Tensor:
    """Compute kernel gradient ∇ψ_σ(z) = (C_2D/σ³) η'(u) z/r.

    This is the EXACT gradient of the normalized mollifier
    ψ_σ(z) = (C_2D/σ²) η(|z|/σ) that `mollifier_2d` evaluates -- the
    same C_2D must appear here, or any weak form pairing ψ (A side)
    with ∇ψ (B side) picks up a spurious uniform factor 1/C_2D.
    That bug shipped: the angle map A^†B rotated normals by
    1/C_2D = π/7 ≈ 0.449 of the geometric rotation (measured on the
    unit circle: factor 0.4461 at every wavenumber, shape residual
    1e-13), so stored angles secularly lagged the evolving curve.

    Args:
        diff: (..., 2) displacement vectors z
        sigma: kernel bandwidth
        kernel: kernel function name

    Returns:
        (..., 2) gradient vectors ∇ψ_σ(z)
    """
    r = diff.norm(dim=-1)  # (...)
    u = r / sigma
    mask = (u < 1.0).to(diff.dtype)

    eta_prime = KERNEL_ETA_PRIME[kernel](u) * mask

    # z/r unit vector (handle r=0)
    mask_r = (r > 1e-10).to(diff.dtype)
    r_safe = torch.where(r > 1e-10, r, torch.ones_like(r))
    z_hat = diff / r_safe.unsqueeze(-1)

    C_2D = MOLLIFIER_2D_CONSTANTS[kernel]
    return (C_2D * sigma ** -3) * (eta_prime * mask_r).unsqueeze(-1) * z_hat


def compute_angle_constraint_matrices(
    positions: torch.Tensor,
    tangents: torch.Tensor,
    masses: torch.Tensor,
    coherence: torch.Tensor,
    sigma: float,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute A, B matrices for weak form angle constraint: A @ Δθ = B @ s.

    Weak form of Δθ = -∂_τ s discretized with test functions φ_ℓ(y) = ψ_σ(y - x_ℓ):
        Σ_i m_eff_i Δθ_i ψ(x_i - x_ℓ) = Σ_i m_eff_i s_i (t_i · ∇ψ(x_i - x_ℓ))

    where m_eff = m * q (effective mass with coherence weighting).

    Args:
        positions: (N, 2) particle positions
        tangents: (N, 2) unit tangent vectors
        masses: (N,) particle masses
        coherence: (N,) coherence values for effective mass weighting
        sigma: kernel bandwidth
        kernel: kernel function name

    Returns:
        A: (N, N) matrix, A_{ℓi} = m_eff_i ψ(x_i - x_ℓ)
        B: (N, N) matrix, B_{ℓi} = m_eff_i (t_i · ∇ψ(x_i - x_ℓ))
    """
    # delegate to the explicit-weight builder (0L-theta: the actual
    # measure weight is stated once; visible_full == m * q here)
    return compute_angle_constraint_matrices_from_weights(
        positions, tangents, masses * coherence, sigma, kernel)


def compute_angle_constraint_matrices_from_weights(
    positions: torch.Tensor,
    tangents: torch.Tensor,
    angle_weights: torch.Tensor,
    sigma: float,
    kernel: Literal["wendland_c2", "biweight",
                    "epanechnikov"] = "wendland_c2",
) -> tuple[torch.Tensor, torch.Tensor]:
    """0L-theta explicit-measure builder: A_{li} = w_i psi(x_i-x_l),
    B_{li} = w_i t_i . grad-psi(x_i-x_l) with the measure weight
    w^angle stated by the caller -- visible_full uses w = m q^full;
    raw_loopwise uses the RAW loopwise mass w = m~^loop (the visible
    weight injects the spurious kinematic term -s d_tau log q on a
    nonuniform-q loop and carries a cross-loop dependence through
    q^full even under a same-loop kernel mask)."""
    diff = positions[None, :, :] - positions[:, None, :]  # (N, N, 2)
    psi = mollifier_2d(diff, sigma, kernel)               # (N, N)
    A = angle_weights[None, :] * psi
    grad = compute_kernel_gradient(diff, sigma, kernel)   # (N, N, 2)
    t_dot_grad = (tangents[None, :, :] * grad).sum(dim=-1)
    B = angle_weights[None, :] * t_dot_grad
    return A, B
