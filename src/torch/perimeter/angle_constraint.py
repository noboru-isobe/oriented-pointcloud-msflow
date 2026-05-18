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

from src.torch.perimeter.coherence_perimeter import mollifier_2d


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
    """Compute kernel gradient ∇ψ_σ(z) = (1/σ³) η'(u) z/r.

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

    return (sigma ** -3) * (eta_prime * mask_r).unsqueeze(-1) * z_hat


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
    # Pairwise differences: diff[ℓ, i] = x_i - x_ℓ
    diff = positions[None, :, :] - positions[:, None, :]  # (N, N, 2)

    # Effective mass
    m_eff = masses * coherence  # (N,)

    # A_{ℓi} = m_eff_i ψ(x_i - x_ℓ)
    psi = mollifier_2d(diff, sigma, kernel)  # (N, N)
    A = m_eff[None, :] * psi  # (N, N)

    # B_{ℓi} = m_eff_i (t_i · ∇ψ(x_i - x_ℓ))
    grad = compute_kernel_gradient(diff, sigma, kernel)  # (N, N, 2)
    t_dot_grad = (tangents[None, :, :] * grad).sum(dim=-1)  # (N, N)
    B = m_eff[None, :] * t_dot_grad  # (N, N)

    return A, B
