"""Regularized mean curvature for point cloud varifolds.

Implements the BLM regularized mean curvature:
    H_ε^Π(x_i) = -(d/n) · (δV * ρ_ε(x_i)) / (‖V‖ · ξ_ε(x_i))

where (ρ, ξ) is a kernel pair satisfying n·ξ(s) = -s·ρ'(s).

Reference:
    B. Buet, G.P. Leonardi, S. Masnou,
    "A Varifold Approach to Surface Approximation",
    Arch. Ration. Mech. Anal. 226(2), 639-694 (2017).
"""

import torch

from .pairwise import compute_pairwise_kernel


def compute_regularized_curvature(
    positions: torch.Tensor,
    normals: torch.Tensor,
    masses: torch.Tensor,
    epsilon: float,
    kernel: str = "wendland_c2",
    loop_labels: torch.Tensor | None = None,
    return_denominator: bool = False,
):
    """Compute BLM regularized mean curvature.

    Uses fixed bandwidth ε and full N×N pairwise computation.
    The kernel pair (ρ, ξ) is derived from KERNEL_GRAD_RATIO:
        ψ(s) = ρ'(s)/s,  ρ'(s) = s·ψ(s),  ξ(s) = -s·ρ'(s)/n

    Args:
        positions: (N, 2) point positions.
        normals: (N, 2) unit outward normals.
        masses: (N,) point masses (effective masses recommended).
        epsilon: Fixed bandwidth for kernel evaluation.
        kernel: Kernel name (key in KERNEL_GRAD_RATIO).

    Returns:
        kappa: (N,) scalar curvature (positive for convex with outward normals).
        H: (N, 2) mean curvature vector.
    """
    n = positions.shape[-1]  # spatial dim
    d = n - 1                # intrinsic dim (codim-1 hypersurface)

    pw = compute_pairwise_kernel(positions, epsilon, kernel)
    s, u, rho_prime = pw.s, pw.unit_diff, pw.rho_prime
    xi = -s * rho_prime / n             # n·ξ = -s·ρ' → ξ = -s·ρ'/n
    if loop_labels is not None:
        # 0L-kappa: same-certified-loop estimation. The union-cloud
        # estimator SIGN-FLIPS in the contact layer (measured: disk
        # true +2.857 vs union -5.609 at endgame-#2 step 1325, while
        # the loopwise value is +2.855) because the projection uses
        # only the QUERY normal n_i -- the opposite antiparallel
        # sheet's contribution does not cancel. Third instance of the
        # cross-loop contamination family (0J coherence / 0K mass /
        # curvature). None keeps the historical path bitwise.
        same = (loop_labels.unsqueeze(1)
                == loop_labels.unsqueeze(0)).to(rho_prime.dtype)
        rho_prime = rho_prime * same
        xi = xi * same

    # Normal projection: P⊥_i = n_i n_i^T
    P_perp = normals.unsqueeze(-1) @ normals.unsqueeze(-2)      # (N, 2, 2)
    proj_u = 2.0 * torch.einsum('nij,nmj->nmi', P_perp, u)     # (N, N, 2)

    # First variation (numerator): Σ_j m_j ρ'(s)/ε^{n+1} · 2 P⊥ u
    w = masses.unsqueeze(0) * rho_prime / epsilon ** (n + 1)     # (N, N)
    first_variation = (w.unsqueeze(-1) * proj_u).sum(dim=1)      # (N, 2)

    # Regularized mass (denominator): Σ_j m_j ξ(s)/ε^n
    w_mass = masses.unsqueeze(0) * xi / epsilon ** n             # (N, N)
    reg_mass_raw = w_mass.sum(dim=1)                             # (N,)
    reg_mass = reg_mass_raw.clamp_min(1e-8)

    # Mean curvature vector: H = -(d/n) · first_variation / reg_mass
    H = -(d / n) * first_variation / reg_mass.unsqueeze(1)       # (N, 2)

    # Scalar curvature: κ = -H · n (positive for convex, outward normals)
    kappa = -(H * normals).sum(dim=1)                            # (N,)

    if return_denominator:
        # 0L-kappa hardening: the PRE-clamp denominator, for the
        # dimensionless certificate beta = eps * B / c_xi (-> 1 in
        # the straight-sheet limit). The absolute 1e-8 clamp is a
        # numerical floor, never a production certificate.
        return kappa, H, reg_mass_raw
    return kappa, H
