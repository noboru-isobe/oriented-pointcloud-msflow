"""Pairwise kernel geometry for point cloud computations.

Shared by regularized curvature (BLM) and tangential redistribution.
"""

import torch
from typing import NamedTuple

from ..oriented_varifold.mass import KERNEL_GRAD_RATIO


class PairwiseKernelResult(NamedTuple):
    """Result of pairwise kernel computation.

    Attributes:
        s: (N, N) normalized distances |x_i - x_j| / bandwidth.
        diff: (N, N, n) direction vectors x_j - x_i.
        unit_diff: (N, N, n) unit direction vectors (x_j - x_i) / |x_j - x_i|.
        psi_vals: (N, N) ψ(s) = η'(s)/s from KERNEL_GRAD_RATIO.
        rho_prime: (N, N) ρ'(s) = s·ψ(s).
    """
    s: torch.Tensor
    diff: torch.Tensor
    unit_diff: torch.Tensor
    psi_vals: torch.Tensor
    rho_prime: torch.Tensor


def compute_pairwise_kernel(
    positions: torch.Tensor,
    bandwidth: float,
    kernel: str,
) -> PairwiseKernelResult:
    """Compute pairwise kernel geometry between all point pairs.

    Self-interaction is excluded by pushing the diagonal outside kernel support.
    Direction convention: diff[i,j] = x_j - x_i (consistent with BLM formula).

    Args:
        positions: (N, n) point positions.
        bandwidth: Kernel bandwidth (δ or ε).
        kernel: Kernel name (key in KERNEL_GRAD_RATIO).

    Returns:
        PairwiseKernelResult with s, diff, unit_diff, psi_vals, rho_prime.
    """
    distances = torch.cdist(positions, positions)           # (N, N)
    distances.fill_diagonal_(bandwidth + 1.0)               # self → outside support
    s = distances / bandwidth                               # (N, N)

    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, n): x_j - x_i
    unit_diff = diff / distances.unsqueeze(-1)              # (N, N, n)

    psi_vals = KERNEL_GRAD_RATIO[kernel](s)                 # (N, N)
    rho_prime = s * psi_vals                                # (N, N): ρ'(s) = s·ψ(s)

    return PairwiseKernelResult(s, diff, unit_diff, psi_vals, rho_prime)
