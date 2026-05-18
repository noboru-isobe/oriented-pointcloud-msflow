"""KeOps-backed pairwise kernel utilities (memory-light path).

This module is opt-in: ``compute_kde_density``, ``compute_scalar_density``
and ``compute_vector_field`` in their respective modules accept a
``backend="naive"|"keops"`` kwarg, and route here only when ``"keops"``
is selected. Existing tests / MS-flow defaults stay on the ``naive``
path so behaviour is unchanged unless callers explicitly opt in.

Memory characteristic: every helper here uses ``pykeops.torch.LazyTensor``
so the full ``(N, N)`` pairwise kernel is **never materialised**; the
reduction (``.sum(dim=1)``) streams without allocating the matrix.
Peak memory at N=10000 measured at ~50 MB vs the naive path's 6.8 GB.

z=0 (self-term) safety: we use ``sqrt(d_sq + 1e-24)`` instead of
``sqrt(d_sq)``. At r=0 this gives ``r ≈ 1e-12``, forward kernel value
``η(≈0) ≈ 1`` (bit-for-bit with the naive ``_SafeKernelEval`` forward),
and the gradient ``∂r/∂point = 0 / sqrt(1e-24) = 0`` by IEEE float (matches
``_SafeKernelEval``'s hand-rolled custom backward). Verified in
``tests/test_keops_parity.py``.

``pykeops`` is imported lazily inside each function so this module is
importable even when KeOps isn't installed — the import error fires
only on the first call.
"""

from __future__ import annotations

from typing import Literal

import torch

_KEOPS_SAFE_SQRT_EPS = 1e-24

_KernelName = Literal["wendland_c2", "biweight", "epanechnikov"]


def _keops_kernel_lazy(
    points: torch.Tensor, scale: float, kernel: _KernelName,
):
    """``(N, N)`` LazyTensor holding ``η(|x_i - x_j| / scale)``.

    Callers multiply by the outer constant (``C_η/δ`` for KDE-style,
    ``C_{2D}/σ²`` for the 2D mollifier) and the per-j weight before
    invoking the final reduction.
    """
    from pykeops.torch import LazyTensor

    P = points.contiguous()
    x_i = LazyTensor(P[:, None, :])
    x_j = LazyTensor(P[None, :, :])
    r = (((x_i - x_j) ** 2).sum(-1) + _KEOPS_SAFE_SQRT_EPS).sqrt()
    u = r / scale
    if kernel == "wendland_c2":
        return (1.0 - u).relu() ** 4 * (4.0 * u + 1.0)
    if kernel == "biweight":
        return ((1.0 - u * u).relu()) ** 2
    if kernel == "epanechnikov":
        return (1.0 - u * u).relu()
    raise ValueError(f"unknown kernel: {kernel!r}")


def kde_density_keops(
    points: torch.Tensor, delta: float, kernel: _KernelName,
) -> torch.Tensor:
    """θ_{δ,N}(x_i) = (1/(N C_η δ)) Σ_j η(|x_i-x_j|/δ).  O(N) memory."""
    # Constants live in mass.py; lazy import avoids circular dep.
    from ..oriented_varifold.mass import KERNEL_CONSTANTS

    N = points.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]
    kvals = _keops_kernel_lazy(points, delta, kernel)
    return kvals.sum(dim=1).squeeze(-1) / (N * C_eta * delta)


def scalar_density_keops(
    positions: torch.Tensor, masses: torch.Tensor,
    sigma: float, kernel: _KernelName,
) -> torch.Tensor:
    """U_σ(x_i) = Σ_j m_j ρ_σ(x_i - x_j) with ρ_σ = (C_{2D}/σ²) η(·/σ).
    O(N) memory."""
    from pykeops.torch import LazyTensor
    from ..perimeter.coherence_perimeter import MOLLIFIER_2D_CONSTANTS

    C_2D = MOLLIFIER_2D_CONSTANTS[kernel]
    kvals = _keops_kernel_lazy(positions, sigma, kernel)
    rho = (C_2D / (sigma * sigma)) * kvals
    m_j = LazyTensor(masses.contiguous()[None, :, None])
    return (rho * m_j).sum(dim=1).squeeze(-1)


def vector_field_keops(
    positions: torch.Tensor, normals: torch.Tensor, masses: torch.Tensor,
    sigma: float, kernel: _KernelName,
) -> torch.Tensor:
    """V_σ(x_i) = Σ_j m_j ρ_σ(x_i - x_j) n_j. O(N) memory.

    The per-j mark ``n_j`` is a 2-vector and KeOps reduces against it
    via a 2-channel lazy tensor of shape ``(1, N, 2)``.
    """
    from pykeops.torch import LazyTensor
    from ..perimeter.coherence_perimeter import MOLLIFIER_2D_CONSTANTS

    C_2D = MOLLIFIER_2D_CONSTANTS[kernel]
    kvals = _keops_kernel_lazy(positions, sigma, kernel)
    rho = (C_2D / (sigma * sigma)) * kvals
    m_j = LazyTensor(masses.contiguous()[None, :, None])
    n_j = LazyTensor(normals.contiguous()[None, :, :])
    return (rho * m_j * n_j).sum(dim=1)
