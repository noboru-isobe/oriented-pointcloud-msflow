"""Parity tests: naive (full N×N) vs KeOps (lazy) backends.

For N=64 (small, so KeOps JIT compile is the dominant cost), check that
both backends agree on:

- forward output (atol=1e-6, rtol=1e-5)
- gradient w.r.t. the inputs (same tolerance)

across all 3 utility functions × 3 kernels = 9 (function, kernel) pairs.

If parity holds, this is strong evidence the ``sqrt(d² + 1e-24)``
regularisation in the KeOps path matches ``_SafeKernelEval``'s
hand-rolled custom backward at the self-term and elsewhere.
"""
from __future__ import annotations

import math
from typing import Callable

import pytest
import torch

from src.torch.oriented_varifold.mass import compute_kde_density
from src.torch.perimeter.coherence_perimeter import (
    compute_scalar_density,
    compute_vector_field,
)


KERNELS = ["wendland_c2", "biweight", "epanechnikov"]
ATOL = 1e-6
RTOL = 1e-5


def _make_inputs(N: int = 64, seed: int = 0):
    """Random points on a perturbed unit circle + random masses + normals."""
    g = torch.Generator().manual_seed(seed)
    phi = torch.rand(N, generator=g, dtype=torch.float64) * 2 * math.pi
    r = 1.0 + 0.05 * torch.rand(N, generator=g, dtype=torch.float64)
    points = torch.stack([r * phi.cos(), r * phi.sin()], dim=1).requires_grad_(True)
    normals = torch.stack([phi.cos(), phi.sin()], dim=1)        # unit outward-ish
    masses = (0.5 + 0.5 * torch.rand(N, generator=g, dtype=torch.float64))
    return points, normals, masses


def _check_pair(
    label: str,
    naive_fn: Callable[[], torch.Tensor],
    keops_fn: Callable[[], torch.Tensor],
    inputs_for_grad: list[torch.Tensor],
):
    """Run both, check forward + gradient parity."""
    # Forward
    out_naive = naive_fn()
    out_keops = keops_fn()
    assert out_naive.shape == out_keops.shape, (
        f"{label}: shape mismatch naive={out_naive.shape} keops={out_keops.shape}")
    torch.testing.assert_close(
        out_naive.detach(), out_keops.detach(), atol=ATOL, rtol=RTOL,
        msg=lambda m: f"{label} forward mismatch: {m}",
    )

    # Gradient (sum loss back to inputs)
    loss_naive = out_naive.sum()
    loss_keops = out_keops.sum()
    grads_naive = torch.autograd.grad(loss_naive, inputs_for_grad, retain_graph=True)
    grads_keops = torch.autograd.grad(loss_keops, inputs_for_grad, retain_graph=True)
    for k, (gn, gk) in enumerate(zip(grads_naive, grads_keops)):
        torch.testing.assert_close(
            gn, gk, atol=ATOL, rtol=RTOL,
            msg=lambda m: f"{label} grad[{k}] mismatch: {m}",
        )


@pytest.mark.parametrize("kernel", KERNELS)
def test_compute_kde_density_parity(kernel):
    points, _normals, _masses = _make_inputs()
    delta = 0.2
    _check_pair(
        f"compute_kde_density[{kernel}]",
        naive_fn=lambda: compute_kde_density(points, delta, kernel, backend="naive"),
        keops_fn=lambda: compute_kde_density(points, delta, kernel, backend="keops"),
        inputs_for_grad=[points],
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_compute_scalar_density_parity(kernel):
    points, _normals, masses = _make_inputs()
    masses = masses.detach().requires_grad_(True)
    sigma = 0.3
    _check_pair(
        f"compute_scalar_density[{kernel}]",
        naive_fn=lambda: compute_scalar_density(points, masses, sigma, kernel, backend="naive"),
        keops_fn=lambda: compute_scalar_density(points, masses, sigma, kernel, backend="keops"),
        inputs_for_grad=[points, masses],
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_compute_vector_field_parity(kernel):
    points, normals, masses = _make_inputs()
    masses = masses.detach().requires_grad_(True)
    sigma = 0.3
    _check_pair(
        f"compute_vector_field[{kernel}]",
        naive_fn=lambda: compute_vector_field(points, normals, masses, sigma, kernel, backend="naive"),
        keops_fn=lambda: compute_vector_field(points, normals, masses, sigma, kernel, backend="keops"),
        inputs_for_grad=[points, masses],
    )
