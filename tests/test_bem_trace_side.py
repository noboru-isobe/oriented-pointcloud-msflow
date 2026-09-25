"""Regression tests pinning down the single-layer jump relation.

The full audit lives in scripts/experiments/bem_static_tests.py; these are the
two cheap assertions that must not silently regress. They guard the *interior*
trace, which the rest of the suite does not cover: the default is still
"legacy_exterior" for artifact reproducibility, so a passing suite otherwise
only proves the legacy path is unchanged.
"""

import math

import pytest
import torch

from src.torch.shapes.generator import (
    annulus_exact_masses,
    annulus_point_counts,
    generate_oriented_annulus,
    generate_oriented_circle,
)
from src.torch.transport.bem_wasserstein import (
    BEMWasserstein,
    build_bem_matrices_point,
)

DTYPE = torch.float64


@pytest.mark.parametrize("n", [128, 256, 512])
def test_interior_trace_vanishes_on_uniform_circle(n):
    """A uniform density on a circle makes the single layer constant inside,
    so the interior trace (+1/2 I + K*) 1 must vanish; the exterior trace must
    not. This is the one-line discriminator between the two conventions.
    """
    v = generate_oriented_circle(n, 1.0, (0.0, 0.0), device="cpu", dtype=DTYPE)
    m = torch.full((n,), 2 * math.pi / n, dtype=DTYPE)
    _, K_star = build_bem_matrices_point(
        v.positions, v.normals, m, 0.1 * m.median().item()
    )
    one = torch.ones(n, dtype=DTYPE)

    interior = (0.5 * one + K_star @ one).abs().max()
    exterior = (-0.5 * one + K_star @ one).abs().max()

    # O(1/N) from the regularised collocation, so scale the tolerance with n.
    assert interior < 2.0 / n
    assert exterior > 0.9


def test_interior_trace_residual_decreases_with_refinement():
    """Guards against a tolerance that merely happens to pass at one N."""
    residuals = []
    for n in (64, 128, 256):
        v = generate_oriented_circle(n, 1.0, (0.0, 0.0), device="cpu", dtype=DTYPE)
        m = torch.full((n,), 2 * math.pi / n, dtype=DTYPE)
        _, K_star = build_bem_matrices_point(
            v.positions, v.normals, m, 0.1 * m.median().item()
        )
        one = torch.ones(n, dtype=DTYPE)
        residuals.append((0.5 * one + K_star @ one).abs().max().item())

    # Halving h should roughly halve the residual.
    for coarse, fine in zip(residuals, residuals[1:]):
        assert fine < 0.6 * coarse


@pytest.mark.parametrize("n_outer", [64, 128])
def test_interior_annulus_radial_energy(n_outer):
    """The annulus has an exact tangent energy Q_E = 2 pi a^2 log(R+/R-).

    W_lin = (dt/2) Q_E, so 2 W_lin/dt should reproduce it. The interior trace
    converges; the exterior trace diverges like O(N) and is asserted to be
    wildly off so the test fails loudly if the two are ever swapped.
    """
    R_out, R_in, dt = 1.0, 0.5, 1e-5
    n_out, n_in = annulus_point_counts(n_outer, R_out, R_in)
    varifold = generate_oriented_annulus(n_outer, R_out, R_in,
                                         device="cpu", dtype=DTYPE)
    masses = annulus_exact_masses(n_out, n_in, R_out, R_in,
                                  device="cpu", dtype=DTYPE)

    a = (1.0 / R_out + 1.0 / R_in) / math.log(R_out / R_in)
    exact = 2 * math.pi * a * a * math.log(R_out / R_in)
    vel = torch.cat([
        torch.full((n_out,), -a / R_out, dtype=DTYPE),
        torch.full((n_in,), +a / R_in, dtype=DTYPE),
    ])
    # The velocity is compatible: 2 pi R+ v+ + 2 pi R- v- = 0.
    assert abs((vel * masses).sum().item()) < 1e-12

    def energy(trace_side):
        bem = BEMWasserstein(method="point", epsilon_scale=0.1,
                             n_endpoints=3, trace_side=trace_side)
        bem.setup_for_step(varifold.positions, varifold.normals, masses, sigma=None)
        bem.setup_coherence(torch.ones(n_out + n_in, dtype=DTYPE))
        W = bem(dt * vel, torch.zeros(n_out + n_in, dtype=DTYPE), dt)
        return (2.0 * W / dt).item()

    assert energy("interior") == pytest.approx(exact, rel=1e-2)
    assert energy("legacy_exterior") > 10 * exact


def test_trace_side_is_validated():
    bem = BEMWasserstein(method="point", trace_side="nonsense")
    v = generate_oriented_circle(32, 1.0, (0.0, 0.0), device="cpu", dtype=DTYPE)
    m = torch.full((32,), 2 * math.pi / 32, dtype=DTYPE)
    with pytest.raises(ValueError, match="trace_side"):
        bem.setup_for_step(v.positions, v.normals, m, sigma=None)


def test_mm_config_wires_trace_side_into_the_solver():
    """MMConfig must reach BEMWasserstein: without this the evolution path
    silently keeps using the exterior trace no matter what an experiment asks
    for.
    """
    from src.torch.solver.mm_step import MMConfig, MMStepper

    assert MMStepper(MMConfig()).bem_wasserstein.trace_side == "legacy_exterior"
    stepper = MMStepper(MMConfig(bem_trace_side="interior"))
    assert stepper.bem_wasserstein.trace_side == "interior"
