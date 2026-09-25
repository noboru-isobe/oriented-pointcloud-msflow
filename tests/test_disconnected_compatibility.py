"""The discrete tangent space admits transport the one-phase metric forbids.

For E = E_1 u E_2 the one-phase tangent norm is finite only when the flux
through each component vanishes separately (div u = 0 forces it). The solver
imposes one global constraint, sum_i s_i w_i = 0, which is strictly weaker.
These tests pin the gap down algebraically -- no optimizer, no evolution.

Known component labels are used as an *oracle*. They measure the defect; they
are not a repair, and they are only meaningful before the components touch.

The full refinement sweep and raw numbers live in
scripts/experiments/bem_disconnected_oracle.py.
"""

import math

import pytest
import torch

from scripts.experiments.bem_disconnected_oracle import (
    IncompatibleVelocity,
    blockwise_oracle_energy,
    exchange_mode,
    flux_matrix,
    nullspace_basis,
    projection_residual,
    two_disks,
)

R1, R2 = 0.5, 1.0
SPACING = 0.05
DTYPE = torch.float64


@pytest.mark.parametrize("gap", [1.0, 4.0])
def test_exchange_mode_admitted_globally_but_not_componentwise(gap):
    """The mode that shrinks the small disk and grows the large one at zero
    net flux sits inside the solver's admissible space and entirely outside
    the correct one.
    """
    _, masses, labels, _ = two_disks(R1, R2, gap, SPACING)
    G = flux_matrix(masses, labels)
    s = exchange_mode(masses, labels, R1, R2)

    assert (G.sum(0) @ s).abs() < 1e-12                    # global flux zero
    assert G @ s == pytest.approx(torch.tensor([-1.0, 1.0], dtype=DTYPE),
                                  abs=1e-12)               # component fluxes

    Q_global = nullspace_basis(G.sum(0, keepdim=True))
    Q_comp = nullspace_basis(G)
    assert Q_global.shape[1] == masses.numel() - 1
    assert Q_comp.shape[1] == masses.numel() - 2

    assert projection_residual(Q_global, s) < 1e-10        # admitted
    assert projection_residual(Q_comp, s) > 0.5            # forbidden


def test_exchange_mode_strictly_decreases_perimeter():
    """DP[s] = -1/R1 + 1/R2 < 0, so this is a genuine descent direction that
    survives the single global constraint and is annihilated by the
    componentwise ones. That is the structural reason the discrete flow can
    exhibit Ostwald-ripening-type transfer the continuum one-phase flow
    cannot.
    """
    _, masses, labels, _ = two_disks(R1, R2, 4.0, SPACING)
    G = flux_matrix(masses, labels)
    s = exchange_mode(masses, labels, R1, R2)

    curvature = torch.where(labels == 0,
                            torch.full_like(masses, 1.0 / R1),
                            torch.full_like(masses, 1.0 / R2))
    grad = curvature * masses

    assert (grad * s).sum().item() == pytest.approx(-1.0 / R1 + 1.0 / R2,
                                                    rel=1e-10)
    assert (grad * s).sum().item() < 0

    scale = grad.norm()
    assert (nullspace_basis(G.sum(0, keepdim=True)).T @ grad).norm() / scale > 0.1
    assert (nullspace_basis(G).T @ grad).norm() / scale < 1e-10


def test_oracle_rejects_incompatible_velocity():
    """It must refuse, not quietly subtract componentwise means -- doing that
    would return a finite cost for a different velocity than the one asked
    about.
    """
    varifold, masses, labels, _ = two_disks(R1, R2, 4.0, SPACING)
    s = exchange_mode(masses, labels, R1, R2)
    with pytest.raises(IncompatibleVelocity, match="compatibility violated"):
        blockwise_oracle_energy(varifold, masses, labels, s)


@pytest.mark.parametrize("gap", [1.0, 4.0])
def test_compatible_mode_energy_is_additive_and_gap_independent(gap):
    """cos(2 theta) has zero mean on each disk, so it is admissible. The exact
    energy of cos(k theta) on a disk of radius R is pi R^2 / k; separated
    disks do not interact, so the total is the sum and does not depend on the
    separation.
    """
    varifold, masses, labels, (n1, n2) = two_disks(R1, R2, gap, SPACING)
    t1 = torch.linspace(0, 2 * math.pi, n1 + 1, dtype=DTYPE)[:-1]
    t2 = torch.linspace(0, 2 * math.pi, n2 + 1, dtype=DTYPE)[:-1]
    vel = torch.cat([torch.cos(2 * t1), torch.cos(2 * t2)])

    assert (flux_matrix(masses, labels) @ vel).abs().max() < 1e-12
    energy = blockwise_oracle_energy(varifold, masses, labels, vel)
    assert energy == pytest.approx(math.pi * (R1 ** 2 + R2 ** 2) / 2, rel=3e-2)


def test_compatible_mode_energy_converges_under_refinement():
    """Guards the tolerance above: the 3% allowance must be discretisation
    error that shrinks, not a fixed offset.
    """
    exact = math.pi * (R1 ** 2 + R2 ** 2) / 2
    errors = []
    for spacing in (0.08, 0.04, 0.02):
        varifold, masses, labels, (n1, n2) = two_disks(R1, R2, 4.0, spacing)
        t1 = torch.linspace(0, 2 * math.pi, n1 + 1, dtype=DTYPE)[:-1]
        t2 = torch.linspace(0, 2 * math.pi, n2 + 1, dtype=DTYPE)[:-1]
        vel = torch.cat([torch.cos(2 * t1), torch.cos(2 * t2)])
        energy = blockwise_oracle_energy(varifold, masses, labels, vel)
        errors.append(abs(energy - exact) / exact)

    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse
    assert errors[-1] < 5e-3
