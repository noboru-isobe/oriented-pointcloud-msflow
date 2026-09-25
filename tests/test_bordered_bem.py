"""0C-3 regressions: the clean q = 1 compatibility-constrained bordered BIE.

Pins the prototype of scripts/experiments/bem_bordered_oracle.py against its
blockwise oracle at coarse resolution (full refinement tables in
results/bordered_0c3). Thresholds carry ~3-8x margins over the measured
values quoted inline. The prototype is static, q = 1, oracle-supplied C, and
deliberately NOT wired into MMStepper.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from bem_bordered_oracle import (  # noqa: E402
    blockwise_solve,
    compatible_velocity,
    exchange_velocity,
    global_solve,
    gauge_fixed_potential_error,
    register_local_geometries,
    run_case,
)
from bem_disconnected_oracle import IncompatibleVelocity  # noqa: E402
from bem_visible_weight_audit import build_geometry_full  # noqa: E402

register_local_geometries()


@pytest.mark.parametrize("name", [
    "two_unequal_disks_g1",
    "two_unequal_ellipses",
    "annulus_plus_ellipse",
])
def test_global_bordered_matches_blockwise_oracle(name):
    """Energy within 5e-3 of the blockwise solve and gauge-fixed potentials
    within 2e-2 per phase (measured: energy ratio deviates by at most 5.9e-4
    at h=0.06, phi error at most 6.7e-3). Includes a noncircular pair and an
    annulus-bearing phase.
    """
    r = run_case(name, 0.06)
    assert abs(r["energy_ratio"] - 1.0) < 5e-3
    assert r["phi_rel_err_max"] < 2e-2
    assert r["r_comp"] < 0.03           # compatible data, measured <= 1.2e-2


def test_exchange_mode_is_rejected_not_absorbed():
    """The bordered variable eta must not soak up an incompatible RHS. The
    exchange mode sits entirely in the left near-null direction
    (r_comp = 1.000 measured), two orders above compatible data.
    """
    varifold, m, phases, _, _, C = build_geometry_full(
        "two_unequal_disks_g1", 0.06)
    with pytest.raises(IncompatibleVelocity, match="compatibility violated"):
        global_solve(varifold, m, phases, C, exchange_velocity(m, phases))


def test_disks_exact_energy_anchor():
    """cos(2 theta) on each disk: exact energy pi (R1^2 + R2^2) / 2.
    Measured ratio 0.9858 at h=0.06."""
    r = run_case("two_unequal_disks_g1", 0.06)
    exact = math.pi * (0.5 ** 2 + 1.0 ** 2) / 2
    assert r["energy_global"] == pytest.approx(exact, rel=0.05)


def test_separation_independence():
    """Disjoint phases do not interact in the continuum, so the energy must
    not depend on the gap. Measured relative difference 3.3e-4 between
    gap ~1 and gap ~4 at h=0.06."""
    e1 = run_case("two_unequal_disks_g1", 0.06)["energy_global"]
    e4 = run_case("two_unequal_disks_g4", 0.06)["energy_global"]
    assert abs(e1 - e4) / abs(e4) < 5e-3


def test_eta_and_potential_error_decay_under_refinement():
    """eta = U0^T v~ by construction (see the script docstring), so its decay
    restates the compatibility-residual decay rather than adding evidence;
    the independent quantities here are the potential error and the oracle
    energy ratio. The refinement check guards every other threshold in this
    file against single-resolution coincidences.
    """
    coarse = run_case("two_unequal_ellipses", 0.08)
    fine = run_case("two_unequal_ellipses", 0.04)
    assert fine["eta_rel"] < coarse["eta_rel"]
    assert fine["r_comp"] < coarse["r_comp"]
    assert fine["phi_rel_err_max"] < coarse["phi_rel_err_max"]
    assert (abs(fine["energy_ratio"] - 1.0)
            < abs(coarse["energy_ratio"] - 1.0) + 1e-4)
