"""0C-5b regressions: label-free component rank inside the MM step.

Pins for scripts/experiments/spectral_rank_auto_0c5b.py (raw tables in
results/spectral_0c5b). Scope: q = 1, u = m, valid separated boundaries.
Measured values quoted inline; thresholds carry margins.

Key design point pinned here: r_comp is NOT rank validation (machine zero
by construction for any selected rank), so the tests check the actual
evidence — detector agreement, gap/absolute margins, oracle parity (same
rank implies the same SVD subspace, hence bitwise-equal steps), and
refusal via AmbiguousComponentRankError when the detectors disagree.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from spectral_rank_auto_0c5b import (  # noqa: E402
    cat,
    circle,
    component_slices,
    ellipse,
    geometries,
    parity_case,
)
from src.torch.transport.bem_wasserstein import (  # noqa: E402
    rank_evidence,
)


def test_rank_evidence_unit_cases():
    """Synthetic spectra spanning all four evidence statuses. The statuses
    are distinct on purpose: weak_gap (criteria AGREE, confidence low) is
    the hysteresis-hold candidate and must not be conflated with genuine
    detector_disagreement."""
    # ascending relative [5e-4, 1e-3, 0.3, 0.5, 1]: clean C = 2, gap 300
    ev = rank_evidence(torch.tensor([1.0, 0.5, 0.3, 1e-3, 5e-4]),
                       4, 0.05, 30.0)
    assert ev.status == "clean" and ev.rank == 2
    assert ev.gap_ratio == pytest.approx(300.0)
    assert ev.abs_levels[1] == pytest.approx(1e-3)

    # both criteria say 2 but the winning ratio is only 12.5 < 30
    ev = rank_evidence(torch.tensor([1.0, 0.5, 0.04, 0.03]), 4, 0.05, 30.0)
    assert ev.status == "weak_gap"
    assert ev.c_abs == ev.c_gap == 2 and ev.rank == 2

    # absolute criterion counts 3 nulls, max-gap splits at 2
    ev = rank_evidence(torch.tensor([1.0, 0.04, 1e-3, 9e-4]), 4, 0.05, 30.0)
    assert ev.status == "detector_disagreement"
    assert ev.c_abs == 3 and ev.c_gap == 2 and ev.rank is None

    # nothing null at all
    ev = rank_evidence(torch.tensor([1.0, 0.8, 0.5, 0.3]), 4, 0.05, 30.0)
    assert ev.status == "no_null_block"


@pytest.mark.parametrize("name", ["disk", "annulus_disk", "three_disks"])
def test_auto_rank_matches_oracle_and_step_is_bitwise_equal(name):
    """Measured (h = 0.05): C detected correctly on all six geometries with
    gap ratios 45.6-153 and absolute levels 2.2e-3..1.2e-2; same rank means
    the same U0, so auto and oracle steps agreed EXACTLY (diff 0.0). The
    three cases here span C = 1, 2, 3 including the (M=3, C=2)
    annulus+disk, which no boundary-count heuristic gets right."""
    v, C_true, slices = geometries()[name]
    row = parity_case(name, v, C_true, slices)
    assert row["C_auto"] == C_true
    assert row["rank_gap_ratio"] > 30.0        # measured >= 45.6
    assert row["rank_abs_level"] < 0.05        # measured <= 1.2e-2
    assert row["max_s_diff"] < 1e-14           # measured 0.0
    assert row["objective_diff"] < 1e-12       # measured 0.0


def test_componentwise_flux_decreases_under_refinement():
    """Measured two-disk fluxes/dt: (-8.6e-5, -2.2e-5) at h = 0.05 ->
    (-2.9e-5, -9.0e-6) at h = 0.03; gap ratio improves 74 -> 128."""
    rows = {}
    for h in (0.05, 0.03):
        parts = [circle(0.5, -1.5, h=h), circle(1.0, 2.0, h=h)]
        rows[h] = parity_case("two_disks", cat(*parts), 2,
                              component_slices(parts))
    for k in range(2):
        assert (abs(rows[0.03]["fluxes_over_dt"][k])
                < abs(rows[0.05]["fluxes_over_dt"][k]))
    assert rows[0.03]["rank_gap_ratio"] > rows[0.05]["rank_gap_ratio"]


def test_short_evolution_holds_rank_two_and_matches_oracle():
    """5 auto-rank steps on the two-ellipse pair (the case with real O(10)
    motion), stepped in lockstep with the oracle run and compared as FULL
    tensors per step: C = 2 at every step, no premature collapse, subspace
    continuation angle far below any transition scale (measured worst
    0.121 deg over 20 steps), and positions/angles equal to machine
    precision -- the scalar-checksum comparison this replaces could not
    establish that."""
    from spectral_rank_auto_0c5b import evolve_pair

    parts = [ellipse(0.7, 0.4, -1.6), ellipse(1.2, 0.8, 1.8)]
    run = evolve_pair("two_ellipses", parts, 5)
    for f in run["frames"]:
        assert f["C"] == 2
        assert f["rank_gap_ratio"] > 30.0
        if f["subspace_angle_deg"] is not None:
            assert f["subspace_angle_deg"] < 5.0
        assert all(l < 0.05 for l in f["leakage_m"])
    assert run["worst_position_diff"] < 1e-14
    assert run["worst_angle_diff"] < 1e-14
