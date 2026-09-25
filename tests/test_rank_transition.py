"""0C-4 regressions: the merger rank transition, clean vs raw carrier.

Pins for scripts/experiments/bem_rank_transition.py (tables in
results/rank_0c4). All at h=0.03; thresholds from measured values with
margins.

The structural findings pinned here:

  * on the CLEAN union boundary the carrier sensor reads the phase count on
    both sides of the merger (C=2 separated, C=1 overlapped, down to
    2 l_res of the tangency), and its null mode after the merger IS the
    phase constant (2-4 deg with corners);
  * on the RAW two-loop carrier the overlap answer is an artifact: crossing
    loops put collocation nodes of the two circles nearly on top of each
    other (min distance ~1e-6 << eps), degenerating the operator; the
    resulting near-exact null mode is unrelated to the phase constant
    (60-70 deg). The count "1" is not a phase reading -- this is the
    conditional-failure branch, in a different form than predicted (the
    prediction was "stays at 2");
  * the label-free reweighted constraint C_u = U_m^T W_m^{-1/2} W_u G
    matches the oracle componentwise space for both u = m and u = q m
    without recovering any labels.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from bem_rank_transition import (  # noqa: E402
    DTYPE,
    R1,
    R2,
    raw_two_loops,
    sensor,
    separated_circles,
    union_boundary,
    validate_reweighted_constraint,
)
from bem_spectral_audit import principal_angles_deg  # noqa: E402

H = 0.03


def test_clean_separated_reads_two_components():
    for g_over_h in (2.0, 4.0):
        varifold, w = separated_circles(g_over_h * H, H)
        dec = sensor(varifold, w)
        sigma = dec["sigma"]
        # two null modes under the absolute threshold, clear gap to the third
        assert (sigma[1] / sigma[-1]).item() < 0.05
        assert (sigma[2] / sigma[-1]).item() > 0.1
        rho = sigma[1:5] / sigma[:4]
        assert int(torch.argmax(rho).item()) + 1 == 2


def test_clean_union_reads_one_component_with_phase_null_mode():
    """After the merger the count must be 1 AND the null mode must actually
    be the phase constant (measured 2.1-3.7 deg; corners cost a couple of
    degrees over the smooth-boundary <1 deg).
    """
    for depth_h in (2.0, 8.0):
        varifold, w = union_boundary(R1 + R2 - depth_h * H, H)
        dec = sensor(varifold, w)
        sigma = dec["sigma"]
        assert (sigma[0] / sigma[-1]).item() < 0.05
        assert (sigma[1] / sigma[-1]).item() > 0.05
        rho = sigma[1:5] / sigma[:4]
        assert int(torch.argmax(rho).item()) + 1 == 1

        U1 = dec["U"][:, -1:]
        B_phase = (dec["sqrt_w"] / dec["sqrt_w"].norm())[:, None]
        assert principal_angles_deg(U1, B_phase).max().item() < 8.0


def test_raw_overlap_null_mode_is_a_crossing_artifact():
    """The raw two-loop cloud after overlap: a near-exact null appears
    (sigma_1/sigma_max ~ 1e-5, vs ~2e-3 for genuine phase nulls) because the
    crossing loops place collocation nodes of the two circles nearly on top
    of each other -- and the mode is NOT the phase constant. So the raw
    carrier does not read phase topology through a transversal merger, in
    either count or subspace.
    """
    d_dist = R1 + R2 - 0.4
    varifold, w, loops = raw_two_loops(d_dist, H)
    dec = sensor(varifold, w)

    # near-duplicate collocation nodes at the crossings
    P = dec["op"]["positions_end"]
    D = torch.cdist(P, P) + 1e9 * torch.eye(len(P), dtype=DTYPE)
    assert D.min().item() < dec["eps"] / 100

    # a null far below the genuine-phase level ...
    sigma = dec["sigma"]
    assert (sigma[0] / sigma[-1]).item() < 1e-3
    # ... whose direction has nothing to do with the phase constant
    U1 = dec["U"][:, -1:]
    B_phase = (dec["sqrt_w"] / dec["sqrt_w"].norm())[:, None]
    assert principal_angles_deg(U1, B_phase).max().item() > 30.0


def test_raw_separated_still_reads_two_components():
    """Before contact the raw representation IS the clean one; the sensor
    must agree with the clean family there."""
    varifold, w, loops = raw_two_loops(R1 + R2 + 4 * H, H)
    dec = sensor(varifold, w)
    sigma = dec["sigma"]
    assert (sigma[1] / sigma[-1]).item() < 0.05
    rho = sigma[1:5] / sigma[:4]
    assert int(torch.argmax(rho).item()) + 1 == 2


def test_reweighted_constraint_matches_oracle_for_both_measures():
    """C_u = U_m^T W_m^{-1/2} W_u G vs the oracle rows B^T W_u G, with a deep
    synthetic dip (q=0.05 on a quarter arc) present: measured 0.187 deg for
    u=m and 0.205 deg for u=qm. No labels are used on the C_u side.
    """
    for r in validate_reweighted_constraint(H):
        assert r["theta_Cu_vs_oracle"] < 1.0, r
