"""0C-5c regressions: carrier operator fixed, m-flux vs qm-flux variants.

Pins for scripts/experiments/spectral_flux_measure_0c5c.py (raw tables in
results/spectral_0c5c). Both variants build panel geometry, quadrature, K*,
rank detection and the bordered solve from the carrier measure m; only the
flux measure of the Neumann data differs (M: v_geom, Q: q v_geom), with
one flux_scale tensor feeding both the compatibility matrix and the BEM
RHS. NO production metric is selected at this stage -- the pins document
the behavior of the two internally consistent variants.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from spectral_flux_measure_0c5c import (  # noqa: E402
    circle,
    component_slices,
    dip_coherence,
    ellipse,
    run_case,
    summarize,
)


def test_variants_coincide_exactly_at_unit_coherence():
    """q = 1 gate (measured |F_M - F_Q| = 0.0): with flux_scale = ones the
    two variants are the same computation."""
    parts = [ellipse(0.7, 0.4, -1.6), ellipse(1.2, 0.8, 1.8)]
    runs = {v: run_case("q1", parts, v, n_steps=1, unit_q=True)
            for v in ("M", "Q")}
    fM = runs["M"]["frames"][0]
    fQ = runs["Q"]["frames"][0]
    assert abs(fM["objective_final"] - fQ["objective_final"]) < 1e-15
    assert fM["leakage_m"] == pytest.approx(fQ["leakage_m"], abs=1e-14)


def test_real_coherence_separated_pair():
    """Separated ellipses with real coherence (q_min = 0.994): identical
    rank sensor, both leakages at discretisation level (measured worst
    7.3e-3)."""
    parts = [ellipse(0.7, 0.4, -1.6), ellipse(1.2, 0.8, 1.8)]
    s = {}
    for v in ("M", "Q"):
        s[v] = summarize(run_case("two_ellipses", parts, v))
    assert s["M"]["C"] == s["Q"]["C"] == [2]
    for v in ("M", "Q"):
        assert s[v]["worst_leak_m"] < 0.05
        assert s[v]["worst_leak_qm"] < 0.05


def test_qdip_divergence_on_the_dipped_component():
    """Synthetic q-dip depth 0.5 on component 1 (the large disk). Leakage
    asserted EXPLICITLY on the dipped, active component -- the undipped
    small disk is nearly stationary and its leakage ratio is denominator
    noise (this is exactly what the earlier max-over-components summary got
    wrong). Measured on component 1: M m-leakage 1.2e-5 with qm-leakage
    2.7e-1; Q qm-leakage 1.6e-4 with m-leakage 1.8e-1. Rank sensor (carrier
    operator) dip-independent."""
    DIPPED = 1
    parts = [circle(0.5, -1.5), circle(1.0, 2.0)]
    fn = dip_coherence(component_slices(parts), DIPPED, 0.5, (2.0, 0.0))
    frames = {}
    for v in ("M", "Q"):
        run = run_case("qdip", parts, v, coherence_fn=fn)
        assert all(f["C"] == 2 for f in run["frames"])
        frames[v] = run["frames"]
    for f in frames["M"] + frames["Q"]:
        # the dipped component is the active one in every frame
        assert (f["activity_m"][DIPPED]
                > 10 * f["activity_m"][1 - DIPPED])
    # each variant preserves ITS OWN measure on the active dipped component
    assert all(f["leakage_m"][DIPPED] < 1e-3 for f in frames["M"])
    assert all(f["leakage_qm"][DIPPED] < 1e-2 for f in frames["Q"])
    # and leaks strongly in the foreign measure there
    assert max(f["leakage_qm"][DIPPED] for f in frames["M"]) > 0.1
    assert max(f["leakage_m"][DIPPED] for f in frames["Q"]) > 0.05


def test_near_fusion_gap_refuses_rank_with_weak_gap_status():
    """gap = 0.08 < 2 l_res: BOTH criteria still say C = 2; what has
    degraded is confidence -- the max-gap winning ratio drops to 10.4
    (< 30) as s_(2)/s_max rises to 4.1e-2. The structured evidence reports
    status=weak_gap (NOT detector_disagreement), which is exactly the
    hysteresis-hold candidate for 0C-5d; the plain auto solver still stops
    -- consistent with the 0C-4 static transition interval
    |g| <~ 2-4 l_res."""
    gap = 0.08
    parts = [ellipse(0.7, 0.4, -0.7 - gap / 2),
             ellipse(1.2, 0.8, 1.2 + gap / 2)]
    run = run_case("near_fusion", parts, "M", n_steps=1)
    assert "ambiguous" in run["frames"][0]
    assert "status=weak_gap" in run["frames"][0]["ambiguous"]
