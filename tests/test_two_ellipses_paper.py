"""Paper-configuration pre-contact pins (two_ellipses_paper_precontact.py;
raw tables in results/two_ellipses_paper).

The paper experiment is gap 0.1 (centers +-0.45), verified against the
archived run's parameters. Measured headlines pinned here on cheap
versions:

  - the isolated corrected relaxation predicts contact at t ~ 0.021
    (0.0208/0.0212/0.0215 for N = 64/128/256; dt-insensitive at N = 128),
    inside the paper window T = 0.05: a genuine PRE-CONTACT contact
    mechanism;
  - at gap ~ sigma the PAIR dynamics is NOT the isolated superposition:
    q_min collapses (0.99 -> 0.17) and the gap closes far faster than the
    isolated prediction -- the coherence-mediated attraction is a real
    part of the model at this scale and the pair run must not be
    validated against superposition there (unlike the resolved gap-0.6
    audits);
  - N = 64 at dt = 1e-4 is beyond the corrected scheme's stability
    boundary (tips at 2.2 h); dt = 2e-5 is stable -- the paper's
    dt = 1e-5 was on the stable side.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from two_ellipses_paper_precontact import (  # noqa: E402
    isolated_run,
    pair_run,
)


@pytest.mark.xfail(
    reason="Retained as a strict SENTINEL for the corrected angle-map "
    "stability boundary: dt=1e-4 sits above the sawtooth threshold of "
    "the outer position-angle iteration (the pre-fix pi/7 "
    "under-rotation was an accidental under-relaxation that masked "
    "it), so this run blows up (neighbor-alternating s/dtheta mode, "
    "onset ~step 21, |s| doubling per step). If this test ever starts "
    "PASSING, the stability domain moved -- investigate before "
    "trusting any dt=1e-4 trajectory. The corrected stable-dt control "
    "is pinned by test_superposition_contact_artifact_pins.",
    strict=True)
def test_dt1e4_remains_outside_postfix_stability_domain():
    """Pre-fix pin for the record: N = 128, dt = 1e-4 to t = 0.03
    measured contact 0.02124, area drift 9.3e-3 -- values produced
    under the buggy angle map, retired."""
    r = isolated_run(128, 1e-4)
    assert r["contact_time"] is not None
    assert r["contact_time"] == pytest.approx(0.0212, abs=0.002)
    assert r["contact_time"] < 0.05          # inside the paper window
    assert r["area_drift_at_end"] < 0.02
    assert r["worst_relative_gradient"] < 1e-5


def test_superposition_contact_artifact_pins():
    """The corrected ISOLATED-RELAXATION CONTROL (C_2D fix), pinned on
    the committed artifacts in results/contact_postfix/ (runs take
    17-33 min; regenerate with scripts/experiments/contact_postfix.py
    --dt <dt>, which also records full provenance in config.json).

    Review-corrected semantics (2026-08-07): these are SUPERPOSITION
    contact predictions from a single isolated ellipse (oracle C=1, no
    redistribution, no deletion, no hidden boundary, no topological
    change) -- NOT actual pair trajectories. At the paper gap
    (g0 ~ sigma) the actual pair is measurably NOT the isolated
    superposition (test_pair_at_paper_gap_shows_coherence_coupling),
    so this value must never be cited as a pair contact time.

    Pinned: at the measured-stable dt=1e-5 the predicted superposition
    contact is t* = 0.02335 inside the paper window; THREE stable time
    levels (1e-5, 5e-6, 2.5e-6 -- the third taken DOWNWARD because
    dt=2e-5 is slowly unstable on the long horizon) give successive
    differences 2.494e-6 and 1.248e-6, ratio 1.998 = observed
    convergence ORDER p ~ 1.0 (consistent with the first-order MM/JKO
    time discretization), Richardson limit t*_inf ~ 0.0233510; area
    drift 1.1e-3 (8x better than the retired pre-fix 9.3e-3 at the
    now-unstable dt=1e-4); sawtooth stability bracket certified
    (unstable at 5e-5 with zigzag corr -0.80, stable at 1e-5)."""
    import json
    import math
    from pathlib import Path

    d = Path(__file__).parent.parent / "results" / "contact_postfix"
    KEY = "predicted_superposition_contact_time"
    r5 = json.loads(
        (d / "superposition_contact_prediction_dt1e-05.json").read_text())
    r6 = json.loads(
        (d / "superposition_contact_prediction_dt5e-06.json").read_text())
    r7 = json.loads(
        (d / "superposition_contact_prediction_dt2.5e-06.json")
        .read_text())
    for r in (r5, r6, r7):
        assert r[KEY] is not None
        assert r[KEY] == pytest.approx(0.02335, abs=5e-4)
        assert r[KEY] < 0.05                     # paper window
        assert r["area_drift_at_end"] < 2e-3
        assert r["worst_relative_gradient"] < 1e-5
    # first-order dt-convergence: successive halving differences with
    # ratio ~2 (measured 1.998 -> order p = 1.00)
    d1 = abs(r5[KEY] - r6[KEY])
    d2 = abs(r6[KEY] - r7[KEY])
    assert d1 < 2e-4 * r5[KEY]
    p_obs = math.log2(d1 / d2)
    assert p_obs == pytest.approx(1.0, abs=0.15)
    # Richardson limit stays inside the pin window
    t_inf = 2 * r7[KEY] - r6[KEY]
    assert t_inf == pytest.approx(0.023351, abs=1e-4)
    sweep = {s["dt"]: s for s in json.loads(
        (d / "sawtooth_dt_sweep.json").read_text())}
    assert sweep[5e-5]["error"] is not None       # unstable: blew up
    assert sweep[5e-5]["min_zigzag"] < -0.5       # sawtooth signature
    assert sweep[1e-5]["error"] is None           # stable
    assert sweep[1e-5]["min_zigzag"] > 0.5        # smooth field
    assert sweep[1e-5]["tail_growth_factor"] < 1.01


def test_pair_at_paper_gap_shows_coherence_coupling():
    """Short pair run (N = 128, oracle C = 2): the gap decreases, q_min
    collapses below 0.6 within a few steps (gap ~ sigma), and the
    superposition defect reaches the 1e-2 scale -- the pair is NOT the
    concatenation of isolated runs at this separation, by measurement."""
    p = pair_run(128, 1e-4, "oracle", t_end=0.001)
    ok = [f for f in p["frames"] if "gap" in f and "stopped" not in f]
    assert len(ok) >= 5
    gaps = [f["gap"] for f in ok]
    assert gaps[-1] < gaps[0]
    assert min(f["q_min"] for f in ok) < 0.6
    assert max(f["dx_iso_right"] for f in ok) > 5e-3
    for f in ok:
        assert f["C"] == 2
