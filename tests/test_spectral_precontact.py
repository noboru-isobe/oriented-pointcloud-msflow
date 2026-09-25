"""0C-5d-1 regressions: pre-contact dynamic audit.

Pins for scripts/experiments/spectral_precontact_0c5d1.py (raw tables in
results/spectral_0c5d1). Short (5-step) versions of the measured 40-cap
runs; measured values quoted inline.

The scientific headline pinned here: from the standard two-ellipse
configuration the CORRECTED metric grows the gap (both major axes lie on
the line of centers, rounding retracts the facing tips; areas and
centroids conserved) -- the pair never approaches contact, consistent with
the old fusion having been induced by the rejected global/exterior metric.
And the superposition control: the pair run equals the concatenated
isolated runs to ~1e-5 after 5 steps -- the finite-N global-BIE
cross-block coupling is real but three orders below the point spacing.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from spectral_precontact_0c5d1 import (  # noqa: E402
    DT,
    cat,
    evolve_pair,
    make_config,
    parts_pair,
)
from spectral_rank_auto_0c5b import circular_angle_diff  # noqa: E402
from src.torch.solver.mm_step import MMStepper  # noqa: E402

STEPS = 5


def test_precontact_pair_grows_gap_and_matches_superposition():
    pair = evolve_pair("M", n_steps=STEPS)
    clean = [f for f in pair["frames"] if "hard_stop" not in f]
    assert len(clean) == STEPS                     # clean C=2 throughout

    # gap grows monotonically (measured 0.6027 -> 0.6095 over 5 steps)
    gaps = [f["gap"] for f in clean]
    assert all(b > a for a, b in zip(gaps, gaps[1:]))
    assert gaps[0] > 0.6

    # rank evidence clean, C_u row-space continuation small
    for f in clean:
        assert f["C"] == 2 and f["rank_status"] == "clean"
        if f["cu_rowspace_angle_deg"] is not None:
            assert f["cu_rowspace_angle_deg"] < 1.0   # measured 0.12-0.17

    # per-component conservation, GEOMETRIC quantities (ordered-polygon
    # area and phase centroid by Green's formula -- the continuum-conserved
    # objects; the earlier version mixed post-step geometry with pre-step
    # carrier masses and a boundary centroid)
    c0 = clean[0]["components"]
    c1 = clean[-1]["components"]
    for a, b in zip(c0, c1):
        assert abs(a["area_geom"] - b["area_geom"]) / abs(
            a["area_geom"]) < 5e-3
        assert max(abs(x - y) for x, y in
                   zip(a["centroid_geom"], b["centroid_geom"])) < 1e-4

    # superposition control: isolated runs concatenated = pair run
    # (measured max dx 3.4e-6 at step 0 growing ~8e-6/step)
    trajs = []
    for p in parts_pair():
        v = p
        stepper = MMStepper(make_config("M", "oracle", 1))
        snaps = []
        for _ in range(STEPS):
            v = stepper.step(v).varifold
            snaps.append(v)
        trajs.append(snaps)
    ref = cat(trajs[0][-1], trajs[1][-1])
    final = pair["snapshots"][-1]
    assert (final.positions - ref.positions).abs().max() < 1e-3
    assert circular_angle_diff(
        final.angles, ref.angles).abs().max() < 1e-3


def test_state_machine_requires_clean_initialisation():
    """Hard stop: the working rank cannot be initialised from non-clean
    evidence. The near-tangent pair at gap = 0.08 gives weak_gap on its
    very first frame (measured in 0C-5c: winning ratio 10.4 < 30) -- the
    stepper must refuse rather than guess a starting rank."""
    import pytest
    from spectral_flux_measure_0c5c import ellipse as mk_ellipse
    from src.torch.transport.bem_wasserstein import (
        AmbiguousComponentRankError,
    )

    gap = 0.08
    v = cat(mk_ellipse(0.7, 0.4, -0.7 - gap / 2),
            mk_ellipse(1.2, 0.8, 1.2 + gap / 2))
    stepper = MMStepper(make_config("M", "state_machine", None))
    with pytest.raises(AmbiguousComponentRankError, match="initialise"):
        stepper.step(v)


def test_superposition_defect_is_cross_block_and_refines():
    """Measured at T = 1e-3 (10 steps): dx(full, iso) = 7.6e-5 equals
    dx(full, blk) while dx(blk, iso) = 4.5e-6 -- zeroing the oracle cross
    blocks of S and K* on the SAME geometry (same masses, coherence, angle
    map) removes ~94% of the pair-vs-isolated defect, and the defect
    decreases under N-refinement (5.2e-5 at h = 0.025). Only with both
    facts may it be called a finite-N BIE cross-block error. Pinned on a
    2-step version."""
    from spectral_precontact_0c5d1 import attribution_control

    rows = attribution_control("M", n_steps=2)
    r = rows[0]
    assert r["dx_blk_iso"] < 0.3 * r["dx_full_iso"]
    assert abs(r["dx_full_blk"] - r["dx_full_iso"]) < 0.3 * r["dx_full_iso"]
    assert rows[1]["dx_full_iso"] < rows[0]["dx_full_iso"]


def test_weak_gap_holds_rank_and_evolution_continues():
    """0C-5d-1.1 headline: same-rank weak evidence must HOLD rank 2 and
    let the evolution genuinely continue (the pre-state-machine version
    stopped the run WITHOUT advancing the geometry): 14 steps must
    produce 14 evolved frames with strictly increasing gap across the
    clean -> weak_gap boundary.

    RE-SCOPED 2026-08-06 (C_2D angle-map fix): the fixture's original
    NATURAL confidence decay (gap ratio 62 -> <30 crossing
    bem_rank_gap_min=30 within ~10 steps while gap grows and q_min
    stays flat) was largely an artifact of the pre-fix under-rotating
    angle map rotting the small component's null block; post-fix the
    ratio declines only 61.9 -> 60.5 in 14 steps -- rank evidence is
    genuinely more stable now. The weak episode is therefore provoked
    via gap_min_override=200 after the clean init frame, so the SAME
    hold-and-continue machine logic is exercised on the real
    evolution. The pre-fix 62 -> <30 decline is history; the decline
    direction is still pinned."""
    n = 14
    pair = evolve_pair("M", n_steps=n, gap_min_override=200.0)
    clean = [f for f in pair["frames"] if "hard_stop" not in f]
    assert len(clean) == n                          # no stop at step ~10
    statuses = [f["rank_status"] for f in clean]
    assert "weak_gap" in statuses                   # the event did occur
    for f in clean:
        assert f["C"] == 2                          # held rank
        assert f["rank_action"] in ("hold_clean", "hold_weak")
        assert f["q_min"] > 0.99                    # NOT a coherence event
    gaps = [f["gap"] for f in clean]
    assert all(b > a for a, b in zip(gaps, gaps[1:]))  # really evolving
    ratios = [f["rank_gap_ratio"] for f in clean]
    abs_levels = [f["rank_abs_level"] for f in clean]
    assert ratios[-1] < ratios[0]                   # measured 62 -> <30
    assert abs_levels[-1] > abs_levels[0]           # confidence decay pinned
