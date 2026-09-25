"""D1 regressions: the shared dead-point rule helper and the D1a-ODE
calibration pins (raw tables in results/d1a_ode).

Measured headline (exact-snapshot calibration, no optimizer/BEM;
CLAIM SCOPE per review: consistency with a common fixed-resolution
dt -> 0 first-deletion radius ON THE SYMMETRIC RADIAL PATH -- active
feedback, progressive deletion and the joint (N, dt) limit remain
open): the lagged offset is, to a few percent, the analytic ONE-STEP
DELAY Rdot(R_del) * dt (n_outer = 128, rho = 0.3, dt = 1e-4: measured
-3.34e-3 vs predicted -3.39e-3); linear-in-dt intercepts of the three
rules coincide within 2e-4..2e-3. semi-implicit sits 20-50x closer to
refreshed than lagged does on THIS path and parameter set (the
threshold is a nonlinear relative quantity -- no additive m/q
decomposition is claimed). first = all deletion by symmetry. Validity:
R_ref/delta = 0.22/0.40/0.75 (rho = 0.2) and 0.32/0.56/1.10
(rho = 0.3) for n_outer = 64/128/256 -- most crossings live inside the
estimator's closure layer; only the finest rho = 0.3 case clears it.
The O(dt) delay is NOT uniform as R_- -> 0 (|Rdot| ~ 1/(R^2 log)):
eta_lag = |Rdot| dt / R_- belongs in the D1a-MM diagnostics.
"""

import math
import sys
from pathlib import Path

import torch

from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
)
from src.torch.shapes.generator import (
    annulus_point_counts,
    generate_oriented_annulus,
)
from src.torch.solver.dead_point_rules import (
    evaluate_dead_points,
    mask_symmetric_difference,
)
from src.torch.transport import compute_coherence

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

DT = torch.float64


def _snap(n_out, n_in, Rp, Rm):
    return generate_oriented_annulus(n_out, Rp, Rm, (0.0, 0.0), "cpu",
                                     DT, n_inner=n_in)


def test_rules_agree_when_geometry_static():
    """With pre == post geometry the three rules are algebraically
    identical: same scores, same masks, empty symmetric differences."""
    n_out, n_in = annulus_point_counts(96, 1.0, 0.5)
    v = _snap(n_out, n_in, 1.0, 0.5)
    delta, tau = compute_recommended_params(v.positions)
    m = compute_masses(v.positions, delta, tau)
    q = compute_coherence(v, m, 0.1, "wendland_c2")
    decs = {r: evaluate_dead_points(v, m, q, r, 0.3, delta_for_kde=delta,
                                    tau_for_kde=tau, sigma=0.1)
            for r in ("legacy_lagged", "semi_implicit", "refreshed")}
    for r, d in decs.items():
        assert torch.allclose(d.score, decs["legacy_lagged"].score,
                              atol=1e-12)
        assert not mask_symmetric_difference(d, decs["legacy_lagged"])
        assert d.n_removed == 0


def test_symmetric_annulus_deletes_all_inner_at_once():
    """Rotation symmetry: every inner point carries the same score, so
    the crossing removes the WHOLE inner ring in one evaluation -- the
    clean calibration case the review predicted."""
    n_out, n_in = annulus_point_counts(128, 1.0, 0.5)
    v0 = _snap(n_out, n_in, 1.0, 0.5)
    delta, tau = compute_recommended_params(v0.positions)
    # safely BELOW the measured refreshed crossing (R_del ~ 0.1382)
    v_pre = _snap(n_out, n_in, 0.9309, 0.1350)
    v_post = _snap(n_out, n_in, 0.9300, 0.1280)
    m = compute_masses(v_pre.positions, delta, tau)
    q = compute_coherence(v_pre, m, 0.1, "wendland_c2")
    dec = evaluate_dead_points(v_post, m, q, "refreshed", 0.3,
                               delta_for_kde=delta, tau_for_kde=tau,
                               sigma=0.1)
    inner = dec.score[n_out:]
    assert (inner.max() - inner.min()) / inner.max() < 1e-10
    # STRICT (review D1a.1-6): the geometry is chosen safely below the
    # measured crossing radius, so the deletion must fire -- the whole
    # inner ring goes, the whole outer ring stays. No vacuous pass.
    assert dec.n_removed == n_in
    assert (~dec.keep_mask[n_out:]).all()
    assert dec.keep_mask[:n_out].all()


def test_degenerate_scores_keep_everything_consistently():
    """max(score) <= 0: keep ALL points with a +inf margin -- the
    previous contract returned an all-True mask alongside all-negative
    margins (self-contradictory; review D1a.1-5). Validation errors are
    ValueError, not stripped asserts."""
    import pytest
    from src.torch.oriented_varifold import OrientedPointCloudVarifold

    v = OrientedPointCloudVarifold(
        positions=torch.randn(8, 2, dtype=DT),
        angles=torch.zeros(8, dtype=DT))
    z = torch.zeros(8, dtype=DT)
    dec = evaluate_dead_points(v, z, z, "legacy_lagged", 0.3)
    assert dec.keep_mask.all() and dec.n_removed == 0
    assert torch.isinf(dec.threshold_margin).all()
    assert (dec.threshold_margin > 0).all()

    with pytest.raises(ValueError):
        evaluate_dead_points(v, z, z, "legacy_lagged", 1.5)
    with pytest.raises(ValueError):
        evaluate_dead_points(v, z, z, "semi_implicit", 0.3)


def test_exact_index_shift_identity():
    """The STRONG pin (review): on the shadow calibration with fixed
    layout and scales, e_lag^{n+1} == e_ref^n is an EXACT tensor
    identity (same function on the same tensors -> bitwise), not an
    asymptotic statement. Margins and masks shift identically."""
    from d1a_ode_calibration import ode_radii

    n_out, n_in = annulus_point_counts(128, 1.0, 0.5)
    v0 = _snap(n_out, n_in, 1.0, 0.5)
    delta, tau = compute_recommended_params(v0.positions)
    dt = 2e-4
    t1, t2 = 20 * dt, 21 * dt
    E1 = _snap(n_out, n_in, *ode_radii(t1))
    E2 = _snap(n_out, n_in, *ode_radii(t2))
    m1 = compute_masses(E1.positions, delta, tau)
    q1 = compute_coherence(E1, m1, 0.1, "wendland_c2")

    ref_n = evaluate_dead_points(E1, m1, q1, "refreshed", 0.3,
                                 delta_for_kde=delta, tau_for_kde=tau,
                                 sigma=0.1)
    lag_n1 = evaluate_dead_points(E2, m1, q1, "legacy_lagged", 0.3,
                                  delta_for_kde=delta, tau_for_kde=tau,
                                  sigma=0.1)
    assert torch.equal(lag_n1.score, ref_n.score)
    assert torch.equal(lag_n1.threshold_margin, ref_n.threshold_margin)
    assert torch.equal(lag_n1.keep_mask, ref_n.keep_mask)


def test_lagged_offset_is_the_one_step_delay():
    """Radius-difference pin, now against the EXACT one-step difference
    R(t_ref + dt) - R(t_ref) at the refreshed-margin root (Rdot*dt is
    only its first-order expansion). Measured agreement is 2-3 digits
    at every dt in the sweep."""
    from d1a_ode_calibration import crossing_radius

    c = crossing_radius(128, 1e-4, 0.3)["crossings"]
    meas = c["legacy_lagged"]["R_del"] - c["refreshed"]["R_del"]
    pred = c["legacy_lagged"]["lag_predicted_exact"]
    assert pred < 0 and meas < 0
    assert abs(meas - pred) / abs(pred) < 0.05


def test_lagged_offset_halves_with_dt():
    """The D1a-ODE headline on a cheap two-dt slice (N = 128,
    rho = 0.3): |R_lag - R_ref| at dt = 4e-4 is ~2x the value at
    dt = 2e-4 (measured 1.42e-2 vs 6.79e-3), while semi-ref stays below
    1e-3 at both."""
    from d1a_ode_calibration import crossing_radius

    r4 = crossing_radius(128, 4e-4, 0.3)["crossings"]
    r2 = crossing_radius(128, 2e-4, 0.3)["crossings"]
    d4 = r4["legacy_lagged"]["R_del"] - r4["refreshed"]["R_del"]
    d2 = r2["legacy_lagged"]["R_del"] - r2["refreshed"]["R_del"]
    assert d4 < 0 and d2 < 0                       # lag deletes LATER
    assert 1.6 < d4 / d2 < 2.5                     # ~linear in dt
    for rr in (r4, r2):
        assert abs(rr["semi_implicit"]["R_del"]
                   - rr["refreshed"]["R_del"]) < 1e-3
        assert all(rr[k]["all_inner_removed"] for k in rr)


def test_mm_shadow_identity_and_window():
    """D1a-MM wiring gate on a short corrected-annulus run: the one-step
    identity e_lag^{n+1} == e_ref^n holds BITWISE on the actual MM
    trajectory (measured: 1422/1422 and 1440/1440 steps on the full
    production-scale runs), and the shadow margins stay positive far
    from the crossing radius."""
    from d1a_mm_shadow import run
    from pathlib import Path as _P
    import tempfile

    d = run(96, 5e-5, 0.001, _P(tempfile.mkdtemp()))
    assert d["identity_checked_steps"] == len(d["frames"]) - 1
    for f in d["frames"]:
        assert f["margin_refreshed_rho0.3"] > 0
        assert f["symdiff_lag_ref_rho0.3"] == 0
        assert f["carrier_resolved"]


def _d1b_scales():
    n_out, n_in = annulus_point_counts(128, 1.0, 0.5)
    v = _snap(n_out, n_in, 1.0, 0.5)
    return (n_out, n_in) + compute_recommended_params(v.positions)


def _assert_completed(r):
    """D1b.1 item 3: fail-closed run pins. MMSolver catches internal
    exceptions and returns a short history, so event-only assertions can
    otherwise pass silently after a later failure (the D0.1 import-path
    incident)."""
    assert r["n_steps_completed"] == r["n_steps_requested"]
    assert r["timeline_errors"] == []


def test_d1b_radial_active_wiring():
    """D1b-R pins (full tables in results/d1b_active): with the rules
    ACTIVE from a near-threshold radial snapshot, (i) the active event
    step equals the rule's own shadow prediction (deletion is wired at
    the intended time level); (ii) rotation symmetry deletes the WHOLE
    inner ring in one event, the tombstone records loop 1, and the
    post-event support is single_closed; (iii) legacy_lagged fires
    exactly ONE step after refreshed (the D1a one-step shift, now in
    the active setting -- a statement about the FIRST event on the
    common trajectory only); (iv) the geometric area jump equals MINUS
    the disappearing hole loop's signed polygon area -- a
    finite-resolution post-processing event that fills the hole
    instantly, NOT continuum closure."""
    from d1b_active_rules import run_rule

    n_out, n_in, delta, tau = _d1b_scales()
    runs = {rule: run_rule(rule, 0.0, 14, delta, tau)
            for rule in ("legacy_lagged", "semi_implicit", "refreshed")}

    for rule, r in runs.items():
        _assert_completed(r)
        assert len(r["events"]) == 1
        assert r["active_first_event"] == r["shadow_first_event"]
        ev = r["events"][0]
        assert ev["n_removed"] == n_in and ev["survivors_inner"] == 0
        assert ev["vanished"] == [1]
        assert sorted(ev["deleted_ids"]) == list(range(n_out,
                                                       n_out + n_in))
        assert r["support_states"][-1][1] == "single_closed"
        # whole-loop extinction leaves a valid support: never sticky ...
        assert not r["requires_reconstruction"]
        # ... but the loop-count change IS a pending topology event that
        # closes the gates until the timeline certifies it (policy B)
        assert len(r["pending_topology_events"]) == 1
        assert r["pending_topology_events"][0]["loop_index"] == 1
        # exact identity Delta A_geom == -A_hole^pre (signed polygon
        # area; review D1b.1 item 2 -- no pi R^2 proxy). The outer loop
        # is untouched by the deletion, so this holds to rounding.
        hole = r["hole_area_pre_first_event"]
        assert hole < 0
        assert abs(r["area_jump_at_first_event"] + hole) \
            <= 1e-10 * abs(hole)
    assert (runs["legacy_lagged"]["active_first_event"]
            == runs["refreshed"]["active_first_event"] + 1)
    assert (runs["semi_implicit"]["active_first_event"]
            == runs["refreshed"]["active_first_event"])


def test_d1b_perturbed_progressive_deletion_sticky_support():
    """D1b-P pins: a k = 2 inner perturbation (eps = 0.03) makes scores
    theta-dependent, so the first event removes only PART of the ring
    (measured 28/64 for refreshed) and the committed support enters the
    open_after_deletion stress track. D1b.1 item 5: that invalidity is
    STICKY -- a later frame that happens to reclassify as closed must
    not restore production permissions; every gate stays closed until an
    explicit certificate is accepted."""
    from src.torch.diagnostics import permissions
    from d1b_active_rules import run_rule

    n_out, n_in, delta, tau = _d1b_scales()
    r, tl = run_rule("refreshed", 0.03, 12, delta, tau,
                     return_timeline=True)
    _assert_completed(r)
    assert r["events"], "no deletion fired in the pinned window"
    ev = r["events"][0]
    assert 0 < ev["n_removed"] < n_in
    assert ev["survivors_inner"] == n_in - ev["n_removed"]
    assert r["active_first_event"] == r["shadow_first_event"]
    states = dict(r["support_states"])
    assert states[ev["step"]] == "open_after_deletion"

    assert tl.requires_reconstruction and r["requires_reconstruction"]
    # a pre-event frame is perfectly valid on its own ...
    early = next(rec.report for rec in tl.records
                 if rec.level == "post_deletion_final" and rec.step == 0)
    assert permissions(early)["allow_rank_refresh"]
    # ... but the sticky trajectory state closes every gate on it
    assert not any(tl.gated_permissions(early).values())


def test_d1b_cross_rule_first_event_masks():
    """D1b.1 item 1: the central semi vs refreshed finding, pinned on
    the COMMON valid pre-deletion geometry with its margins. Measured:
    refreshed removes 28, semi_implicit 24; the symmetric difference is
    ONE symmetry orbit of 4 inner particles (theta ~ +-0.93, +-2.21).
    HONEST CLASSIFICATION (review): those 4 points sit within 1.5e-4 of
    the threshold for BOTH rules (margins -4.2e-5 refreshed vs +1.5e-4
    semi), so the first-event mask difference is a NEAR-THRESHOLD
    effect -- sensitive to rho_dead, dt and rounding -- not yet a
    robust production-rule discriminator. The pin asserts the sign
    structure AND the smallness, so a future change that turns this
    into a robust O(1e-2) split will be flagged as loudly as one that
    erases it. legacy@step10 == refreshed@step9 is pinned bitwise
    (first-event one-step shift on the common trajectory; later event
    pairs like 31/32 are post-divergence stress coincidences and are
    deliberately NOT pinned)."""
    from d1b_active_rules import cross_rule_first_event

    n_out, n_in, delta, tau = _d1b_scales()
    cr = cross_rule_first_event(delta, tau)

    # RE-PINNED 2026-08-06 (C_2D angle-map fix): the docstring above
    # anticipated exactly this -- the 4-point split was a NEAR-THRESHOLD
    # effect, and the corrected angle map (compute_kernel_gradient was
    # missing the C_2D mollifier normalization; normals under-rotated
    # by pi/7) moved those 4 points onto the remove side of BOTH rules.
    # Measured post-fix: refreshed == semi == legacy@10 == the same 28
    # ids, symmetric difference EMPTY. The rules now agree on the first
    # event; the one-step lag structure (legacy fires one step later)
    # is unchanged and still pinned bitwise. This CONFIRMS the honest
    # classification: the split never was a production-rule
    # discriminator. Pre-fix values (28/24, orbit of 4 at
    # theta ~ +-0.93, +-2.21, margins -4.2e-5 / +1.5e-4) remain in the
    # docstring as the historical record.
    ref, semi = cr["removed_refreshed"], cr["removed_semi"]
    lag = cr["removed_legacy_step10"]
    assert len(ref) == 28 and len(semi) == 28 and len(lag) == 28
    inner = set(range(n_out, n_out + n_in))
    assert set(ref) <= inner and set(semi) <= inner and set(lag) <= inner
    assert set(semi) == set(ref)         # post-fix: rules agree
    diff = cr["symmetric_difference_ids"]
    assert len(diff) == 0

    assert cr["legacy10_equals_refreshed9_score"]
    assert cr["legacy10_equals_refreshed9_mask"]
    assert lag == ref


def test_d1b_loop_extinction_certificate():
    """D1 closure review (policy B): EVERY boundary-loop-count change is
    a pending topology event that closes the gates; the timeline itself
    revalidates the event FROM ITS OWN RECORDS
    (certify_and_accept_loop_extinction) -- a directly instantiated,
    foreign or wrong-step certificate can clear nothing. A phase-rank
    change and a partial deletion are rejected; resolving the event
    never launders a separate sticky invalid episode; the used
    quantities and rule payload on the raw record are the ones the rule
    actually consumed."""
    import pytest
    import torch as _t
    from src.torch.diagnostics import permissions
    from d1b_active_rules import run_rule

    n_out, n_in, delta, tau = _d1b_scales()
    r, tl = run_rule("refreshed", 0.0, 12, delta, tau,
                     return_timeline=True)
    e0 = r["events"][0]["step"]
    valid = next(rec.report for rec in tl.records
                 if rec.level == "post_deletion_final" and rec.step == 0)

    # the un-certified loop-count change closes every gate on a frame
    # that is perfectly valid on its own
    assert len(tl.pending_topology_events) == 1
    assert permissions(valid)["allow_rank_refresh"]
    assert not any(tl.gated_permissions(valid).values())

    # timeline used-quantity contract (review): the raw record stores
    # what the REFRESHED rule consumed -- current-geometry m,q, not the
    # legacy pre-MM tensors -- plus the actual score/threshold/margin
    raw = next(x for x in tl.records
               if x.level == "post_deletion_raw" and x.step == e0)
    cand = next(x for x in tl.records
                if x.level == "post_mm_candidate" and x.step == e0)
    assert raw.rank_fields["dead_point_rule"] == "refreshed"
    assert _t.equal(raw.mq_used, raw.rank_fields["dead_point_score"])
    assert not _t.equal(raw.m_used, cand.m_used)   # refreshed != pre-MM
    kept = _t.tensor(raw.keep_mask)
    assert (raw.rank_fields["dead_point_score"][kept]
            >= raw.rank_fields["dead_point_threshold"]).all()

    # wrong step / rank change are rejected, the event stays pending
    with pytest.raises(ValueError):
        tl.certify_and_accept_loop_extinction(
            e0 + 1, phase_rank_before=1, phase_rank_after=1)
    with pytest.raises(ValueError):
        tl.certify_and_accept_loop_extinction(
            e0, phase_rank_before=2, phase_rank_after=1)
    assert len(tl.pending_topology_events) == 1

    # the timeline's own revalidation resolves it and reopens the gates
    cert = tl.certify_and_accept_loop_extinction(
        e0, phase_rank_before=1, phase_rank_after=1)
    assert cert.loop_index == 1 and cert.n_removed == n_in
    assert cert.hole_area_pre < 0
    assert abs(cert.area_jump + cert.hole_area_pre) \
        <= 1e-10 * abs(cert.hole_area_pre)
    assert not tl.pending_topology_events
    assert tl.gated_permissions(valid) == permissions(valid)

    # resolving an event never clears a sticky invalid episode
    tl.requires_reconstruction = True
    assert not any(tl.gated_permissions(valid).values())

    # partial deletion: no loop vanishes -> no pending extinction event,
    # nothing to certify; the sticky flag from the open support stands
    rp, tlp = run_rule("refreshed", 0.03, 10, delta, tau,
                       return_timeline=True)
    ep = rp["events"][0]["step"]
    assert tlp.pending_topology_events == []
    assert tlp.requires_reconstruction
    with pytest.raises(ValueError):
        tlp.certify_and_accept_loop_extinction(
            ep, phase_rank_before=1, phase_rank_after=1)


def test_d1b_progressive_all_rules_fail_closed():
    """Review (D1 closure): the pytest completion/error pins must cover
    the ACTIVE progressive runs of all three rules, not refreshed only.
    Science needs only the first event, so 12 steps suffice -- the
    31/32/44 stress tails stay in the committed JSON, deliberately
    unpinned."""
    from d1b_active_rules import run_rule

    n_out, n_in, delta, tau = _d1b_scales()
    first = {}
    for rule in ("legacy_lagged", "semi_implicit", "refreshed"):
        r = run_rule(rule, 0.03, 12, delta, tau)
        _assert_completed(r)
        assert r["events"], f"{rule}: no deletion in the pinned window"
        ev = r["events"][0]
        assert 0 < ev["n_removed"] < n_in
        assert r["active_first_event"] == r["shadow_first_event"]
        assert r["requires_reconstruction"]
        first[rule] = (ev["step"], ev["n_removed"])
    # re-pinned 2026-08-06 with the C_2D angle-map fix: semi_implicit's
    # first event grew 24 -> 28 (the near-threshold orbit of 4 joined;
    # see test_d1b_cross_rule_first_event_masks); step structure intact
    assert first["refreshed"] == (9, 28)
    assert first["semi_implicit"] == (9, 28)
    assert first["legacy_lagged"] == (10, 28)


def test_d3_bundle_pins():
    """D3 pins (raw tables in results/d3_bundles): the full-bundle
    audit on the radial extinction track.
      (i)   the three legacy-deletion bundles (every-step / adaptive /
            fixed-interval redistribution) delete the whole inner ring
            at the SAME step with certificate-clean extinction --
            deletion timing is robust to a 5x invocation reduction;
      (ii)  the refreshed bundle fires exactly one step earlier (the
            D1 one-step shift survives the full bundle);
      (iii) the extinction certificate validates for EVERY bundle now
            that its pre-record is post_redistribution (the geometry
            deletion acted on) and its jump is candidate-free
            (deletion-only, raw-record based) -- both discovered
            through D3 wiring;
      (iv)  adaptive == every-step BITWISE here: the global density-CV
            trigger fires permanently on the shrunk annulus
            (inter-loop density contrast, a real trigger blind spot).
    Cheap slice: two bundles on the radial track."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d2b0_timelevel_shadow import snapshot
    from d3_bundles import run_bundle

    v0, _, _ = snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    le = run_bundle("LE", 0.0, 12, delta, tau)
    lf = run_bundle("LF", 0.0, 12, delta, tau)
    for r in (le, lf):
        assert r["n_steps_completed"] == 12
        assert r["timeline_errors"] == []
        assert r["events"] and r["events"][0]["survivors_inner"] == 0
        assert r["extinction_certificate"] is not None
        assert "error" not in r["extinction_certificate"]
        assert not r["pending_topology_events"]     # certified
    assert le["first_event"] == lf["first_event"]   # timing robust
    assert lf["regular_redist_invocations"] \
        < 0.3 * le["regular_redist_invocations"]


def test_p2_support_gate_enforcement():
    """P2 (production integration): with enforce_support_gates=True the
    solver CONSUMES the sticky gates. Radial whole-loop extinction is
    certified INLINE and the run continues to completion on the
    certified single loop (pending events resolved); the partial
    deletion's open support closes the gate and the solver stops
    CONTINUATION right after the committed event step with
    stop_stage='support_gate'. Default (False) leaves trajectories
    untouched (goldens cover bitwise)."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d1a_mm_shadow import make_config
    from d1b_active_rules import start_cloud
    from d2b0_timelevel_shadow import snapshot
    from src.torch.diagnostics import SupportTimeline
    from src.torch.solver.mm_solver import MMSolver

    delta, tau = compute_recommended_params(snapshot(0.5)[0].positions)

    def _run(eps, n_steps):
        v, n_out, n_in = start_cloud(eps)
        cfg = make_config(2e-5, delta, tau)
        cfg.remove_dead_points = True
        cfg.dead_point_threshold = 0.3
        cfg.dead_point_rule = "refreshed"
        cfg.enforce_support_gates = True
        solver = MMSolver(cfg)
        solver.support_timeline = SupportTimeline(
            initial_loops=[torch.arange(n_out),
                           torch.arange(n_out, n_out + n_in)])
        hist = solver.solve(v, n_steps)
        return hist, solver.support_timeline

    # radial: extinction certified inline, run continues to completion
    hist, tl = _run(0.0, 13)
    assert hist.n_completed == 13
    assert hist.stop_step is None
    assert not tl.pending_topology_events        # resolved inline
    assert len(tl.accepted_certificates) == 1

    # partial: open support closes the gate right after the event
    hist, tl = _run(0.03, 13)
    assert hist.stop_stage == "support_gate"
    assert hist.stop_exception_type == "SupportGateClosed"
    assert hist.n_completed == hist.stop_step + 1   # step committed
    assert hist.n_completed < 13                    # stopped early
    assert tl.requires_reconstruction


def test_p21_gate_semantics():
    """P2.1 regressions (review): (i) enforce_support_gates=True with no
    attached timeline fails closed BEFORE stepping; (ii) a whole-loop
    extinction with NO rank evidence (legacy solver, no oracle rank)
    stops instead of certifying via None == None; (iii) a classifier
    failure (no support report on the committed record) stops
    structurally; (iv) after an invalid-terminal gate stop the terminal
    frame is an OBSERVATION -- get_final_valid_varifold() returns the
    last valid committed state, not the open-support frame."""
    import pytest
    from unittest.mock import patch
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d1a_mm_shadow import make_config
    from d1b_active_rules import start_cloud
    from d2b0_timelevel_shadow import snapshot
    from src.torch.diagnostics import SupportTimeline
    from src.torch.solver.mm_solver import MMSolver

    delta, tau = compute_recommended_params(snapshot(0.5)[0].positions)

    def _cfg(**kw):
        cfg = make_config(2e-5, delta, tau)
        cfg.remove_dead_points = True
        cfg.dead_point_threshold = 0.3
        cfg.dead_point_rule = "refreshed"
        cfg.enforce_support_gates = True
        for k, val in kw.items():
            setattr(cfg, k, val)
        return cfg

    def _tl(n_out, n_in):
        return SupportTimeline(
            initial_loops=[torch.arange(n_out),
                           torch.arange(n_out, n_out + n_in)])

    # (i) no timeline -> preflight ValueError, zero steps run
    v, n_out, n_in = start_cloud(0.0)
    with pytest.raises(ValueError, match="SupportTimeline"):
        MMSolver(_cfg()).solve(v, 1)

    # (ii) extinction with no rank evidence: legacy solver mode carries
    # neither a rank decision nor an oracle rank -> fail closed at the
    # event (None == None must never certify)
    cfg = _cfg(bem_solver_mode="legacy_global_projection",
               bem_rank_mode="oracle", bem_component_rank=None)
    solver = MMSolver(cfg)
    solver.support_timeline = _tl(n_out, n_in)
    hist = solver.solve(v, 13)
    assert hist.stop_stage == "support_gate"
    assert "NO rank evidence" in hist.stop_message
    assert solver.support_timeline.pending_topology_events  # unresolved

    # (iii) classifier failure -> no report on the committed record ->
    # structured stop (fail closed), not a silent continuation
    v2, _, _ = start_cloud(0.0)
    solver = MMSolver(_cfg())
    solver.support_timeline = _tl(n_out, n_in)
    with patch("src.torch.diagnostics.timeline.classify_support",
               side_effect=RuntimeError("injected classifier failure")):
        hist = solver.solve(v2, 2)
    assert hist.stop_stage == "support_gate"
    assert "no support report" in hist.stop_message

    # (iv) invalid terminal frame vs last valid state
    v3, _, _ = start_cloud(0.03)
    solver = MMSolver(_cfg())
    solver.support_timeline = _tl(n_out, n_in)
    hist = solver.solve(v3, 13)
    assert hist.stop_stage == "support_gate"
    assert not hist.terminal_candidate_valid
    assert hist.last_valid_step == hist.stop_step - 1
    v_term = hist.get_final_varifold()
    v_valid = hist.get_final_valid_varifold()
    assert v_term.n_points < v_valid.n_points     # terminal lost points
