"""D2a regressions: the shared redistribution-rule helper and the
operator-level calibration pins (raw tables in results/d2a_redist).

Measured headline (operator level, no MM step; production parameters
step_size 0.01, tol 1e-4, max_disp_ratio 0.05, delta_redist 0.5 delta):

  - legacy_hybrid reproduces production `redistribute_points` BITWISE
    (wiring gate) and COINCIDES with post_mm_frozen at operator level --
    their difference is purely the staleness of q across an MM step,
    which only the in-solver audit (D2b) can measure.
  - At production-scale iteration counts (10-50) all three rules reduce
    the spacing CV with small drift; substep_refreshed consistently has
    2-10x SMALLER geometric drift (Hausdorff, area, perimeter) at
    nearly equal CV reduction. NOTE the frame subtlety: legacy moves
    exactly along the FIXED input tangents, so its per-update normal
    component w.r.t. the INPUT frame is tiny -- the support drift shows
    up in dH/dA/dP, not in max|dx.n|(input frame).
  - Under long SINGLE-CALL iteration, from the warped fixture and at
    the tested step/clipping parameters, the fixed-tangent explicit
    iteration fails to reach the prescribed tolerance and becomes
    geometrically unstable (circle, 2000 iters: CV_l 0.25 -> 1.15,
    area +25%, perimeter +69%). CLAIM SCOPE (review): this does NOT
    mean legacy has no reparametrization fixed point (uniform-density
    configurations are fixed points), and it does not by itself
    predict production dt-refinement accumulation -- production
    refreshes angles BETWEEN invocations (repeated-call control:
    D2a.1). substep_refreshed is substantially more stable (2000
    iters: dA +0.5%) but does NOT converge either: dH ~ 0.082,
    dP/P ~ 5.3%, its theta-CV rebounds, converged=False. Consequence:
    projection-to-tolerance is not established for EITHER current
    operator at these parameters. Long-iteration results are partly
    CLIPPING-DOMINATED (update norms pinned at max_disp for long
    stretches) -- the D2a.1 step-size sweep at fixed pseudo-time
    separates clipping effects from the vector field itself.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.solver.redistribute import redistribute_points
from src.torch.solver.redistribution_rules import (
    REDISTRIBUTION_RULES,
    redistribute_with_rule,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"
                       / "experiments"))

from d2a_redistribution_calibration import (  # noqa: E402
    DELTA_RATIO,
    MAX_DISP_RATIO,
    SIGMA,
    STEP_SIZE,
    TOL,
    curve_metrics,
    make_fixture,
    sampled_polyline_hausdorff,
)


def _setup(name):
    P0, ang0 = make_fixture(name)
    delta, tau = compute_recommended_params(P0)
    kw = dict(delta=delta, step_size=STEP_SIZE, tol=TOL,
              max_disp_ratio=MAX_DISP_RATIO, mass_tau=tau,
              delta_redist=DELTA_RATIO * delta)
    return P0, ang0, kw


def test_legacy_branch_is_production_bitwise():
    """Operator wiring gate: the shared helper's legacy branch must be
    the production redistribute_points BITWISE on the same inputs --
    otherwise the D2 audit audits something else."""
    P0, ang0, kw = _setup("ellipse")
    q = torch.full((P0.shape[0],), 0.9, dtype=P0.dtype)
    p1, a1, _ = redistribute_points(
        P0, ang0, kw["delta"], "wendland_c2", 10, STEP_SIZE, TOL,
        MAX_DISP_RATIO, kw["mass_tau"], kw["delta_redist"], coherence=q)
    p2, a2, _ = redistribute_with_rule(
        P0, ang0, "legacy_hybrid", coherence=q, n_iters=10, **kw)
    assert torch.equal(p1, p2) and torch.equal(a1, a2)


def test_legacy_equals_frozen_at_operator_level():
    """SCOPING pin: with q(input) supplied to legacy, legacy_hybrid and
    post_mm_frozen coincide bitwise at operator level -- the axis
    between them is the q staleness across an MM step (D2b), not the
    operator structure."""
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.oriented_varifold.mass import compute_masses
    from src.torch.transport import compute_coherence

    P0, ang0, kw = _setup("circle")
    m0 = compute_masses(P0, kw["delta"], kw["mass_tau"])
    q0 = compute_coherence(
        OrientedPointCloudVarifold(positions=P0, angles=ang0),
        m0, SIGMA, "wendland_c2")
    p1, a1, _ = redistribute_with_rule(
        P0, ang0, "legacy_hybrid", coherence=q0, n_iters=20, **kw)
    p2, a2, _ = redistribute_with_rule(
        P0, ang0, "post_mm_frozen", sigma=SIGMA, n_iters=20, **kw)
    assert torch.equal(p1, p2) and torch.equal(a1, a2)


def test_rules_reduce_spacing_cv_with_bounded_drift():
    """Production-scale invocation (50 iters): every rule reduces the
    spacing CV on every fixture, with the geometric support essentially
    unmoved (dH < 1e-2, |dA| < 1e-2); substep_refreshed's perimeter
    drift is smaller than legacy's on the ellipse (measured 3.1e-4 vs
    9.5e-3)."""
    for name in ("circle", "ellipse", "flower"):
        P0, ang0, kw = _setup(name)
        before = curve_metrics(P0)
        after = {}
        for rule in REDISTRIBUTION_RULES:
            P1, A1, _ = redistribute_with_rule(
                P0, ang0, rule,
                coherence=None, sigma=SIGMA, n_iters=50, **kw)
            m = curve_metrics(P1)
            assert m["cv_l"] < before["cv_l"]
            assert sampled_polyline_hausdorff(P0, P1) < 1e-2
            assert abs(m["area"] - before["area"]) < 1e-2
            after[rule] = m
        if name == "ellipse":
            assert (abs(after["substep_refreshed"]["perimeter"]
                        - before["perimeter"])
                    < 0.2 * abs(after["legacy_hybrid"]["perimeter"]
                                - before["perimeter"]))


def test_legacy_single_call_long_iteration_is_unstable():
    """NARROWED claim (review): from the warped circle at the tested
    step/clipping parameters, the single-call fixed-tangent iteration
    accumulates geometric drift without reaching tolerance (800 iters:
    dP +0.39 = 6% of 2pi and growing; spacing CV rebounds), while
    substep_refreshed at the same budget stays far closer to the curve
    (|dP| < 0.05) -- 'substantially more stable', NOT 'converged'.
    Both runs force q == 1 (q_override), removing the coherence
    confound the earlier version had (legacy q=None vs substep real
    q)."""
    P0, ang0, kw = _setup("circle")
    before = curve_metrics(P0)
    res = {}
    for rule in ("legacy_hybrid", "substep_refreshed"):
        P1, A1, _ = redistribute_with_rule(
            P0, ang0, rule, q_override="unit",
            n_iters=800, **kw)
        m = curve_metrics(P1)
        res[rule] = dict(cv_l=m["cv_l"],
                         dP=m["perimeter"] - before["perimeter"],
                         dH=sampled_polyline_hausdorff(P0, P1))
    leg, sub = res["legacy_hybrid"], res["substep_refreshed"]
    assert leg["dP"] > 0.2 and leg["dH"] > 0.06          # unstable
    assert abs(sub["dP"]) < 0.05 and sub["dH"] < 0.03    # far closer
    assert sub["cv_l"] < leg["cv_l"]                     # CV rebound


def test_rule_validation():
    P0, ang0, kw = _setup("circle")
    with pytest.raises(ValueError):
        redistribute_with_rule(P0, ang0, "bogus", **kw)
    with pytest.raises(ValueError):
        redistribute_with_rule(P0, ang0, "substep_refreshed", **kw)
    with pytest.raises(ValueError):
        redistribute_with_rule(P0, ang0, "legacy_hybrid",
                               q_override="fixed", **kw)
    with pytest.raises(ValueError):
        redistribute_with_rule(P0, ang0, "legacy_hybrid",
                               curvature_visible_qm=True, sigma=SIGMA,
                               **kw)


def test_repeated_production_calls_rehabilitate_legacy():
    """D2a.1 headline (section C): the production invocation pattern --
    SHORT calls with the angle correction applied between them, so
    tangents are rebuilt from corrected angles at every call -- is the
    most stable configuration tested. 80 x 10 legacy calls reach BETTER
    spacing CV than any single-call run (0.026) with ~100x less drift
    than the 1 x 800 single call (dP/P 5.4e-4 vs 6.2e-2). The
    single-call instability is therefore an artifact of tangent
    freezing over long stretches, which production avoids BY
    CONSTRUCTION; it does not, at this budget, indict the production
    invocation pattern. SCOPE (review): an operator-level result on one
    q ~ 1 warped circle with no MM step -- it does not yet establish
    dt-refinement stability of the full solver, noncircular stability,
    or q-dip behavior (D2b). The current operator is a discrete
    reparametrization map, not a flow discretization: splitting one
    call changes the result, so it is not semigroup-like."""
    P0, ang0, kw = _setup("circle")
    before = curve_metrics(P0)

    P_long, _, _ = redistribute_with_rule(
        P0, ang0, "legacy_hybrid", q_override="unit", n_iters=800, **kw)
    dP_long = abs(curve_metrics(P_long)["perimeter"]
                  - before["perimeter"]) / before["perimeter"]

    Pk, Ak = P0.clone(), ang0.clone()
    for _ in range(80):
        Pk, Ak, _ = redistribute_with_rule(
            Pk, Ak, "legacy_hybrid", q_override="unit", n_iters=10, **kw)
    mk = curve_metrics(Pk)
    dP_rep = abs(mk["perimeter"] - before["perimeter"]) \
        / before["perimeter"]

    assert mk["cv_l"] < 0.05                       # better equalization
    assert dP_rep < 2e-3                           # essentially no drift
    assert dP_rep < 0.1 * dP_long                  # >> single call
    assert sampled_polyline_hausdorff(P0, Pk) < 2e-3


def test_substep_is_call_boundary_invariant():
    """substep_refreshed refreshes everything inside the loop, so
    splitting one call into several changes NOTHING (bitwise) -- the
    call-boundary angle correction that rehabilitates legacy is already
    internal to it."""
    P0, ang0, kw = _setup("circle")
    p1, a1, _ = redistribute_with_rule(
        P0, ang0, "substep_refreshed", q_override="unit", n_iters=20,
        **kw)
    pk, ak = P0.clone(), ang0.clone()
    for _ in range(2):
        pk, ak, _ = redistribute_with_rule(
            pk, ak, "substep_refreshed", q_override="unit", n_iters=10,
            **kw)
    assert torch.equal(p1, pk) and torch.equal(a1, ak)


def test_fixed_pseudo_time_is_not_an_invariant():
    """D2a.1 section B (wording per review): at fixed redistribution
    pseudo-time s = n * lambda, HALVING lambda (doubling n) INCREASES
    the drift. The clip is a GLOBAL rescaling -- whenever it is active
    the entire update field is scaled so its maximum norm equals
    max_disp, independent of lambda -- so the accumulated displacement
    is governed by the iteration count, not by s: the current clipped
    map cannot be parameterized by s alone. (This does NOT preclude an
    underlying tangential flow; the unclipped / jointly-scaled regime
    is what D2b's dt_scaled_flow candidate must test, scaling BOTH the
    step and the clip.)"""
    P0, ang0, kw = _setup("circle")
    before = curve_metrics(P0)
    dP = {}
    for lam, n_it in ((0.01, 400), (0.005, 800)):
        kw_l = dict(kw, step_size=lam)
        P1, _, info = redistribute_with_rule(
            P0, ang0, "legacy_hybrid", q_override="unit",
            n_iters=n_it, **kw_l)
        dP[lam] = abs(curve_metrics(P1)["perimeter"]
                      - before["perimeter"])
        assert any(r["clip_active"] for r in info["trace"])
    assert dP[0.005] > dP[0.01]


def test_repeated_call_map_saturates_without_degradation():
    """D2a.1 fix 4: the repeated production-sized legacy map SATURATES
    -- CV_l plateaus (~0.015 by K=160) and the drift stops growing
    (dP/P ~ 5.4e-4 flat from K=80 through K=320). No long-horizon
    degradation of the K-call map on this fixture."""
    P0, ang0, kw = _setup("circle")
    before = curve_metrics(P0)
    marks, res = (80, 160, 320), {}
    Pk, Ak = P0.clone(), ang0.clone()
    for k in range(1, max(marks) + 1):
        Pk, Ak, _ = redistribute_with_rule(
            Pk, Ak, "legacy_hybrid", q_override="unit", n_iters=10, **kw)
        if k in marks:
            m = curve_metrics(Pk)
            res[k] = dict(cv_l=m["cv_l"],
                          dP=abs(m["perimeter"] - before["perimeter"]))
    # NOTE: the long-horizon state is summation-order sensitive -- the
    # branchy clipped map amplifies thread-count rounding differences
    # over 1600+ subiterations into different near-fixed configurations
    # (measured cv at K=160/320: 0.015/0.015 at OMP=2 but 0.049/0.073
    # at OMP=1). The environment-robust pins are: the warp stays
    # RELAXED (cv far below the initial 0.25 at every checkpoint) and
    # the geometric drift SATURATES (dP flat, not growing with K).
    assert res[160]["cv_l"] < 0.12 and res[320]["cv_l"] < 0.12
    assert res[320]["dP"] < 1.2 * res[80]["dP"]      # drift not growing


def test_unit_override_forces_unit_q_in_visible_qm_curvature():
    """Review fix 1: q_override='unit' must force q == 1 EVERYWHERE --
    with the visible-qm curvature knob the result must be bitwise the
    raw-m run (w_curv = m * 1), never a real-q recomputation."""
    P0, ang0, kw = _setup("flower")
    p1, a1, _ = redistribute_with_rule(
        P0, ang0, "substep_refreshed", q_override="unit",
        n_iters=30, **kw)
    p2, a2, _ = redistribute_with_rule(
        P0, ang0, "substep_refreshed", q_override="unit",
        curvature_visible_qm=True, n_iters=30, **kw)
    assert torch.equal(p1, p2) and torch.equal(a1, a2)


def test_q_confound_absent_on_circle():
    """D2a.1 section A: on the q ~ 1 circle fixture the unit-q and
    fixed-real-q controls agree to ~1% -- the legacy long-iteration
    instability attribution to tangent freezing survives the coherence
    control."""
    P0, ang0, kw = _setup("circle")
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.oriented_varifold.mass import compute_masses
    from src.torch.transport import compute_coherence
    m0 = compute_masses(P0, kw["delta"], kw["mass_tau"])
    q0 = compute_coherence(
        OrientedPointCloudVarifold(positions=P0, angles=ang0),
        m0, SIGMA, "wendland_c2")
    before = curve_metrics(P0)
    dP = {}
    for mode, extra in (("unit", dict(q_override="unit")),
                        ("fixed", dict(q_override="fixed",
                                       coherence=q0))):
        P1, _, _ = redistribute_with_rule(
            P0, ang0, "legacy_hybrid", n_iters=200, **kw, **extra)
        dP[mode] = curve_metrics(P1)["perimeter"] - before["perimeter"]
    assert abs(dP["unit"] - dP["fixed"]) < 0.02 * abs(dP["unit"])


def test_d2b0_shadow_axes_at_small_inner_ring():
    """D2b-0.1 pins (raw tables in results/d2b_redist) at R_- = 0.06,
    warped sampling, INSIDE the carrier closure layer (algorithmic
    audit data):
      (i)   on the exact PRE-MM snapshot legacy and frozen coincide
            EXACTLY (q_stale == q(input): the operator-level identity);
            on the post-MM candidate the staleness displacement is
            small on the local-spacing scale at this depth;
      (ii)  ONE production-sized legacy call is destructive on the
            small warped inner ring (inner dP/P > 0.1) while substep
            stays controlled (< 0.05) and equalizes it; at THIS radius
            both outputs remain valid_closed;
      (iii) cause attribution (controls): with the clip INACTIVE
            (lambda = 0.001) legacy still drifts far more than substep
            -- the global clip is NOT the primary cause; the failure is
            consistent with the fixed-tangent field under a global
            delta_redist that dwarfs the loop (delta_redist/R ~ 2);
      (iv)  the visible-qm curvature pairing activates at real q < 1
            but stays small."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d2b0_timelevel_shadow import audit_snapshot, snapshot

    v0, _, _ = snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    s = audit_snapshot(0.06, 0.895, delta, tau)
    assert "mm_step_error" not in s

    assert s["eta_step"] < 0.1                  # inside the step gate
    assert s["support_post"]["geometry_status"] == "valid_closed"
    assert s["staleness_inner"]["inf"] < 0.05
    assert s["staleness_outer"]["inf"] < 1e-5   # outer ring: none

    pre, post = s["shadow_pre_mm"], s["shadow_post_mm"]
    assert pre["dx_legacy_vs_frozen_inf"] == 0.0        # exact identity
    assert post["dx_legacy_vs_frozen_over_h_min_inner"] < 0.1
    assert post["dx_frozen_vs_substep_inf"] > \
        100 * post["dx_legacy_vs_frozen_inf"]

    leg = post["drift_legacy_hybrid"]
    sub = post["drift_substep_refreshed"]
    assert leg["inner"]["dP_rel"] > 0.1
    assert sub["inner"]["dP_rel"] < 0.05
    assert sub["inner"]["cv_l_after"] < sub["inner"]["cv_l_before"]
    assert leg["inner"]["cv_l_after"] > leg["inner"]["cv_l_before"]
    assert leg["validity"]["geometry_status"] == "valid_closed"
    assert sub["validity"]["geometry_status"] == "valid_closed"

    ctl = post["controls"]["small_step_lam0.001"]
    assert ctl["legacy_hybrid"]["clip_active_iters"] == 0
    assert ctl["legacy_hybrid"]["max_dP_rel"] > \
        5 * ctl["substep_refreshed"]["max_dP_rel"]

    assert 0 < post["dang_substep_rawm_vs_visibleqm_inf"] < 0.01


def test_solver_wiring_active_rules():
    """D2b-1 wiring smoke: the solver routes redistribution through the
    shared helper for every rule (legacy bitwise via the D0 goldens);
    frozen/substep run actively without errors and the
    post_redistribution record carries the rule payload."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d1a_mm_shadow import make_config
    from d2b0_timelevel_shadow import snapshot
    from src.torch.diagnostics import SupportTimeline
    from src.torch.solver.mm_solver import MMSolver

    v, n_out, n_in = snapshot(0.5, warp=0.35)
    delta, tau = compute_recommended_params(v.positions)
    for rule in ("post_mm_frozen", "substep_refreshed"):
        cfg = make_config(2e-5, delta, tau)
        cfg.redistribute = True
        cfg.redistribution_rule = rule
        solver = MMSolver(cfg)
        solver.support_timeline = SupportTimeline(
            initial_loops=[torch.arange(n_out),
                           torch.arange(n_out, n_out + n_in)])
        hist = solver.solve(v, 2)
        tl = solver.support_timeline
        assert hist.n_completed == 2
        assert all(r.error is None for r in tl.records)
        red = next(r for r in tl.records
                   if r.level == "post_redistribution"
                   and r.stage_executed)
        assert red.rank_fields["redistribution_rule"] == rule
        assert red.geometry_changed


def test_tangent_source_validation_and_default():
    """tangent_source knobs are audit-only; the default 'angles' path
    is bitwise the production behavior (covered by the parity test).
    'local_pca' is the permutation-invariant control; the
    ordered-loop oracle is quarantined (needs an explicit cyclic
    order); bogus values fail loudly."""
    P0, ang0, kw = _setup("circle")
    with pytest.raises(ValueError):
        redistribute_with_rule(P0, ang0, "legacy_hybrid",
                               tangent_source="bogus", **kw)
    p1, _, _ = redistribute_with_rule(
        P0, ang0, "legacy_hybrid", q_override="unit", n_iters=5, **kw)
    p2, _, _ = redistribute_with_rule(
        P0, ang0, "legacy_hybrid", q_override="unit", n_iters=5,
        tangent_source="local_pca", **kw)
    assert not torch.equal(p1, p2)    # the knob genuinely changes it


def test_d2b1_dt_invariance_of_cumulative_drift():
    """D2b-1R headline pins (full tables in results/d2b_redist,
    resolved track, warped annulus, deletion OFF): the every-step
    schedule is dt-refinement STABLE -- doubling the invocation count
    (dt 2e-5 -> 1e-5 at fixed T) leaves the cumulative inner-loop
    redistribution drift essentially unchanged (measured +2.6%),
    because the drift is the ONE-TIME cost of relaxing the initial
    sampling warp, not a per-invocation cost (uniform spacing is near
    a fixed point of the map). The F4 dt-dependence concern does not
    materialize on the resolved track. legacy and frozen stay within
    |dR| ~ 1e-7 (staleness invisible when redistribution keeps the
    sampling uniform)."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d2b0_timelevel_shadow import snapshot
    from d2b1_active_rules import run_one

    v0, _, _ = snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    a = run_one("legacy_hybrid", 0.35, 2e-5, 40, delta, tau)
    b = run_one("legacy_hybrid", 0.35, 1e-5, 80, delta, tau)
    c = run_one("post_mm_frozen", 0.35, 2e-5, 40, delta, tau)

    for r in (a, b, c):
        assert r["n_steps_completed"] == r["n_steps_requested"]
        assert r["timeline_errors"] == []
        assert r["cv_l_inner_final"] < 0.02      # warp relaxed
    assert b["invocations"] == 2 * a["invocations"]
    # dt invariance: doubling invocations adds < 10% cumulative drift
    assert b["sum_step_stage_dH_inner"] < 1.1 * a["sum_step_stage_dH_inner"]
    # staleness invisible on the resolved track
    assert abs(a["R_in_final"] - c["R_in_final"]) < 1e-6


def test_post_removal_redistribution_nonlegacy_fixture():
    """Review correction 6: an ACTUAL deletion event followed by the
    post-removal redistribution under a nonlegacy rule -- exercises the
    sliced stale-q path, the surviving-point count and the timeline
    payload through the shared helper. Radial near-threshold annulus:
    the whole inner ring deletes, then the post-removal substep
    redistribution runs on the surviving outer ring."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d1a_mm_shadow import make_config
    from d1b_active_rules import start_cloud
    from src.torch.diagnostics import SupportTimeline
    from src.torch.solver.mm_solver import MMSolver
    from d2b0_timelevel_shadow import snapshot

    v, n_out, n_in = start_cloud(0.0)
    delta, tau = compute_recommended_params(snapshot(0.5)[0].positions)
    cfg = make_config(2e-5, delta, tau)
    cfg.remove_dead_points = True
    cfg.dead_point_threshold = 0.3
    cfg.dead_point_rule = "refreshed"
    cfg.redistribute = True
    cfg.redistribution_rule = "substep_refreshed"
    cfg.redistribute_n_iters_after_removal = 3
    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(n_out),
                       torch.arange(n_out, n_out + n_in)])
    hist = solver.solve(v, 13)
    tl = solver.support_timeline
    assert hist.n_completed == 13
    assert all(r.error is None for r in tl.records)

    ev = next(r for r in tl.records
              if r.level == "post_deletion_raw" and r.n_removed > 0)
    assert ev.n_removed == n_in           # whole inner ring
    fin = next(r for r in tl.records
               if r.level == "post_deletion_final"
               and r.step == ev.step)
    assert fin.n_points == n_out          # survivors only
    # the post-removal substep redistribution actually moved the ring
    assert not torch.equal(fin.positions, ev.positions)


def test_cause_separation_axes_pinned():
    """D2b-1 correction pins (raw tables in results/d2b_redist):
    axis B (fixed q, matched local step u = 0.25 h_min, cap disabled):
    the PERMUTATION-INVARIANT local-PCA tangent refresh (delta_tan =
    3 h_min) removes most of the fixed-tangent drift (measured 1.7e-4
    vs 3.0e-3, within 2x of the ordered-loop ORACLE 9.7e-5): a LOCAL
    tangent refresh is SUFFICIENT to remove most of the fixed-tangent
    drift IN THIS CONTROLLED EXPERIMENT (review: NOT yet proven to be
    the dominant cause of the full-substep bundle -- the two are not
    nested ablations) -- PROVIDED the tangent bandwidth is local: at
    delta_tan = delta_redist the PCA neighbourhood covers the whole
    sub-bandwidth ring and degrades below fixed tangents (bandwidth
    sweep in JSON). axis C: an APPROXIMATELY QUADRATIC small-step
    onset in u/h at the upper end of the tested range (7e-4 at
    u/h=0.1 vs 4.6e-2 at u/h=1; sub-quadratic at the smallest u/h
    where the wanted resampling floor contributes) -- no exact power
    law is claimed."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d2b0_timelevel_shadow import cause_separation, snapshot

    v0, _, _ = snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    cs = cause_separation(0.03, delta, tau)
    B = cs["axis_B_tangent"]
    assert B["local_pca_refresh"]["inner_dP_rel"] \
        < 0.2 * B["fixed_tangent"]["inner_dP_rel"]
    assert B["ordered_loop_oracle"]["inner_dP_rel"] \
        < B["local_pca_refresh"]["inner_dP_rel"]
    C = cs["axis_C_step_scale"]
    assert C["u_over_h1.0"]["inner_dP_rel"] \
        > 10 * C["u_over_h0.1"]["inner_dP_rel"]
    assert C["u_over_h0.1"]["inner_dP_rel"] \
        > C["u_over_h0.025"]["inner_dP_rel"]


def test_d2b2_schedules_fire_correctly():
    """D2b-2 scheduling smoke: fixed_physical_interval fires at the
    dt-independent physical rate (Delta t_post = 1e-4 -> every 5th
    step at dt = 2e-5); adaptive_cv_trigger fires while the warp is
    unrelaxed (CV_theta,excl = 0.56 >> 0.05) and stays quiet on a
    uniform sampling (CV 0.0005 < 0.05). every_step reproduces the
    legacy interval logic (goldens cover bitwise)."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    import d2b0_timelevel_shadow as d2b0
    from d2b1_active_rules import run_one

    v0, n_out, _ = d2b0.snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)

    r = run_one("legacy_hybrid", 0.35, 2e-5, 10, delta, tau,
                schedule="fixed_physical_interval", phys_interval=1e-4)
    assert r["invocations"] == 2 and r["n_steps_completed"] == 10

    r = run_one("legacy_hybrid", 0.35, 2e-5, 6, delta, tau,
                schedule="adaptive_cv_trigger", trigger_cv=0.05)
    assert r["invocations"] == 6          # warp unrelaxed: fires
    r = run_one("legacy_hybrid", 0.0, 2e-5, 6, delta, tau,
                schedule="adaptive_cv_trigger", trigger_cv=0.05)
    assert r["invocations"] == 0          # uniform: quiet


def test_adaptive_trigger_metric_is_permutation_invariant():
    """The adaptive trigger consumes the self-excluded KDE-density CV
    -- a permutation-INVARIANT scalar (review: never ordered
    edge-length CV)."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d2b0_timelevel_shadow import snapshot
    from src.torch.solver.redistribute import (
        _compute_density_log_gradient)

    torch.manual_seed(3)
    v, n_out, _ = snapshot(0.4, warp=0.35)
    P = v.positions
    perm = torch.randperm(P.shape[0])
    _, th1 = _compute_density_log_gradient(P, 0.12, "wendland_c2")
    _, th2 = _compute_density_log_gradient(P[perm], 0.12, "wendland_c2")
    cv1 = (th1.std() / th1.mean()).item()
    cv2 = (th2.std() / th2.mean()).item()
    assert abs(cv1 - cv2) < 1e-13


def test_d2b2_tradeoff_headlines_pinned():
    """D2b-2 headline pins, read from the COMMITTED raw JSON (the runs
    are 200-800 MM steps each -- rerunning in pytest is not
    proportionate; regeneration must reproduce these):
      (i)   fixed_physical_interval invokes at a dt-INDEPENDENT rate
            (inv == 40 at every dt);
      (ii)  the warp-relaxation radius bias of the redistribution
            (~9e-4 on the warped fixture) is dt-STABLE across three dt
            levels (spread < 5e-5: an initialization-layer effect, not
            an accumulating one) ...
      (iii) ... and production-coupled refinement (N and the
            recommended bandwidths move together) reduces it ~2.5x at
            N=256 (3.5e-4). NARROWED (review): no 1/N rate and no
            joint-limit claim from two N levels with N-dependent
            bandwidths -- the D2b-2.1 refinement rows (N=512 +
            fixed-delta/sigma control) address the separation;
      (iv)  on an already-uniform sampling the adaptive trigger skips
            most calls (34 vs 200) and leaves ~50x less net support
            difference than every_step;
      (v)   zero timeline errors anywhere."""
    import json
    from pathlib import Path as _P
    p = _P(__file__).parent.parent / "results" / "d2b_redist" \
        / "d2b2_schedule_tradeoff.json"
    d = json.loads(p.read_text())

    fixed = {k: v for k, v in d.items()
             if isinstance(v, dict)
             and v.get("schedule") == "fixed_physical_interval"
             and k.startswith("A|")}
    assert fixed and all(v["invocations"] == 40 for v in fixed.values())

    errs_w = [d[f"A|dt{dt}|warp0.35|every_step"]["R_fit_inner_err"]
              for dt in ("2e-05", "1e-05", "5e-06")]
    assert max(errs_w) - min(errs_w) < 5e-5          # dt-stable bias

    assert d["C|every_step"]["R_fit_inner_err"] \
        < 0.5 * d["A|dt2e-05|warp0.35|every_step"]["R_fit_inner_err"]

    ada = d["A|dt2e-05|warp0.0|adaptive_cv_trigger"]
    eve = d["A|dt2e-05|warp0.0|every_step"]
    assert ada["invocations"] < 0.5 * eve["invocations"]
    assert ada["net_final_dH_vs_none"]["inner"] \
        < 0.1 * eve["net_final_dH_vs_none"]["inner"]

    for k, v in d.items():
        if isinstance(v, dict) and "timeline_errors" in v:
            assert v["timeline_errors"] == [], k


def test_d2b21_refinement_classifies_the_warp_layer():
    """D2b-2.1 refinement pins (committed JSON; the generator is now
    FAIL-CLOSED -- incomplete runs carry valid=False with an
    invalid_reason and E=None, and rates are computed only between
    valid rows): the warp layer is CONSISTENT WITH a
    bandwidth-regularization initialization layer -- at FIXED physical
    delta/sigma it does not decay under N refinement, while the
    production-coupled series decays 128->256 then stalls (sigma fixed
    at 0.1 throughout; E is a TOTAL radius error, so no stronger causal
    claim). B_red gives the redistribution differential against the
    matching no-redistribution run."""
    import json
    from pathlib import Path as _P
    p = _P(__file__).parent.parent / "results" / "d2b_redist" \
        / "d2b2_schedule_tradeoff.json"
    ref = json.loads(p.read_text())["warp_bias_refinement"]
    pc, fd = ref["production_coupled"], ref["fixed_delta_sigma"]
    assert pc["p_128_256"] is not None and pc["p_128_256"] > 1.0
    assert pc["p_256_512"] is not None and pc["p_256_512"] < 0.5
    if fd["p_128_256"] is not None:
        assert fd["p_128_256"] < 0.2                     # no decay
    # fail-closed contract: any invalid row has no E and a reason
    for mode in (pc, fd):
        for k, row in mode.items():
            if isinstance(row, dict) and row.get("valid") is False:
                assert row["E"] is None
                assert row["invalid_reason"] is not None
    assert "bandwidth-regularization" in ref["note"]


def test_schedule_api_fail_closed():
    """D2b-2.1: unknown schedule strings must never silently run as the
    adaptive trigger; fixed interval needs a finite positive interval;
    adaptive needs a finite nonnegative threshold."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d1a_mm_shadow import make_config
    from d2b0_timelevel_shadow import snapshot
    from src.torch.solver.mm_solver import MMSolver

    v, n_out, n_in = snapshot(0.5)
    delta, tau = compute_recommended_params(v.positions)

    def _cfg(**kw):
        cfg = make_config(2e-5, delta, tau)
        cfg.redistribute = True
        for k, val in kw.items():
            setattr(cfg, k, val)
        return cfg

    with pytest.raises(ValueError):
        MMSolver(_cfg(redistribution_schedule="bogus")).solve(v, 1)
    with pytest.raises(ValueError):
        MMSolver(_cfg(redistribution_schedule="fixed_physical_interval",
                      redistribute_physical_interval=0.0)).solve(v, 1)
    with pytest.raises(ValueError):
        MMSolver(_cfg(redistribution_schedule="adaptive_cv_trigger",
                      redistribute_trigger_cv=float("nan"))).solve(v, 1)


def test_steps_since_last_invocation_recorded_correctly():
    """Closure patch 5: the firing record reports the gap to the
    PREVIOUS invocation (the old order updated the marker first, so
    firing records always read 0)."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d1a_mm_shadow import make_config
    from d2b0_timelevel_shadow import snapshot
    from src.torch.diagnostics import SupportTimeline
    from src.torch.solver.mm_solver import MMSolver

    v, n_out, n_in = snapshot(0.5, warp=0.35)
    delta, tau = compute_recommended_params(v.positions)
    cfg = make_config(2e-5, delta, tau)
    cfg.redistribute = True
    cfg.redistribution_schedule = "fixed_physical_interval"
    cfg.redistribute_physical_interval = 1e-4      # every 5th step
    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(n_out),
                       torch.arange(n_out, n_out + n_in)])
    solver.solve(v, 10)
    fired = [r for r in solver.support_timeline.records
             if r.level == "post_redistribution" and r.stage_executed]
    assert len(fired) == 2
    assert fired[0].rank_fields["steps_since_last_invocation"] is None
    assert fired[1].rank_fields["steps_since_last_invocation"] == 5


def test_b_red_differential_pinned():
    """0D closure review: the B_red headline pinned directly -- the
    redistribution differential against the matching no-redistribution
    run exists at every production-coupled resolution and keeps
    DECAYING through n_outer = 512 (while total E stalls there): the
    total-error stall is not a floor created by the redistribution
    differential."""
    import json
    from pathlib import Path as _P
    p = _P(__file__).parent.parent / "results" / "d2b_redist" \
        / "d2b2_schedule_tradeoff.json"
    pc = json.loads(p.read_text())["warp_bias_refinement"][
        "production_coupled"]
    b128 = pc["n128"]["B_red"]
    b256 = pc["n256"]["B_red"]
    b512 = pc["n512"]["B_red"]
    assert b128 is not None and b256 is not None and b512 is not None
    assert 0 < abs(b512) < abs(b256) < abs(b128)
