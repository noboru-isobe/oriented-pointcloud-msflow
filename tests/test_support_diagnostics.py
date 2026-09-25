"""0E-core fixtures: the fail-closed support classification layer.

The six review-specified geometries plus the invariance requirements:
particle-ID permutation invariance, diagnostics-only (no trajectory
effect by construction -- the layer never writes), current-quantity
usage, and fail-closed behaviour on unclassifiable input.
"""

import math
import sys
from pathlib import Path

import torch

from src.torch.diagnostics import (
    classify_support,
    segment_contact_report,
    segment_crossings,
)
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_ellipse,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))
from spectral_rank_auto_0c5b import cat  # noqa: E402

DT = torch.float64
H = 0.05
SIGMA = 0.1


def two_ellipses(gap, h=H):
    a, b = 0.4, 1.0
    peri = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    n = round(peri / h)
    parts = [generate_oriented_ellipse(n, a, b, (-(a + gap / 2), 0.0),
                                       "cpu", DT, "arc_length"),
             generate_oriented_ellipse(n, a, b, (a + gap / 2, 0.0),
                                       "cpu", DT, "arc_length")]
    v = cat(*parts)
    comps = [torch.arange(n), torch.arange(n, 2 * n)]
    return v, comps


def test_separated_closed_and_carrier_free():
    """Separated pair with delta < gap: separated_closed and
    omega_cross = 0 exactly."""
    v, comps = two_ellipses(0.6)
    r = classify_support(v, comps, delta=0.2, sigma=SIGMA, h=H,
                         eps_bem=0.005, mass_tau=0.065)
    assert r.state == "separated_closed"
    assert r.geometry_status == "valid_closed"
    assert r.interaction_regime == "separated"
    assert r.omega_cross_quantiles[-1] == 0.0
    assert r.loopwise_carrier_ratio_sum == 1.0
    assert r.open_endpoints == 0 and r.cross_intersections == 0


def test_contact_layer_when_delta_exceeds_gap():
    """Still a valid separated support, but the carrier-overlap bias is
    active: contact_layer, with facing arcs identified."""
    v, comps = two_ellipses(0.1, h=0.036)      # the N = 128 audit scale:
    r = classify_support(v, comps, delta=0.18, sigma=SIGMA, h=0.036,
                         eps_bem=0.005,        # gap resolvable, delta > gap
                         mass_tau=0.065)
    assert r.state == "contact_layer"
    assert r.within_carrier_range and r.carrier_bias_active
    assert r.omega_cross_quantiles[-1] > 0.0
    assert r.loopwise_carrier_ratio_sum < 1.0
    assert len(r.facing_ids) > 0


def test_touching_or_unresolved():
    v, comps = two_ellipses(0.08)
    r = classify_support(v, comps, delta=0.18, sigma=SIGMA, h=H,
                         eps_bem=0.06)   # resolution scale ~ gap
    assert r.state == "touching_or_unresolved"


def test_reconstructed_merged_closed():
    """A single closed positively oriented curve (the clean-union
    stand-in: one circle)."""
    n = round(2 * math.pi / H)
    v = generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DT)
    r = classify_support(v, [torch.arange(n)], delta=0.2, sigma=SIGMA,
                         h=H, eps_bem=0.005)
    # a bare single closed loop is NOT accepted as a reconstruction --
    # the certificate must come from the caller
    assert r.state == "single_closed"
    r = classify_support(v, [torch.arange(n)], delta=0.2, sigma=SIGMA,
                         h=H, eps_bem=0.005,
                         provenance="reconstructed",
                         reconstruction_certified=True)
    assert r.state == "reconstructed_merged_closed"
    assert r.orientation_consistent
    assert abs(r.areas_geom[0] - math.pi) < 0.01


def test_invalid_crossing_raw_loops():
    """Two overlapping full circles kept as raw loops: cross
    intersections detected, state invalid_crossing."""
    n = round(2 * math.pi / H)
    v = cat(generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DT),
            generate_oriented_circle(round(2 * math.pi * 0.8 / H), 0.8,
                                     (1.5, 0.0), "cpu", DT))
    n2 = v.n_points - n
    r = classify_support(v, [torch.arange(n), torch.arange(n, n + n2)],
                         delta=0.2, sigma=SIGMA, h=H, eps_bem=0.005)
    assert r.state == "invalid_crossing"
    assert r.cross_intersections >= 2


def test_open_after_deletion():
    """Deleting a contiguous arc leaves degree-one endpoints; the
    closed-boundary theory no longer applies and the state says so.
    The causal name is attached ONLY when the caller attests that a
    deletion event actually removed points this step."""
    n = round(2 * math.pi / H)
    v0 = generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DT)
    keep = torch.ones(n, dtype=torch.bool)
    keep[10:30] = False
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    v = OrientedPointCloudVarifold(positions=v0.positions[keep],
                                   angles=v0.angles[keep])
    r = classify_support(v, [torch.arange(v.n_points)], delta=0.2,
                         sigma=SIGMA, h=H, eps_bem=0.005,
                         deletion_this_step=True)
    assert r.state == "open_after_deletion"
    assert r.open_endpoints >= 2


def test_open_support_without_deletion():
    """P1.2 reviewer correction: the SAME open geometry with NO deletion
    event this step must NOT claim a deletion cause -- it is
    "open_support" (e.g. a diverged frame read as scattered points).
    Both names are sticky-invalid; only the causal claim differs."""
    n = round(2 * math.pi / H)
    v0 = generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DT)
    keep = torch.ones(n, dtype=torch.bool)
    keep[10:30] = False
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    v = OrientedPointCloudVarifold(positions=v0.positions[keep],
                                   angles=v0.angles[keep])
    r = classify_support(v, [torch.arange(v.n_points)], delta=0.2,
                         sigma=SIGMA, h=H, eps_bem=0.005)
    assert r.state == "open_support"
    assert r.geometry_status == "open"
    from src.torch.diagnostics.support import STICKY_INVALID_STATES
    assert "open_support" in STICKY_INVALID_STATES
    assert "open_after_deletion" in STICKY_INVALID_STATES


def test_open_support_closes_gates_sticky():
    """The cause-unattributed open state must close every gate exactly
    like open_after_deletion (sticky), through the TIMELINE wiring: a
    frame whose geometry is open with NO deletion this step is recorded
    as open_support and latches requires_reconstruction."""
    from src.torch.diagnostics.timeline import SupportTimeline
    from src.torch.oriented_varifold import OrientedPointCloudVarifold

    n = round(2 * math.pi / H)
    v0 = generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DT)
    tl = SupportTimeline(initial_loops=[torch.arange(n)],
                         compute_current_fields=False,
                         store_pointwise_fields=False)
    rec0 = tl.record(0, "post_deletion_final", v0, delta=0.2, sigma=SIGMA)
    assert rec0.report.state == "single_closed"

    # tear the geometry WITHOUT any deletion: pull an arc far away
    pos = v0.positions.clone()
    pos[10:30] += torch.tensor([5.0, 5.0], dtype=DT)
    v1 = OrientedPointCloudVarifold(positions=pos, angles=v0.angles)
    rec1 = tl.record(1, "post_deletion_final", v1, delta=0.2, sigma=SIGMA)
    assert rec1.report.state == "open_support"
    assert tl.requires_reconstruction
    assert not any(tl.gated_permissions(rec0.report).values())


def test_annulus_is_not_misread_as_two_phases():
    """An annulus: two boundary rings of ONE phase component. The layer
    reports geometry only -- two closed non-intersecting curves with the
    inner ring's signed area negative (a hole, noted but not condemned);
    the phase count is the rank sensor's job, not this layer's."""
    n_out = round(2 * math.pi / H)
    v = generate_oriented_annulus(n_out, 1.0, 0.5, (0.0, 0.0), "cpu", DT)
    n_in = v.n_points - n_out
    r = classify_support(v, [torch.arange(n_out),
                             torch.arange(n_out, n_out + n_in)],
                         delta=0.15, sigma=SIGMA, h=H, eps_bem=0.005)
    assert r.state == "separated_closed"
    assert r.areas_geom[0] > 0 and r.areas_geom[1] < 0
    assert any("hole" in note for note in r.notes)


def test_particle_id_permutation_invariance():
    """Storage order must not matter as long as the per-component curve
    order is supplied."""
    v, comps = two_ellipses(0.1, h=0.036)
    r0 = classify_support(v, comps, delta=0.18, sigma=SIGMA, h=0.036)

    perm = torch.randperm(v.n_points)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(v.n_points)
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    v2 = OrientedPointCloudVarifold(positions=v.positions[perm],
                                    angles=v.angles[perm])
    comps2 = [inv[c] for c in comps]
    r1 = classify_support(v2, comps2, delta=0.18, sigma=SIGMA, h=0.036,
                          particle_ids=perm.argsort().argsort())
    assert r1.state == r0.state
    assert abs(r1.min_gap - r0.min_gap) < 1e-14
    assert abs(r1.areas_geom[0] - r0.areas_geom[0]) < 1e-12
    assert abs(r1.omega_cross_quantiles[-1]
               - r0.omega_cross_quantiles[-1]) < 1e-12


def test_fail_closed_on_garbage():
    """A component that is neither closed nor a single arc (random
    points) must not be classified as anything valid."""
    torch.manual_seed(3)
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    v = OrientedPointCloudVarifold(
        positions=torch.randn(40, 2, dtype=DT),
        angles=torch.zeros(40, dtype=DT))
    r = classify_support(v, [torch.arange(40)], delta=0.2, sigma=SIGMA,
                         h=H)
    assert r.state in ("unknown", "open_support", "open_after_deletion",
                       "invalid_crossing")
    assert r.state != "reconstructed_merged_closed"


def test_segment_crossings_unit():
    P = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                     dtype=DT)
    assert segment_crossings(P) == 0            # square: simple
    Q = P + torch.tensor([0.5, 0.5], dtype=DT)  # overlapping square
    assert segment_crossings(P, Q) == 2
    bow = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                       dtype=DT)
    assert segment_crossings(bow) >= 1          # self-crossing bowtie


def test_tangent_and_collinear_contacts_detected():
    """The strict proper-crossing test misses tangency and collinear
    overlap; the tolerance-aware segment report must not (review 3).
    Two externally tangent circles with PHASE-SHIFTED sampling (the
    touch point falls mid-segment, so vertex distances stay positive)."""
    n = 100
    th1 = torch.arange(n, dtype=DT) / n * 2 * math.pi
    th2 = th1 + math.pi / n                     # shifted sampling
    P = torch.stack([torch.cos(th1) - 1.0, torch.sin(th1)], 1)
    Q = torch.stack([torch.cos(th2) + 1.0, torch.sin(th2)], 1)
    rep = segment_contact_report(P, Q, tol=0.01)
    assert rep["crossings"] == 0                # no proper crossing
    assert rep["near_contacts"] > 0             # tangency caught by tol
    assert rep["min_distance"] < 0.01

    # collinear overlap
    A = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
                     dtype=DT)
    B = torch.tensor([[1.0, 0.0], [3.0, 0.0], [3.0, -1.0], [1.0, -1.0]],
                     dtype=DT)
    rep = segment_contact_report(A, B, tol=0.0)
    assert rep["collinear_overlaps"] > 0


def test_orientation_validation_reversed_and_broken():
    """Review 4: reversed vertex order and deliberately broken annulus
    orientation must fail validity; a concave outer boundary must pass
    (centroid-based tests break there, tangent-based must not)."""
    from src.torch.oriented_varifold import OrientedPointCloudVarifold

    n = round(2 * math.pi / H)
    v0 = generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DT)
    # reverse the vertex order but keep the (now inconsistent) normals
    v_rev = OrientedPointCloudVarifold(
        positions=v0.positions.flip(0), angles=v0.angles.flip(0))
    r = classify_support(v_rev, [torch.arange(n)], delta=0.2,
                         sigma=SIGMA, h=H)
    assert not r.orientation_consistent
    assert r.state != "single_closed"

    # annulus with the INNER ring's normals flipped outward: parity says
    # hole, normals say outer -> invalid
    n_out = round(2 * math.pi / H)
    va = generate_oriented_annulus(n_out, 1.0, 0.5, (0.0, 0.0), "cpu", DT)
    n_in = va.n_points - n_out
    angles = va.angles.clone()
    angles[n_out:] += math.pi
    v_bad = OrientedPointCloudVarifold(positions=va.positions,
                                       angles=angles)
    r = classify_support(v_bad, [torch.arange(n_out),
                                 torch.arange(n_out, n_out + n_in)],
                         delta=0.15, sigma=SIGMA, h=H)
    assert not r.orientation_consistent

    # concave crescent-like outer boundary, correctly oriented: valid
    t = torch.arange(200, dtype=DT) / 200 * 2 * math.pi
    rr = 1.0 + 0.6 * torch.cos(2 * t)           # strongly non-convex
    P = torch.stack([rr * torch.cos(t), rr * torch.sin(t)], 1)
    tang = P.roll(-1, 0) - P.roll(1, 0)
    ang = torch.atan2(-tang[:, 0], tang[:, 1])  # outward normal of CCW
    v_c = OrientedPointCloudVarifold(positions=P, angles=ang)
    r = classify_support(v_c, [torch.arange(200)], delta=0.2,
                         sigma=SIGMA, h=0.06)
    assert r.orientation_consistent
    assert r.geometry_status == "valid_closed"


def test_same_loop_neck_detected():
    """Review 6: a single closed dumbbell with a thin neck -- geographic
    proximity of arc-far sheets -- must not sail through as a plain
    valid single loop; the nonlocal self-gap flags it."""
    t = torch.arange(400, dtype=DT) / 400 * 2 * math.pi
    rr = 1.0 - 0.92 * torch.sin(t) ** 2         # pinched at t = pi/2, 3pi/2
    P = torch.stack([rr * torch.cos(t), rr * torch.sin(t)], 1)
    tang = P.roll(-1, 0) - P.roll(1, 0)
    ang = torch.atan2(-tang[:, 0], tang[:, 1])  # outward normal of CCW
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    v = OrientedPointCloudVarifold(positions=P, angles=ang)
    r = classify_support(v, [torch.arange(400)], delta=0.2, sigma=SIGMA,
                         h=0.05, eps_bem=0.05)
    assert r.min_self_gap_nonlocal is not None
    assert r.min_self_gap_nonlocal < 0.2
    assert r.interaction_regime == "touching"
    assert r.state == "touching_or_unresolved"


def test_permission_gates():
    """Review: three separate permissions, not one boolean; only a
    certified reconstruction authorizes post-contact evolution."""
    from src.torch.diagnostics import permissions

    v, comps = two_ellipses(0.6)
    sep = classify_support(v, comps, delta=0.2, sigma=SIGMA, h=H)
    p = permissions(sep)
    # a separated valid support may evolve and REFRESH the basis, but a
    # rank CHANGE is a topology event -- no certificate, no change
    assert p["allow_same_rank_evolution"] and p["allow_rank_refresh"]
    assert not p["allow_rank_change"]
    assert not p["allow_postcontact_evolution"]

    v, comps = two_ellipses(0.1, h=0.036)
    cl = classify_support(v, comps, delta=0.18, sigma=SIGMA, h=0.036)
    p = permissions(cl)
    assert p["allow_same_rank_evolution"] and p["allow_rank_refresh"]
    assert not p["allow_rank_change"]
    assert not p["allow_postcontact_evolution"]

    n = round(2 * math.pi / H)
    vc = generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DT)
    cert = classify_support(vc, [torch.arange(n)], delta=0.2,
                            sigma=SIGMA, h=H,
                            provenance="reconstructed",
                            reconstruction_certified=True)
    p = permissions(cert)
    assert p["allow_rank_change"] and p["allow_postcontact_evolution"]
    # the boolean alone must NOT unlock the gate: certificate is tied to
    # provenance == "reconstructed"
    boolonly = classify_support(vc, [torch.arange(n)], delta=0.2,
                                sigma=SIGMA, h=H, provenance="raw",
                                reconstruction_certified=True)
    pb = permissions(boolonly)
    assert not pb["allow_rank_change"]
    assert not pb["allow_postcontact_evolution"]
    uncert = classify_support(vc, [torch.arange(n)], delta=0.2,
                              sigma=SIGMA, h=H)
    assert not permissions(uncert)["allow_postcontact_evolution"]


def test_timeline_records_and_leaves_trajectory_unchanged():
    """The 0E time-level wiring is diagnostics ONLY. Strengthened claim
    (review 0E-wiring.1 item 8): ALL per-step positions/angles and every
    scalar history must be bitwise equal with and without the timeline;
    all FIVE levels must be emitted on EVERY step, identity records
    included, with pre_mm present at step 0."""
    from src.torch.diagnostics import SupportTimeline
    from src.torch.diagnostics.timeline import TIME_LEVELS
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.mm_step import MMConfig

    n = 64
    v = generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DT)
    cfg = MMConfig(time_step=1e-4, use_unit_coherence=True,
                   optimizer_method="trust-ncg", optimizer_tol=1e-8,
                   optimizer_max_iter=200)

    ref = MMSolver(cfg).solve(v, 3)
    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(n)])
    out = solver.solve(v, 3)

    assert len(ref._varifolds) == len(out._varifolds)
    for a, b in zip(ref._varifolds, out._varifolds):
        assert torch.equal(a.positions, b.positions)
        assert torch.equal(a.angles, b.angles)
    assert torch.equal(ref.perimeters, out.perimeters)
    assert torch.equal(ref.wassersteins, out.wassersteins)
    assert torch.equal(ref.objectives, out.objectives)
    assert torch.equal(ref.volumes, out.volumes)

    recs = solver.support_timeline.records
    for step in range(3):
        levels = [r.level for r in recs if r.step == step]
        assert levels == list(TIME_LEVELS)      # all five, in order
    r0 = [r for r in recs if r.step == 0 and r.level == "pre_mm"][0]
    assert r0.delta_used is not None            # step-0 scales computed
    post = [r for r in recs if r.level == "post_mm_candidate"][0]
    assert post.report is not None and post.error is None
    assert post.report.state == "single_closed"
    assert post.m_used is not None and post.m_current is not None
    assert post.q_current_diagnostic is not None
    assert post.rank_fields["optimizer_message"] is not None
    assert post.rank_fields["objective_initial"] is not None
    ident = [r for r in recs if r.level == "post_redistribution"][0]
    assert ident.stage_enabled is False and ident.stage_executed is False
    delid = [r for r in recs if r.level == "post_deletion_raw"][0]
    assert delid.stage_enabled is False and delid.n_removed == 0


def test_timeline_records_actual_deletion():
    """A bunched cluster of near-duplicate points has ~1/6 the carrier
    mass and is removed by the dead-point rule: the timeline must record
    the keep mask, update persistent IDs, keep tombstone counts, and
    emit post_deletion_raw with n_removed > 0."""
    import math as m
    from src.torch.diagnostics import SupportTimeline
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.mm_step import MMConfig

    n = 48
    th = (torch.arange(n, dtype=DT) + 0.5) / n * 2 * m.pi
    # 12 extra points bunched inside one gap, in curve order (measured
    # mass ratio 0.22 < the 0.3 threshold)
    th_extra = th[10] + (torch.arange(12, dtype=DT) + 1) / 13 * (2 * m.pi / n)
    th_all = torch.cat([th[:11], th_extra, th[11:]])
    v = OrientedPointCloudVarifold(
        positions=torch.stack([torch.cos(th_all), torch.sin(th_all)], 1),
        angles=th_all.clone())
    N = v.n_points

    cfg = MMConfig(time_step=1e-6, use_unit_coherence=True,
                   remove_dead_points=True, dead_point_threshold=0.3,
                   optimizer_method="trust-ncg", optimizer_tol=1e-6,
                   optimizer_max_iter=50)
    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(N)])
    solver.solve(v, 1)

    recs = solver.support_timeline.records
    raw = [r for r in recs if r.level == "post_deletion_raw"][0]
    assert raw.stage_executed and raw.n_removed > 0
    assert raw.keep_mask is not None and not all(raw.keep_mask)
    assert len(raw.particle_ids) == N - raw.n_removed
    assert raw.loop_survivor_counts[0] == N - raw.n_removed
    final = [r for r in recs if r.level == "post_deletion_final"][0]
    assert final.n_points == N - raw.n_removed


def test_timeline_tombstones_and_partition_check():
    """Synthetic: killing one whole loop must surface in
    vanished_loop_ids, not silently drop from the classifier input; and
    initial loops that do not partition the IDs are rejected."""
    import pytest
    from src.torch.diagnostics import SupportTimeline

    tl = SupportTimeline(initial_loops=[torch.arange(10),
                                        torch.arange(10, 16)])
    mask = torch.ones(16, dtype=torch.bool)
    mask[10:] = False                       # loop 1 disappears entirely
    tl.apply_keep_mask(mask)
    counts, vanished, under = tl._loop_status()
    assert counts == [10, 0]
    assert vanished == [1]
    assert len(tl.current_loops()) == 1     # classifier sees one loop...
    # ...but the record keeps the tombstone (checked via _loop_status)

    with pytest.raises(AssertionError):
        SupportTimeline(initial_loops=[torch.arange(5),
                                       torch.arange(3, 8)])  # overlap


def test_post_removal_redistribution_not_hidden():
    """0E-wiring.2 item 5: with deletion AND post-removal redistribution
    both active, post_deletion_raw and post_deletion_final must have the
    same surviving IDs but DIFFERENT geometries, with final marked as
    the committed state and its geometry_changed measured from actual
    tensor differences."""
    import math as m
    from src.torch.diagnostics import SupportTimeline
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.mm_step import MMConfig

    n = 48
    th = (torch.arange(n, dtype=DT) + 0.5) / n * 2 * m.pi
    th_extra = th[10] + (torch.arange(12, dtype=DT) + 1) / 13 * (2 * m.pi / n)
    th_all = torch.cat([th[:11], th_extra, th[11:]])
    v = OrientedPointCloudVarifold(
        positions=torch.stack([torch.cos(th_all), torch.sin(th_all)], 1),
        angles=th_all.clone())
    N = v.n_points

    cfg = MMConfig(time_step=1e-6, use_unit_coherence=True,
                   remove_dead_points=True, dead_point_threshold=0.3,
                   redistribute=True, redistribute_n_iters=2,
                   redistribute_n_iters_after_removal=3,
                   optimizer_method="trust-ncg", optimizer_tol=1e-6,
                   optimizer_max_iter=50)
    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(N)])
    solver.solve(v, 1)

    recs = solver.support_timeline.records
    raw = [r for r in recs if r.level == "post_deletion_raw"][0]
    fin = [r for r in recs if r.level == "post_deletion_final"][0]
    assert raw.n_removed > 0
    assert raw.particle_ids == fin.particle_ids       # same survivors
    assert raw.n_points == fin.n_points
    assert not torch.equal(raw.positions, fin.positions)  # moved
    assert fin.geometry_changed and fin.stage_executed
    # committed semantics: only pre_mm and post_deletion_final commit
    for r in recs:
        expected = r.level in ("pre_mm", "post_deletion_final")
        assert r.committed == expected, (r.level, r.committed)
    # used quantities present at the deletion level (the legacy score)
    assert raw.m_used is not None and raw.q_used is not None
    assert len(raw.m_used) == N                       # pre-deletion sized


def test_perpoint_bandwidth_deletion_unsafe_documented():
    """BACKLOG (review, 0E-wiring.2 acceptance): the non-adaptive
    per-point bandwidth cache is NOT deletion-safe -- after N changes the
    cached (N_old,) tensor is neither sliced nor invalidated. All 0D
    drivers therefore pin mass_bandwidth_type='global'. This test
    documents the constraint; the fix (invalidate + recompute on N_n
    points after deletion) is a separate work item."""
    import math as m
    import pytest
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.mm_step import MMConfig

    n = 48
    th = (torch.arange(n, dtype=DT) + 0.5) / n * 2 * m.pi
    th_extra = th[10] + (torch.arange(12, dtype=DT) + 1) / 13 * (2 * m.pi / n)
    th_all = torch.cat([th[:11], th_extra, th[11:]])
    v = OrientedPointCloudVarifold(
        positions=torch.stack([torch.cos(th_all), torch.sin(th_all)], 1),
        angles=th_all.clone())

    from src.torch.solver.mm_step import MMStepper

    cfg = MMConfig(time_step=1e-6, use_unit_coherence=True,
                   mass_bandwidth_type="per_point_knn",
                   optimizer_method="trust-ncg", optimizer_tol=1e-6,
                   optimizer_max_iter=50)
    # the PRECISE invariant violation (review D0.1 item 7): the cached
    # per-point bandwidth tensor keeps its old length after N changes,
    # so the very next setup on the reduced cloud must raise a shape
    # error -- not merely "the run stops early for some reason"
    stepper = MMStepper(cfg)
    stepper._setup_step(v)
    assert stepper._mass_delta_for_kde.shape[0] == v.n_points
    keep = torch.ones(v.n_points, dtype=torch.bool)
    keep[11:23] = False
    v_small = OrientedPointCloudVarifold(positions=v.positions[keep],
                                         angles=v.angles[keep])
    with pytest.raises(RuntimeError):
        stepper._setup_step(v_small)
