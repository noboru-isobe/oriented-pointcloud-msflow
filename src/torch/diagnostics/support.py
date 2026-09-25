"""0E-core: fail-closed support diagnostics (0E-core.1 revision).

Scope: a VALIDATOR on caller-supplied ordered boundary loops. It neither
recovers loops from an unordered cloud nor constructs merged boundaries;
it answers, per configuration and without touching any trajectory: is
this a valid phase boundary, in which interaction regime, and with what
provenance?

Three ORTHOGONAL axes (review: geometry, regime and history must not be
mixed in one enum):

    geometry_status     valid_closed | open | intersecting | degenerate
                        | unknown
    interaction_regime  separated | carrier_contact | unresolved
                        | touching | none        (none: single loop, no
                                                  nonlocal proximity)
    provenance          caller-supplied string (initial / raw / post_mm /
                        post_redistribution / post_deletion /
                        reconstructed); NEVER inferred here.

The legacy combined `state` is derived from the axes;
`reconstructed_merged_closed` additionally REQUIRES the caller's
reconstruction certificate -- looking like one simple closed loop is not
evidence that a merger was correctly reconstructed. A certificate-less
valid single loop reports `single_closed`.

Terminology (review): the ordered inputs are BOUNDARY LOOPS (M of them).
The phase-component count C is the BEM spectrum's job and is not inferred
here -- an annulus has M = 2 loops and C = 1 phase.

Carrier conventions: the loopwise carrier ratio is a PRE-CONTACT ORACLE
diagnostic only, computed with the ACTUAL mass_tau/kernel passed in (no
ad hoc tau). "Within carrier range" (g <= delta) and "carrier bias
ACTIVE" (omega_cross above roundoff) are separate fields: at delta = g
the compactly supported Wendland contribution is exactly zero (pinned in
0C-5d-1.2a), so range-boundary proximity alone must not trigger
`carrier_contact`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from src.torch.oriented_varifold import mass as _mass
from src.torch.oriented_varifold.mass import (
    compute_kde_density,
    compute_masses,
)

_T = torch.Tensor

SUPPORT_STATES = (
    "separated_closed",
    "contact_layer",
    "touching_or_unresolved",
    "invalid_crossing",
    "open_after_deletion",
    "open_support",
    "single_closed",
    "reconstructed_merged_closed",
    "unknown",
)

OPEN_GAP_FACTOR = 3.0     # consecutive-spacing factor marking an open end
OMEGA_ROUNDOFF = 1e-12    # kernel-support roundoff floor
TOUCH_FACTOR = 2.0        # gap below TOUCH_FACTOR * l_res is unresolvable
ARC_EXCLUDE_FACTOR = 2.0  # same-loop points closer than this * delta in
                          # arc length count as local, not nonlocal
QUANTS = (0.0, 0.1, 0.5, 0.9, 1.0)


@dataclass(frozen=True)
class SupportReport:
    # orthogonal axes
    geometry_status: str
    interaction_regime: str
    provenance: str
    reconstruction_certified: bool
    state: str                          # derived, legacy-compatible
    # loops (M), NOT phase components (C -- the BEM spectrum's job)
    n_boundary_loops: int
    closed_flags: List[bool]
    open_endpoints: int
    self_intersections: int
    cross_intersections: int
    near_contacts: int                  # seg-seg dist < tol, no crossing
    collinear_overlaps: int
    # gaps: segment-segment based
    min_gap: Optional[float]            # between loops; None if M == 1
    min_self_gap_nonlocal: Optional[float]  # same-loop, arc-far pairs
    gap_over_delta: Optional[float]
    gap_over_sigma: Optional[float]
    gap_over_resolution: Optional[float]
    l_res_contact: Optional[float]      # max(local facing spacings, eps)
    # carrier contact layer
    within_carrier_range: Optional[bool]
    carrier_bias_active: Optional[bool]
    omega_cross_quantiles: Optional[list]
    omega_nonlocal_quantiles: Optional[list]   # same-loop included
    loopwise_carrier_ratio_sum: Optional[float]
    facing_ids: Optional[list]
    facing_m_quantiles: Optional[list]
    facing_q_quantiles: Optional[list]
    facing_mq_quantiles: Optional[list]
    # geometry
    areas_geom: List[float]             # holes negative (nesting parity)
    centroids_geom: List[list]
    nesting_depths: List[int]
    orientation_consistent: bool
    # BEM / step validity
    min_collocation_over_eps: Optional[float]  # TRUE endpoint nodes only
    step_over_h: Optional[float]
    step_over_gap: Optional[float]
    step_over_eps: Optional[float]
    notes: list = field(default_factory=list)


def _quantiles(x: _T) -> list:
    if x.numel() == 0:
        return []
    return [x.quantile(q).item() for q in QUANTS]


def polygon_stats(P: _T):
    """Shoelace area (vertex-order sign) and phase centroid."""
    X, Y = P[:, 0], P[:, 1]
    Xn, Yn = X.roll(-1), Y.roll(-1)
    c = X * Yn - Xn * Y
    A = 0.5 * c.sum()
    gx = ((X + Xn) * c).sum() / (6 * A)
    gy = ((Y + Yn) * c).sum() / (6 * A)
    return A.item(), [gx.item(), gy.item()]


def _detect_closed(P: _T):
    seg = (P.roll(-1, 0) - P).norm(dim=1)
    med = seg.median()
    breaks = int((seg > OPEN_GAP_FACTOR * med).sum())
    return breaks == 0, (2 * breaks if breaks else 0)


def _segments(P: _T, closed: bool):
    if closed:
        return P, P.roll(-1, 0)
    return P[:-1], P[1:]


def segment_segment_distance(A0: _T, A1: _T, B0: _T, B1: _T) -> _T:
    """Pairwise min distance matrix between 2D segment sets (endpoint-to-
    segment feet on both sides -- exact for non-crossing segments; proper
    crossings are handled separately and reported as distance-0 contact
    by the crossing test)."""
    def pt_seg(P, S0, S1):
        d = S1 - S0
        L2 = (d * d).sum(-1).clamp_min(1e-300)
        t = (((P - S0) * d).sum(-1) / L2).clamp(0.0, 1.0)
        proj = S0 + t[..., None] * d
        return (P - proj).norm(dim=-1)

    a0 = A0[:, None, :]
    a1 = A1[:, None, :]
    b0 = B0[None, :, :].expand(A0.shape[0], -1, -1)
    b1 = B1[None, :, :].expand(A0.shape[0], -1, -1)
    cands = torch.stack([
        pt_seg(a0, b0, b1), pt_seg(a1, b0, b1),
        pt_seg(b0, a0.expand_as(b0), a1.expand_as(b0)),
        pt_seg(b1, a0.expand_as(b1), a1.expand_as(b1)),
    ])
    return cands.min(0).values


def _proper_crossings_matrix(A0, A1, B0, B1):
    def orient(p, q, r):
        return ((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
                - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

    a0, a1 = A0[:, None, :], A1[:, None, :]
    b0, b1 = B0[None, :, :], B1[None, :, :]
    d1 = orient(a0, a1, b0)
    d2 = orient(a0, a1, b1)
    d3 = orient(b0, b1, a0)
    d4 = orient(b0, b1, a1)
    crossing = (d1 * d2 < 0) & (d3 * d4 < 0)
    # collinear overlap: all four orientations ~ 0 AND the segments'
    # bounding intervals overlap
    scale = ((A1 - A0).norm(dim=-1)[:, None]
             * (B1 - B0).norm(dim=-1)[None, :]).clamp_min(1e-300)
    coll = ((d1.abs() + d2.abs() + d3.abs() + d4.abs()) / scale < 1e-9)
    overlap_x = (torch.minimum(a0[..., 0], a1[..., 0])
                 <= torch.maximum(b0[..., 0], b1[..., 0]) + 1e-12) & \
                (torch.minimum(b0[..., 0], b1[..., 0])
                 <= torch.maximum(a0[..., 0], a1[..., 0]) + 1e-12)
    overlap_y = (torch.minimum(a0[..., 1], a1[..., 1])
                 <= torch.maximum(b0[..., 1], b1[..., 1]) + 1e-12) & \
                (torch.minimum(b0[..., 1], b1[..., 1])
                 <= torch.maximum(a0[..., 1], a1[..., 1]) + 1e-12)
    return crossing, coll & overlap_x & overlap_y


def segment_contact_report(P: _T, Q: Optional[_T] = None,
                           closed_p: bool = True, closed_q: bool = True,
                           tol: float = 0.0) -> dict:
    """Tolerance-aware contact between ordered curves: proper crossings,
    collinear overlaps, near contacts (segment-segment distance < tol
    without crossing -- catches tangency and endpoint touch that the
    strict crossing test misses), and the min segment-segment distance.
    Self mode (Q=None) excludes adjacent segments."""
    A0, A1 = _segments(P, closed_p)
    if Q is None:
        B0, B1 = A0, A1
    else:
        B0, B1 = _segments(Q, closed_q)
    crossing, coll = _proper_crossings_matrix(A0, A1, B0, B1)
    dist = segment_segment_distance(A0, A1, B0, B1)
    if Q is None:
        n = A0.shape[0]
        idx = torch.arange(n)
        diff = (idx[:, None] - idx[None, :]).abs()
        if closed_p:
            diff = torch.minimum(diff, n - diff)
        adj = diff <= 1
        crossing = crossing & ~adj
        coll = coll & ~adj
        dist = dist.masked_fill(adj, float("inf"))
        return dict(
            crossings=int(crossing.sum().item()) // 2,
            collinear_overlaps=int(coll.sum().item()) // 2,
            near_contacts=int(((dist < tol) & ~crossing).sum().item()) // 2,
            min_distance=dist.min().item())
    return dict(
        crossings=int(crossing.sum().item()),
        collinear_overlaps=int(coll.sum().item()),
        near_contacts=int(((dist < tol) & ~crossing).sum().item()),
        min_distance=dist.min().item())


def segment_crossings(P: _T, Q: Optional[_T] = None,
                      closed_p: bool = True, closed_q: bool = True) -> int:
    """Back-compatible proper-crossing count."""
    return segment_contact_report(P, Q, closed_p, closed_q)["crossings"]


def _point_in_polygon(pt: _T, P: _T) -> bool:
    """Even-odd ray casting."""
    x, y = pt[0].item(), pt[1].item()
    X, Y = P[:, 0], P[:, 1]
    Xn, Yn = X.roll(-1), Y.roll(-1)
    cond = ((Y > y) != (Yn > y))
    denom = (Yn - Y)
    xin = X + (y - Y) * (Xn - X) / torch.where(
        denom.abs() < 1e-300, torch.full_like(denom, 1e-300), denom)
    hits = cond & (xin > x)
    return bool(hits.sum().item() % 2)


def _loop_orientation(P: _T, normals: _T):
    """(median sign of cross(t, n), consistency fraction) from LOCAL
    tangents -- centroid directions are meaningless for concave loops.
    For a CCW loop with outward normals, cross(t, n) = -1."""
    t = P.roll(-1, 0) - P.roll(1, 0)
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-300)
    cross = t[:, 0] * normals[:, 1] - t[:, 1] * normals[:, 0]
    frac = max((cross > 0).float().mean().item(),
               (cross < 0).float().mean().item())
    sign = 1 if cross.median() > 0 else -1
    return sign, frac


def classify_support(
    varifold,
    boundary_loops: List[_T],
    delta: float,
    sigma: float,
    h: float,
    eps_bem: Optional[float] = None,
    masses: Optional[_T] = None,
    coherence: Optional[_T] = None,
    displacements: Optional[_T] = None,
    particle_ids: Optional[_T] = None,
    mass_tau: Optional[float] = None,
    mass_kernel: str = "wendland_c2",
    provenance: str = "raw",
    reconstruction_certified: bool = False,
    endpoint_positions: Optional[_T] = None,
    endpoints_per_particle: int = 3,
    deletion_this_step: bool = False,
) -> SupportReport:
    """Classify the current support; diagnostics only, fail closed.

    boundary_loops: ordered particle-index tensors in curve order, one
    per boundary loop (M loops; the phase count C is NOT inferred here).
    mass_tau/mass_kernel: the ACTUAL production values -- required for
    the oracle carrier ratio (skipped, with a note, if absent).
    endpoint_positions: the true BEM collocation nodes (N*K, 2) from
    EndpointOperator; particle centers are NOT a substitute.
    """
    pos = varifold.positions
    normals = varifold.normals
    ids = (particle_ids if particle_ids is not None
           else torch.arange(pos.shape[0]))
    notes = []
    contact_tol = eps_bem or 0.0

    loops = [pos[c] for c in boundary_loops]
    M = len(loops)
    closed, opens = zip(*(_detect_closed(P) for P in loops))
    open_endpoints = int(sum(opens))

    # --- contact structure (segment-based) ---------------------------
    self_x = coll_x = near_x = 0
    for P, cl in zip(loops, closed):
        rep = segment_contact_report(P, closed_p=cl, tol=contact_tol)
        self_x += rep["crossings"]
        coll_x += rep["collinear_overlaps"]
        near_x += rep["near_contacts"]
    cross_x = 0
    min_gap = None
    pair_min = {}
    for i in range(M):
        for j in range(i + 1, M):
            rep = segment_contact_report(loops[i], loops[j], closed[i],
                                         closed[j], tol=contact_tol)
            cross_x += rep["crossings"]
            coll_x += rep["collinear_overlaps"]
            near_x += rep["near_contacts"]
            pair_min[(i, j)] = rep["min_distance"]
            min_gap = (rep["min_distance"] if min_gap is None
                       else min(min_gap, rep["min_distance"]))

    # --- same-loop NONLOCAL proximity (C-shapes, necks) ---------------
    min_self_gap = None
    for P, cl in zip(loops, closed):
        n = P.shape[0]
        if n < 8:
            continue
        seg_med = (P.roll(-1, 0) - P).norm(dim=1).median().item()
        k_excl = max(2, int(round(ARC_EXCLUDE_FACTOR * delta / seg_med)))
        A0, A1 = _segments(P, cl)
        dist = segment_segment_distance(A0, A1, A0, A1)
        m = A0.shape[0]
        idx = torch.arange(m)
        diff = (idx[:, None] - idx[None, :]).abs()
        if cl:
            diff = torch.minimum(diff, m - diff)
        dist = dist.masked_fill(diff <= k_excl, float("inf"))
        val = dist.min().item()
        if val != float("inf"):
            min_self_gap = (val if min_self_gap is None
                            else min(min_self_gap, val))

    # --- orientation: local tangent-normal consistency + nesting ------
    depths = []
    for i, P in enumerate(loops):
        d = 0
        for j, Qp in enumerate(loops):
            if i != j and closed[j] and _point_in_polygon(P[0], Qp):
                d += 1
        depths.append(d)
    areas, cents = [], []
    orientation_ok = True
    for i, (c, P, cl) in enumerate(zip(boundary_loops, loops, closed)):
        if not cl:
            areas.append(float("nan"))
            cents.append([float("nan")] * 2)
            continue
        a_sh, cxy = polygon_stats(P)
        if a_sh < 0:
            # production convention: every loop is stored CCW (the
            # generators guarantee it); a CW loop mid-run signals
            # bookkeeping corruption even if the normals happen to be
            # geometrically consistent
            orientation_ok = False
            notes.append(f"loop {i}: vertex order reversed "
                         "(convention: CCW)")
        sign, frac = _loop_orientation(P, normals[c])
        if frac < 0.95:
            orientation_ok = False
            notes.append(f"loop {i}: tangent-normal sign inconsistent "
                         f"({frac:.2f})")
        # outward normals <=> cross(t, n) = -sign(shoelace)
        outward = (sign == (-1 if a_sh > 0 else 1))
        is_hole = depths[i] % 2 == 1
        if is_hole == outward:
            orientation_ok = False
            notes.append(f"loop {i}: normal orientation contradicts "
                         f"nesting parity (depth {depths[i]}, "
                         f"outward={outward})")
        if is_hole:
            notes.append(f"loop {i}: hole (nesting depth {depths[i]})")
        areas.append(-abs(a_sh) if is_hole else abs(a_sh))
        cents.append(cxy)

    # --- carrier contact layer ---------------------------------------
    gap_delta = gap_sigma = gap_res = None
    within_range = bias_active = None
    omega_q = omega_nl_q = ratio = None
    facing_ids = fm = fq = fmq = None
    l_res_contact = None
    if M >= 2 and min_gap is not None:
        gap_delta = min_gap / delta
        gap_sigma = min_gap / sigma
        # local facing spacings around the closest loop pair (a single
        # global h misrepresents unequal or partially deleted loops)
        (i0, j0) = min(pair_min, key=pair_min.get)
        h_loc = []
        for k in (i0, j0):
            P = loops[k]
            other = loops[j0 if k == i0 else i0]
            d = torch.cdist(P, other).min(1).values
            face = d < d.min() + 2 * delta
            segl = (P.roll(-1, 0) - P).norm(dim=1)
            h_loc.append(segl[face].median().item()
                         if face.any() else segl.median().item())
        l_res_contact = max(h_loc + ([eps_bem] if eps_bem else []))
        gap_res = min_gap / l_res_contact if l_res_contact > 0 else None

        N = pos.shape[0]
        th_all = compute_kde_density(pos, delta, mass_kernel)
        th_self = torch.zeros_like(th_all)
        for c in boundary_loops:
            th_self[c] = compute_kde_density(pos[c], delta, mass_kernel) \
                * (c.numel() / N)
        omega = (1 - th_self / th_all).clamp(min=0.0)
        omega = torch.where(omega > OMEGA_ROUNDOFF, omega,
                            torch.zeros_like(omega))
        omega_q = _quantiles(omega)
        within_range = bool(gap_delta <= 1.0 + 1e-12)
        bias_active = bool(omega_q[-1] > 0)
        facing_mask = omega > 0
        facing_ids = ids[facing_mask].tolist()
        if masses is not None:
            fm = _quantiles(masses[facing_mask])
            if coherence is not None:
                fq = _quantiles(coherence[facing_mask])
                fmq = _quantiles((masses * coherence)[facing_mask])
        if mass_tau is not None:
            m_gl = compute_masses(pos, delta, mass_tau, mass_kernel)
            m_cw = torch.zeros_like(m_gl)
            for c in boundary_loops:
                m_cw[c] = compute_masses(pos[c], delta, mass_tau,
                                         mass_kernel)
            ratio = (m_gl.sum() / m_cw.sum()).item()
        else:
            notes.append("carrier ratio skipped: no mass_tau supplied")

    # nonlocal omega including SAME-loop sheets (C-shape / neck)
    if pos.shape[0] >= 8:
        kern_fn = getattr(_mass, mass_kernel, None)
        if kern_fn is not None:
            seg_all = torch.cat([(P.roll(-1, 0) - P).norm(dim=1)
                                 for P in loops])
            med_h = seg_all.median().item()
            k_arc = max(2, int(round(ARC_EXCLUDE_FACTOR * delta / med_h)))
            d_all = torch.cdist(pos, pos)
            local = torch.zeros_like(d_all, dtype=torch.bool)
            for c in boundary_loops:
                n = c.numel()
                idx = torch.arange(n)
                diff = (idx[:, None] - idx[None, :]).abs()
                diff = torch.minimum(diff, n - diff)
                local[c[:, None], c[None, :]] = diff <= k_arc
            eta = kern_fn(d_all / delta)
            tot = eta.sum(1).clamp_min(1e-300)
            nl = (eta * (~local)).sum(1) / tot
            nl = torch.where(nl > OMEGA_ROUNDOFF, nl,
                             torch.zeros_like(nl))
            omega_nl_q = _quantiles(nl)

    # --- step validity / collocation ---------------------------------
    step_h = step_gap = step_eps = None
    if displacements is not None:
        smax = displacements.abs().max().item()
        step_h = smax / h
        step_gap = (smax / min_gap) if min_gap else None
        step_eps = (smax / eps_bem) if eps_bem else None
    min_coll_eps = None
    if eps_bem and endpoint_positions is not None:
        # TRUE BEM collocation nodes; same-particle and same-loop
        # adjacent-particle pairs excluded (their proximity is by design;
        # the artificial-null-mode danger is NONLOCAL near-duplication)
        E = endpoint_positions
        K = endpoints_per_particle
        pid = torch.arange(E.shape[0]) // K
        d = torch.cdist(E, E)
        excl = pid[:, None] == pid[None, :]
        for c in boundary_loops:
            n = c.numel()
            idx = torch.arange(n)
            diff = (idx[:, None] - idx[None, :]).abs()
            diff = torch.minimum(diff, n - diff)
            pa = diff <= 1
            eidx = (c[:, None] * K
                    + torch.arange(K)[None, :]).reshape(-1)
            excl[eidx[:, None], eidx[None, :]] |= \
                pa.repeat_interleave(K, 0).repeat_interleave(K, 1)
        d = d.masked_fill(excl, float("inf"))
        min_coll_eps = d.min().item() / eps_bem
    elif eps_bem:
        notes.append("collocation check skipped: endpoint positions not "
                     "supplied (particle centers are no substitute)")

    # --- classification along the orthogonal axes --------------------
    if open_endpoints > 0:
        geometry = "open"
    elif self_x > 0 or cross_x > 0 or coll_x > 0:
        geometry = "intersecting"
    elif all(closed) and orientation_ok:
        geometry = "valid_closed"
    else:
        geometry = "degenerate"

    res_scale = max([h] + ([eps_bem] if eps_bem else []))
    if M == 1:
        regime = "none"
        if (min_self_gap is not None
                and min_self_gap <= TOUCH_FACTOR * res_scale):
            regime = "touching"
            notes.append("same-loop nonlocal near-contact (neck)")
    elif min_gap is None:
        regime = "none"
    elif near_x > 0:
        regime = "touching"
    elif l_res_contact and min_gap <= TOUCH_FACTOR * l_res_contact:
        regime = "unresolved"
    elif bias_active:
        regime = "carrier_contact"
    else:
        regime = "separated"

    state = "unknown"
    if geometry == "open":
        # cause-dependent naming (P1.2 reviewer): "open_after_deletion"
        # is ONLY attached when a deletion event actually removed points
        # at this step (threaded from the timeline); an open geometry
        # with no deletion -- e.g. a diverged frame read as scattered
        # points -- is "open_support". Both are sticky-invalid; the gate
        # behaviour is identical, only the causal claim differs.
        state = ("open_after_deletion" if deletion_this_step
                 else "open_support")
    elif geometry == "intersecting":
        state = "invalid_crossing"
    elif geometry == "valid_closed":
        if M == 1:
            if regime == "touching":
                state = "touching_or_unresolved"
            elif reconstruction_certified:
                state = "reconstructed_merged_closed"
            else:
                state = "single_closed"
        else:
            if regime in ("touching", "unresolved"):
                state = "touching_or_unresolved"
            elif regime == "carrier_contact":
                state = "contact_layer"
            elif regime == "separated":
                state = "separated_closed"
    if state == "unknown":
        notes.append("fail closed: no safe classification")

    return SupportReport(
        geometry_status=geometry, interaction_regime=regime,
        provenance=provenance,
        reconstruction_certified=reconstruction_certified,
        state=state,
        n_boundary_loops=M, closed_flags=list(closed),
        open_endpoints=open_endpoints, self_intersections=self_x,
        cross_intersections=cross_x, near_contacts=near_x,
        collinear_overlaps=coll_x,
        min_gap=min_gap, min_self_gap_nonlocal=min_self_gap,
        gap_over_delta=gap_delta, gap_over_sigma=gap_sigma,
        gap_over_resolution=gap_res, l_res_contact=l_res_contact,
        within_carrier_range=within_range,
        carrier_bias_active=bias_active,
        omega_cross_quantiles=omega_q,
        omega_nonlocal_quantiles=omega_nl_q,
        loopwise_carrier_ratio_sum=ratio,
        facing_ids=facing_ids, facing_m_quantiles=fm,
        facing_q_quantiles=fq, facing_mq_quantiles=fmq,
        areas_geom=areas, centroids_geom=cents, nesting_depths=depths,
        orientation_consistent=orientation_ok,
        min_collocation_over_eps=min_coll_eps,
        step_over_h=step_h, step_over_gap=step_gap,
        step_over_eps=step_eps, notes=notes)


def permissions(report: SupportReport) -> dict:
    """Separate gates (review, 0E-wiring.1): rank REFRESH (recomputing
    the basis at the SAME working rank) is routine on any valid
    separated/contact-layer support; rank CHANGE (e.g. C: 2 -> 1) and
    post-contact evolution are topology events and require the
    reconstruction certificate TIED to provenance == "reconstructed" --
    a raw-provenance caller must not obtain them by setting a boolean."""
    valid = report.geometry_status == "valid_closed"
    certified = (report.reconstruction_certified
                 and report.provenance == "reconstructed")
    routine = valid and report.interaction_regime in (
        "separated", "carrier_contact", "none")
    return dict(
        allow_same_rank_evolution=routine,
        allow_rank_refresh=routine,
        allow_rank_change=valid and certified,
        allow_postcontact_evolution=valid and certified,
    )


# ---------------------------------------------------------------------------
# D1b.1 item 6: whole-boundary-loop extinction certificate.
#
# The annulus hole removal is NOT a merger: the boundary-loop count drops
# M: 2 -> 1 while the phase rank stays C = 1, so it must not ride on the
# merger-reconstruction certificate (which attests a C: 2 -> 1 topology
# event on a rebuilt boundary). This certificate attests the much simpler
# event "one whole hole loop was deleted in a single post-processing
# event, everything else untouched" and is what a caller supplies to keep
# evolving the surviving support as a valid scientific trajectory.
# ---------------------------------------------------------------------------

STICKY_INVALID_STATES = (
    "open_after_deletion",
    "open_support",
    "invalid_crossing",
    "touching_or_unresolved",
    "unknown",
)


@dataclass(frozen=True)
class WholeBoundaryLoopExtinctionCertificate:
    loop_index: int             # index into the timeline's initial loops
    step: int                   # solver step of the deletion event
    n_removed: int              # == full survivor count of that loop
    hole_area_pre: float        # signed area of the loop just before (<0)
    area_jump: float            # measured total geometric area jump
    phase_rank_before: int
    phase_rank_after: int


def certify_loop_extinction(pre_record, raw_record, final_record, *,
                            phase_rank_before: int,
                            phase_rank_after: int,
                            area_rtol: float = 1e-9,
                            ) -> WholeBoundaryLoopExtinctionCertificate:
    """Validate a whole-loop extinction event from three timeline records
    of the SAME step (post_mm_candidate, post_deletion_raw,
    post_deletion_final) and mint the certificate. Raises ValueError on
    any violated condition -- a partial deletion, an outer-loop casualty,
    a positively oriented loop or a phase-rank change must never be
    certified by this path.
    """
    if not (pre_record.step == raw_record.step == final_record.step):
        raise ValueError("records must belong to the same step")
    if raw_record.n_removed <= 0:
        raise ValueError("no deletion event in the raw record")

    pre_counts = pre_record.loop_survivor_counts
    raw_counts = raw_record.loop_survivor_counts
    new_vanished = [li for li in raw_record.vanished_loop_ids
                    if li not in pre_record.vanished_loop_ids]
    if len(new_vanished) != 1:
        raise ValueError(
            f"expected exactly one loop to vanish in this event, got "
            f"{new_vanished}")
    li = new_vanished[0]

    # whole loop in ONE event; every other loop fully kept
    if raw_record.n_removed != pre_counts[li]:
        raise ValueError(
            f"partial deletion: removed {raw_record.n_removed} of "
            f"{pre_counts[li]} loop-{li} survivors -- not an extinction")
    for lj, (a, b) in enumerate(zip(pre_counts, raw_counts)):
        if lj != li and a != b:
            raise ValueError(f"loop {lj} lost points in the same event")

    # hole orientation: the vanishing loop must carry NEGATIVE signed
    # area (nesting-parity convention). areas_geom lists ACTIVE loops in
    # initial order, so map through the active set at the pre level.
    if pre_record.report is None:
        raise ValueError("pre record carries no support report")
    active_pre = [k for k, c in enumerate(pre_counts) if c >= 3]
    hole_area = pre_record.report.areas_geom[active_pre.index(li)]
    if not hole_area < 0.0:
        raise ValueError(
            f"vanishing loop {li} has signed area {hole_area} >= 0 -- "
            "not a hole; this certificate covers hole extinction only")

    # post support must be a VALID closed boundary with the loop gone
    if final_record.report is None:
        raise ValueError("final record carries no support report")
    if final_record.report.geometry_status != "valid_closed":
        raise ValueError(
            f"post-event support is {final_record.report.geometry_status},"
            " not valid_closed")
    if li not in final_record.vanished_loop_ids:
        raise ValueError(f"loop {li} is not vanished in the final record")

    # geometric area jump == -(signed hole area), no pi R^2 proxy.
    # Measured candidate -> RAW (the deletion-only jump): with
    # post-removal redistribution enabled the FINAL record additionally
    # carries the redistribution's own area drift on the surviving
    # loops (~1e-4 relative, discovered in D3), which is not part of
    # the extinction event. Without post-removal redistribution
    # raw == final and this is unchanged.
    if raw_record.report is None:
        raise ValueError("raw record carries no support report")
    jump = (sum(raw_record.report.areas_geom)
            - sum(pre_record.report.areas_geom))
    if abs(jump + hole_area) > area_rtol * abs(hole_area):
        raise ValueError(
            f"area jump {jump} does not match -hole area {-hole_area}")

    # NOT a merger: the phase rank must be unchanged
    if phase_rank_before != phase_rank_after:
        raise ValueError(
            f"phase rank changed {phase_rank_before} -> "
            f"{phase_rank_after}: this is a topology event for the "
            "merger-reconstruction certificate, not loop extinction")

    return WholeBoundaryLoopExtinctionCertificate(
        loop_index=li, step=raw_record.step,
        n_removed=raw_record.n_removed,
        hole_area_pre=hole_area, area_jump=jump,
        phase_rank_before=phase_rank_before,
        phase_rank_after=phase_rank_after)
