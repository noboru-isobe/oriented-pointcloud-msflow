"""0L-A: certified cyclic loop order + exact polygon-area calculus.

The contact layer exposed a divergence between the oriented-current
volume V_cur = 1/2 sum m~ x.n and the true geometric area: stored
normals tilt away from the geometric normal (x.n = r cos phi), the
componentwise volume constraint pins V_cur, and geometry inflates by
1/<cos phi> (endgame #2, +7% before merger). The reviewer-mandated
replacement conserves the EXACT differential of the polygon area
instead of an empirical tilt correction:

    A(X)      = 1/2 sum_i x_i x x_{i+1}            (shoelace)
    DA(X)[dx] = sum_i g_i . dx_i,
    g_i       = 1/2 J_cw(x_{i+1} - x_{i-1}),  J_cw(a,b) = (b, -a)

Normal-graph restriction dx_i = s_i n_i gives the row
C^geom_i = g_i . n_i = m^poly_i (n_i . nu^poly_i) with the exact dual
quantities m^poly_i = 1/2 |x_{i+1}-x_{i-1}|, nu^poly_i = g_i/|g_i| --
i.e. the LOCAL coefficient m^poly cos(phi), never a scalar
1/<cos phi>. A is a function of positions ONLY (no dtheta column),
and it is exactly quadratic:

    A(X + dX) - A(X) = DA(X)[dX] + Q_A(dX),
    Q_A(dX) = 1/2 sum_i dx_i x dx_{i+1}

so a step with DA = 0 drifts EXACTLY by the second-order term Q_A --
the 0L-A2 acceptance gate is |Delta A - Q_A| <= eps_alg, not a vague
"reduced rush".

Scope: loops that are a RADIAL GRAPH about the chosen sort center
(the loop centroid). This is certified, not assumed (fail-closed
LoopOrderError):
  1. center certificate: the center lies inside the polygon and the
     polygon winds exactly once about it (wind = +-1);
  2. no polygon self-intersection (segment test);
  3. local adjacency: every polygon edge stays within the certified
     loop-graph scale (no long chord across a concavity);
  4. normal transversality: n_i . nu^poly_i > 0 for ALL i -- zero or
     negative means the normal-graph parametrization itself is
     locally degenerate;
  5. the construction is particle-permutation invariant and
     cyclic-shift invariant; reversal flips only the area sign
     (pinned in tests, not re-checked at runtime).
General (non-radial-graph / local-contact) cycle reconstruction is
future contact-complex work.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_T = torch.Tensor


class LoopOrderError(RuntimeError):
    """Fail-closed: the cyclic-order certificate did not hold."""


def _j_cw(v: _T) -> _T:
    """J_cw(a, b) = (b, -a) applied along the last dimension."""
    return torch.stack([v[..., 1], -v[..., 0]], dim=-1)


def _cross_z(a: _T, b: _T) -> _T:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@dataclass(frozen=True)
class LoopOrder:
    """Certified cyclic order of ONE loop.

    order: (n,) long tensor of GLOBAL particle indices in cyclic
    order, oriented so that the polygon dual normal nu^poly aligns
    with the stored normals (mass-weighted majority; transversality
    then certifies EVERY point). With that orientation the signed
    polygon area reproduces the loop's contribution to the bulk
    volume (outward-normal loops positive, inward-normal loops
    negative), so bulk rows are plain sums over member loops."""
    order: _T
    center: _T
    min_transversality: float
    mean_transversality: float
    max_edge_over_scale: float


def _polygon_winding_about(points: _T, c: _T) -> float:
    d = points - c[None, :]
    ang = torch.atan2(d[:, 1], d[:, 0])
    dang = torch.remainder(ang.roll(-1) - ang + torch.pi,
                           2 * torch.pi) - torch.pi
    return float(dang.sum() / (2 * torch.pi))


def _point_in_polygon(points: _T, c: _T) -> bool:
    x, y = points[:, 0], points[:, 1]
    x2, y2 = x.roll(-1), y.roll(-1)
    cx, cy = float(c[0]), float(c[1])
    crosses = ((y > cy) != (y2 > cy)) & (
        cx < x + (cy - y) * (x2 - x) / (y2 - y + 1e-300))
    return bool(crosses.sum() % 2 == 1)


def _has_self_intersection(points: _T) -> bool:
    """O(n^2) proper-crossing test over the closed polyline."""
    n = points.shape[0]
    p = points
    q = points.roll(-1, 0)
    d = q - p
    for i in range(n):
        # skip adjacent edges (share a vertex)
        hi = n if i > 0 else n - 1
        if i + 2 >= hi:
            continue
        js = torch.arange(i + 2, hi)
        r = p[i]
        s = d[i]
        denom = _cross_z(s.expand_as(d[js]), d[js])
        rel = p[js] - r
        t = _cross_z(rel, d[js])
        u = _cross_z(rel, s.expand_as(rel))
        with torch.no_grad():
            mask = denom.abs() > 1e-30
            t_ = t[mask] / denom[mask]
            u_ = u[mask] / denom[mask]
            if bool((((t_ > 0) & (t_ < 1)) & ((u_ > 0)
                                              & (u_ < 1))).any()):
                return True
    return False


def certified_cyclic_order(
    positions: _T,
    loop_labels: _T,
    normals: _T,
    masses: _T,
    graph_scale: float,
    adjacency_factor: float = 4.5,
) -> dict[int, LoopOrder]:
    """Certified per-loop cyclic order (see module doc for the five
    certificate conditions). graph_scale is the certified loop-graph
    spacing (median raw mass); polygon edges must stay within
    adjacency_factor * graph_scale -- the same envelope family as the
    3-scale loop-graph certificate (upper factor 4.5), so an order
    that jumps across a concavity is rejected rather than silently
    forming a wrong simple polygon."""
    out: dict[int, LoopOrder] = {}
    for lb in loop_labels.unique():
        sel = (loop_labels == lb).nonzero().flatten()
        pts = positions[sel]
        if pts.shape[0] < 3:
            raise LoopOrderError(
                f"loop {int(lb)} has {pts.shape[0]} < 3 points -- no "
                f"polygon order")
        c = pts.mean(dim=0)
        ang = torch.atan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        perm = ang.argsort()
        pts_o = pts[perm]
        # orientation: align nu^poly with stored normals
        # (mass-weighted majority vote; per-point transversality is
        # certified below)
        g = 0.5 * _j_cw(pts_o.roll(-1, 0) - pts_o.roll(1, 0))
        trans = (g * normals[sel][perm]).sum(dim=1)
        m_l = masses[sel][perm]
        if float((m_l * torch.sign(trans)).sum()) < 0:
            perm = perm.flip(0)
            pts_o = pts[perm]
            g = 0.5 * _j_cw(pts_o.roll(-1, 0) - pts_o.roll(1, 0))
            trans = (g * normals[sel][perm]).sum(dim=1)
            m_l = masses[sel][perm]
        # certificate 1: center inside + unit winding
        wind = _polygon_winding_about(pts_o, c)
        if not _point_in_polygon(pts_o, c):
            raise LoopOrderError(
                f"loop {int(lb)}: sort center outside polygon -- not "
                f"a radial graph about the centroid")
        if abs(abs(wind) - 1.0) > 1e-9:
            raise LoopOrderError(
                f"loop {int(lb)}: polygon winds {wind:.6f} (not +-1) "
                f"about the sort center")
        # certificate 2: simple polygon
        if _has_self_intersection(pts_o):
            raise LoopOrderError(
                f"loop {int(lb)}: ordered polygon self-intersects")
        # certificate 3: local adjacency within the graph envelope
        edge = (pts_o.roll(-1, 0) - pts_o).norm(dim=1)
        max_ratio = float(edge.max() / (adjacency_factor
                                        * graph_scale))
        if max_ratio > 1.0:
            raise LoopOrderError(
                f"loop {int(lb)}: polygon edge {float(edge.max()):.4g}"
                f" exceeds the adjacency envelope "
                f"{adjacency_factor} * {graph_scale:.4g} -- possible "
                f"chord across a concavity")
        # certificate 4: pointwise normal transversality
        m_poly = g.norm(dim=1)
        cosphi = trans / m_poly.clamp_min(1e-300)
        min_t = float(cosphi.min())
        if min_t <= 0.0:
            raise LoopOrderError(
                f"loop {int(lb)}: normal transversality violated "
                f"(min n . nu_poly = {min_t:.4f} <= 0) -- the normal "
                f"graph parametrization is locally degenerate")
        mean_t = float((m_l * cosphi).sum() / m_l.sum())
        out[int(lb)] = LoopOrder(
            order=sel[perm],
            center=c,
            min_transversality=min_t,
            mean_transversality=mean_t,
            max_edge_over_scale=max_ratio,
        )
    return out


def polygon_gradient(positions: _T, order: _T) -> _T:
    """Exact area gradient at the ordered vertices, returned in the
    GLOBAL index frame: g_i = 1/2 J_cw(x_{i+1} - x_{i-1}); zero for
    particles outside the loop."""
    p = positions[order]
    g = 0.5 * _j_cw(p.roll(-1, 0) - p.roll(1, 0))
    out = torch.zeros_like(positions)
    out[order] = g
    return out


def polygon_dual(positions: _T, order: _T) -> tuple[_T, _T]:
    """(m^poly, nu^poly) on the ordered loop (local frame):
    m^poly_i = |g_i|, nu^poly_i = g_i / |g_i|."""
    p = positions[order]
    g = 0.5 * _j_cw(p.roll(-1, 0) - p.roll(1, 0))
    m_poly = g.norm(dim=1)
    return m_poly, g / m_poly.clamp_min(1e-300)[:, None]


def polygon_area(positions: _T, order: _T) -> _T:
    """Signed polygon area 1/2 sum x_i x x_{i+1} of the ordered loop.
    Internally asserts the exact discrete divergence identity
    A = 1/2 sum m^poly x . nu^poly (same expression rearranged --
    guards the sign/coefficient wiring, reviewer 0L section 4)."""
    p = positions[order]
    a_sho = 0.5 * _cross_z(p, p.roll(-1, 0)).sum()
    g = 0.5 * _j_cw(p.roll(-1, 0) - p.roll(1, 0))
    a_div = 0.5 * (p * g).sum()
    if not torch.isclose(a_sho, a_div, rtol=1e-10,
                         atol=1e-14 * max(1.0, float(a_sho.abs()))):
        raise AssertionError(
            f"polygon divergence identity broke: shoelace "
            f"{float(a_sho):.16e} vs 1/2 sum x.g {float(a_div):.16e}")
    return a_sho


def area_variation_rows(
    positions: _T,
    normals: _T,
    bulk_label_per_particle: _T,
    orders: dict[int, LoopOrder],
    loop_labels: _T,
    n_bulk: int,
) -> _T:
    """0L physical bulk rows over the concatenated (s, dtheta)
    variables, shape (B, 2N): row b, entry i = g_i . n_i summed over
    the member loops of bulk b (signs automatic from the certified
    orientation), dtheta block EXACTLY zero (the polygon area is a
    positions-only functional)."""
    N = positions.shape[0]
    c_geom = torch.zeros(N, dtype=positions.dtype,
                         device=positions.device)
    for lb, lo in orders.items():
        g = polygon_gradient(positions, lo.order)
        sel = loop_labels == lb
        c_geom[sel] = (g[sel] * normals[sel]).sum(dim=1)
    rows = []
    zeros = torch.zeros(N, dtype=positions.dtype,
                        device=positions.device)
    for b in range(n_bulk):
        ind = (bulk_label_per_particle == b).to(positions.dtype)
        rows.append(torch.cat([c_geom * ind, zeros]))
    return torch.stack(rows) if rows else \
        torch.zeros(0, 2 * N, dtype=positions.dtype,
                    device=positions.device)


def polygon_first_moments(positions: _T, order: _T) -> tuple[_T, _T]:
    """Signed polygon first spatial moments M_k = int_E x_k dx
    (shoelace-moment formula on the certified order; the orientation
    sign is automatic, matching polygon_area)."""
    p = positions[order]
    x, y = p[:, 0], p[:, 1]
    x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
    cr = x * y2 - x2 * y
    return (((x + x2) * cr).sum() / 6.0,
            ((y + y2) * cr).sum() / 6.0)


def first_moment_variation_rows(
    positions: _T,
    normals: _T,
    bulk_label_per_particle: _T,
    orders: dict[int, LoopOrder],
    loop_labels: _T,
    n_bulk: int,
) -> _T:
    """L0J-M (reviewer 2026-08-21): exact differentials of the signed
    polygon FIRST MOMENTS over the (s, dtheta) variables, shape
    (2B, 2N) with row order (b0_x, b0_y, b1_x, ...). Row entry
    i = grad_{x_i} M_k . n_i (autograd of the cubic shoelace-moment
    polynomial -- exact), dtheta block EXACTLY zero (the moment is a
    positions-only functional). These rows put the one-phase MS exact
    invariant d/dt int_E x dx = 0 into the admissible tangent space --
    they do not modify the energy; the first moment is constrained to
    first order (the barycenter is then preserved up to the existing
    area-functional error)."""
    N = positions.shape[0]
    gx = torch.zeros(N, dtype=positions.dtype, device=positions.device)
    gy = torch.zeros(N, dtype=positions.dtype, device=positions.device)
    for lb, lo in orders.items():
        p_req = positions.detach().clone().requires_grad_(True)
        Mx, My = polygon_first_moments(p_req, lo.order)
        gMx = torch.autograd.grad(Mx, p_req, retain_graph=True)[0]
        gMy = torch.autograd.grad(My, p_req)[0]
        sel = loop_labels == lb
        gx[sel] = (gMx[sel] * normals[sel]).sum(dim=1)
        gy[sel] = (gMy[sel] * normals[sel]).sum(dim=1)
    zeros = torch.zeros(N, dtype=positions.dtype,
                        device=positions.device)
    rows = []
    for b in range(n_bulk):
        ind = (bulk_label_per_particle == b).to(positions.dtype)
        rows.append(torch.cat([gx * ind, zeros]))
        rows.append(torch.cat([gy * ind, zeros]))
    return torch.stack(rows) if rows else \
        torch.zeros(0, 2 * N, dtype=positions.dtype,
                    device=positions.device)


def area_quadratic_term(delta_x: _T, order: _T) -> _T:
    """Exact second-order area term of one loop:
    Q_A(dX) = 1/2 sum_i dx_i x dx_{i+1} (ordered frame). Together
    with DA this reproduces the finite step change EXACTLY:
    A(X + dX) - A(X) = DA(X)[dX] + Q_A(dX)."""
    d = delta_x[order]
    return 0.5 * _cross_z(d, d.roll(-1, 0)).sum()
