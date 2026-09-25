"""L4 / S0: certified arc splice -- the zero-time representation
canonicalization that changes the DERIVED LOOP COMPLEX from two
cycles to one cycle (reviewer 2026-08-24 GO; the operational
point-cloud topology change).

Generalization contract: this is the partial-window generalization of
the Comp-4 conservative ghost compression (ghost_compression.py).
There the certified annihilating pair covers BOTH FULL loops, the
remainders are empty and the survivor is the pre-existing outer loop
(volume transferred by homothety). Here the pair is a facing ARC pair
(the certified h-window); the survivor is the crosswise reconnection
of the two remainders, and the volume/moment transfer is a least-norm
normal displacement closed by Newton. A future unification folds
Comp-4 into `remainders empty -> homothety onto mask_out`.

Construction (reviewer minimal spec):
  1. remove the facing arcs W1 subset Gamma1, W2 subset Gamma2
     (contiguous runs in certified order, cyclically canonicalized);
  2. reconnect the two remaining open arcs crosswise at the window
     ends; BOTH candidate pairings are built and the certificates
     (simple Jordan + outward orientation) select the unique valid
     one -- never chosen by construction;
  3. cubic Hermite bridges from the endpoint positions and certified
     tangents, resampled at the existing spacing h; bridge normals
     from the curve tangents with the loop's outward convention;
  4. conservation correction: least-norm (mass-weighted) normal
     displacement with the exact polygon rows [C_A; C_Mx; C_My],
     Newton-closed so merged area AND first moment match the
     PRE-SPLICE values to eps_alg.

Event certificates (all-or-nothing, fail-closed):
  single loop / simple polygon / outward orientation / |dA| <=
  eps_alg / |dM| <= eps_alg / P#_new <= P#_pre + margin (the
  annihilating window carries positive sharp perimeter -- a splice
  must not create energy) / spacing envelope / normal transversality.
"""

from __future__ import annotations

import math

import torch

_T = torch.Tensor

EPS_ALG = 1e-10          # geometric closure tolerance of the Newton
                         # correction (float64 polygon algebra)


class ArcSpliceError(RuntimeError):
    pass


def _polygon_A_M(pos: _T):
    x, y = pos[:, 0], pos[:, 1]
    x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
    cr = x * y2 - x2 * y
    A = 0.5 * cr.sum()
    Mx = ((x + x2) * cr).sum() / 6.0
    My = ((y + y2) * cr).sum() / 6.0
    return A, Mx, My


def _has_self_intersection(points: _T) -> bool:
    from ..transport.loop_geometry import _has_self_intersection as f
    return f(points)


def _hermite_bridge(p0: _T, t0: _T, p1: _T, t1: _T, h: float):
    """Cubic Hermite from p0 (unit tangent t0) to p1 (unit tangent
    t1), resampled at ~h. Returns (interior points, interior unit
    tangents) -- possibly empty for a short bridge."""
    chord = float((p1 - p0).norm())
    n_seg = max(int(round(chord / h)), 1)
    if n_seg < 2:
        return (torch.empty(0, 2, dtype=p0.dtype),
                torch.empty(0, 2, dtype=p0.dtype))
    s = torch.linspace(0.0, 1.0, n_seg + 1, dtype=p0.dtype)[1:-1]
    m0, m1 = t0 * chord, t1 * chord          # tangent scaling
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    pts = (h00[:, None] * p0 + h10[:, None] * m0
           + h01[:, None] * p1 + h11[:, None] * m1)
    d00 = 6 * s ** 2 - 6 * s
    d10 = 3 * s ** 2 - 4 * s + 1
    d01 = -6 * s ** 2 + 6 * s
    d11 = 3 * s ** 2 - 2 * s
    tan = (d00[:, None] * p0 + d10[:, None] * m0
           + d01[:, None] * p1 + d11[:, None] * m1)
    tan = tan / tan.norm(dim=1, keepdim=True)
    return pts, tan


def _tangent_sign(pos: _T, ang: _T) -> float:
    """Outward-normal convention of the stored cloud: sign s such
    that normal = J_s tangent with J_s = rotation by -s pi/2 (s=+1:
    normal = tangent rotated -90 deg, the CCW/outward convention)."""
    t = pos.roll(-1, 0) - pos.roll(1, 0)
    t = t / t.norm(dim=1, keepdim=True)
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    # candidate s=+1: n = (t_y, -t_x)
    plus = (nor[:, 0] * t[:, 1] - nor[:, 1] * t[:, 0]).sum()
    return 1.0 if float(plus) > 0 else -1.0


def arc_splice(pos: _T, ang: _T, labels: _T, w1_idx: _T, w2_idx: _T,
               h: float, eps_alg: float = EPS_ALG,
               p_sharp=None) -> dict:
    """S(X): remove the certified facing arcs, reconnect crosswise,
    bridge, conserve. w1_idx / w2_idx are the window particle ids of
    loop 0 / loop 1 as CONTIGUOUS runs in certified (array) order
    (cyclically canonicalized by the caller). Returns dict with the
    new cloud, the certificates and full telemetry."""
    dt_ = pos.dtype
    A_pre, Mx_pre, My_pre = (sum(t) for t in zip(
        *[_polygon_A_M(pos[labels == lb]) for lb in (0, 1)]))
    rec: dict = dict(A_pre=float(A_pre), M_pre=(float(Mx_pre),
                                                float(My_pre)))

    def remainder(lb, w_idx):
        mem = (labels == lb).nonzero().flatten()
        n = mem.shape[0]
        loc = torch.searchsorted(mem, w_idx)          # local indices
        if not torch.equal(mem[loc], w_idx):
            raise ArcSpliceError("window ids not inside the loop")
        inw = torch.zeros(n, dtype=torch.bool)
        inw[loc] = True
        # contiguous cyclic run of the COMPLEMENT
        starts = [i for i in range(n)
                  if not inw[i] and inw[(i - 1) % n]]
        if len(starts) != 1:
            raise ArcSpliceError(
                f"loop {lb}: window complement has {len(starts)} runs")
        k = int((~inw).sum())
        order = [(starts[0] + j) % n for j in range(k)]
        return mem[torch.tensor(order, dtype=torch.long)]

    r1 = remainder(0, w1_idx)
    r2 = remainder(1, w2_idx)
    rec["n_removed"] = int(w1_idx.shape[0] + w2_idx.shape[0])
    sgn = _tangent_sign(pos[labels == 0], ang[labels == 0])

    def endpoint_tangent(ids, at_start):
        """Tangent along traversal from a 4-point least-squares fit
        (a 2-point difference kinks the junction: the S2-long run
        showed the seeded kappa ~ 28 kink sharpening to ~74 under the
        coherence-suppressed mobility and folding at step ~1315)."""
        k = min(4, len(ids))
        sel = ids[:k] if at_start else ids[-k:]
        q = pos[sel]
        tt = torch.arange(k, dtype=q.dtype) - (k - 1) / 2.0
        t = ((q - q.mean(dim=0)) * tt[:, None]).sum(dim=0)
        return t / t.norm()

    def build(order_b):
        """Assemble candidate: r1 + bridge + (r2 in order_b) + bridge."""
        rb = r2 if order_b == "forward" else r2.flip(0)
        tb_in = endpoint_tangent(rb, True)
        tb_out = endpoint_tangent(rb, False)
        t1_out = endpoint_tangent(r1, False)
        t1_in = endpoint_tangent(r1, True)
        br1, tanbr1 = _hermite_bridge(pos[r1[-1]], t1_out,
                                      pos[rb[0]], tb_in, h)
        br2, tanbr2 = _hermite_bridge(pos[rb[-1]], tb_out,
                                      pos[r1[0]], t1_in, h)
        P = torch.cat([pos[r1], br1, pos[rb], br2])
        ang_rb = (ang[rb] if order_b == "forward"
                  else ang[rb] + math.pi)      # reversal flips normal
        def nang(tan):
            # normal from tangent with the source convention
            nx = sgn * tan[:, 1]
            ny = -sgn * tan[:, 0]
            return torch.atan2(ny, nx)
        AN = torch.cat([ang[r1], nang(tanbr1), ang_rb, nang(tanbr2)])
        prov = torch.cat([r1, torch.full((br1.shape[0],), -1,
                                         dtype=torch.long),
                          rb, torch.full((br2.shape[0],), -1,
                                         dtype=torch.long)])
        return P, AN, prov

    cands = {}
    for ob in ("forward", "reversed"):
        P, AN, prov = build(ob)
        simple = not _has_self_intersection(P)
        A_new = float(_polygon_A_M(P)[0])
        # outward orientation: polygon normal (traversal convention
        # sgn) must align with the stored normals (majority)
        t = P.roll(-1, 0) - P.roll(1, 0)
        nor_poly = torch.stack([sgn * t[:, 1], -sgn * t[:, 0]], 1)
        nor_st = torch.stack([AN.cos(), AN.sin()], 1)
        align = float((nor_poly * nor_st).sum(dim=1).sign().sum())
        cands[ob] = dict(P=P, AN=AN, prov=prov, simple=simple,
                         A=A_new, align=align,
                         valid=simple and A_new > 0 and align > 0)
    rec["candidates"] = {k: dict(simple=v["simple"], A=v["A"],
                                 align=v["align"], valid=v["valid"])
                         for k, v in cands.items()}
    valid = [k for k, v in cands.items() if v["valid"]]
    if len(valid) != 1:
        raise ArcSpliceError(
            f"crosswise pairing not uniquely certified: valid = "
            f"{valid} ({rec['candidates']})")
    rec["pairing"] = valid[0]
    P, AN, prov = (cands[valid[0]][k] for k in ("P", "AN", "prov"))
    rec["n_new"] = int(P.shape[0])
    rec["n_bridge"] = int((prov < 0).sum())
    # ---- junction smoothing (S2-long lesson) --------------------
    # local curvature smoothing around the bridge junctions BEFORE
    # the conservation projection: a residual kink is not healed by
    # the flow (its anti-aligned walls are coherence-suppressed) --
    # it sharpens and eventually folds. 12 damped Laplacian sweeps on
    # the scar neighborhoods; the global A/M Newton projection below
    # restores the invariants exactly.
    scar = set()
    for j in (prov < 0).nonzero().flatten().tolist():
        for o in range(-6, 7):
            scar.add((j + o) % P.shape[0])
    scar = torch.tensor(sorted(scar), dtype=torch.long)
    for _ in range(12):
        lap = 0.5 * (P.roll(-1, 0) + P.roll(1, 0)) - P
        P[scar] = P[scar] + 0.5 * lap[scar]
    # re-derive scar angles from the smoothed polygon tangents
    tsm = P.roll(-1, 0) - P.roll(1, 0)
    tsm = tsm / tsm.norm(dim=1, keepdim=True)
    nx = sgn * tsm[:, 1]
    ny = -sgn * tsm[:, 0]
    AN = AN.clone()
    AN[scar] = torch.atan2(ny[scar], nx[scar])
    def kmax_of(Q):
        a, b, c = Q.roll(1, 0), Q, Q.roll(-1, 0)
        cr = (b - a)[:, 0] * (c - a)[:, 1] \
            - (b - a)[:, 1] * (c - a)[:, 0]
        den = (b - a).norm(dim=1) * (c - b).norm(dim=1) \
            * (c - a).norm(dim=1)
        return float((2 * cr / den.clamp_min(1e-30)).abs().max())
    rec["kappa_max_post_smooth"] = kmax_of(P)
    # ---- conservation correction (least-norm + Newton) ----------
    nor = torch.stack([AN.cos(), AN.sin()], 1)
    X = P.clone()
    for it in range(6):
        Xv = X.clone().requires_grad_(True)
        A, Mx, My = _polygon_A_M(Xv)
        res = torch.stack([A - A_pre, Mx - Mx_pre, My - My_pre])
        if float(res.abs().max()) <= eps_alg:
            break
        rows = []
        for q in (A, Mx, My):
            gr = torch.autograd.grad(q, Xv, retain_graph=True)[0]
            rows.append((gr * nor).sum(dim=1))       # normal-projected
        R = torch.stack(rows)                        # (3, N)
        G = R @ R.T
        lam = torch.linalg.solve(G, -res.detach())
        s = (R.T @ lam)
        X = X + s[:, None] * nor
    A1, Mx1, My1 = _polygon_A_M(X)
    rec["corr_residual"] = (float(A1 - A_pre), float(Mx1 - Mx_pre),
                            float(My1 - My_pre))
    rec["corr_disp_over_h"] = float(((X - P).norm(dim=1)).max() / h)
    # ---- event certificates -------------------------------------
    cert = {}
    cert["simple"] = not _has_self_intersection(X)
    cert["orientation_area"] = float(_polygon_A_M(X)[0]) > 0
    cert["area"] = abs(rec["corr_residual"][0]) <= eps_alg
    cert["moment"] = max(abs(rec["corr_residual"][1]),
                         abs(rec["corr_residual"][2])) <= eps_alg
    edge = (X.roll(-1, 0) - X).norm(dim=1)
    rec["edge_over_h"] = (float(edge.min() / h), float(edge.max() / h))
    cert["spacing"] = edge.max() <= 4.5 * h and edge.min() > 0.05 * h
    t = X.roll(-1, 0) - X.roll(1, 0)
    nor_poly = torch.stack([sgn * t[:, 1], -sgn * t[:, 0]], 1)
    nor_poly = nor_poly / nor_poly.norm(dim=1, keepdim=True)
    cosphi = (nor_poly * nor).sum(dim=1)
    rec["transversality_min"] = float(cosphi.min())
    cert["transversality"] = float(cosphi.min()) > 0.0
    if p_sharp is not None:
        P_pre = float(p_sharp(pos, ang, labels))
        P_new = float(p_sharp(X, AN, torch.zeros(X.shape[0],
                                                 dtype=torch.long)))
        rec["P_sharp_pre"], rec["P_sharp_new"] = P_pre, P_new
        cert["perimeter_drop"] = P_new <= P_pre + 1e-9
    rec["certificates"] = cert
    rec["certified"] = all(cert.values())
    rec["positions"], rec["angles"], rec["provenance"] = X, AN, prov
    return rec
