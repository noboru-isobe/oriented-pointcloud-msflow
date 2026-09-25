"""P1-1: order-free neck telemetry for the pinch-off study
(reviewer program 2026-08-26).

Primary observable: the order-free neck width w_fit -- facing-pair
graph inside an ORACLE ROI (the generator knows the neck axis and
initial center; this is benchmark ground-truth region selection, no
topology information reaches the solver), connected component
crossing the neck center, per-pair normal gaps, local quadratic fit
around the minimum. Reported alongside: w_point (raw min) and
w_segment (the ordered-shadow `min_self_gap` oracle, separately).

Lobe areas by fixed-separator current quadrature (order-free):
    A_L = sum_i m_i min(x_i - x*, 0) n_{x,i},
    A_R = sum_i m_i max(x_i - x*, 0) n_{x,i},
with the EXACT discrete identity A_L + A_R = A_curr where
A_curr = sum_i m_i (x_i - x*) n_{x,i} (machine-precision pin).
J_drain = -dA_L/dt (evaluated by the driver across steps).
"""

from __future__ import annotations

import math

import torch

_T = torch.Tensor


def facing_pairs(pos: _T, nor: _T, h: float, roi: _T,
                 gamma: float = 0.9, c_tang: float = 1.5):
    """Anti-parallel pairs inside the ROI: n_i . n_j <= -gamma and
    tangential offset |(x_j - x_i) . t_i| <= c_tang h. Returns (I, J)
    index tensors (i < j)."""
    idx = roi.nonzero().flatten()
    p, n = pos[idx], nor[idx]
    t = torch.stack([-n[:, 1], n[:, 0]], 1)
    dots = n @ n.T
    dx = p[None, :, :] - p[:, None, :]
    tang = (dx * t[:, None, :]).sum(-1).abs()
    norm = (dx * n[:, None, :]).sum(-1).abs()
    ok = (dots <= -gamma) & (tang <= c_tang * h) & (norm > 1e-12)
    iu = torch.triu_indices(len(idx), len(idx), offset=1)
    mask = ok[iu[0], iu[1]] & ok[iu[1], iu[0]]
    return idx[iu[0][mask]], idx[iu[1][mask]]


def _components(I: _T, J: _T, n_nodes: int):
    """Union-find over the pair graph; returns labels per node id
    appearing in (I, J)."""
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            parent[a] = parent.get(parent[a], parent[a])
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for a, b in zip(I.tolist(), J.tolist()):
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)
    return {a: find(a) for a in parent}


def neck_width(pos: _T, nor: _T, m: _T, h: float,
               axis=(1.0, 0.0), center: float = 0.0,
               roi_half: float = 1.5, gamma: float = 0.9,
               c_tang: float = 1.5, fit_half: float = 0.12):
    """Order-free neck width. Returns dict(w_point, w_fit, n_pairs,
    n_component, s_min) or status='no_pairs'/'no_crossing_component'.
    """
    e = torch.tensor(axis, dtype=pos.dtype)
    e = e / e.norm()
    s_all = pos @ e - center
    roi = s_all.abs() <= roi_half
    I, J = facing_pairs(pos, nor, h, roi, gamma, c_tang)
    if len(I) == 0:
        return dict(status="no_pairs")
    comp = _components(I, J, pos.shape[0])
    # component crossing the center: contains axis coords of both
    # signs (within a slack of h); choose the largest such
    from collections import defaultdict
    members = defaultdict(set)
    for a, r in comp.items():
        members[r].add(a)
    best = None
    for r, mem in members.items():
        ss = s_all[torch.tensor(sorted(mem), dtype=torch.long)]
        if float(ss.min()) < h and float(ss.max()) > -h:
            if best is None or len(mem) > len(members[best]):
                best = r
    if best is None:
        return dict(status="no_crossing_component")
    keep = torch.tensor(
        [comp[int(a)] == best for a, b in zip(I, J)],
        dtype=torch.bool)
    I, J = I[keep], J[keep]
    # gap along the pair BISECTOR normal (n_i - n_j)/|.|: exact on
    # anti-parallel walls and first-order insensitive to the wall
    # tilt in the blend (the raw n_i projection shaved ~2% there)
    nb = nor[I] - nor[J]
    nb = nb / nb.norm(dim=1, keepdim=True)
    gaps = ((pos[J] - pos[I]) * nb).sum(-1).abs()
    s_mid = 0.5 * (s_all[I] + s_all[J])
    w_point = float(gaps.min())
    # per-bin minima (bin width h): deterministic under particle
    # permutation and robust to tangential-offset scatter
    bins = torch.round(s_mid / h)
    ub = bins.unique()
    b_s = torch.tensor([float(s_mid[bins == b].mean()) for b in ub],
                       dtype=pos.dtype)
    b_w = torch.tensor([float(gaps[bins == b].min()) for b in ub],
                       dtype=pos.dtype)
    i0 = int(b_w.argmin())
    s0 = float(b_s[i0])
    sel = (b_s - s0).abs() <= fit_half
    if int(sel.sum()) >= 5:
        X = torch.stack([torch.ones(int(sel.sum()), dtype=pos.dtype),
                         b_s[sel] - s0, (b_s[sel] - s0) ** 2], 1)
        beta = torch.linalg.lstsq(X, b_w[sel][:, None]).solution
        b0, b1, b2 = (float(x) for x in beta.flatten())
        w_fit = b0 - b1 * b1 / (4 * b2) if b2 > 1e-9 else b0
    else:
        w_fit = w_point
    # sanity guard: an ill-conditioned parabola (b2 ~ 0 with a large
    # linear term, e.g. sparse bins without redistribution) can throw
    # the vertex far outside the data; the vertex is only trusted
    # within [0.5, 2] x w_point, else fall back to the raw minimum
    if not (0.5 * w_point <= w_fit <= 2.0 * w_point):
        return dict(status="ok", w_point=w_point, w_fit=w_point,
                    w_fit_guarded=True, n_pairs=int(len(I)),
                    n_component=len(members[best]), s_min=s0)
    return dict(status="ok", w_point=w_point, w_fit=float(w_fit),
                n_pairs=int(len(I)),
                n_component=len(members[best]), s_min=s0)


def lobe_areas(pos: _T, nor: _T, m: _T, x_star: float = 0.0):
    """Fixed-separator lobe areas by order-free current quadrature.
    Identity A_L + A_R == A_curr holds to machine precision by
    construction."""
    xs = pos[:, 0] - x_star
    nx = nor[:, 0]
    A_L = float((m * torch.minimum(xs, torch.zeros_like(xs)) * nx)
                .sum())
    A_R = float((m * torch.maximum(xs, torch.zeros_like(xs)) * nx)
                .sum())
    A_curr = float((m * xs * nx).sum())
    return dict(A_L=A_L, A_R=A_R, A_curr=A_curr,
                identity_residual=A_L + A_R - A_curr)


def w_segment_shadow(pos: _T, delta: float):
    """Ordered-shadow neck width: same-loop segment-segment min gap
    with arc-neighbour exclusion (wraps the support-diagnostics
    convention; ARRAY order = generation order -- shadow oracle
    only, never the primary)."""
    n = pos.shape[0]
    a = pos
    b = pos.roll(-1, 0)
    # midpoint distance proxy between non-neighbouring segments
    mid = 0.5 * (a + b)
    d = torch.cdist(mid, mid)
    idx = torch.arange(n)
    arc = (idx[None, :] - idx[:, None]).abs()
    arc = torch.minimum(arc, n - arc)
    seg = (b - a).norm(dim=1).median()
    k_excl = max(int(math.ceil(2.0 * delta / float(seg))), 3)
    d = torch.where(arc > k_excl, d, torch.full_like(d, float("inf")))
    return float(d.min())


# ---------------------------------------------------------------------
# P1-3 refinement stage additions (reviewer STOP 2 verdict, 2026-08-30)
# ---------------------------------------------------------------------

def neck_width_strips(pos: _T, nor: _T, h: float, axis=(1.0, 0.0),
                      roi_half: float = 1.5, gamma: float = 0.9,
                      c_tang: float = 1.5, fit_half: float = 0.12,
                      amb_tol: float = 0.02):
    """State-functional global-waist estimator (no history): facing
    pairs -> axial-midpoint bins (width h) -> contiguous occupied
    bins = contact strips -> per strip a local quadratic fit of the
    bin-minimum gap around its minimum -> (w_c, x_c). The strip with
    the smallest w_c is the waist; if the runner-up is within amb_tol
    (relative) the result is 'ambiguous' (fail-closed). x_waist is an
    OUTPUT of the estimator, never an input."""
    e = torch.tensor(axis, dtype=pos.dtype)
    e = e / e.norm()
    s_all = pos @ e
    roi = s_all.abs() <= roi_half
    I, J = facing_pairs(pos, nor, h, roi, gamma, c_tang)
    if len(I) == 0:
        return dict(status="no_pairs")
    nb = nor[I] - nor[J]
    nb = nb / nb.norm(dim=1, keepdim=True)
    gaps = ((pos[J] - pos[I]) * nb).sum(-1).abs()
    s_mid = 0.5 * (s_all[I] + s_all[J])
    bins = torch.round(s_mid / h)
    ub = bins.unique()
    b_s = torch.tensor([float(s_mid[bins == b].mean()) for b in ub],
                       dtype=pos.dtype)
    b_w = torch.tensor([float(gaps[bins == b].min()) for b in ub],
                       dtype=pos.dtype)
    # contiguous occupied bins -> strips
    strips, cur = [], [0]
    for i in range(1, len(ub)):
        if int(ub[i]) - int(ub[i - 1]) <= 1:
            cur.append(i)
        else:
            strips.append(cur)
            cur = [i]
    strips.append(cur)
    cands = []
    for st in strips:
        idx = torch.tensor(st, dtype=torch.long)
        ss, wwv = b_s[idx], b_w[idx]
        i0 = int(wwv.argmin())
        s0, w0 = float(ss[i0]), float(wwv[i0])
        sel = (ss - s0).abs() <= fit_half
        w_c, x_c = w0, s0
        if int(sel.sum()) >= 5:
            X = torch.stack([torch.ones(int(sel.sum()), dtype=pos.dtype),
                             ss[sel] - s0, (ss[sel] - s0) ** 2], 1)
            beta = torch.linalg.lstsq(X, wwv[sel][:, None]).solution
            b0, b1, b2 = (float(x) for x in beta.flatten())
            if b2 > 1e-9:
                xv = -b1 / (2 * b2)
                wv = b0 - b1 * b1 / (4 * b2)
                if abs(xv) <= fit_half and 0.5 * w0 <= wv <= 2.0 * w0:
                    w_c, x_c = wv, s0 + xv
        # plateau centre: mean axial position of the strip bins within
        # 1% of the minimum (well-defined on a flat neck, where the
        # single minimum bin is arbitrary)
        plat = wwv <= w0 * 1.01
        cands.append((w_c, x_c, w0, len(st), float(ss[plat].mean())))
    cands.sort(key=lambda c: c[0])
    w_c, x_c, w_pt, n_bins, x_plat = cands[0]
    status = "ok"
    if len(cands) > 1 and cands[1][0] - w_c <= amb_tol * w_c:
        status = "ambiguous"
    return dict(status=status, w_strip=w_c, x_waist=x_c,
                x_plateau=x_plat,
                w_strip_point=w_pt, n_strips=len(strips),
                n_bins=n_bins, runner_up=(cands[1][0] if len(cands) > 1
                                          else None))


def neck_control_area(pos: _T, nor: _T, m: _T, ell_N: float,
                      x_c: float = 0.0):
    """A_N = |E cap {|x - x_c| < ell_N}| by order-free current
    quadrature: A_N = oint clip(x - x_c, -ell_N, ell_N) n_x ds."""
    xs = (pos[:, 0] - x_c).clamp(-ell_N, ell_N)
    return float((m * xs * nor[:, 0]).sum())


def delta_split(P_geom: float, A: float):
    """Perimeter budget margin for a symmetric equal-area split:
    P - 4 sqrt(pi A / 2) (isoperimetric lower bound of the split
    state). Negative + non-increasing perimeter => symmetric
    equal-area pinch-off impossible thereafter."""
    import math
    return P_geom - 4.0 * math.sqrt(math.pi * A / 2.0)


def half_width_profile(pos: _T, nor: _T, h: float):
    """Bin-wise half-width profile from the UPPER wall (n_y > 0):
    H(x) = max y in bin. Returns (x_bins, H) sorted."""
    up = nor[:, 1] > 0.2
    x, y = pos[up, 0], pos[up, 1]
    b = torch.round(x / h)
    ub = b.unique()
    xs = torch.tensor([float(x[b == k].mean()) for k in ub],
                      dtype=pos.dtype)
    Hs = torch.tensor([float(y[b == k].max()) for k in ub],
                      dtype=pos.dtype)
    o = xs.argsort()
    return xs[o], Hs[o]


def p_eff_dynamic(pos: _T, nor: _T, h: float, w: float,
                  x_c: float = 0.0, mult: float = 1.5,
                  fit_half: float = 0.2, deg: int = 4):
    """Dynamic CENV end-curvature data, per side: ell_pm = distance
    from x_c (the neck plateau centre) to where the upper-wall
    half-width profile H = mult * w/2 (bin-profile crossing), then a
    quartic fit of the RAW upper-wall points over |x - ell| <= fit_half
    gives H(ell), H''(ell) and P_eff_pm = ell^2 H''(ell) / H(ell).
    Validated on the t = 0 mod capsule: 5.843 vs generator 5.840
    (quadratic/cubic fits under-estimate by 7-13%). Shadow telemetry."""
    import numpy as np
    xs, Hs = half_width_profile(pos, nor, h)
    up = nor[:, 1] > 0.2
    xr, yr = pos[up, 0].numpy(), pos[up, 1].numpy()
    target = mult * w / 2
    out = {}
    for side, sgn in (("r", +1), ("l", -1)):
        s = sgn * (xs - x_c)
        sel = s > 0
        ss, HH = s[sel], Hs[sel]
        o = ss.argsort()
        ss, HH = ss[o], HH[o]
        cross = (HH[:-1] < target) & (HH[1:] >= target)
        if not bool(cross.any()):
            out[f"p_eff_{side}"] = None
            out[f"ell_{side}"] = None
            continue
        i = int(cross.nonzero()[0])
        f = float((target - HH[i]) / (HH[i + 1] - HH[i] + 1e-300))
        ell = float(ss[i] + f * (ss[i + 1] - ss[i]))
        sr = sgn * (xr - x_c)
        win = np.abs(sr - ell) <= fit_half
        if int(win.sum()) < deg + 4:
            out[f"p_eff_{side}"] = None
            out[f"ell_{side}"] = ell
            continue
        c = np.polyfit(sr[win] - ell, yr[win], deg)
        H0 = float(np.polyval(c, 0.0))
        H2 = float(np.polyval(np.polyder(c, 2), 0.0))
        out[f"p_eff_{side}"] = ell * ell * H2 / H0
        out[f"ell_{side}"] = ell
    return out
