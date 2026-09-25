"""Phase P dumbbell generators (plan approved 2026-08-26; reviewer
corrections applied): Cassini ovals and C^2 capsule dumbbells for the
pinch-off feasibility study.

Design constraints from the review:
- the capsule junctions must be at least C^2 (kappa is Dirichlet data
  in MS; a curvature jump seeds its own transient) -- implemented as
  half-width profiles H(x) that are piecewise {circle graph | MONOTONE
  cubic-in-H' blend | flat neck} with H, H', H'' matched at both blend
  ends; the blend length is DERIVED from (R, w) by the height closure
  (free-length blends can undershoot the neck width -- caught by the
  neck-telemetry pin);
- every shape is rescaled to a COMMON area A0 (default: the
  two-ellipses merged area pi*(2ab), R_eq = sqrt(A0/pi) ~ 0.894), so
  runs are comparable and the production scales h_ref/sigma/eps_fill
  keep their meaning;
- the CENV end-pressure surrogate P_eff = ell^2 H''(ell)/H(ell) is
  computed FROM THE GENERATED PROFILE (left and right separately)
  under the fixed convention ell = the |x| (from the neck center)
  where H first reaches 1.5 * (w/2); the mapping to CENV's P is
  heuristic and is recorded as such;
- the perimeter budget P0 >= 2 sqrt(pi A0)(sqrt f + sqrt(1-f)) is
  computed so inadmissible fixtures can be excluded up front.

Sampling: each analytic piece is densely sampled, concatenated in CCW
order, resampled uniformly by cumulative arc length at spacing ~h;
normals and curvature are re-evaluated from the analytic piece data
at the resampled parameters (no finite-difference normals).
"""

from __future__ import annotations

import math

import torch

_T = torch.Tensor
DT = torch.float64


# ------------------------------------------------------------------
# C^2 capsule dumbbell half-width profile
# ------------------------------------------------------------------

class CapsuleProfile:
    """Half-width profile H(x) of a capsule dumbbell: flat neck of
    half-width w/2 on |x| <= x_n, monotone C^2 blends on
    [x_n, x_m] (and mirrored / left-lobe analogue), circle graphs up
    to the 30-degree points of the lobes. Lobe circles: radius R_l /
    R_r, centers on the x-axis positioned so the circle graph point
    x_m sits at slope 1/2 (c - x_m = R/sqrt(5))."""

    def __init__(self, R_l: float, R_r: float, L: float, w: float,
                 blend: "float | None" = None):
        """blend is DERIVED per side from (R, w) by the monotone
        cubic height closure below; the argument is accepted for API
        compatibility and ignored."""
        self.w = w
        self.x_n = L / 2.0                      # neck end (flat part)
        self.blend = None                       # derived (right side)

        def side(R, sgn):
            y_m = 2.0 * R / math.sqrt(5.0)
            ddy_m = -R * R / y_m ** 3           # circle-graph H''
            # MONOTONE C^2 blend (2026-08-26 estimator finding: a
            # free-length quintic in H undershoots the neck
            # half-width, relocating the true waist into the
            # junction). Work at the H' level with a CUBIC
            # g(s) = s^2 (c2 + c3 s):  g(0) = g'(0) = 0 (C^2 with
            # the flat neck), g(1) = slope_m, g'(1) = ddy_m l_b (C^2
            # with the circle graph). Since c2 = 3 slope_m - d_hat
            # > 0 and c2 + c3 = slope_m > 0, the linear factor is
            # positive at both ends => g >= 0 on [0,1] PROVABLY:
            # H monotone, waist exactly on the flat neck. The height
            # closure  integral(g) l_b = y_m - w/2  then DERIVES the
            # blend length l_b (unique positive root of
            # slope_m l/2 + |ddy_m| l^2/12 = y_m - w/2); the blend
            # is no longer a free knob -- P_eff is tuned via R and L.
            slope_m = 0.5                       # circle graph at x_m
            Dw = y_m - w / 2.0
            if Dw <= 0:
                raise ValueError("capsule: neck as wide as the lobe "
                                 "graph height (w/2 >= 2R/sqrt(5))")
            aa = -ddy_m / 12.0
            bb = slope_m / 2.0
            l_b = (-bb + math.sqrt(bb * bb + 4 * aa * Dw)) / (2 * aa)
            self.blend = l_b
            x_m = sgn * (self.x_n + l_b)
            c = x_m + sgn * R / math.sqrt(5.0)  # lobe center
            d_hat = ddy_m * l_b
            c2 = 3 * slope_m - d_hat
            c3 = d_hat - 2 * slope_m

            def g(sv):
                return sv * sv * (c2 + c3 * sv)

            def dg(sv):
                return 2 * c2 * sv + 3 * c3 * sv ** 2

            def q(x):
                sv = (sgn * x - self.x_n) / l_b
                return w / 2.0 + l_b * (c2 * sv ** 3 / 3
                                        + c3 * sv ** 4 / 4)

            def dq(x):
                sv = (sgn * x - self.x_n) / l_b
                return sgn * g(sv)

            def ddq(x):
                sv = (sgn * x - self.x_n) / l_b
                return dg(sv) / l_b
            return dict(R=R, c=c, x_m=x_m, y_m=y_m, q=q, dq=dq,
                        ddq=ddq, blend=l_b)
        self.r = side(R_r, +1)
        self.l = side(R_l, -1)

    def H(self, x: float) -> float:
        if abs(x) <= self.x_n:
            return self.w / 2.0
        s = self.r if x > 0 else self.l
        if (x > 0 and x <= s["x_m"]) or (x < 0 and x >= s["x_m"]):
            return float(s["q"](x))
        return math.sqrt(max(s["R"] ** 2 - (x - s["c"]) ** 2, 0.0))

    def dH(self, x: float) -> float:
        if abs(x) <= self.x_n:
            return 0.0
        s = self.r if x > 0 else self.l
        if (x > 0 and x <= s["x_m"]) or (x < 0 and x >= s["x_m"]):
            return float(s["dq"](x))
        return -(x - s["c"]) / self.H(x)

    def ddH(self, x: float) -> float:
        if abs(x) <= self.x_n:
            return 0.0
        s = self.r if x > 0 else self.l
        if (x > 0 and x <= s["x_m"]) or (x < 0 and x >= s["x_m"]):
            return float(s["ddq"](x))
        return -s["R"] ** 2 / self.H(x) ** 3

    def graph_range(self):
        """x-range on which the boundary is the graph y = +-H(x):
        out to the 45-degree points of the lobes (slope +-1 there;
        the arcs take over beyond)."""
        xr = self.r["c"] + self.r["R"] / math.sqrt(2.0)
        xl = self.l["c"] - self.l["R"] / math.sqrt(2.0)
        return xl, xr


def p_eff_of_profile(H, ddH, w: float, side: int,
                     mult: float = 1.5, x_max: float = 10.0) -> float:
    """CENV end-pressure surrogate (heuristic, fixed convention):
    ell = the |x| where H first reaches mult*(w/2) on the given side
    (bisection on the analytic profile), P_eff = ell^2 ddH(ell)/H(ell).
    """
    target = mult * w / 2.0
    lo, hi = 0.0, x_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if H(side * mid) < target:
            lo = mid
        else:
            hi = mid
    ell = 0.5 * (lo + hi)
    return ell * ell * ddH(side * ell) / H(side * ell)


# ------------------------------------------------------------------
# dense piecewise boundary -> arc-length resampled oriented cloud
# ------------------------------------------------------------------

def _resample(pieces, h: float):
    """pieces: list of (x, y, nx, ny, kappa) dense tensors in CCW
    order. Concatenate, resample at spacing ~h by cumulative arc
    length (linear interpolation of positions; normals re-normalized;
    kappa interpolated)."""
    X = torch.cat([p[0] for p in pieces])
    Y = torch.cat([p[1] for p in pieces])
    NX = torch.cat([p[2] for p in pieces])
    NY = torch.cat([p[3] for p in pieces])
    K = torch.cat([p[4] for p in pieces])
    d = torch.hypot(X.roll(-1) - X, Y.roll(-1) - Y)
    s = torch.cat([torch.zeros(1, dtype=DT), d.cumsum(0)])[:-1]
    L_tot = float(s[-1] + d[-1])
    n_out = max(int(round(L_tot / h)), 8)
    s_new = torch.arange(n_out, dtype=DT) * (L_tot / n_out)
    idx = torch.searchsorted(s, s_new, right=True) - 1
    idx = idx.clamp(0, len(s) - 1)
    nxt = (idx + 1) % len(s)
    seg = d[idx].clamp_min(1e-300)
    t = ((s_new - s[idx]) / seg).clamp(0.0, 1.0)
    def lerp(A):
        return A[idx] * (1 - t) + A[nxt] * t
    pos = torch.stack([lerp(X), lerp(Y)], 1)
    nor = torch.stack([lerp(NX), lerp(NY)], 1)
    nor = nor / nor.norm(dim=1, keepdim=True)
    ang = torch.atan2(nor[:, 1], nor[:, 0])
    return pos, ang, lerp(K), L_tot


def _graph_piece(profile, x0, x1, n, top: bool):
    """Dense samples of y = +-H(x) between x0 and x1 (traversal
    direction = from x0 to x1), outward normal and curvature from the
    analytic profile."""
    xs = torch.linspace(x0, x1, n, dtype=DT)
    H = torch.tensor([profile.H(float(x)) for x in xs], dtype=DT)
    dH = torch.tensor([profile.dH(float(x)) for x in xs], dtype=DT)
    ddH = torch.tensor([profile.ddH(float(x)) for x in xs], dtype=DT)
    sgn = 1.0 if top else -1.0
    y = sgn * H
    # tangent along traversal: (1, sgn dH) * sign(dx)
    dirx = 1.0 if x1 > x0 else -1.0
    tx = torch.full_like(xs, dirx)
    ty = sgn * dH * dirx
    tn = torch.hypot(tx, ty)
    # outward normal (interior |y| < H): CCW convention n = (t_y, -t_x)
    nx, ny = ty / tn, -tx / tn
    # curvature w.r.t. the OUTWARD normal: kappa = -H''/(1+H'^2)^{3/2}
    # for BOTH graphs (top y=+H with normal +y, bottom y=-H with
    # normal -y: the sgn factors cancel -- the first implementation
    # carried a sign error on the bottom piece, caught by the
    # cloud-vs-analytic curvature pin)
    kappa = -ddH / (1 + dH ** 2) ** 1.5
    return xs, y, nx, ny, kappa


def _arc_piece(c, R, a0, a1, n):
    """CCW arc of circle (center (c,0), radius R) from angle a0 to
    a1, outward normal radial, kappa = 1/R."""
    t = torch.linspace(a0, a1, n, dtype=DT)
    x = c + R * t.cos()
    y = R * t.sin()
    return (x, y, t.cos(), t.sin(),
            torch.full_like(t, 1.0 / R))


def build_capsule_dumbbell(R_l: float, R_r: float, L: float,
                           w: float, h: float,
                           blend: "float | None" = None,
                           area_target: "float | None" = None,
                           n_dense: int = 4000):
    """C^2 capsule dumbbell as an oriented point cloud. Returns dict:
    positions, angles, kappa (analytic, resampled), profile,
    p_eff (left, right), area, perimeter, scale (applied), and the
    junction curvature-continuity diagnostics."""
    pr = CapsuleProfile(R_l, R_r, L, w, blend)
    xl, xr = pr.graph_range()
    a45 = math.pi / 4.0
    pieces = [
        _graph_piece(pr, xl, xr, n_dense, top=False),        # bottom
        _arc_piece(pr.r["c"], pr.r["R"], -a45, a45, n_dense),
        _graph_piece(pr, xr, xl, n_dense, top=True),         # top
        _arc_piece(pr.l["c"], pr.l["R"], math.pi - a45,
                   math.pi + a45, n_dense),
    ]
    pos_d = torch.cat([torch.stack([p[0], p[1]], 1) for p in pieces])
    # area / perimeter of the dense polyline (shoelace)
    x, y = pos_d[:, 0], pos_d[:, 1]
    A = float(0.5 * (x * y.roll(-1) - x.roll(-1) * y).sum())
    scale = 1.0
    if area_target is not None:
        scale = math.sqrt(area_target / A)
        pr = CapsuleProfile(R_l * scale, R_r * scale, L * scale,
                            w * scale)
        xl, xr = pr.graph_range()
        pieces = [
            _graph_piece(pr, xl, xr, n_dense, top=False),
            _arc_piece(pr.r["c"], pr.r["R"], -a45, a45, n_dense),
            _graph_piece(pr, xr, xl, n_dense, top=True),
            _arc_piece(pr.l["c"], pr.l["R"], math.pi - a45,
                       math.pi + a45, n_dense),
        ]
    pos, ang, kap, P = _resample(pieces, h)
    x, y = pos[:, 0], pos[:, 1]
    A_f = float(0.5 * (x * y.roll(-1) - x.roll(-1) * y).sum())
    # junction curvature continuity (analytic): sample kappa along the
    # top graph densely and take the max |d kappa / ds|
    xs = torch.linspace(xl, xr, 20000, dtype=DT)
    kk = torch.tensor([
        -(pr.ddH(float(xx)) / (1 + pr.dH(float(xx)) ** 2) ** 1.5)
        for xx in xs], dtype=DT)
    ds = (xs[1] - xs[0]) * (1 + torch.tensor(
        [pr.dH(float(xx)) for xx in xs], dtype=DT) ** 2).sqrt()
    dk = (kk[1:] - kk[:-1]).abs() / ds[:-1]
    return dict(
        positions=pos, angles=ang, kappa=kap,
        profile=pr, scale=scale,
        w=pr.w, L=2 * pr.x_n, blend=pr.blend,
        p_eff_left=p_eff_of_profile(pr.H, pr.ddH, pr.w, -1,
                                    x_max=abs(pr.l["c"])),
        p_eff_right=p_eff_of_profile(pr.H, pr.ddH, pr.w, +1,
                                     x_max=abs(pr.r["c"])),
        area=A_f, perimeter=P,
        kappa_jump_max=float((kk[1:] - kk[:-1]).abs().max()),
        dkappa_ds_max=float(dk.max()))


# ------------------------------------------------------------------
# Cassini oval
# ------------------------------------------------------------------

def build_cassini(a: float, b: float, h: float,
                  area_target: "float | None" = None,
                  n_dense: int = 200000):
    """Cassini oval |z-a||z+a| = b^2 (b > a: single connected
    dumbbell), polar form r^2 = a^2 cos 2t + sqrt(b^4 - a^4 sin^2 2t).
    Analytic normals from the gradient of F = |z^2 - a^2|^2 - b^4;
    neck half-width sqrt(b^2 - a^2)."""
    assert b > a > 0
    t = torch.linspace(0, 2 * math.pi, n_dense + 1, dtype=DT)[:-1]
    S = torch.sqrt(b ** 4 - a ** 4 * torch.sin(2 * t) ** 2)
    r2 = a * a * torch.cos(2 * t) + S
    r = r2.clamp_min(0).sqrt()
    x, y = r * t.cos(), r * t.sin()
    A = float(0.5 * (x * y.roll(-1) - x.roll(-1) * y).sum())
    scale = 1.0
    if area_target is not None:
        scale = math.sqrt(area_target / A)
        a, b = a * scale, b * scale
        x, y = x * scale, y * scale
    # outward normal: grad F, F = (x^2+y^2)^2 - 2a^2(x^2-y^2) + a^4 - b^4
    q2 = x * x + y * y
    gx = 4 * x * q2 - 4 * a * a * x
    gy = 4 * y * q2 + 4 * a * a * y
    gn = torch.hypot(gx, gy).clamp_min(1e-300)
    nx, ny = gx / gn, gy / gn
    # curvature via divergence of the unit normal along the dense
    # curve (finite difference of the analytic normal -- adequate at
    # this density, used only as telemetry)
    tx, ty = -ny, nx
    dnx = nx.roll(-1) - nx.roll(1)
    dny = ny.roll(-1) - ny.roll(1)
    dsx = x.roll(-1) - x.roll(1)
    dsy = y.roll(-1) - y.roll(1)
    ds = torch.hypot(dsx, dsy).clamp_min(1e-300)
    kap = (dnx * tx + dny * ty) / ds
    pieces = [(x, y, nx, ny, kap)]
    pos, ang, kout, P = _resample(pieces, h)
    xx, yy = pos[:, 0], pos[:, 1]
    A_f = float(0.5 * (xx * yy.roll(-1) - xx.roll(-1) * yy).sum())
    w_half = math.sqrt(b * b - a * a)
    return dict(positions=pos, angles=ang, kappa=kout, a=a, b=b,
                scale=scale, w=2 * w_half, area=A_f, perimeter=P)


def perimeter_budget(A0: float, f: float = 0.5) -> float:
    """Minimum perimeter compatible with splitting into fractions
    (f, 1-f) at fixed area (reviewer eq 10.1)."""
    return 2 * math.sqrt(math.pi * A0) * (math.sqrt(f)
                                          + math.sqrt(1 - f))
