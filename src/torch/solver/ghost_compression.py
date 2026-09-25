"""Comp-4 (reviewer 2026-08-19): conservative ghost compression -- the
zero-time canonicalization of the raw representation after a committed
bulk quotient. Scope: ONE certified annihilating pair + ONE surviving
outer loop (fixed-N exact-merger class); shadow-certified,
resolution-dependent -- NOT an exact gauge transformation at finite
resolution.

The compression map C removes the ghost pair atomically and transfers
its signed volume to the surviving outer loop by a homothety about the
outer POLYGON centroid z (the polygon centroid is algebraically
preserved only for z = c_poly):

    x_i -> z + c (x_i - z),        c_* = sqrt(V_sharp / A_out),

where the PRIMARY event conservation functional is the same one the
State III hard gate reads (driver vol_geom_* series):

    V_sharp(X) = A_out + A_disk - A_hole      (star-shoelace areas,
                 orientation-aware signed combination -- never a plain
                 sum of absolute areas),

so F(c) = c^2 A_out - V_sharp is exactly quadratic and c_* is
analytic; the bisection path exists only for injected non-polynomial
functionals (bracket + local monotonicity required, fail-closed
otherwise). The current-bookkeeping volume 0.5 sum m x.n and the
radius formula are SHADOW quantities. EPS_ALG is an ALGEBRAIC
tolerance (float64 root/consistency precision), not a calibrated
physical envelope. The transfer magnitude ||C(X)-X||_inf / h_wall is
telemetry, never gated; the hard event gate is the transfer-map
residual of the committed candidate against C(X_source)."""

from __future__ import annotations

import math

import torch

_T = torch.Tensor

EPS_ALG = 1e-12


class GhostCompressionError(RuntimeError):
    """Fail-closed guard of the compression map/certificates."""


# ---------------------------------------------------------------------
# geometry (star-shaped sheets; the sort matches the driver's
# estimator-independent gate functional bitwise)
# ---------------------------------------------------------------------

def _sorted_by_angle(pts: _T) -> _T:
    c = pts.mean(dim=0)
    d = pts - c
    return pts[torch.atan2(d[:, 1], d[:, 0]).argsort()]


def star_shoelace_area(pts: _T) -> float:
    """|shoelace| after the angular sort about the centroid -- the
    SAME functional as the driver's shoelace_area (State III gate)."""
    if pts.shape[0] < 3:
        raise GhostCompressionError("loop with < 3 particles")
    p = _sorted_by_angle(pts)
    x, y = p[:, 0], p[:, 1]
    x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
    return float(0.5 * torch.abs((x * y2 - x2 * y).sum()))


def signed_shoelace(pts: _T) -> float:
    """Signed shoelace in the GIVEN cyclic order (source order is
    transported, never re-derived)."""
    x, y = pts[:, 0], pts[:, 1]
    x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
    return float(0.5 * (x * y2 - x2 * y).sum())


def polygon_centroid(pts: _T) -> _T:
    """Polygon (area) centroid in the angular-sorted order; falls back
    to the vertex mean only for degenerate (near-zero area) input."""
    p = _sorted_by_angle(pts)
    x, y = p[:, 0], p[:, 1]
    x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
    cross = x * y2 - x2 * y
    a = 0.5 * cross.sum()
    if float(a.abs()) < 1e-300:
        return pts.mean(dim=0)
    cx = ((x + x2) * cross).sum() / (6.0 * a)
    cy = ((y + y2) * cross).sum() / (6.0 * a)
    return torch.stack([cx, cy])


def loop_orientation_sign(pos: _T, nor: _T, mask: _T) -> int:
    """+1 for an outward sheet, -1 for inward: sign of the mean radial
    normal component about the loop centroid (estimator-independent,
    reads the varifold orientation, not masses)."""
    x = pos[mask]
    n = nor[mask]
    c = x.mean(dim=0)
    s = float(((x - c) * n).sum(dim=1).mean())
    if s == 0.0:
        raise GhostCompressionError("degenerate loop orientation")
    return 1 if s > 0 else -1


def merged_geom_volume(pos: _T, nor: _T, masks: "list[_T]") -> float:
    """V_sharp = sum over loops of (orientation sign) * star area --
    for the exact-merger triple this is A_out + A_disk - A_hole, i.e.
    EXACTLY the State III hard-gate combination."""
    V = 0.0
    for m in masks:
        V += loop_orientation_sign(pos, nor, m) \
            * star_shoelace_area(pos[m])
    return V


# ---------------------------------------------------------------------
# scalar volume root
# ---------------------------------------------------------------------

def solve_transfer_scale(V_target: float, A_out: float,
                         F=None, eps_alg: float = EPS_ALG) -> dict:
    """c_* with F(c_*) = 0 for the homothety of the outer loop.

    Default (shoelace) branch: F(c) = c^2 A_out - V_target is exactly
    quadratic -> analytic root, the residual check is a confirmation.
    Injected-functional branch (F callable, F(c) increasing in c near
    the root): analytic c0 as the initial guess, bracket expansion,
    local monotonicity check, bisection; no bracket -> fail-closed.
    Structure certificate (exact merger: V_pair < 0): 0 < c_* <= 1."""
    if V_target <= 0.0 or A_out <= 0.0:
        raise GhostCompressionError(
            f"non-positive volume/area (V {V_target}, A_out {A_out})")
    c0 = math.sqrt(V_target / A_out)
    if F is None:
        c = c0
        resid = c * c * A_out - V_target
        out = dict(c=c, residual=resid, iterations=0, branch="analytic")
    else:
        lo, hi = c0 * (1.0 - 1e-3), c0 * (1.0 + 1e-3)
        flo, fhi = F(lo), F(hi)
        n_exp = 0
        while flo * fhi > 0.0 and n_exp < 60:
            lo *= 0.999
            hi *= 1.001
            flo, fhi = F(lo), F(hi)
            n_exp += 1
        if flo * fhi > 0.0:
            raise GhostCompressionError(
                f"no bracket for the transfer scale around c0 = {c0}")
        if not (flo < 0.0 < fhi):
            raise GhostCompressionError(
                "F is not locally increasing across the bracket "
                f"(F({lo}) = {flo}, F({hi}) = {fhi})")
        it = 0
        while hi - lo > eps_alg * c0 and it < 200:
            mid = 0.5 * (lo + hi)
            if F(mid) < 0.0:
                lo = mid
            else:
                hi = mid
            it += 1
        c = 0.5 * (lo + hi)
        out = dict(c=c, residual=F(c), iterations=it, branch="bisection")
    if abs(out["residual"]) > eps_alg * max(V_target, 1.0):
        raise GhostCompressionError(
            f"transfer-scale residual {out['residual']:.3e} > eps_alg")
    if not (0.0 < out["c"] <= 1.0 + eps_alg):
        raise GhostCompressionError(
            f"transfer scale c = {out['c']} outside (0, 1] -- "
            "orientation / pair assignment / volume bookkeeping is "
            "inconsistent (exact-merger structure certificate)")
    return out


# ---------------------------------------------------------------------
# certificates
# ---------------------------------------------------------------------

def polygon_is_simple(pts: _T) -> bool:
    """O(n^2) segment-intersection test on the closed polyline in the
    GIVEN order (transported source order)."""
    n = pts.shape[0]
    a = pts
    b = torch.roll(pts, -1, dims=0)

    def cross(o, p, q):
        return ((p[:, 0] - o[:, 0]) * (q[:, 1] - o[:, 1])
                - (p[:, 1] - o[:, 1]) * (q[:, 0] - o[:, 0]))
    for i in range(n):
        hi = n if i > 0 else n - 1
        if i + 2 >= hi:
            continue
        js = torch.arange(i + 2, hi)
        o = a[i].expand(js.numel(), 2)
        p = b[i].expand(js.numel(), 2)
        d1 = cross(o, p, a[js])
        d2 = cross(o, p, b[js])
        d3 = cross(a[js], b[js], o)
        d4 = cross(a[js], b[js], p)
        hit = (d1 * d2 < 0) & (d3 * d4 < 0)
        if bool(hit.any()):
            return False
    return True


def compression_certificates(pos_src_out: _T, pos_comp: _T, z: _T,
                             eps_alg: float = EPS_ALG) -> dict:
    """Event-level certificates for the homothety (reviewer sec. 13):
    source/compressed polygons simple in the transported order,
    orientation sign preserved, cyclic (angular-about-z) order
    identical, polygon centroid preserved (algebraic for z = c_poly).
    Positive homothety preserves all of these in exact arithmetic --
    the checks guard bookkeeping errors, not geometry."""
    out = {}
    out["source_simple"] = polygon_is_simple(pos_src_out)
    out["compressed_simple"] = polygon_is_simple(pos_comp)
    sa_s = signed_shoelace(pos_src_out)
    sa_c = signed_shoelace(pos_comp)
    out["orientation_preserved"] = bool(sa_s * sa_c > 0)
    ds = pos_src_out - z
    dc = pos_comp - z
    ang_s = torch.atan2(ds[:, 1], ds[:, 0]).argsort()
    ang_c = torch.atan2(dc[:, 1], dc[:, 0]).argsort()
    out["cyclic_order_preserved"] = bool(torch.equal(ang_s, ang_c))
    c_comp = polygon_centroid(pos_comp)
    scale = float(pos_src_out.norm(dim=1).max())
    out["centroid_residual"] = float((c_comp - z).norm())
    out["centroid_preserved"] = out["centroid_residual"] \
        <= eps_alg * max(scale, 1.0) * 1e3
    out["all"] = all(out[k] for k in
                     ("source_simple", "compressed_simple",
                      "orientation_preserved", "cyclic_order_preserved",
                      "centroid_preserved"))
    return out


def transfer_map_residual(candidate_pos: _T, c_of_source_pos: _T,
                          h_wall: float) -> float:
    """Hard event gate: the committed ZERO-TIME candidate must equal
    the defined conservative map output (never compared against the
    one-step-evolved state -- that contains legitimate MM
    displacement)."""
    return float((candidate_pos - c_of_source_pos).abs().max()) / h_wall


# ---------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------

def conservative_compression(pos: _T, ang: _T, nor: _T,
                             mask_out: _T, mask_a: _T, mask_b: _T,
                             eps_alg: float = EPS_ALG) -> dict:
    """C(X): remove the ghost pair, scale the outer loop about its
    polygon centroid so the PRIMARY geometric merged volume is
    conserved to eps_alg. Angles are unchanged (a homothety preserves
    tangent directions). Returns positions/angles of the compressed
    cloud + full telemetry."""
    for name, m in (("mask_out", mask_out), ("mask_a", mask_a),
                    ("mask_b", mask_b)):
        if not bool(m.any()):
            raise GhostCompressionError(f"{name} is empty")
    union = mask_out | mask_a | mask_b
    if not bool(union.all()):
        raise GhostCompressionError(
            "pair + outer masks do not cover the cloud "
            f"({int(union.sum())} of {pos.shape[0]}) -- unknown loop")
    if bool((mask_out & mask_a).any()) or bool((mask_out & mask_b).any()) \
            or bool((mask_a & mask_b).any()):
        raise GhostCompressionError("masks overlap")
    V_sharp = merged_geom_volume(pos, nor, [mask_out, mask_a, mask_b])
    A_out = star_shoelace_area(pos[mask_out])
    if loop_orientation_sign(pos, nor, mask_out) != 1:
        raise GhostCompressionError("outer loop is not outward")
    z = polygon_centroid(pos[mask_out])
    root = solve_transfer_scale(V_sharp, A_out, eps_alg=eps_alg)
    c = root["c"]
    pos_out = pos[mask_out]
    pos_comp = z + c * (pos_out - z)
    V_comp = star_shoelace_area(pos_comp)
    if abs(V_comp - V_sharp) > eps_alg * max(V_sharp, 1.0) * 10.0:
        raise GhostCompressionError(
            f"compressed geometric volume {V_comp!r} != target "
            f"{V_sharp!r} beyond eps_alg")
    certs = compression_certificates(pos_out, pos_comp, z,
                                     eps_alg=eps_alg)
    if not certs["all"]:
        raise GhostCompressionError(
            f"compression certificates failed: "
            f"{ {k: v for k, v in certs.items() if k != 'all'} }")
    transfer_inf = float((pos_comp - pos_out).abs().max())
    return dict(
        positions=pos_comp, angles=ang[mask_out].clone(),
        scale=c, root=root, z=[float(z[0]), float(z[1])],
        V_sharp_geom=V_sharp, A_out=A_out, V_comp_geom=V_comp,
        V_pair_geom=V_sharp - A_out,
        transfer_inf=transfer_inf, certificates=certs)


# ---------------------------------------------------------------------
# representation distances (telemetry)
# ---------------------------------------------------------------------

def current_distance(pos_a: _T, nor_a: _T, w_a: _T,
                     pos_b: _T, nor_b: _T, w_b: _T,
                     sigma: float) -> float:
    """||psi_sigma*(T_a) - psi_sigma*(T_b)||^2 / ||psi_sigma*T_a||^2
    for currents with the supplied per-particle weights (w = m for
    D_T_raw; w = m q for a visible current -- the CALLER names which
    coherence was used)."""
    from ..transport.contact_certificate import mollified_current_gram
    pos = torch.cat([pos_a, pos_b])
    nor = torch.cat([nor_a, nor_b])
    w = torch.cat([w_a, w_b])
    ia = torch.arange(pos_a.shape[0])
    ib = torch.arange(pos_a.shape[0], pos.shape[0])
    e_a = mollified_current_gram(pos, nor, w, sigma, ia, ia)
    e_b = mollified_current_gram(pos, nor, w, sigma, ib, ib)
    e_ab = mollified_current_gram(pos, nor, w, sigma, ia, ib)
    if e_a <= 0.0:
        raise GhostCompressionError("reference current has no energy")
    return (e_a + e_b - 2.0 * e_ab) / e_a


def phase_field_distance(rho_ref: _T, rho_other: _T) -> dict:
    """D_rho,1 / D_rho,2: relative L1/L2 field distances on the SAME
    grid (both fields must come from FRESH setups with the same frozen
    target -- never from a stale per-stepper cache)."""
    if rho_ref.shape != rho_other.shape:
        raise GhostCompressionError("phase fields on different grids")
    d = (rho_other - rho_ref).abs()
    l1 = float(d.sum() / rho_ref.abs().sum())
    l2 = float(d.pow(2).sum().sqrt() / rho_ref.pow(2).sum().sqrt())
    return dict(D_rho_1=l1, D_rho_2=l2)
