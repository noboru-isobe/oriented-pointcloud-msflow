"""P1-1D: 1D lubrication solver for the Hele-Shaw neck model
(reviewer program section 5),

    h_t + (h h_xxx)_x = 0   on (-1, 1),
    h(+-1) = 1,  h_xx(+-1) = P            (CENV boundary conditions)

or the lifted-capsule variant with profile-derived boundary data.

Semi-implicit conservative scheme: mobility lagged,
    (I + dt D1 diag(h^n) D3) h^{n+1} = h^n + BC terms,
with ghost points enforcing the four boundary conditions. Known
structure used as pins (CENV, CMP 363:139):
  - steady states are exactly the parabolas h_ss = 1 + (P/2)(x^2-1),
    positive iff P < 2  ->  P = 1 must converge to h_ss (min 1/2);
  - P > 2  ->  inf h -> 0 in finite or infinite time (P = 4 pinches);
  - while h > 0 no singularity (smoothness) -- energy
    E = int h_x^2 /2? (their Lyapunov is int (h_xx - P)... we pin
    monotone decrease of the surface energy int h_x^2/2 dx + flux
    consistency instead of a specific functional).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def solve_neck(P: float, n: int = 401, dt: float = 1e-6,
               t_end: float = 0.05, h0=None, record_every: int = 50,
               pinch_floor: float = 1e-3):
    """Semi-implicit step with lagged mobility, pentadiagonal solve
    (scipy solve_banded; the dense version timed out). Returns
    dict(t, min_h, mass, energy, profiles, status)."""
    from scipy.linalg import solve_banded
    x = np.linspace(-1.0, 1.0, n)
    dx = x[1] - x[0]
    h = np.ones(n) if h0 is None else np.array(h0, dtype=float)
    assert h.shape == (n,)
    t = 0.0
    out = dict(t=[], min_h=[], mass=[], energy=[])
    profiles = []
    step = 0
    status = "end"
    coef0 = dt / dx / dx ** 3
    W = np.array([-1.0, 3.0, -3.0, 1.0])
    while t < t_end:
        hg_l = 2 * h[0] - h[1] + P * dx * dx
        hg_r = 2 * h[-1] - h[-2] + P * dx * dx
        hf = 0.5 * (h[:-1] + h[1:])               # faces (n-1,)
        cf = coef0 * hf                            # face coefficients
        # banded matrix ab[u + i - j, j] = M[i, j], (l, u) = (2, 2)
        ab = np.zeros((5, n))
        ab[2, :] = 1.0                             # identity diagonal
        rhs = h.copy()
        # face f (between nodes f, f+1), stencil nodes f-1..f+2 with
        # weights W: +into row f, -into row f+1
        f_idx = np.arange(n - 1)
        for k, wj in enumerate(W):
            j = f_idx + (k - 1)                    # column index
            contrib = cf * wj
            # row f
            valid = (j >= 0) & (j < n)
            rows, cols, cc = f_idx[valid], j[valid], contrib[valid]
            np.add.at(ab, (2 + rows - cols, cols), cc)
            gl = ~valid & (j < 0)
            gr = ~valid & (j >= n)
            rhs[f_idx[gl]] -= contrib[gl] * hg_l
            rhs[f_idx[gr]] -= contrib[gr] * hg_r
            # row f+1
            r2 = f_idx + 1
            valid2 = (j >= 0) & (j < n) & (r2 < n) \
                & (np.abs(r2 - j) <= 2)
            rows, cols, cc = r2[valid2], j[valid2], contrib[valid2]
            np.add.at(ab, (2 + rows - cols, cols), -cc)
            gl2 = (j < 0) & (r2 < n)
            gr2 = (j >= n) & (r2 < n)
            rhs[r2[gl2]] += contrib[gl2] * hg_l
            rhs[r2[gr2]] += contrib[gr2] * hg_r
        # Dirichlet rows 0 and n-1
        for r, v in ((0, 1.0), (n - 1, 1.0)):
            for off in (-2, -1, 0, 1, 2):
                j = r + off
                if 0 <= j < n:
                    ab[2 + r - j, j] = 1.0 if j == r else 0.0
            rhs[r] = v
        h_new = solve_banded((2, 2), ab, rhs)
        dh = float(np.abs(h_new - h).max())
        h = h_new
        t += dt
        step += 1
        if step % record_every == 0 or h.min() <= pinch_floor:
            out["t"].append(t)
            out["min_h"].append(float(h.min()))
            out["mass"].append(float(np.trapz(h, x)))
            out["energy"].append(float(np.trapz(
                np.gradient(h, dx) ** 2, x) / 2))
            if len(profiles) < 200:
                profiles.append(h.copy())
        if h.min() <= pinch_floor:
            status = "pinch_floor"
            break
        if dh < 1e-12 * (dt / 1e-6):
            status = "steady"
            break
        if not np.isfinite(h).all():
            status = "nan"
            break
    return dict(x=x, h=h, status=status, profiles=profiles, **out)


def classify_min_h(ts, ws, floor: float):
    """Fit finite-time-like w ~ C (T*-t)^alpha vs infinite-time-like
    w ~ C exp(-lam t) / C t^-beta on the tail; return dict of fits
    (report-only -- classification thresholds live with the
    reviewer)."""
    ts, ws = np.array(ts), np.array(ws)
    sel = ws > floor
    ts, ws = ts[sel], ws[sel]
    if len(ts) < 8:
        return dict(status="too_short")
    tail = slice(len(ts) // 2, None)
    # exponential fit
    ce = np.polyfit(ts[tail], np.log(ws[tail]), 1)
    resid_e = float(np.mean((np.log(ws[tail])
                             - np.polyval(ce, ts[tail])) ** 2))
    # finite-time fit: grid-search T* > t_max
    best = None
    for Tst in np.linspace(ts[-1] * 1.001, ts[-1] * 3.0, 60):
        cf = np.polyfit(np.log(Tst - ts[tail]), np.log(ws[tail]), 1)
        r = float(np.mean((np.log(ws[tail]) - np.polyval(
            cf, np.log(Tst - ts[tail]))) ** 2))
        if best is None or r < best[0]:
            best = (r, Tst, cf)
    return dict(exp_rate=float(-ce[0]), exp_resid=resid_e,
                ft_resid=best[0], ft_Tstar=float(best[1]),
                ft_alpha=float(best[2][0]))


def main():
    out = Path("results/pinchoff")
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    x = np.linspace(-1, 1, 401)
    # P = 1 (< 2): the exact steady state h_ss = 1 + (P/2)(x^2-1)
    # (a) stationarity: started ON h_ss, stays (scheme consistency)
    h_ss = 1 + 0.5 * (x ** 2 - 1)
    ra = solve_neck(P=1.0, t_end=5e-3, h0=h_ss)
    res["P1_stationary"] = dict(
        status=ra["status"],
        drift=float(np.abs(ra["h"] - h_ss).max()))
    print("P=1 stationary:", res["P1_stationary"], flush=True)
    # (b) attraction: perturbed start decays toward h_ss
    h_pert = h_ss + 0.10 * (1 - x ** 2) ** 2
    rb = solve_neck(P=1.0, t_end=0.4, dt=1e-5, h0=h_pert)
    e0 = float(np.abs(h_pert - h_ss).max())
    e1 = float(np.abs(rb["h"] - h_ss).max())
    res["P1_attracts"] = dict(status=rb["status"], err0=e0, err1=e1,
                              decayed=bool(e1 < 0.2 * e0))
    print("P=1 attraction:", res["P1_attracts"], flush=True)
    # P = 4 (> 2): finite-time-like pinch, T* refinement-stable
    r4 = solve_neck(P=4.0, t_end=0.5)
    res["P4"] = dict(status=r4["status"], min_h=float(r4["h"].min()),
                     t_reached=r4["t"][-1] if r4["t"] else None,
                     fits=classify_min_h(r4["t"], r4["min_h"], 1e-3))
    print("P=4:", res["P4"], flush=True)
    r4b = solve_neck(P=4.0, t_end=0.5, dt=5e-7, n=601)
    res["P4_ref"] = dict(status=r4b["status"],
                         fits=classify_min_h(r4b["t"], r4b["min_h"],
                                             1e-3))
    print("P=4 refined:", res["P4_ref"], flush=True)
    # P = 2.2 (just above threshold): thinning classification near
    # the CENV boundary (finite-or-infinite -- recorded, not pinned)
    r22 = solve_neck(P=2.2, t_end=3.0, dt=1e-5)
    res["P2_2"] = dict(status=r22["status"],
                       min_h=float(r22["h"].min()),
                       fits=classify_min_h(r22["t"], r22["min_h"],
                                           1e-3))
    print("P=2.2:", res["P2_2"], flush=True)
    (out / "lubrication1d_known_cases.json").write_text(
        json.dumps(res, indent=1, default=str))
    print("wrote", out / "lubrication1d_known_cases.json")


if __name__ == "__main__":
    main()
