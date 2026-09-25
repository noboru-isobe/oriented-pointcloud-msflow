"""L0J-1b: source-frozen shape stability + half-seed dynamic
linearity of the q^CC energy template (reviewer 2026-08-21 spec).

Shape subspace (pre-registered, reviewer sec. 6): 6 low-order tapered
normal-displacement modes per contact side (taper * {1, sin 2 pi s,
cos 2 pi s, sin 4 pi s, cos 4 pi s, sin 6 pi s} on the window arc
coordinate), restricted to the numerical null space of the SMOOTH
constraints
    D A_1 = D A_2 = 0,  D bar(E_1) = D bar(E_2) = 0,  D g_alpha = 0,
with the source-frozen smooth gap (eq. 6.1)
    g_alpha(Y) = sum_{(i,j) in Pi} w_ij (y_j - y_i) . nbar_ij / sum w
over the complex's mutual-NN pairs (weights and mean normals frozen).
The physical mean-gap-closing mode is EXCLUDED by D g = 0 and measured
separately as a positive control.

Hessian decomposition (finite differences at t = 1e-3 h; the W form is
an exact quadratic priced by the production grid metric):
    H_J = H_P^arm + H_W,   arms F / W / C / S share H_W.
Acceptance reading: lambda_min(H_J^CC | S_shape) against the clean
control (the same machinery on the step-500 pre-interaction state).

Half-seed dynamic linearity (768 only): seed = the restricted
lambda_min eigenmode of arm C; amplitudes {0, a/2, a}, a = 0.1 h;
10 production steps under contact_complex_renormalized; response
linearity and growth sign.

Usage:
    uv run python scripts/experiments/l0j_stability_audit.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import build_config  # noqa: E402
from l0j_locality_audit import _V, arms_q  # noqa: E402
from quotient_switch_stability import production_args  # noqa: E402
from two_ellipses_benchmark import SIGMA, resolve_m  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)

DT = torch.float64
KERNEL = "wendland_c2"


def window_modes(pos, sides):
    """12 tapered normal-amplitude modes (6 per side) as scalar fields
    over ALL particles (zero outside the window)."""
    n = pos.shape[0]
    modes = []
    for side in sides:
        idx = side.particles          # cyclic order within the side
        k = idx.numel()
        s = torch.linspace(0, 1, k, dtype=DT)
        taper = torch.sin(math.pi * s) ** 2
        for f in (torch.ones_like(s),
                  torch.sin(2 * math.pi * s),
                  torch.cos(2 * math.pi * s),
                  torch.sin(4 * math.pi * s),
                  torch.cos(4 * math.pi * s),
                  torch.sin(6 * math.pi * s)):
            phi = torch.zeros(n, dtype=DT)
            phi[idx] = taper * f
            modes.append(phi)
    return modes


def frozen_gap(pos0, nor0, m0, sides):
    """Source-frozen smooth gap functional (eq. 6.1): mutual pairs by
    NN from side a to side b, w = m_i, nbar = normalized (n_i - n_j)/2
    (facing normals are anti-aligned, so nbar points from a to b)."""
    sa, sb = sides
    D = torch.cdist(pos0[sa.particles], pos0[sb.particles])
    pi = D.argmin(dim=1)
    ii = sa.particles
    jj = sb.particles[pi]
    w = m0[ii].clone()
    nbar = nor0[ii] - nor0[jj]
    nbar = nbar / nbar.norm(dim=1, keepdim=True)

    def g(Y):
        return ((w * ((Y[jj] - Y[ii]) * nbar).sum(1)).sum() / w.sum())
    return g


def area_fn(sel):
    def f(Y):
        p = Y[sel]
        x, y = p[:, 0], p[:, 1]
        x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
        return 0.5 * (x * y2 - x2 * y).sum()
    return f


def bary_fn(sel, comp):
    def f(Y):
        p = Y[sel]
        x, y = p[:, 0], p[:, 1]
        x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
        cr = x * y2 - x2 * y
        a = 0.5 * cr.sum()
        if comp == 0:
            return ((x + x2) * cr).sum() / (6.0 * a)
        return ((y + y2) * cr).sum() / (6.0 * a)
    return f


def null_space_basis(pos, nor, modes, constraints, t=1e-6):
    """Constraint Jacobian on mode coefficients by central differences
    of the smooth functionals; SVD null space."""
    K = len(modes)
    C = torch.zeros(len(constraints), K, dtype=DT)
    for k, phi in enumerate(modes):
        d = nor * phi.unsqueeze(1)
        for c, fn in enumerate(constraints):
            C[c, k] = (fn(pos + t * d) - fn(pos - t * d)) / (2 * t)
    U, S, Vh = torch.linalg.svd(C)
    rank = int((S > S.max() * 1e-10).sum())
    N = Vh[rank:].T                     # K x (K - rank)
    return N, rank


def hessians(pos, nor, lbl, m0, qs, gm, dt_step, modes, N, t_h):
    """Restricted H_P per arm + shared H_W on the null-space basis."""
    d = N.shape[1]
    psi = [sum(float(N[k, j]) * modes[k] for k in range(len(modes)))
           for j in range(d)]

    def m_of(Y):
        nor_y = nor            # normal-displacement: angles unchanged
        return resolve_m(Y, nor_y, *_DT_CACHE)

    def P_arm(Y, q):
        return float((m_of(Y) * q).sum())

    H_P = {a: torch.zeros(d, d, dtype=DT) for a in qs}
    base = {a: P_arm(pos, qs[a]) for a in qs}
    evals = {}

    def pp(vec):
        key = tuple(round(float(x), 12) for x in vec)
        if key not in evals:
            Y = pos + sum(float(v) * t_h * (nor * psi[j].unsqueeze(1))
                          for j, v in enumerate(vec))
            evals[key] = {a: P_arm(Y, qs[a]) for a in qs}
        return evals[key]

    for i in range(d):
        ei = torch.zeros(d); ei[i] = 1.0
        for j in range(i, d):
            ej = torch.zeros(d); ej[j] = 1.0
            if i == j:
                pl, mi = pp(ei), pp(-ei)
                for a in qs:
                    H_P[a][i, i] = (pl[a] + mi[a] - 2 * base[a]) / t_h ** 2
            else:
                ppp, pmm = pp(ei + ej), pp(-ei - ej)
                ppm, pmp = pp(ei - ej), pp(-ei + ej)
                for a in qs:
                    H_P[a][i, j] = H_P[a][j, i] = (
                        ppp[a] + pmm[a] - ppm[a] - pmp[a]) / (4 * t_h ** 2)
    H_W = torch.zeros(d, d, dtype=DT)
    zer = torch.zeros(pos.shape[0], dtype=DT)
    Q = {}

    def qform(j1, j2, s1, s2):
        # projected admissible-space form (0L-B0 cell-C style): the
        # audit modes satisfy the POLYGON area constraints, whose
        # discrete grid-compatibility residual (~1e-5) would trip the
        # fail-closed forward; the metric prices the projected flux
        v = s1 * psi[j1] + s2 * psi[j2] if j2 is not None \
            else s1 * psi[j1]
        with torch.no_grad():
            drho = gm.flux(v, zer)
            b = gm.poisson.project_componentwise_zero_mean(drho)
            phi = gm.poisson.solve(b)[0]
            return (0.5 / dt_step) * float(gm.poisson.inner(b, phi))
    for i in range(d):
        Q[(i,)] = qform(i, None, 1.0, 0.0)
    for i in range(d):
        for j in range(i, d):
            if i == j:
                H_W[i, i] = 2.0 * Q[(i,)]
            else:
                qpp = qform(i, j, 1.0, 1.0)
                H_W[i, j] = H_W[j, i] = qpp - Q[(i,)] - Q[(j,)]
    return H_P, H_W


_DT_CACHE = None


def analyze_state(pos, ang, lbl, grid, label, do_gap_control=True,
                  sides_override=None):
    global _DT_CACHE
    from src.torch.solver.mm_step import MMStepper
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    delta, tau = map(float, compute_recommended_params(pos))
    _DT_CACHE = (delta, tau)
    m = resolve_m(pos, nor, delta, tau)
    qs, res = arms_q(pos, nor, m, lbl)
    if sides_override is not None:
        # clean control: the FINAL window's particle sets transported
        # (fixed N) to a pre-interaction state where no complex exists
        sides = sides_override
    else:
        sides = res["cx"]["complexes"][0].sides
    modes = window_modes(pos, sides)
    g_fn = frozen_gap(pos, nor, m, sides)
    m1 = lbl == 0
    constraints = [area_fn(m1), area_fn(~m1),
                   bary_fn(m1, 0), bary_fn(m1, 1),
                   bary_fn(~m1, 0), bary_fn(~m1, 1), g_fn]
    N, rank = null_space_basis(pos, nor, modes, constraints)
    # production grid metric on the source state
    cfg = build_config(production_args(
        grid=grid, redist_monotone=True,
        q_mode="contact_complex_renormalized"), delta, tau)
    st = MMStepper(cfg)
    st._setup_step(OrientedPointCloudVarifold(positions=pos.clone(),
                                              angles=ang.clone()))
    gm = st.grid_wasserstein
    h = float(m.median())
    H_P, H_W = hessians(pos, nor, lbl, m, qs, gm, cfg.time_step,
                        modes, N, t_h=1e-3 * h)
    out = dict(label=label, rank=int(rank), dim=int(N.shape[1]),
               h=h, sides=[(s.loop, s.n) for s in sides])
    for a in ("F", "W", "C", "S"):
        HJ = H_P[a] + H_W
        ev, V = torch.linalg.eigh(HJ)
        out[f"lam_min_J_{a}"] = float(ev[0])
        out[f"lam_min_P_{a}"] = float(
            torch.linalg.eigh(H_P[a]).eigenvalues[0])
    out["lam_min_W_metric"] = float(
        torch.linalg.eigh(H_W).eigenvalues[0])
    # gap-closing positive control (excluded from the subspace)
    if do_gap_control:
        gap_dir = torch.zeros(pos.shape[0], dtype=DT)
        for side in sides:
            k = side.particles.numel()
            s = torch.linspace(0, 1, k, dtype=DT)
            gap_dir[side.particles] = torch.sin(math.pi * s) ** 2
        t_h = 1e-3 * h
        zer = torch.zeros(pos.shape[0], dtype=DT)
        with torch.no_grad():
            drho = gm.flux(gap_dir, zer)
            b = gm.poisson.project_componentwise_zero_mean(drho)
            phi = gm.poisson.solve(b)[0]
            Wq = 2.0 * (0.5 / cfg.time_step) \
                * float(gm.poisson.inner(b, phi))

        def P_at(Y, q):
            return float((resolve_m(Y, nor, *_DT_CACHE) * q).sum())
        dvec = nor * gap_dir.unsqueeze(1)
        ctrl = {}
        for a in ("F", "W", "C"):
            Pp = P_at(pos + t_h * dvec, qs[a])
            Pm = P_at(pos - t_h * dvec, qs[a])
            P0 = float((m * qs[a]).sum())
            ctrl[a] = dict(H_P=(Pp + Pm - 2 * P0) / t_h ** 2, H_W=Wq)
        out["gap_closing_control"] = ctrl
    # eigenmode of arm C for the half-seed run
    HJ_C = H_P["C"] + H_W
    ev, V = torch.linalg.eigh(HJ_C)
    seed = sum(float((N @ V[:, 0])[k]) * modes[k]
               for k in range(len(modes)))
    out["_seed"] = seed
    out["_nor"] = nor
    return out


def half_seed(pos, ang, lbl, grid, seed_field, nor, h, n_steps=10):
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.mm_step import MMStepper
    delta, tau = map(float, compute_recommended_params(pos))
    cfg = build_config(production_args(
        grid=grid, redist_monotone=True,
        q_mode="contact_complex_renormalized"), delta, tau)
    a = 0.1 * h
    unit = seed_field / seed_field.norm()
    runs = {}
    for name, amp in (("zero", 0.0), ("half", a / 2), ("full", a)):
        p0 = pos + amp * (nor * unit.unsqueeze(1))
        v = OrientedPointCloudVarifold(positions=p0.clone(),
                                       angles=ang.clone())
        solver = MMSolver(cfg)
        st = MMStepper(cfg)
        traj = []
        for k in range(n_steps):
            res, v = solver._advance_committed(st, v)
            traj.append(v.positions.clone())
        runs[name] = traj
    amp_series = {}
    for name in ("half", "full"):
        amps = []
        for k in range(n_steps):
            dpos = runs[name][k] - runs["zero"][k]
            amps.append(float((dpos * (nor * unit.unsqueeze(1)))
                              .sum(1).mul(1.0).sum()))
        amp_series[name] = amps
    return amp_series


def main():
    torch.set_default_dtype(DT)
    rep = {}
    ck = torch.load("results/two_ellipses/two_ellipses_l0_768_states.pt",
                    weights_only=True)
    n = ck[max(ck)]["positions"].shape[0] // 2
    lbl = torch.zeros(2 * n, dtype=torch.long)
    lbl[n:] = 1
    # contact state (768 stop)
    st_c = ck[max(ck)]
    out_c = analyze_state(st_c["positions"], st_c["angles"], lbl, 768,
                          "contact_768")
    seed, nor = out_c.pop("_seed"), out_c.pop("_nor")
    rep["contact_768"] = out_c
    print(json.dumps({k: v for k, v in out_c.items()
                      if not k.startswith("_")}, indent=1,
                     default=str))
    # clean control: step 500 (pre-interaction), same particle windows
    # -- windows re-derived would be empty, so we reuse the machinery
    # only if a complex exists; otherwise measure H on the same modes
    # defined by the FINAL window particle indices at the 500 state
    try:
        # pre-interaction control: the t=0 cloud with the loops shifted
        # +-0.01 apart (gap 0.12 > sigma; the trajectory itself starts
        # at gap 0.1 - rounding = epsilon INSIDE the support, where a
        # fringe satellite pair already trips the |I| >= 2 guard)
        p0 = ck[0]["positions"].clone()
        p0[:n, 0] -= 0.01
        p0[n:, 0] += 0.01
        st_0 = {"positions": p0, "angles": ck[0]["angles"]}
        from src.torch.perimeter.contact_complex import (
            certify_loop_orders,
            sigma_contact_complex,
        )
        nor_c = torch.stack([st_c["angles"].cos(),
                             st_c["angles"].sin()], 1)
        delta_c, tau_c = map(float,
                             compute_recommended_params(
                                 st_c["positions"]))
        m_c = resolve_m(st_c["positions"], nor_c, delta_c, tau_c)
        orders_c = certify_loop_orders(st_c["positions"], lbl)
        cx_c = sigma_contact_complex(st_c["positions"], nor_c, m_c,
                                     lbl, orders_c, SIGMA, KERNEL)
        out_0 = analyze_state(
            st_0["positions"], st_0["angles"], lbl, 768, "clean_500",
            do_gap_control=False,
            sides_override=cx_c["complexes"][0].sides)
        out_0.pop("_seed", None)
        out_0.pop("_nor", None)
        rep["clean_500"] = out_0
        print(json.dumps(out_0, indent=1, default=str))
    except Exception as e:                    # noqa: BLE001
        rep["clean_500"] = f"not evaluable: {type(e).__name__}: {e}"
        print(rep["clean_500"])
    # half-seed dynamic linearity (arm C production)
    amps = half_seed(st_c["positions"], st_c["angles"], lbl, 768,
                     seed, nor, out_c["h"])
    rep["half_seed"] = amps
    for name, series in amps.items():
        print(name, ["%.3e" % x for x in series])
    ratio = [f / h_ for f, h_ in zip(amps["full"], amps["half"])
             if abs(h_) > 0]
    print("linearity ratios (expect ~2):",
          ["%.2f" % r for r in ratio[:6]])
    Path("results/reports/l0j_stability_audit.json").write_text(
        json.dumps(rep, indent=1, default=str))
    print("wrote results/reports/l0j_stability_audit.json")


if __name__ == "__main__":
    main()
