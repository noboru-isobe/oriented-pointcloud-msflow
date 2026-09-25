"""Phase I decisive experiments for the weak-UOT boundary metric.

Marginals are the coherence-VISIBLE measures alpha_i = m_i q_i (the
same measure the perimeter reads); the plan's conditional barycenters
define normal displacements; the cost is an H^{-1/2}-type quadratic
form on the visible cloud, plus KL marginal relaxation and an
augmented-Lagrangian first-order volume constraint:

    D^2 = inf_{pi>=0} <M,pi> + 1/2 r^T Qhat_X r + 1/2 c^T Qhat_Y c
          + zeta g + rho/2 g^2
          + lam KL(pi 1|alpha) + lam KL(pi^T 1|beta),
    r_i = sum_j pi_ij d_ij,  c_j = sum_i pi_ij d_ij,  g = sum_i r_i.

Kernels under test (reviewer-corrected protocol, all pins
pre-registered in the plan):
  * graph fractional (Ltilde + kappa^2)^{-1/2}: NEGATIVE control --
    on the annulus radial mode (constant per component = graph
    kernel) the analytic value is 6 pi / kappa, an artifact.
  * single-layer G_ik = -(1/2pi) log(|x_i-x_k|/ell0), diagonal
    -(1/2pi)(log(ell_i/ell0) - 3/2) with GEOMETRIC panel length
    ell_i; the inner objective uses the zero-charge projection
    Ghat = P G P (convex during AL, ell0 rank-one term annihilated).
    A' (annulus radial) analytic target: 2 pi log(R+/R-) -- the
    cross-component low-mode coupling the graph cannot have.
    A0 (EXACT circle translation Y = X + eps e1) true value
    pi a^2 R^2; raw SL predicts x1/2. A' passing is therefore a
    DIAGNOSIS (SL has the coupling), not adoption.

Usage:
    uv run python scripts/experiments/weak_uot_decisive.py --phase1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
)
from src.torch.perimeter.coherence_perimeter import (
    compute_recommended_sigma,
)
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
)
from src.torch.transport.bem_wasserstein import compute_coherence

DT = torch.float64
R_PLUS, R_MINUS = 1.0, 0.5
OUT = Path("results/reports")


# ------------------------------------------------------------ geometry

def annulus_pair(n_out: int, eps: float):
    """X = annulus (R+, R-); Y = EXACT-volume-preserving radial
    displacement: R+^e = R+ + eps, R-^e = sqrt(R-^2 + (R+^e)^2 - R+^2).
    Same n_inner passed explicitly (same index correspondence)."""
    X = generate_oriented_annulus(n_out, R_PLUS, R_MINUS,
                                  (0.0, 0.0), "cpu", DT)
    n_in = X.n_points - n_out
    rp_e = R_PLUS + eps
    rm_e = math.sqrt(R_MINUS ** 2 + rp_e ** 2 - R_PLUS ** 2)
    Y = generate_oriented_annulus(n_out, rp_e, rm_e, (0.0, 0.0),
                                  "cpu", DT, n_inner=n_in)
    geo = dict(
        n_out=n_out, n_in=n_in, rp_e=rp_e, rm_e=rm_e,
        d_rm=rm_e - R_MINUS,
        area_X=math.pi * (R_PLUS ** 2 - R_MINUS ** 2),
        area_Y=math.pi * (rp_e ** 2 - rm_e ** 2),
        # reviewer pin: linearized residual is O(eps^2), NOT a
        # geometry bug: g_geom = pi((dRm)^2 - eps^2)
        g_geom_analytic=math.pi * ((rm_e - R_MINUS) ** 2 - eps ** 2),
    )
    # exact per-point normal displacement (outer normal +rhat,
    # inner normal -rhat)
    u = np.empty(X.n_points)
    u[:n_out] = eps
    u[n_out:] = -(rm_e - R_MINUS)
    return X, Y, u, geo


def circle_pair(n: int, eps: float, a: float = 1.0, radius: float = 1.0):
    """A0: EXACT translation Y = X + eps a e1 (reviewer form).
    W2^2 = pi R^2 eps^2 a^2 exactly; boundary normal displacement
    u(theta) = a cos(theta) (per unit eps)."""
    X = generate_oriented_circle(n, radius, (0.0, 0.0), "cpu", DT)
    shift = torch.zeros_like(X.positions)
    shift[:, 0] = eps * a
    Y = OrientedPointCloudVarifold(positions=X.positions + shift,
                                   angles=X.angles.clone())
    u = eps * a * np.cos(X.angles.numpy())
    return X, Y, u


# ------------------------------------------------------------ weights

def weights_oracle_annulus(n_out, n_in, rp, rm):
    a = np.empty(n_out + n_in)
    a[:n_out] = 2.0 * math.pi * rp / n_out
    a[n_out:] = 2.0 * math.pi * rm / n_in
    return a


def weights_oracle_circle(n, radius):
    return np.full(n, 2.0 * math.pi * radius / n)


def weights_production(v, sigma=None):
    delta, tau = compute_recommended_params(v.positions)
    m = compute_masses(v.positions, delta, tau, "wendland_c2")
    if sigma is None:
        sigma = compute_recommended_sigma(v.positions)
    q = compute_coherence(v, m, sigma, "wendland_c2")
    meta = dict(delta=float(delta), tau=float(tau),
                sigma=float(sigma), q_min=float(q.min()),
                q_mean=float(q.mean()))
    return (m * q).numpy(), meta


def panel_lengths_annulus(n_out, n_in, rp, rm):
    ell = np.empty(n_out + n_in)
    ell[:n_out] = 2.0 * math.pi * rp / n_out
    ell[n_out:] = 2.0 * math.pi * rm / n_in
    return ell


# ------------------------------------------------------------ kernels

def graph_Q(pos, nrm, alpha, ell, kappa):
    """Q for r = alpha u: |u|^2 = (sqrt(a)u)^T (Lt+k^2)^{-1/2}
    (sqrt(a)u). Constant-per-component modes are in ker(Lt), so
    their cost is kappa^{-1} sum alpha u^2 -- the negative-control
    analytic value."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    t = np.clip(d / ell, 0.0, 1.0)
    eta = (1.0 - t) ** 4 * (4.0 * t + 1.0)
    omega = np.clip(nrm @ nrm.T, 0.0, 1.0)
    W = eta * omega * np.sqrt(np.outer(alpha, alpha))
    np.fill_diagonal(W, 0.0)
    L = np.diag(W.sum(1)) - W
    isa = 1.0 / np.sqrt(alpha)
    Lt = 0.5 * (L * np.outer(isa, isa)
                + (L * np.outer(isa, isa)).T)
    evals, evecs = np.linalg.eigh(Lt)
    frac = evecs @ np.diag((np.clip(evals, 0.0, None)
                            + kappa ** 2) ** -0.5) @ evecs.T
    Q = frac * np.outer(isa, isa)
    return 0.5 * (Q + Q.T)


def single_layer_G(pos, panel_len, ell0):
    """G_ik = -(1/2pi) log(|x_i-x_k|/ell0); diagonal =
    -(1/2pi)(log(ell_i/ell0) - 3/2) -- SAME kernel with ell0 on
    both (reviewer mandatory fix 1)."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    G = -np.log(d / ell0) / (2.0 * math.pi)
    np.fill_diagonal(G, -(np.log(panel_len / ell0) - 1.5)
                     / (2.0 * math.pi))
    return 0.5 * (G + G.T)


def zero_charge_projection(G):
    """Ghat = P G P, P = I - (1/N) 1 1^T (reviewer mandatory fix 2:
    keeps the AL inner problem convex; the ell0 rank-one term is
    annihilated; on g = 0 the physical energy is unchanged)."""
    n = G.shape[0]
    P = np.eye(n) - np.full((n, n), 1.0 / n)
    Gh = P @ G @ P
    return 0.5 * (Gh + Gh.T)


def zero_charge_min_eig(G):
    """lambda_min(Z^T G Z) with Z an orthonormal basis of 1^perp
    (P G P keeps a structural zero eigenvalue, so use Z)."""
    n = G.shape[0]
    ones = np.ones((n, 1)) / math.sqrt(n)
    Z = np.linalg.qr(np.eye(n) - ones @ ones.T)[0][:, : n - 1]
    ev = np.linalg.eigvalsh(Z.T @ G @ Z)
    norm = float(np.abs(np.linalg.eigvalsh(G)).max())
    return float(ev.min()), norm


# ------------------------------------------------------------ weak UOT

def kl_div(r, a):
    r = np.clip(r, 1e-300, None)
    return float((r * np.log(r / a) - r + a).sum())


def solve_uot(alpha, beta, D, admissible, Q_X, Q_Y, lam,
              M_tie=None, big=1e3, rho=1e4, n_al=12,
              al_tol=1e-8, num_iter=20000):
    """Augmented-Lagrangian outer loop around POT's
    lbfgsb_unbalanced with the custom quadratic weak regularizer."""
    import ot

    M = np.where(admissible, 0.0 if M_tie is None else M_tie, big)
    zeta = 0.0
    plan = (np.outer(alpha, beta) / beta.sum()) * admissible
    hist = []
    for it in range(n_al):
        def f(G):
            r = (G * D).sum(1)
            c = (G * D).sum(0)
            g = r.sum()
            return (0.5 * r @ (Q_X @ r) + 0.5 * c @ (Q_Y @ c)
                    + zeta * g + 0.5 * rho * g * g)

        def df(G):
            r = (G * D).sum(1)
            c = (G * D).sum(0)
            g = r.sum()
            return D * ((Q_X @ r)[:, None] + (Q_Y @ c)[None, :]
                        + zeta + rho * g)

        plan, log = ot.unbalanced.lbfgsb_unbalanced(
            alpha, beta, M, reg=1.0, reg_m=(lam, lam),
            reg_div=(f, df), regm_div="kl", G0=plan,
            numItermax=num_iter, stopThr=1e-16, log=True)
        r = (plan * D).sum(1)
        g = float(r.sum())
        denom = float((np.abs(plan * D)).sum()) + 1e-300
        hist.append(dict(it=it, g=g, g_rel=g / denom,
                         success=bool(log["res"].success),
                         nit=int(log["res"].nit)))
        if abs(g) / denom < al_tol:
            break
        zeta += rho * g

    r = (plan * D).sum(1)
    c = (plan * D).sum(0)
    g = float(r.sum())
    decomp = dict(
        J_kin_X=float(0.5 * r @ (Q_X @ r)),
        J_kin_Y=float(0.5 * c @ (Q_Y @ c)),
        J_linear=float((M * plan).sum()),
        J_KL_X=lam * kl_div(plan.sum(1), alpha),
        J_KL_Y=lam * kl_div(plan.sum(0), beta),
        J_volume=float(zeta * g + 0.5 * rho * g * g),
        g_over_pid=abs(g) / (float(np.abs(plan * D).sum()) + 1e-300),
        transported_mass=float(plan.sum()),
        marginal_err_X=float(np.abs(plan.sum(1) - alpha).sum()
                             / alpha.sum()),
        marginal_err_Y=float(np.abs(plan.sum(0) - beta).sum()
                             / beta.sum()),
        forbidden_mass=float((plan * (~admissible)).sum()),
        al_iters=len(hist),
        al_hist=hist,
    )
    decomp["J_kin"] = decomp["J_kin_X"] + decomp["J_kin_Y"]
    decomp["J_total"] = (decomp["J_kin"] + decomp["J_linear"]
                         + decomp["J_KL_X"] + decomp["J_KL_Y"]
                         + decomp["J_volume"])
    return plan, decomp


def direct_value(u, alpha_X, alpha_Y, Q_X, Q_Y):
    """No-OT reference: plug the known displacement into both
    quadratic forms (same-index correspondence)."""
    rX = alpha_X * u
    rY = alpha_Y * u
    return float(0.5 * rX @ (Q_X @ rX) + 0.5 * rY @ (Q_Y @ rY))


# --------------------------------------------------------- Phase I-0

def phase_i0(report):
    ok = True
    eps, n_out = 1e-2, 64
    X, Y, u, geo = annulus_pair(n_out, eps)
    n = X.n_points
    aX = weights_oracle_annulus(n_out, geo["n_in"], R_PLUS, R_MINUS)
    aY = weights_oracle_annulus(n_out, geo["n_in"], geo["rp_e"],
                                geo["rm_e"])
    pX = X.positions.numpy()
    nX = X.normals.numpy()
    ellX = panel_lengths_annulus(n_out, geo["n_in"], R_PLUS, R_MINUS)

    # pin 1: exact volume + the three residuals kept apart
    vol_gap = geo["area_Y"] - geo["area_X"]
    g_geom = float((aX * u).sum())
    rec = dict(area_gap=vol_gap, g_geom_linearized=g_geom,
               g_geom_analytic=geo["g_geom_analytic"])
    rec["pass"] = bool(abs(vol_gap) < 1e-12
                       and abs(g_geom - geo["g_geom_analytic"])
                       < 1e-8)
    ok &= rec["pass"]
    report["pin1_exact_volume"] = rec

    # pin 3: SL zero-charge spectrum (Z basis)
    G = single_layer_G(pX, ellX, 1.0)
    lam_min, gnorm = zero_charge_min_eig(G)
    rec = dict(lambda_min=lam_min, G_norm=gnorm,
               ratio=lam_min / gnorm)
    rec["pass"] = bool(lam_min >= -1e-10 * gnorm)
    ok &= rec["pass"]
    report["pin3_zero_charge_spectrum"] = rec

    # pin 4: ell0 invariance on a zero-charge vector
    r0 = aX * u
    r0 = r0 - r0.mean()                      # strict zero charge
    vals = {}
    for ell0 in (0.5, 1.0, 2.0):
        Gl = single_layer_G(pX, ellX, ell0)
        vals[str(ell0)] = float(r0 @ (Gl @ r0))
    v = np.array(list(vals.values()))
    rec = dict(energies=vals,
               rel_spread=float((v.max() - v.min())
                                / max(abs(v.mean()), 1e-300)))
    rec["pass"] = bool(rec["rel_spread"] < 1e-10)
    ok &= rec["pass"]
    report["pin4_ell0_invariance"] = rec

    # pin 5: graph constant mode = 6 pi / kappa
    rec = {}
    for kappa in (0.1, 0.01):
        Q = graph_Q(pX, nX, aX, 0.12, kappa)
        val = float((aX * u) @ (Q @ (aX * u))) / eps ** 2
        target = 6.0 * math.pi / kappa
        rec[f"kappa={kappa:g}"] = dict(
            value_over_eps2=val, target=target,
            rel_err=abs(val - target) / target)
    rec["pass"] = bool(all(x["rel_err"] < 2e-2
                           for x in rec.values()
                           if isinstance(x, dict)))
    ok &= rec["pass"]
    report["pin5_graph_constant_mode"] = rec

    # pin 2: central-difference gradient check of the FULL (f, df)
    # (SL kernel, projected, AL terms with fixed zeta, rho)
    Gh = zero_charge_projection(G)
    aY_arr = aY
    diff = (Y.positions.numpy()[None, :, :] - pX[:, None, :])
    dist = np.linalg.norm(diff, axis=2)
    nY = Y.normals.numpy()
    nbar = nX[:, None, :] + nY[None, :, :]
    nbar /= np.maximum(np.linalg.norm(nbar, axis=2, keepdims=True),
                       1e-300)
    D = (diff * nbar).sum(2)
    admissible = (dist <= 0.1) & ((nX @ nY.T) >= 0.5)
    GY = single_layer_G(Y.positions.numpy(),
                        panel_lengths_annulus(
                            n_out, geo["n_in"], geo["rp_e"],
                            geo["rm_e"]), 1.0)
    GhY = zero_charge_projection(GY)
    zeta, rho = 0.3, 10.0

    def f_full(P):
        r = (P * D).sum(1)
        c = (P * D).sum(0)
        g = r.sum()
        return (0.5 * r @ (Gh @ r) + 0.5 * c @ (GhY @ c)
                + zeta * g + 0.5 * rho * g * g)

    def df_full(P):
        r = (P * D).sum(1)
        c = (P * D).sum(0)
        g = r.sum()
        return D * ((Gh @ r)[:, None] + (GhY @ c)[None, :]
                    + zeta + rho * g)

    rng = np.random.default_rng(0)
    P0 = 0.5 * (np.outer(aX, aY_arr) / aY_arr.sum()) * admissible
    H = rng.standard_normal(P0.shape) * P0        # zero off-support
    errs = {}
    for t in (1e-4, 1e-5, 1e-6):
        num = (f_full(P0 + t * H) - f_full(P0 - t * H)) / (2 * t)
        ana = float((df_full(P0) * H).sum())
        errs[f"t={t:g}"] = abs(num - ana) / max(1.0, abs(num),
                                                abs(ana))
    e = list(errs.values())
    # pass = agreement at machine precision (errors then GROW as t
    # shrinks -- fp cancellation, the signature of an exact analytic
    # gradient) OR clean O(t^2) decay above the fp floor
    rec = dict(errors=errs,
               machine_precision=bool(min(e) < 1e-10),
               order2=bool(e[0] < 1e-6 and e[1] <= e[0]))
    rec["pass"] = rec["machine_precision"] or rec["order2"]
    ok &= rec["pass"]
    report["pin2_gradient_check"] = rec

    report["phase_i0_pass"] = bool(ok)
    return ok


# --------------------------------------------------------- Phase I-1

def phase_i1(report):
    eps = 1e-2
    rows = []
    for n_out in (128, 256):
        X, Y, u, geo = annulus_pair(n_out, eps)
        aX = weights_oracle_annulus(n_out, geo["n_in"], R_PLUS,
                                    R_MINUS)
        aY = weights_oracle_annulus(n_out, geo["n_in"], geo["rp_e"],
                                    geo["rm_e"])
        pX, nX = X.positions.numpy(), X.normals.numpy()
        pY = Y.positions.numpy()
        ellX = panel_lengths_annulus(n_out, geo["n_in"], R_PLUS,
                                     R_MINUS)
        ellY = panel_lengths_annulus(n_out, geo["n_in"], geo["rp_e"],
                                     geo["rm_e"])
        target = 2.0 * math.pi * math.log(R_PLUS / R_MINUS)
        for kappa in (0.1, 0.01):
            QX = graph_Q(pX, nX, aX, 0.12, kappa)
            QY = graph_Q(pY, Y.normals.numpy(), aY, 0.12, kappa)
            val = direct_value(u, aX, aY, QX, QY) / eps ** 2
            rows.append(dict(test="Aprime", kernel=f"graph k={kappa:g}",
                             n_out=n_out, value=val,
                             target=6.0 * math.pi / kappa,
                             physical_target=target))
        GX = zero_charge_projection(single_layer_G(pX, ellX, 1.0))
        GY = zero_charge_projection(single_layer_G(pY, ellY, 1.0))
        val = direct_value(u, aX, aY, GX, GY) / eps ** 2
        rows.append(dict(test="Aprime", kernel="single_layer",
                         n_out=n_out, value=val, target=target))

        # A0: exact circle translation
        Xc, Yc, uc = circle_pair(n_out, eps)
        ac = weights_oracle_circle(n_out, 1.0)
        ellc = np.full(n_out, 2.0 * math.pi / n_out)
        Gc = zero_charge_projection(
            single_layer_G(Xc.positions.numpy(), ellc, 1.0))
        GcY = zero_charge_projection(
            single_layer_G(Yc.positions.numpy(), ellc, 1.0))
        val = direct_value(uc / eps, ac, ac, Gc, GcY)
        rows.append(dict(test="A0", kernel="single_layer",
                         n_out=n_out, value=val,
                         target_true=math.pi,
                         target_raw_sl=0.5 * math.pi))
    report["phase_i1_direct"] = rows
    for r in rows:
        print(f"I-1 {r['test']:7s} {r['kernel']:16s} N={r['n_out']:4d}"
              f"  value={r['value']:10.4f}  "
              + " ".join(f"{k}={v:.4f}" for k, v in r.items()
                         if k.startswith("target")), flush=True)
    return rows


# --------------------------------------------------------- Phase I-2

def build_config(n_out, eps, weights):
    X, Y, u, geo = annulus_pair(n_out, eps)
    if weights == "oracle":
        aX = weights_oracle_annulus(n_out, geo["n_in"], R_PLUS,
                                    R_MINUS)
        aY = weights_oracle_annulus(n_out, geo["n_in"], geo["rp_e"],
                                    geo["rm_e"])
        wmeta = {}
    else:
        aX, mX = weights_production(X)
        aY, mY = weights_production(Y)
        wmeta = dict(X=mX, Y=mY)
    pX, nX = X.positions.numpy(), X.normals.numpy()
    pY, nY = Y.positions.numpy(), Y.normals.numpy()
    diff = pY[None, :, :] - pX[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    ndot = nX @ nY.T
    nbar = nX[:, None, :] + nY[None, :, :]
    nbar /= np.maximum(np.linalg.norm(nbar, axis=2, keepdims=True),
                       1e-300)
    D = (diff * nbar).sum(2)
    tang = diff - D[:, :, None] * nbar
    tie_base = (np.linalg.norm(tang, axis=2) ** 2
                + (0.05 ** 2) * (1.0 - ndot))
    ellX = panel_lengths_annulus(n_out, geo["n_in"], R_PLUS, R_MINUS)
    ellY = panel_lengths_annulus(n_out, geo["n_in"], geo["rp_e"],
                                 geo["rm_e"])
    GX = zero_charge_projection(single_layer_G(pX, ellX, 1.0))
    GY = zero_charge_projection(single_layer_G(pY, ellY, 1.0))
    same_index = np.eye(X.n_points, dtype=bool)
    local = (dist <= 0.1) & (ndot >= 0.5)
    return dict(X=X, Y=Y, u=u, geo=geo, aX=aX, aY=aY, D=D,
                tie=tie_base, GX=GX, GY=GY, same=same_index,
                local=local, wmeta=wmeta)


def phase_i2(report):
    target = 2.0 * math.pi * math.log(R_PLUS / R_MINUS)
    rows = []

    def run(cfg_name, c, admissible, M_tie, eps, lam, kernel_QX,
            kernel_QY, kernel_name):
        plan, dec = solve_uot(c["aX"], c["aY"], c["D"], admissible,
                              kernel_QX, kernel_QY, lam,
                              M_tie=M_tie)
        row = dict(config=cfg_name, kernel=kernel_name, eps=eps,
                   lam=lam, n=c["X"].n_points,
                   J_kin_over_eps2=dec["J_kin"] / eps ** 2,
                   target=target,
                   direct_over_eps2=direct_value(
                       c["u"], c["aX"], c["aY"], kernel_QX,
                       kernel_QY) / eps ** 2,
                   **{k: v for k, v in dec.items() if k != "al_hist"},
                   al_hist=dec["al_hist"][-1])
        rows.append(row)
        print(f"I-2 {cfg_name:34s} J_kin/e2={row['J_kin_over_eps2']:9.4f}"
              f" direct={row['direct_over_eps2']:9.4f}"
              f" (target {target:.4f}) g_rel={dec['g_over_pid']:.1e}"
              f" margX={dec['marginal_err_X']:.1e}", flush=True)

    # main sweep: SL kernel, same-index edges, oracle weights
    for n_out in (128, 256):
        for eps in (1e-2, 5e-3, 2.5e-3):
            c = build_config(n_out, eps, "oracle")
            run(f"SL/same/oracle N={n_out} e={eps:g}", c, c["same"],
                None, eps, eps, c["GX"], c["GY"], "single_layer")

    # base config variants
    n_out, eps = 128, 1e-2
    c = build_config(n_out, eps, "oracle")
    # local edges + tie-breaker sweep
    for eta in (0.0, 1e-8, 1e-6):
        run(f"SL/local/oracle eta={eta:g}", c, c["local"],
            eta * c["tie"] if eta > 0 else None, eps, eps,
            c["GX"], c["GY"], "single_layer")
    # production weights
    cp = build_config(n_out, eps, "production")
    run("SL/same/production", cp, cp["same"], None, eps, eps,
        cp["GX"], cp["GY"], "single_layer")
    report["production_weight_meta"] = cp["wmeta"]
    # graph negative control
    QX = graph_Q(c["X"].positions.numpy(), c["X"].normals.numpy(),
                 c["aX"], 0.12, 0.1)
    QY = graph_Q(c["Y"].positions.numpy(), c["Y"].normals.numpy(),
                 c["aY"], 0.12, 0.1)
    run("graph(k=0.1)/same/oracle", c, c["same"], None, eps, eps,
        QX, QY, "graph k=0.1")

    report["phase_i2_uot"] = rows
    report["reduction_note"] = (
        "full (N x eps) sweep only for SL/same-index/oracle; "
        "local-edge, production-weight and graph-control variants "
        "at the base config (N=128, eps=1e-2) -- recorded, not silent")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", action="store_true")
    ap.add_argument("--phase2", action="store_true")
    ap.add_argument("--phase3a", action="store_true")
    ap.add_argument("--phase3b", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase1:
        report = {}
        ok = phase_i0(report)
        print("Phase I-0 pins:",
              {k: v.get("pass") for k, v in report.items()
               if isinstance(v, dict) and "pass" in v}, flush=True)
        if not ok:
            print("PHASE I-0 FAILED -- aborting before I-1/I-2")
        else:
            phase_i1(report)
            phase_i2(report)
        out = OUT / "phase1.json"
        out.write_text(json.dumps(report, indent=1, default=float))
        print(f"wrote {out}")
    elif args.phase2:
        report = {}
        phase_ii(report)
        out = OUT / "phase2.json"
        out.write_text(json.dumps(report, indent=1, default=float))
        print(f"wrote {out}")
    elif args.phase3a:
        phase_iiia()
    elif args.phase3b:
        phase_iiib()
    else:
        ap.print_help()




# ================================================================
# Phase II: discretization-aggressive spectral-equivalence tiers
# (reviewer 3rd comment). Tiers priced as effective penetration
# depths ell(k) against the exact per-mode depth R/k:
#   1-band:  ell* = sqrt(ell_min ell_max)      (plain linear UOT)
#   band:    wbar_b = 1/sqrt(2 Lambda_b) on dyadic GEV bands
#   lowrank: ell_min + (lambda_r^{-1/2} - ell_min)_+ on r <= K0
#   hybrid (II-2b): per-mode SL Rayleigh weights x (1 for modes
#     carrying per-component charge, 2 for component-neutral modes)
#     -- promotes the measured Phase-I x1/x1/2 split to a rule.
# High-frequency floor everywhere: w = max(lambda^{-1/2}, c_sig*sigma).
# ================================================================


def graph_laplacian(pos, nrm, alpha, ell):
    """Normalized graph Laplacian: L phi = lam M phi approximates
    -Delta_boundary with unit coefficient (global second-moment
    calibration gamma so that on a uniform circle lam_k ~ k^2/R^2)."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    t = np.clip(d / ell, 0.0, 1.0)
    eta = (1.0 - t) ** 4 * (4.0 * t + 1.0)
    omega = np.clip(nrm @ nrm.T, 0.0, 1.0)
    W = eta * omega * np.sqrt(np.outer(alpha, alpha))
    np.fill_diagonal(W, 0.0)
    gamma = (W * d ** 2).sum(1) / (2.0 * alpha)
    W = W / gamma.mean()
    L = np.diag(W.sum(1)) - W
    return 0.5 * (L + L.T), float(gamma.mean()), float(gamma.std())


def gev_modes(L, alpha):
    """Generalized eigenproblem L phi = lam diag(alpha) phi with
    M-orthonormal eigenvectors."""
    from scipy.linalg import eigh
    lam, phi = eigh(L, np.diag(alpha))
    return np.clip(lam, 0.0, None), phi


def graph_components(pos, nrm, alpha, ell):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    adj = (d < ell) & ((nrm @ nrm.T) > 0.0)
    n = len(alpha)
    labels = -np.ones(n, dtype=int)
    c = 0
    for s in range(n):
        if labels[s] >= 0:
            continue
        stack = [s]
        labels[s] = c
        while stack:
            i = stack.pop()
            for j in np.nonzero(adj[i])[0]:
                if labels[j] < 0:
                    labels[j] = c
                    stack.append(j)
        c += 1
    return labels, c


def tier_weights(lam, sigma, c_sig=1.0, ell_max_cap=None):
    """Per-mode penetration depths for the band and exact-fractional
    references, with the high-frequency floor and a zero-mode cap."""
    floor = c_sig * sigma
    with np.errstate(divide="ignore"):
        w_frac = np.where(lam > 0, lam ** -0.5, np.inf)
    if ell_max_cap is None:
        pos = lam[lam > 1e-12]
        ell_max_cap = float(pos.min() ** -0.5) if pos.size else 1.0
    w_frac = np.minimum(w_frac, ell_max_cap)
    w_frac = np.maximum(w_frac, floor)
    # dyadic bands [Lam, 4 Lam): geometric-mean weight 1/sqrt(2 Lam)
    w_band = np.empty_like(w_frac)
    lam_pos = np.where(lam > 1e-12, lam, np.nan)
    with np.errstate(invalid="ignore"):
        b = np.floor(np.log(lam_pos) / np.log(4.0))
    lam_b = 4.0 ** b
    w_band = np.where(np.isnan(lam_b), ell_max_cap,
                      1.0 / np.sqrt(2.0 * lam_b))
    w_band = np.clip(w_band, floor, ell_max_cap)
    return w_frac, w_band, floor, ell_max_cap


def modal_Q(phi, alpha, weights):
    """Quadratic form on r = alpha*u: sum_r w_r (phi_r^T r)^2."""
    return (phi * weights[None, :]) @ phi.T


def hybrid_Q(phi, lam, alpha, Ghat, labels, ncomp, floor, ell_max):
    """II-2b hybrid, block-corrected: exact SL = a^T S a with
    S_rs = (M phi_r)^T Ghat (M phi_s). The cross-component coupling
    lives in the OFF-DIAGONAL of the zero-mode block (component
    constants), so that block keeps the FULL S; component-neutral
    modes keep the diagonal x2 rule (Phase-I measured split);
    everything clipped to [floor, ell_max] spectrally."""
    Mphi = phi * alpha[:, None]
    zero = lam < 1e-10 * max(lam.max(), 1.0)
    S = np.zeros((len(lam), len(lam)))
    # full SL block on zero modes (IIIa-0: upper cap ell_max added --
    # docstring/implementation mismatch flagged by the reviewer)
    if zero.any():
        Mz = Mphi[:, zero]
        B = Mz.T @ Ghat @ Mz
        ev, evec = np.linalg.eigh(0.5 * (B + B.T))
        ev = np.clip(ev, floor, ell_max)
        S[np.ix_(zero, zero)] = evec @ np.diag(ev) @ evec.T
    # diagonal x2/x1 on the rest
    rest = ~zero
    if rest.any():
        w_sl = np.einsum("ir,ij,jr->r", Mphi[:, rest], Ghat,
                         Mphi[:, rest])
        charge = np.zeros((ncomp, int(rest.sum())))
        for c in range(ncomp):
            charge[c] = Mphi[:, rest][labels == c].sum(0)
        cf = (np.abs(charge).max(0)
              / np.maximum(np.abs(Mphi[:, rest]).sum(0), 1e-300))
        factor = np.where(cf > 0.1, 1.0, 2.0)
        S[np.ix_(rest, rest)] = np.diag(
            np.clip(factor * w_sl, floor, ell_max))
    return phi @ S @ phi.T


def circle_mode_pair(n, k, eps, radius=1.0):
    X = generate_oriented_circle(n, radius, (0.0, 0.0), "cpu", DT)
    th = X.angles.numpy()
    u = eps * np.cos(k * th)
    pos = X.positions.numpy() * (1.0 + (u / radius))[:, None]
    Y = OrientedPointCloudVarifold(
        positions=torch.tensor(pos, dtype=DT),
        angles=X.angles.clone())
    return X, Y, u


def phase_ii(report):
    from scipy.linalg import eigh  # noqa: F401 (import check)

    sigma, c_sig = 0.1, 1.0
    R = 1.0
    n = 256
    eps = 1e-2
    out = {}

    # ---------------- II-0: GEV pins on the circle
    X = generate_oriented_circle(n, R, (0.0, 0.0), "cpu", DT)
    aX = weights_oracle_circle(n, R)
    pX, nX = X.positions.numpy(), X.normals.numpy()
    L, gmean, gstd = graph_laplacian(pX, nX, aX, 0.12)
    lam, phi = gev_modes(L, aX)
    # circle: lam_{2j-1},lam_{2j} ~ j^2/R^2 (cos/sin pairs)
    ks = np.arange(1, 9)
    lam_k = lam[1:17:2]
    ratios = lam_k / (ks ** 2 / R ** 2)
    out["ii0"] = dict(
        gamma_mean=gmean, gamma_rel_std=gstd / gmean,
        lam_over_k2=[float(x) for x in ratios],
        pin_lambda_k=bool(np.all(np.abs(ratios - 1.0) < 0.05)),
        pin_ratio_free=bool(np.all(
            np.abs(lam_k / lam_k[0] / ks ** 2 - 1.0) < 0.05)),
    )
    pos_lam = lam[lam > 1e-12]
    ell_max = float(pos_lam.min() ** -0.5)
    ell_star = math.sqrt(c_sig * sigma * ell_max)
    k_eff = int((np.sqrt(pos_lam) ** -1 >= sigma).sum() // 2)
    out["ii0"].update(ell_max=ell_max, ell_star=ell_star,
                      K_eff=k_eff)
    print(f"II-0 gamma_rel_std={gstd / gmean:.3f} "
          f"lam/k2={np.round(ratios, 3)} ell_max={ell_max:.3f} "
          f"ell*={ell_star:.3f} K_eff={k_eff}", flush=True)

    # ---------------- II-1: circle modes k=1..16, direct per tier
    w_frac, w_band, floor, _ = tier_weights(lam, sigma, c_sig,
                                            ell_max)
    Q_band = modal_Q(phi, aX, w_band)
    Q_frac = modal_Q(phi, aX, w_frac)
    GX = zero_charge_projection(
        single_layer_G(pX, np.full(n, 2 * math.pi * R / n), 1.0))
    labels, ncomp = graph_components(pX, nX, aX, 0.12)
    Q_hyb = hybrid_Q(phi, lam, aX, GX, labels, ncomp, floor,
                     ell_max)

    rows = []
    for k in range(1, 17):
        _, Yk, u = circle_mode_pair(n, k, eps)
        r = aX * u
        q_ref = math.pi * R ** 2 / k
        q_dpt = math.pi * R      # Q_partial/eps^2 (per unit a=1)
        vals = dict(
            ref=q_ref,
            one_band=ell_star * q_dpt,
            band=float(r @ (Q_band @ r)) / eps ** 2,
            frac=float(r @ (Q_frac @ r)) / eps ** 2,
            lowrank=max(min(lam[2 * k - 1] ** -0.5
                            if lam[2 * k - 1] > 0 else ell_max,
                            ell_max), floor) * q_dpt
            if k <= 5 else floor * q_dpt,
            hybrid=float(r @ (Q_hyb @ r)) / eps ** 2,
        )
        row = dict(k=k, **vals,
                   log_ratio={t: math.log(v / q_ref)
                              for t, v in vals.items()
                              if t != "ref"})
        rows.append(row)
    out["ii1_circle_modes"] = rows
    k_lim = min(k_eff, 16)
    maxlog = {t: max(abs(r["log_ratio"][t]) for r in rows
                     if r["k"] <= k_lim)
              for t in ("one_band", "band", "frac", "lowrank",
                        "hybrid")}
    out["ii1_maxlog_upto_Keff"] = maxlog
    print("II-1 max|log Q/Qref| (k<=K_eff=%d): %s" % (
        k_lim, {t: round(v, 3) for t, v in maxlog.items()}),
        flush=True)

    # UOT spot checks (plan reconstruction) for k in {1, 4}
    spot = []
    for k in (1, 4):
        Xk, Yk, u = circle_mode_pair(n, k, eps)
        pY = Yk.positions.numpy()
        diff = pY[None, :, :] - pX[:, None, :]
        dist = np.linalg.norm(diff, axis=2)
        nY = Yk.normals.numpy()
        nbar = nX[:, None, :] + nY[None, :, :]
        nbar /= np.maximum(np.linalg.norm(nbar, axis=2,
                                          keepdims=True), 1e-300)
        Dm = (diff * nbar).sum(2)
        adm = np.eye(n, dtype=bool)
        # 1-band: linear cost, zero quadratic
        Z = np.zeros((n, n))
        _, dec1 = solve_uot(aX, aX, Dm, adm, Z, Z, eps,
                            M_tie=ell_star * Dm ** 2)
        # hybrid: quadratic
        _, dech = solve_uot(aX, aX, Dm, adm, Q_hyb, Q_hyb, eps)
        spot.append(dict(k=k,
                         one_band_uot=dec1["J_linear"] / eps ** 2,
                         hybrid_uot=dech["J_kin"] / eps ** 2,
                         ref=math.pi / k))
        print(f"II-1 UOT k={k}: 1band={spot[-1]['one_band_uot']:.3f}"
              f" hybrid={spot[-1]['hybrid_uot']:.3f}"
              f" ref={math.pi / k:.3f}", flush=True)
    out["ii1_uot_spot"] = spot

    # ---------------- II-2: annulus radial per tier
    Xa, Ya, ua, geo = annulus_pair(128, eps)
    aA = weights_oracle_annulus(128, geo["n_in"], R_PLUS, R_MINUS)
    pA, nA = Xa.positions.numpy(), Xa.normals.numpy()
    La, _, _ = graph_laplacian(pA, nA, aA, 0.12)
    lamA, phiA = gev_modes(La, aA)
    posA = lamA[lamA > 1e-12]
    ell_maxA = float(posA.min() ** -0.5)
    ell_starA = math.sqrt(c_sig * sigma * ell_maxA)
    labelsA, ncompA = graph_components(pA, nA, aA, 0.12)
    wf, wb, floorA, _ = tier_weights(lamA, sigma, c_sig, ell_maxA)
    ellA = panel_lengths_annulus(128, geo["n_in"], R_PLUS, R_MINUS)
    GA = zero_charge_projection(single_layer_G(pA, ellA, 1.0))
    Q_hybA = hybrid_Q(phiA, lamA, aA, GA, labelsA, ncompA,
                      floorA, ell_maxA)
    rA = aA * ua
    target = 2.0 * math.pi * math.log(2.0)
    q_dptA = float((aA * ua ** 2).sum()) / eps ** 2   # = 6 pi
    vals = dict(
        ref=target,
        one_band=ell_starA * q_dptA,
        band=float(rA @ (modal_Q(phiA, aA, wb) @ rA)) / eps ** 2,
        hybrid=float(rA @ (Q_hybA @ rA)) / eps ** 2,
    )
    out["ii2_annulus_radial"] = dict(
        n_components=ncompA, ell_max=ell_maxA, ell_star=ell_starA,
        q_partial_over_eps2=q_dptA, values=vals,
        log_ratio={t: math.log(v / target) for t, v in vals.items()
                   if t != "ref"})
    print("II-2 annulus radial:", {t: round(v, 3)
                                   for t, v in vals.items()},
          flush=True)

    # ---------------- II-3: resampling invariance (k=2 mode)
    res_rows = []
    for tier in ("one_band", "hybrid"):
        vals_n = {}
        for nn in (128, 181, 256):
            Xn = generate_oriented_circle(nn, R, (0.0, 0.0), "cpu",
                                          DT)
            an = weights_oracle_circle(nn, R)
            pn, nrm_n = Xn.positions.numpy(), Xn.normals.numpy()
            _, Yn, un = circle_mode_pair(nn, 2, eps)
            rn = an * un
            if tier == "one_band":
                Ln, _, _ = graph_laplacian(pn, nrm_n, an, 0.12)
                lam_n, _ = gev_modes(Ln, an)
                pos_n = lam_n[lam_n > 1e-12]
                ells = math.sqrt(c_sig * sigma
                                 * float(pos_n.min() ** -0.5))
                vals_n[nn] = ells * float((an * un ** 2).sum()) \
                    / eps ** 2
            else:
                Ln, _, _ = graph_laplacian(pn, nrm_n, an, 0.12)
                lam_n, phi_n = gev_modes(Ln, an)
                pos_n = lam_n[lam_n > 1e-12]
                emx = float(pos_n.min() ** -0.5)
                _, _, flr, _ = tier_weights(lam_n, sigma, c_sig,
                                            emx)
                Gn = zero_charge_projection(single_layer_G(
                    pn, np.full(nn, 2 * math.pi * R / nn), 1.0))
                lbl, nc = graph_components(pn, nrm_n, an, 0.12)
                Qh = hybrid_Q(phi_n, lam_n, an, Gn, lbl, nc, flr,
                              emx)
                vals_n[nn] = float(rn @ (Qh @ rn)) / eps ** 2
        v = np.array(list(vals_n.values()))
        res_rows.append(dict(tier=tier,
                             values={str(k): float(x)
                                     for k, x in vals_n.items()},
                             rel_spread=float((v.max() - v.min())
                                              / v.mean())))
        print(f"II-3 {tier}: {vals_n} spread="
              f"{res_rows[-1]['rel_spread']:.3f}", flush=True)
    out["ii3_resampling"] = res_rows

    # ---------------- II-4: merger checkpoint q-quotient
    ck = torch.load("results/exact_merger/"
                    "exact_merger_control_states.pt",
                    weights_only=True)
    st = ck[1200]
    Vm = OrientedPointCloudVarifold(positions=st["positions"],
                                    angles=st["angles"])
    rr = Vm.positions.norm(dim=1).numpy()
    wall = rr < 0.6
    out["ii4_merger_checkpoint"] = dict(
        step=1200, n_points=Vm.n_points, n_wall=int(wall.sum()),
        m_wall_fraction=None, by_sigma={})
    # two-scale point: q must be read at the PRODUCTION sigma, not
    # the h_NN auto value (which shrinks below the wall separation
    # on the merged cloud and stops seeing the cancellation)
    for sig in (None, 0.05, 0.1):
        am, meta_m = weights_production(Vm, sigma=sig)
        key = "auto" if sig is None else f"{sig:g}"
        out["ii4_merger_checkpoint"]["by_sigma"][key] = dict(
            alpha_wall_fraction=float(am[wall].sum() / am.sum()),
            **meta_m)
    # raw-mass comparison for contrast
    delta, tau = compute_recommended_params(Vm.positions)
    m_raw = compute_masses(Vm.positions, delta, tau,
                           "wendland_c2").numpy()
    out["ii4_merger_checkpoint"]["m_wall_fraction"] = float(
        m_raw[wall].sum() / m_raw.sum())
    print("II-4 wall alpha-frac by sigma:",
          {k: round(v["alpha_wall_fraction"], 4)
           for k, v in out["ii4_merger_checkpoint"]["by_sigma"]
           .items()},
          "raw m-frac=%.4f" % out["ii4_merger_checkpoint"]
          ["m_wall_fraction"], flush=True)

    # ---------------- II-5: two-ellipses checkpoint stability
    ck2 = torch.load("results/grid_metric/"
                     "pair_contact_otto_v2_states.pt",
                     weights_only=True)
    step2 = max(k for k in ck2.keys() if k <= 950)
    st2 = ck2[step2]
    V2 = OrientedPointCloudVarifold(positions=st2["positions"],
                                    angles=st2["angles"])
    a2, meta2 = weights_production(V2)
    p2, n2 = V2.positions.numpy(), V2.normals.numpy()
    L2, _, _ = graph_laplacian(p2, n2, np.clip(a2, 1e-12, None),
                               0.12)
    lam2, _ = gev_modes(L2, np.clip(a2, 1e-12, None))
    pos2 = lam2[lam2 > 1e-12]
    out["ii5_two_ellipses"] = dict(
        step=step2, n=V2.n_points,
        lam1=float(pos2.min()), ell_max=float(pos2.min() ** -0.5),
        lam_top=float(lam2.max()),
        n_near_zero=int((lam2 < 1e-12).sum()),
        weight_meta=meta2)
    print("II-5 step %d: ell_max=%.3f n_zero_modes=%d"
          % (step2, out["ii5_two_ellipses"]["ell_max"],
             out["ii5_two_ellipses"]["n_near_zero"]), flush=True)

    report["phase_ii"] = out


# ================================================================
# Phase IIIa: static closure tests (reviewer-approved protocol).
# ================================================================


def bulk_incidence(pos, nrm, m, labels, ncomp, spacing,
                   delta_fracs=(1.5, 2.5, 4.0), n_probe=8):
    """Componentwise winding signature (reviewer blocker fix): the
    material-side probes of boundary component b read the vector
    (round(median w_a(p)))_a of per-component windings; equal
    signatures = same bulk component. Fail-closed (returns None) if
    the signature is unstable across probe depths."""
    sigs = []
    eta2 = (0.5 * spacing) ** 2
    for frac in delta_fracs:
        sig = []
        for b in range(ncomp):
            idx = np.nonzero(labels == b)[0]
            sel = idx[np.linspace(0, len(idx) - 1,
                                  min(n_probe, len(idx))).astype(int)]
            probes = pos[sel] - frac * spacing * nrm[sel]
            row = []
            for a in range(ncomp):
                ia = labels == a
                diff = pos[ia][None, :, :] - probes[:, None, :]
                den = (diff ** 2).sum(2) + eta2
                w = ((m[ia][None, :]
                      * (nrm[ia][None, :, :] * diff).sum(2) / den)
                     .sum(1) / (2.0 * math.pi))
                row.append(int(round(float(np.median(w)))))
            sig.append(tuple(row))
        sigs.append(sig)
    if not all(s == sigs[0] for s in sigs[1:]):
        return None
    uniq = {}
    bulk = np.empty(ncomp, dtype=int)
    for b, s in enumerate(sigs[0]):
        bulk[b] = uniq.setdefault(s, len(uniq))
    return bulk


def make_two_disks(n_per, R, centers, dtype=DT):
    parts, sgn = [], []
    for c in centers:
        parts.append(generate_oriented_circle(n_per, R, c, "cpu",
                                              dtype))
    return OrientedPointCloudVarifold(
        positions=torch.cat([p.positions for p in parts]),
        angles=torch.cat([p.angles for p in parts]))


def make_disk_two_holes(n_out, dtype=DT):
    outer = generate_oriented_circle(n_out, 1.0, (0.0, 0.0), "cpu",
                                     dtype)
    holes = []
    for c in ((-0.4, 0.0), (0.4, 0.0)):
        h = generate_oriented_circle(max(16, n_out // 5), 0.2, c,
                                     "cpu", dtype)
        holes.append(OrientedPointCloudVarifold(
            positions=h.positions,
            angles=h.angles + math.pi))    # normals into the hole
    return OrientedPointCloudVarifold(
        positions=torch.cat([outer.positions]
                            + [h.positions for h in holes]),
        angles=torch.cat([outer.angles]
                         + [h.angles for h in holes]))


def make_two_annuli(n_out, dtype=DT):
    parts = []
    for c in ((-1.1, 0.0), (1.1, 0.0)):
        parts.append(generate_oriented_annulus(n_out, 0.5, 0.25, c,
                                               "cpu", dtype))
    return OrientedPointCloudVarifold(
        positions=torch.cat([p.positions for p in parts]),
        angles=torch.cat([p.angles for p in parts]))


def oracle_alpha_from_labels(pos, labels, ncomp):
    """arclength weights per graph component (closed loops)."""
    a = np.empty(len(labels))
    for c in range(ncomp):
        idx = labels == c
        pts = pos[idx]
        centroid = pts.mean(0)
        rad = np.linalg.norm(pts - centroid, axis=1).mean()
        a[idx] = 2.0 * math.pi * rad / idx.sum()
    return a


def phase_iiia_incidence(report):
    """IIIa-1a: unit-test pins for the winding-signature incidence."""
    cases = {}
    fixtures = {
        "annulus": (generate_oriented_annulus(
            128, 1.0, 0.5, (0.0, 0.0), "cpu", DT), 1),
        "two_disks": (make_two_disks(96, 0.5,
                                     ((-1.0, 0.0), (1.0, 0.0))), 2),
        "disk_two_holes": (make_disk_two_holes(160), 1),
        "two_annuli": (make_two_annuli(96), 2),
    }
    ok = True
    for name, (v, n_bulk_true) in fixtures.items():
        pos, nrm = v.positions.numpy(), v.normals.numpy()
        labels, ncomp = graph_components(pos, nrm,
                                         np.ones(v.n_points), 0.12)
        alpha = oracle_alpha_from_labels(pos, labels, ncomp)
        spacing = float(np.median(alpha))
        bulk = bulk_incidence(pos, nrm, alpha, labels, ncomp,
                              spacing)
        n_bulk = int(bulk.max()) + 1 if bulk is not None else -1
        cases[name] = dict(n_boundary=ncomp, n_bulk=n_bulk,
                           expected=n_bulk_true,
                           ok=bool(n_bulk == n_bulk_true))
        ok &= cases[name]["ok"]
        print(f"IIIa-1a {name:16s} boundary={ncomp} bulk={n_bulk} "
              f"(expect {n_bulk_true}) {'OK' if cases[name]['ok'] else 'FAIL'}",
              flush=True)
    report["iiia1a_incidence"] = dict(cases=cases, all_ok=bool(ok))
    return ok


def solve_uot_cw(alpha, beta, D, admissible, Q_X, Q_Y, lam,
                 bulk_row=None, M_tie=None, big=1e3, rho=1e4,
                 n_al=15, al_tol=1e-8, num_iter=20000):
    """solve_uot with COMPONENTWISE (per-bulk) AL volume
    constraints: g_c = sum_{i in bulk c} r_i, zeta vector."""
    import ot

    n_bulk = 1 if bulk_row is None else int(bulk_row.max()) + 1
    if bulk_row is None:
        bulk_row = np.zeros(len(alpha), dtype=int)
    Bmask = np.stack([(bulk_row == c) for c in range(n_bulk)])
    M = np.where(admissible, 0.0 if M_tie is None else M_tie, big)
    zeta = np.zeros(n_bulk)
    plan = (np.outer(alpha, beta) / beta.sum()) * admissible
    hist = []
    for it in range(n_al):
        def f(G):
            r = (G * D).sum(1)
            gc = Bmask @ r
            c = (G * D).sum(0)
            return (0.5 * r @ (Q_X @ r) + 0.5 * c @ (Q_Y @ c)
                    + zeta @ gc + 0.5 * rho * (gc ** 2).sum())

        def df(G):
            r = (G * D).sum(1)
            gc = Bmask @ r
            c = (G * D).sum(0)
            per_row = (zeta + rho * gc)[bulk_row]
            return D * ((Q_X @ r)[:, None] + (Q_Y @ c)[None, :]
                        + per_row[:, None])

        plan, log = ot.unbalanced.lbfgsb_unbalanced(
            alpha, beta, M, reg=1.0, reg_m=(lam, lam),
            reg_div=(f, df), regm_div="kl", G0=plan,
            numItermax=num_iter, stopThr=1e-16, log=True)
        r = (plan * D).sum(1)
        gc = Bmask @ r
        denom = float(np.abs(plan * D).sum()) + 1e-300
        hist.append(dict(it=it, g_max=float(np.abs(gc).max()),
                         g_rel=float(np.abs(gc).max()) / denom))
        if np.abs(gc).max() / denom < al_tol:
            break
        zeta += rho * gc

    r = (plan * D).sum(1)
    c = (plan * D).sum(0)
    gc = Bmask @ r
    dec = dict(
        J_kin_X=float(0.5 * r @ (Q_X @ r)),
        J_kin_Y=float(0.5 * c @ (Q_Y @ c)),
        J_linear=float((M * plan).sum()),
        J_KL_X=lam * kl_div(plan.sum(1), alpha),
        J_KL_Y=lam * kl_div(plan.sum(0), beta),
        J_volume=float(zeta @ gc + 0.5 * rho * (gc ** 2).sum()),
        g_rel=float(np.abs(gc).max())
        / (float(np.abs(plan * D).sum()) + 1e-300),
        transported_mass_per_bulk=[
            float(plan[bulk_row == cb].sum())
            for cb in range(n_bulk)],
        marginal_err_X=float(np.abs(plan.sum(1) - alpha).sum()
                             / alpha.sum()),
        al_iters=len(hist),
    )
    dec["J_kin"] = dec["J_kin_X"] + dec["J_kin_Y"]
    dec["J_total"] = (dec["J_kin"] + dec["J_linear"] + dec["J_KL_X"]
                      + dec["J_KL_Y"] + dec["J_volume"])
    return plan, dec


def band_block_Q(phi, lam, alpha, Ghat, labels, bulk_of_comp,
                 floor, ell_max, band_ratio=4.0):
    """S0-b mandatory fix: basis-invariant band-block hybrid.
    Bands = zero modes (grouped BY BULK component) + dyadic lambda
    intervals. Per band: S_B = (M Phi_B)^T Ghat (M Phi_B),
    D_B = P_ch + sqrt(2)(I - P_ch) with P_ch the projection onto the
    row space of the per-bulk charge map C_B; S_B^hyb = D_B S_B D_B,
    spectrally clipped to [floor, ell_max]. Invariant under
    Phi_B -> Phi_B O; keeps off-diagonal cross terms."""
    n = len(lam)
    Mphi = phi * alpha[:, None]
    ncomp = int(labels.max()) + 1
    n_bulk = int(bulk_of_comp.max()) + 1
    bulk_row = bulk_of_comp[labels]
    zero = lam < 1e-10 * max(float(lam.max()), 1.0)
    # assign zero modes to bulks by mass concentration
    bands = []
    if zero.any():
        zi = np.nonzero(zero)[0]
        conc = np.zeros((n_bulk, len(zi)))
        for c in range(n_bulk):
            conc[c] = np.abs(Mphi[:, zi][bulk_row == c]).sum(0)
        owner = conc.argmax(0)
        for c in range(n_bulk):
            grp = zi[owner == c]
            if grp.size:
                bands.append(grp)
    pos = ~zero
    if pos.any():
        lam_pos = lam[pos]
        idx_pos = np.nonzero(pos)[0]
        b_id = np.floor(np.log(lam_pos)
                        / math.log(band_ratio)).astype(int)
        for b in np.unique(b_id):
            bands.append(idx_pos[b_id == b])
    S = np.zeros((n, n))
    for grp in bands:
        MB = Mphi[:, grp]
        S_B = MB.T @ Ghat @ MB
        S_B = 0.5 * (S_B + S_B.T)
        # charge rows over BOUNDARY components (the x1 rule is "the
        # mode carries charge on some boundary component" -- e.g.
        # the annulus radial mode is bulk-neutral but per-ring
        # charged); rank threshold is ABSOLUTE on the mass scale
        # (a relative-to-max threshold promotes numerically-zero
        # charges to spurious x1 directions)
        C_B = np.stack([MB[labels == b].sum(0)
                        for b in range(ncomp)])
        U_, s_, Vt = np.linalg.svd(C_B, full_matrices=False)
        rank = int((s_ > 1e-8 * alpha.sum()).sum())
        P_ch = Vt[:rank].T @ Vt[:rank] if rank else             np.zeros((len(grp), len(grp)))
        D_B = P_ch + math.sqrt(2.0) * (np.eye(len(grp)) - P_ch)
        S_h = D_B @ S_B @ D_B
        ev, evec = np.linalg.eigh(0.5 * (S_h + S_h.T))
        ev = np.clip(ev, floor, ell_max)
        S[np.ix_(grp, grp)] = evec @ np.diag(ev) @ evec.T
    return phi @ S @ phi.T


def build_operator(v, alpha, ell_graph=0.12, sigma=0.1,
                   panel=None):
    """Common assembly: graph, GEV, components, incidence, SL,
    band-block hybrid Q."""
    pos, nrm = v.positions.numpy(), v.normals.numpy()
    labels, ncomp = graph_components(pos, nrm, alpha, ell_graph)
    spacing = float(np.median(alpha))
    bulk = bulk_incidence(pos, nrm, alpha, labels, ncomp, spacing)
    if bulk is None:
        raise RuntimeError("incidence fail-closed")
    L, _, _ = graph_laplacian(pos, nrm, alpha, ell_graph)
    lam, phi = gev_modes(L, alpha)
    lam_pos = lam[lam > 1e-12]
    ell_max = float(lam_pos.min() ** -0.5) if lam_pos.size else 1.0
    floor = sigma
    if panel is None:
        panel = alpha.copy()
    Gh = zero_charge_projection(single_layer_G(pos, panel, 1.0))
    Q = band_block_Q(phi, lam, alpha, Gh, labels, bulk, floor,
                     ell_max)
    return dict(pos=pos, nrm=nrm, labels=labels, ncomp=ncomp,
                bulk=bulk, lam=lam, phi=phi, Gh=Gh, Q=Q,
                ell_max=ell_max, floor=floor,
                bulk_row=bulk[labels])


def phase_iiia_d0(report):
    """IIIa-1b: two-disk exchange. Before-fix finite cost, after
    componentwise constraints J_total/eps^2 ~ 1/eps."""
    R = 0.5
    centers = ((-1.0, 0.0), (1.0, 0.0))
    n_per = 96
    rows = []
    for eps in (1e-2, 5e-3):
        R1 = R + eps
        R2 = math.sqrt(2 * R * R - R1 * R1)
        X = make_two_disks(n_per, R, centers)
        Y = OrientedPointCloudVarifold(
            positions=torch.cat([
                generate_oriented_circle(n_per, R1, centers[0],
                                         "cpu", DT).positions,
                generate_oriented_circle(n_per, R2, centers[1],
                                         "cpu", DT).positions]),
            angles=X.angles.clone())
        u = np.empty(X.n_points)
        u[:n_per] = eps
        u[n_per:] = -(R - R2)
        aX = np.concatenate([np.full(n_per, 2 * math.pi * R / n_per)]
                            * 2)
        aY = np.concatenate([
            np.full(n_per, 2 * math.pi * R1 / n_per),
            np.full(n_per, 2 * math.pi * R2 / n_per)])
        opX = build_operator(X, aX)
        opY = build_operator(Y, aY)
        diff = (Y.positions.numpy()[None, :, :]
                - X.positions.numpy()[:, None, :])
        nbar = (X.normals.numpy()[:, None, :]
                + Y.normals.numpy()[None, :, :])
        nbar /= np.maximum(np.linalg.norm(nbar, axis=2,
                                          keepdims=True), 1e-300)
        Dm = (diff * nbar).sum(2)
        adm = np.eye(X.n_points, dtype=bool)
        # direct (operator) value of the exchange mode
        rX = aX * u
        direct = float(rX @ (opX["Q"] @ rX)) / eps ** 2
        # BEFORE: global constraint (bulk=None)
        _, dec_g = solve_uot_cw(aX, aY, Dm, adm, opX["Q"], opY["Q"],
                                eps, bulk_row=None)
        # AFTER: componentwise constraint via incidence
        _, dec_c = solve_uot_cw(aX, aY, Dm, adm, opX["Q"], opY["Q"],
                                eps, bulk_row=opX["bulk_row"])
        rows.append(dict(eps=eps, n_bulk=int(opX["bulk"].max()) + 1,
                         direct_over_eps2=direct,
                         global_J_total_over_eps2=dec_g["J_total"]
                         / eps ** 2,
                         cw_J_total_over_eps2=dec_c["J_total"]
                         / eps ** 2,
                         cw_J_kin_over_eps2=dec_c["J_kin"] / eps ** 2,
                         cw_J_KL_over_eps2=(dec_c["J_KL_X"]
                                            + dec_c["J_KL_Y"])
                         / eps ** 2,
                         cw_g_rel=dec_c["g_rel"],
                         cw_transported=dec_c[
                             "transported_mass_per_bulk"]))
        print(f"IIIa-1b D0 eps={eps:g}: direct={direct:.3f} "
              f"global_Jtot/e2={rows[-1]['global_J_total_over_eps2']:.3f} "
              f"cw_Jtot/e2={rows[-1]['cw_J_total_over_eps2']:.3f}",
              flush=True)
    ratio = (rows[1]["cw_J_total_over_eps2"]
             / rows[0]["cw_J_total_over_eps2"])
    report["iiia1b_d0"] = dict(
        rows=rows, cw_scaling_ratio_eps_halved=float(ratio),
        expect="ratio ~ 2 (J_total/eps^2 ~ 1/eps)",
        pass_inadmissible=bool(ratio > 1.5))
    print(f"IIIa-1b D0 scaling ratio (eps halved) = {ratio:.2f} "
          f"(expect ~2)", flush=True)


def phase_iiia_u0(report):
    """IIIa-2: real hybrid-annulus UOT solve (band-block Q)."""
    target = 2.0 * math.pi * math.log(2.0)
    rows = []
    for n_out in (128, 256):
        for eps in (1e-2, 5e-3, 2.5e-3):
            X, Y, u, geo = annulus_pair(n_out, eps)
            aX = weights_oracle_annulus(n_out, geo["n_in"], R_PLUS,
                                        R_MINUS)
            aY = weights_oracle_annulus(n_out, geo["n_in"],
                                        geo["rp_e"], geo["rm_e"])
            opX = build_operator(X, aX,
                                 panel=panel_lengths_annulus(
                                     n_out, geo["n_in"], R_PLUS,
                                     R_MINUS))
            opY = build_operator(Y, aY,
                                 panel=panel_lengths_annulus(
                                     n_out, geo["n_in"],
                                     geo["rp_e"], geo["rm_e"]))
            diff = (Y.positions.numpy()[None, :, :]
                    - X.positions.numpy()[:, None, :])
            nbar = (X.normals.numpy()[:, None, :]
                    + Y.normals.numpy()[None, :, :])
            nbar /= np.maximum(np.linalg.norm(nbar, axis=2,
                                              keepdims=True),
                               1e-300)
            Dm = (diff * nbar).sum(2)
            adm = np.eye(X.n_points, dtype=bool)
            rX = aX * u
            direct = float(rX @ (opX["Q"] @ rX)) / eps ** 2
            _, dec = solve_uot_cw(aX, aY, Dm, adm, opX["Q"],
                                  opY["Q"], eps,
                                  bulk_row=opX["bulk_row"])
            rows.append(dict(n_out=n_out, eps=eps,
                             direct_over_eps2=direct,
                             J_kin_over_eps2=dec["J_kin"] / eps ** 2,
                             g_rel=dec["g_rel"],
                             marginal_err=dec["marginal_err_X"],
                             target=target))
            print(f"IIIa-2 U0 N={n_out} e={eps:g}: "
                  f"J_kin/e2={rows[-1]['J_kin_over_eps2']:.4f} "
                  f"direct={direct:.4f} target={target:.4f}",
                  flush=True)
    report["iiia2_u0"] = rows


def _coherence_perimeter(v, sigma):
    delta, tau = compute_recommended_params(v.positions)
    m = compute_masses(v.positions, delta, tau, "wendland_c2")
    q = compute_coherence(v, m, sigma, "wendland_c2")
    return float((m * q).sum())


def phase_iiia_q0(report):
    """IIIa-3: radial whole-sheet contraction on checkpoint 1200
    (no pair matching), Rayleigh response, curvature diagnostics,
    d/sigma collapse, alpha-active-set sensitivity."""
    ck = torch.load("results/exact_merger/"
                    "exact_merger_control_states.pt",
                    weights_only=True)
    st = ck[1200]
    V0 = OrientedPointCloudVarifold(positions=st["positions"],
                                    angles=st["angles"])
    r_all = V0.positions.norm(dim=1)
    wall = (r_all < 0.6).numpy()
    rhat = (V0.positions / r_all.unsqueeze(1)).numpy()
    nr = (V0.normals.numpy() * rhat).sum(1)
    rw = r_all.numpy()[wall]
    rbar = 0.5 * (rw[nr[wall] > 0].mean() + rw[nr[wall] < 0].mean())
    d0 = abs(rw[nr[wall] > 0].mean() - rw[nr[wall] < 0].mean())
    sheet_sign = np.where(r_all.numpy() > rbar, 1.0, -1.0)

    def contracted(theta):
        pos = V0.positions.numpy().copy()
        rr = np.linalg.norm(pos, axis=1)
        tgt = rbar + sheet_sign * theta * d0 / 2.0
        pos[wall] = (pos[wall] / rr[wall, None]) * tgt[wall, None]
        return OrientedPointCloudVarifold(
            positions=torch.tensor(pos, dtype=DT),
            angles=V0.angles.clone())

    thetas = [1.0, 0.5, 0.25, 0.125, 0.0]
    rows = []
    for theta in thetas:
        Vt = contracted(theta)
        d = theta * d0
        for sigma in (0.05, 0.1, 0.2):
            am, meta = weights_production(Vt, sigma=sigma)
            row = dict(theta=theta, d=d, sigma=sigma,
                       d_over_sigma=d / sigma,
                       alpha_wall_frac=float(am[wall].sum()
                                             / am.sum()),
                       q_wall_mean=None)
            delta, tau = compute_recommended_params(Vt.positions)
            m = compute_masses(Vt.positions, delta, tau,
                               "wendland_c2")
            q = compute_coherence(Vt, m, sigma,
                                  "wendland_c2").numpy()
            row["q_wall_mean"] = float(q[wall].mean())
            # operator Rayleigh of the collapse velocity at sigma=0.1
            if sigma == 0.1:
                try:
                    aa = np.clip(am, 1e-10, None)
                    op = build_operator(Vt, aa, sigma=sigma)
                    u_col = np.where(wall,
                                     -sheet_sign * nr, 0.0)
                    rv = aa * u_col
                    den = float((aa * u_col ** 2).sum())
                    row["rayleigh_collapse"] = (
                        float(rv @ (op["Q"] @ rv)) / den
                        if den > 1e-14 else 0.0)
                    row["n_bulk"] = int(op["bulk"].max()) + 1
                except RuntimeError as exc:
                    row["rayleigh_collapse"] = str(exc)
            rows.append(row)
        print(f"IIIa-3 Q0 theta={theta:5.3f} d/sig(0.1)="
              f"{d / 0.1:5.2f}: "
              + " ".join(f"a_frac(s={s:g})="
                         f"{[x for x in rows if x['theta'] == theta and x['sigma'] == s][0]['alpha_wall_frac']:.3f}"
                         for s in (0.05, 0.1, 0.2)), flush=True)

    # curvature diagnostics of F = coherent perimeter at sigma=0.1
    curv = []
    rng = np.random.default_rng(1)
    for theta in (1.0, 0.25, 0.0):
        Vt = contracted(theta)
        base = _coherence_perimeter(Vt, 0.1)
        dirs = {
            "separation": np.where(wall[:, None],
                                   sheet_sign[:, None] * rhat, 0.0),
            "common_translation": np.where(
                wall[:, None], np.array([1.0, 0.0]), 0.0),
            "random_wall": np.where(
                wall[:, None],
                rng.standard_normal((V0.n_points, 2)), 0.0),
        }
        t = 1e-3
        row = dict(theta=theta)
        for name, xi in dirs.items():
            xin = xi / max(np.linalg.norm(xi), 1e-300)
            Vp = OrientedPointCloudVarifold(
                positions=Vt.positions + t * torch.tensor(xin,
                                                          dtype=DT),
                angles=Vt.angles.clone())
            Vm2 = OrientedPointCloudVarifold(
                positions=Vt.positions - t * torch.tensor(xin,
                                                          dtype=DT),
                angles=Vt.angles.clone())
            fp = _coherence_perimeter(Vp, 0.1)
            fm = _coherence_perimeter(Vm2, 0.1)
            row[name] = (fp + fm - 2 * base) / (t * t)
        curv.append(row)
        print(f"IIIa-3 curvature theta={theta}: "
              + " ".join(f"{k}={v:.3e}" for k, v in row.items()
                         if k != "theta"), flush=True)

    # alpha-active-set sensitivity at theta=0.125
    Vt = contracted(0.125)
    am, _ = weights_production(Vt, sigma=0.1)
    sens = {}
    u_test = np.where(~wall, 1.0, 0.0)     # outer dilation mode
    for tau_a in (1e-4, 1e-3, 1e-2):
        act = am >= tau_a * am.mean()
        Va = OrientedPointCloudVarifold(
            positions=Vt.positions[torch.tensor(act)],
            angles=Vt.angles[torch.tensor(act)])
        aa = np.clip(am[act], 1e-12, None)
        try:
            op = build_operator(Va, aa, sigma=0.1)
            rv = aa * u_test[act]
            sens[f"{tau_a:g}"] = dict(
                n_active=int(act.sum()),
                rayleigh=float(rv @ (op["Q"] @ rv))
                / float((aa * u_test[act] ** 2).sum()))
        except RuntimeError as exc:
            sens[f"{tau_a:g}"] = dict(n_active=int(act.sum()),
                                      error=str(exc))
    report["iiia3_q0"] = dict(d0=float(d0), rbar=float(rbar),
                              rows=rows, curvature=curv,
                              active_set_sensitivity=sens)
    print("IIIa-3 active-set:", sens, flush=True)


def phase_iiia_s0(report):
    """IIIa-4: S0-a visibility collapse continuity; S0-b basis
    rotation invariance (per-mode rule vs band-block)."""
    # S0-b on the circle: rotate a degenerate pair
    n = 256
    X = generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DT)
    aX = weights_oracle_circle(n, 1.0)
    pos, nrm = X.positions.numpy(), X.normals.numpy()
    labels, ncomp = graph_components(pos, nrm, aX, 0.12)
    bulk = bulk_incidence(pos, nrm, aX, labels, ncomp,
                          float(np.median(aX)))
    L, _, _ = graph_laplacian(pos, nrm, aX, 0.12)
    lam, phi = gev_modes(L, aX)
    lam_pos = lam[lam > 1e-12]
    ell_max = float(lam_pos.min() ** -0.5)
    Gh = zero_charge_projection(single_layer_G(
        pos, np.full(n, 2 * math.pi / n), 1.0))
    # rotate the k=3 degenerate pair (indices 5, 6)
    rng = np.random.default_rng(2)
    th = rng.uniform(0, 2 * math.pi)
    O = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th), math.cos(th)]])
    phi_rot = phi.copy()
    phi_rot[:, 5:7] = phi[:, 5:7] @ O

    def relnorm(QA, QB):
        M12 = np.sqrt(aX)
        A = QA * np.outer(M12, M12)
        B = QB * np.outer(M12, M12)
        return float(np.linalg.norm(A - B, 2)
                     / np.linalg.norm(A, 2))

    Q_pm = hybrid_Q(phi, lam, aX, Gh, labels, ncomp, 0.1, ell_max)
    Q_pm_rot = hybrid_Q(phi_rot, lam, aX, Gh, labels, ncomp, 0.1,
                        ell_max)
    Q_bb = band_block_Q(phi, lam, aX, Gh, labels, bulk, 0.1,
                        ell_max)
    Q_bb_rot = band_block_Q(phi_rot, lam, aX, Gh, labels, bulk,
                            0.1, ell_max)
    s0b = dict(per_mode_rotation_relnorm=relnorm(Q_pm, Q_pm_rot),
               band_block_rotation_relnorm=relnorm(Q_bb, Q_bb_rot))
    print(f"IIIa-4 S0-b rotation invariance: per-mode "
          f"{s0b['per_mode_rotation_relnorm']:.2e} vs band-block "
          f"{s0b['band_block_rotation_relnorm']:.2e}", flush=True)

    # band-block regression: circle modes + annulus radial targets
    reg = {}
    rows = []
    for k in (1, 2, 4, 8):
        _, _, u = circle_mode_pair(n, k, 1e-2)
        r = aX * u
        rows.append(dict(k=k,
                         value=float(r @ (Q_bb @ r)) / 1e-4,
                         ref=math.pi / k))
    reg["circle"] = rows
    Xa, Ya, ua, geo = annulus_pair(128, 1e-2)
    aA = weights_oracle_annulus(128, geo["n_in"], R_PLUS, R_MINUS)
    opA = build_operator(Xa, aA,
                         panel=panel_lengths_annulus(
                             128, geo["n_in"], R_PLUS, R_MINUS))
    rA = aA * ua
    reg["annulus_radial"] = dict(
        value=float(rA @ (opA["Q"] @ rA)) / 1e-4,
        ref=2 * math.pi * math.log(2.0))
    maxlog = max(abs(math.log(r["value"] / r["ref"]))
                 for r in rows)
    reg["circle_maxlog"] = maxlog
    print(f"IIIa-4 band-block regression: circle maxlog={maxlog:.3f}"
          f" annulus={reg['annulus_radial']['value']:.3f}"
          f" (ref {reg['annulus_radial']['ref']:.3f})", flush=True)

    # S0-a: visibility collapse on the wall (synthetic alpha scale)
    ck = torch.load("results/exact_merger/"
                    "exact_merger_control_states.pt",
                    weights_only=True)
    st = ck[1200]
    Vw = OrientedPointCloudVarifold(positions=st["positions"],
                                    angles=st["angles"])
    wall = (Vw.positions.norm(dim=1) < 0.6).numpy()
    am, _ = weights_production(Vw, sigma=0.1)
    u_test = np.where(~wall, 1.0, 0.0)
    s0a = []
    for t in (1.0, 0.3, 0.1, 0.03, 0.01):
        av = am.copy()
        av[wall] *= t
        act = av >= 1e-3 * av.mean()
        Va = OrientedPointCloudVarifold(
            positions=Vw.positions[torch.tensor(act)],
            angles=Vw.angles[torch.tensor(act)])
        aa = np.clip(av[act], 1e-12, None)
        try:
            op = build_operator(Va, aa, sigma=0.1)
            rv = aa * u_test[act]
            s0a.append(dict(t=t, n_active=int(act.sum()),
                            rayleigh=float(rv @ (op["Q"] @ rv))
                            / float((aa * u_test[act] ** 2).sum())))
        except RuntimeError as exc:
            s0a.append(dict(t=t, n_active=int(act.sum()),
                            error=str(exc)))
    print("IIIa-4 S0-a visibility collapse:",
          [(x["t"], x.get("rayleigh")) for x in s0a], flush=True)
    report["iiia4_s0"] = dict(s0b=s0b, band_block_regression=reg,
                              s0a=s0a)


def phase_iiia_crossblock(report):
    """IIIa-5: P-Q cross term on the eccentric annulus, grid metric
    as the reference (with its own resolution check)."""
    from src.torch.transport.grid_wasserstein import (
        GridMetricConfig, GridWassersteinMetric)
    from src.torch.transport.phase_grid import PhaseGridConfig

    n_out, n_in = 256, 128
    outer = generate_oriented_circle(n_out, 1.0, (0.0, 0.0), "cpu",
                                     DT)
    inner = generate_oriented_circle(n_in, 0.5, (0.15, 0.0), "cpu",
                                     DT)
    v = OrientedPointCloudVarifold(
        positions=torch.cat([outer.positions, inner.positions]),
        angles=torch.cat([outer.angles + 0.0,
                          inner.angles + math.pi]))
    a = np.concatenate([np.full(n_out, 2 * math.pi / n_out),
                        np.full(n_in, 2 * math.pi * 0.5 / n_in)])
    m_t = torch.tensor(a, dtype=DT)
    q_t = torch.ones(v.n_points, dtype=DT)
    vol = math.pi * (1.0 - 0.25)
    th_out = outer.angles.numpy()
    u0 = np.concatenate([np.full(n_out, 1.0),
                         np.full(n_in, -2.0)])      # global neutral
    uperp = np.concatenate([np.cos(th_out), np.zeros(n_in)])

    def make_gm(grid):
        cfg = GridMetricConfig(phase=PhaseGridConfig(
            grid_shape=(grid, grid), fill_epsilon=0.04,
            support_threshold=3e-3))
        gm = GridWassersteinMetric(cfg)
        gm.setup_for_step(v, m_t, q_t, vol)
        return gm

    def gm_Q(gm, u):
        s = torch.tensor(u, dtype=DT)
        return 2.0 * float(gm(s, torch.zeros(v.n_points, dtype=DT),
                              1.0))

    res = {}
    u0_used = None
    for grid in (512, 768):
        gm = make_gm(grid)
        zeros = torch.zeros(v.n_points, dtype=DT)
        e_out = np.concatenate([np.ones(n_out), np.zeros(n_in)])
        e_in = np.concatenate([np.zeros(n_out), np.ones(n_in)])

        def comp_res(u):
            drho = gm.flux(torch.tensor(u, dtype=DT), zeros)
            # neutrality in the OPERATOR's own yardstick (active-set
            # compatibility residual), not the full-grid integral
            return float(gm.poisson.compatibility_residuals(
                drho)[0])

        c_bal = -comp_res(e_out) / comp_res(e_in)
        u0_used = e_out + c_bal * e_in       # discretely neutral
        Q00 = gm_Q(gm, u0_used)
        Qpp = gm_Q(gm, uperp)
        Qmix = gm_Q(gm, u0_used + uperp)
        C = Qmix - Q00 - Qpp
        res[str(grid)] = dict(Q_u0=Q00, Q_uperp=Qpp, Q_mix=Qmix,
                              C_ref=C,
                              eta_cross=abs(C) / (Q00 + Qpp))
        print(f"IIIa-5 grid={grid}: Q0={Q00:.4f} Qp={Qpp:.4f} "
              f"C_ref={C:+.4f} eta={res[str(grid)]['eta_cross']:.3f}",
              flush=True)
    # hybrid side
    op = build_operator(v, a, panel=a.copy())
    r0 = a * (u0_used if u0_used is not None else u0)
    rp = a * uperp
    Ch = float((r0 + rp) @ (op["Q"] @ (r0 + rp))
               - r0 @ (op["Q"] @ r0) - rp @ (op["Q"] @ rp))
    res["hybrid_C"] = Ch
    res["hybrid_eta"] = abs(Ch) / (
        float(r0 @ (op["Q"] @ r0)) + float(rp @ (op["Q"] @ rp)))
    print(f"IIIa-5 hybrid: C={Ch:+.4f} eta={res['hybrid_eta']:.3f}",
          flush=True)
    report["iiia5_crossblock"] = res


def phase_iiia_nullgauge(report):
    """IIIa-6: jitter low-alpha atoms, measure response of visible
    measure / perimeter / hybrid Rayleigh."""
    ck = torch.load("results/exact_merger/"
                    "exact_merger_control_states.pt",
                    weights_only=True)
    st = ck[1200]
    V = OrientedPointCloudVarifold(positions=st["positions"],
                                   angles=st["angles"])
    wall = (V.positions.norm(dim=1) < 0.6).numpy()
    am, _ = weights_production(V, sigma=0.1)
    low = np.argsort(am)[:20]
    rng = np.random.default_rng(3)
    pos_j = V.positions.numpy().copy()
    pos_j[low] += 0.05 * rng.standard_normal((len(low), 2))
    Vj = OrientedPointCloudVarifold(
        positions=torch.tensor(pos_j, dtype=DT),
        angles=V.angles.clone())
    am_j, _ = weights_production(Vj, sigma=0.1)
    P0 = _coherence_perimeter(V, 0.1)
    Pj = _coherence_perimeter(Vj, 0.1)
    report["iiia6_nullgauge"] = dict(
        n_jittered=len(low),
        alpha_l1_change=float(np.abs(am_j - am).sum()
                              / am.sum()),
        perimeter_rel_change=float(abs(Pj - P0) / P0),
        jitter_scale=0.05)
    print(f"IIIa-6 null-gauge: |dalpha|_1/|alpha|_1="
          f"{report['iiia6_nullgauge']['alpha_l1_change']:.3e} "
          f"dP/P={report['iiia6_nullgauge']['perimeter_rel_change']:.3e}",
          flush=True)


def phase_iiia():
    report = {}
    ok = phase_iiia_incidence(report)
    if not ok:
        print("INCIDENCE UNIT TESTS FAILED -- aborting")
    else:
        phase_iiia_d0(report)
        phase_iiia_u0(report)
        phase_iiia_q0(report)
        phase_iiia_s0(report)
        phase_iiia_crossblock(report)
        phase_iiia_nullgauge(report)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "phase3a.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out}")




# ================================================================
# Phase IIIb: bulk direct sum + extended resolved block (cross
# correction), reviewer conditional-GO protocol.
# ================================================================

RANK_TOL_FACTOR = 1e-8      # sigma_j > RANK_TOL_FACTOR * sum(alpha);
                            # FIXED across all geometries/resolutions
                            # (IIIa bug: relative tolerances promote
                            # numerically-zero charges)


def charge_projector(C, alpha_sum):
    """P_ch = V_r V_r^T with the explicit absolute rank rule."""
    U_, s_, Vt = np.linalg.svd(C, full_matrices=False)
    rank = int((s_ > RANK_TOL_FACTOR * alpha_sum).sum())
    if rank == 0:
        return np.zeros((C.shape[1], C.shape[1])), 0
    Vr = Vt[:rank]
    return Vr.T @ Vr, rank


def extended_block_Q_bulk(pos, nrm, alpha, labels_local, panel,
                          sigma, ell_graph=0.12):
    """Per-BULK operator: GEV on the sub-cloud, SL with per-bulk
    zero-charge projection, extended resolved block
    S^ext = D S D on I^sigma = {lam <= sigma^-2} (full off-diagonals
    kept -- the cross correction), dyadic band + floor beyond.
    Returns (Q_b, diag_info)."""
    L, _, _ = graph_laplacian(pos, nrm, alpha, ell_graph)
    lam, phi = gev_modes(L, alpha)
    lam_pos = lam[lam > 1e-12]
    ell_max = float(lam_pos.min() ** -0.5) if lam_pos.size else 1.0
    floor = sigma
    Gh = zero_charge_projection(single_layer_G(pos, panel, 1.0))
    Mphi = phi * alpha[:, None]
    n = len(lam)
    ncomp_local = int(labels_local.max()) + 1
    resolved = lam <= sigma ** -2
    S = np.zeros((n, n))
    info = dict(n_resolved=int(resolved.sum()), n_modes=n,
                ell_max=ell_max)
    if resolved.any():
        idx = np.nonzero(resolved)[0]
        MB = Mphi[:, idx]
        S_B = MB.T @ Gh @ MB
        S_B = 0.5 * (S_B + S_B.T)
        C_B = np.stack([MB[labels_local == b].sum(0)
                        for b in range(ncomp_local)])
        P_ch, rank = charge_projector(C_B, float(alpha.sum()))
        D_B = P_ch + math.sqrt(2.0) * (np.eye(len(idx)) - P_ch)
        S_h = D_B @ S_B @ D_B
        ev, evec = np.linalg.eigh(0.5 * (S_h + S_h.T))
        ev = np.clip(ev, floor, ell_max)
        S[np.ix_(idx, idx)] = evec @ np.diag(ev) @ evec.T
        info.update(charge_rank=rank,
                    P_ch=P_ch, resolved_idx=idx, S_B=S_B, D_B=D_B,
                    Mphi_res=MB)
    un = ~resolved
    if un.any():
        # unresolved: dyadic band + floor (diagonal in bands as
        # before; these are below the resolution scale)
        idxu = np.nonzero(un)[0]
        lam_u = lam[idxu]
        b_id = np.floor(np.log(np.maximum(lam_u, 1e-300))
                        / math.log(4.0)).astype(int)
        for b in np.unique(b_id):
            grp = idxu[b_id == b]
            MB = Mphi[:, grp]
            S_B = MB.T @ Gh @ MB
            S_B = 0.5 * (S_B + S_B.T)
            ev, evec = np.linalg.eigh(2.0 * S_B)   # neutral x2
            ev = np.clip(ev, floor, ell_max)
            S[np.ix_(grp, grp)] = evec @ np.diag(ev) @ evec.T
    Q_b = phi @ S @ phi.T
    return 0.5 * (Q_b + Q_b.T), info


def build_operator_v3(v, alpha, m_raw=None, sigma=0.1,
                      ell_graph=0.12, panel=None):
    """IIIb operator: bulk DIRECT SUM Q = oplus_b Q^(b). Incidence
    from RAW m (winding identity carrier); B_vis recorded alongside.
    Fail-closed on incidence instability or raw/vis mismatch."""
    pos, nrm = v.positions.numpy(), v.normals.numpy()
    if m_raw is None:
        m_raw = alpha.copy()
    if panel is None:
        panel = alpha.copy()
    labels, ncomp = graph_components(pos, nrm, alpha, ell_graph)
    spacing = float(np.median(panel))
    bulk_raw = bulk_incidence(pos, nrm, m_raw, labels, ncomp,
                              spacing)
    bulk_vis = bulk_incidence(pos, nrm, alpha, labels, ncomp,
                              spacing)
    if bulk_raw is None:
        raise RuntimeError("incidence fail-closed (raw)")
    mismatch = (bulk_vis is None
                or not np.array_equal(bulk_raw, bulk_vis))
    bulk = bulk_raw
    n_bulk = int(bulk.max()) + 1
    bulk_row = bulk[labels]
    N = v.n_points
    Q = np.zeros((N, N))
    infos = []
    for b in range(n_bulk):
        sel = bulk_row == b
        # local boundary-component labels within the bulk
        comps_b = np.unique(labels[sel])
        relab = {c: i for i, c in enumerate(comps_b)}
        labels_local = np.array([relab[c] for c in labels[sel]])
        Q_b, info = extended_block_Q_bulk(
            pos[sel], nrm[sel], alpha[sel], labels_local,
            panel[sel], sigma, ell_graph)
        Q[np.ix_(sel, sel)] = Q_b
        infos.append(info)
    return dict(Q=Q, labels=labels, ncomp=ncomp, bulk=bulk,
                bulk_row=bulk_row, n_bulk=n_bulk,
                incidence_mismatch=bool(mismatch), infos=infos,
                pos=pos, nrm=nrm)


def solve_uot_cw2(alpha, beta, D, admissible, Q_X, Q_Y, lam,
                  bulk_row_X=None, bulk_row_Y=None, M_tie=None,
                  big=1e3, rho=1e4, n_al=15, al_tol=1e-10,
                  num_iter=20000):
    """IIIb-0-1: componentwise AL volume constraints on BOTH sides
    (B_X r = 0 and B_Y c = 0). Also returns the reviewer's D0
    metrics (zero-plan objective ratio, absolute g_c)."""
    import ot

    nX, nY = len(alpha), len(beta)
    if bulk_row_X is None:
        bulk_row_X = np.zeros(nX, dtype=int)
    if bulk_row_Y is None:
        bulk_row_Y = np.zeros(nY, dtype=int)
    nbX = int(bulk_row_X.max()) + 1
    nbY = int(bulk_row_Y.max()) + 1
    BX = np.stack([(bulk_row_X == c) for c in range(nbX)])
    BY = np.stack([(bulk_row_Y == c) for c in range(nbY)])
    M = np.where(admissible, 0.0 if M_tie is None else M_tie, big)
    zX = np.zeros(nbX)
    zY = np.zeros(nbY)
    plan = (np.outer(alpha, beta) / beta.sum()) * admissible
    for it in range(n_al):
        def f(G):
            r = (G * D).sum(1)
            c = (G * D).sum(0)
            gX = BX @ r
            gY = BY @ c
            return (0.5 * r @ (Q_X @ r) + 0.5 * c @ (Q_Y @ c)
                    + zX @ gX + 0.5 * rho * (gX ** 2).sum()
                    + zY @ gY + 0.5 * rho * (gY ** 2).sum())

        def df(G):
            r = (G * D).sum(1)
            c = (G * D).sum(0)
            gX = BX @ r
            gY = BY @ c
            rowt = (zX + rho * gX)[bulk_row_X]
            colt = (zY + rho * gY)[bulk_row_Y]
            return D * ((Q_X @ r)[:, None] + (Q_Y @ c)[None, :]
                        + rowt[:, None] + colt[None, :])

        plan, log = ot.unbalanced.lbfgsb_unbalanced(
            alpha, beta, M, reg=1.0, reg_m=(lam, lam),
            reg_div=(f, df), regm_div="kl", G0=plan,
            numItermax=num_iter, stopThr=1e-16, log=True)
        r = (plan * D).sum(1)
        c = (plan * D).sum(0)
        gX = BX @ r
        gY = BY @ c
        g_abs = max(float(np.abs(gX).max()), float(np.abs(gY).max()))
        if g_abs < al_tol * max(1.0, float(np.abs(plan * D).sum())):
            break
        zX += rho * gX
        zY += rho * gY

    r = (plan * D).sum(1)
    c = (plan * D).sum(0)
    J0_zero_plan = lam * (alpha.sum() + beta.sum())
    dec = dict(
        J_kin_X=float(0.5 * r @ (Q_X @ r)),
        J_kin_Y=float(0.5 * c @ (Q_Y @ c)),
        J_linear=float((M * plan).sum()),
        J_KL_X=lam * kl_div(plan.sum(1), alpha),
        J_KL_Y=lam * kl_div(plan.sum(0), beta),
        g_abs_X=[float(x) for x in (BX @ r)],
        g_abs_Y=[float(x) for x in (BY @ c)],
        transported_mass_ratio=float(plan.sum()
                                     / min(alpha.sum(),
                                           beta.sum())),
        J0_zero_plan=float(J0_zero_plan),
        al_iters=it + 1,
    )
    dec["J_kin"] = dec["J_kin_X"] + dec["J_kin_Y"]
    dec["J_total"] = (dec["J_kin"] + dec["J_linear"]
                      + dec["J_KL_X"] + dec["J_KL_Y"])
    dec["J_total_over_J0"] = dec["J_total"] / J0_zero_plan
    return plan, dec


# ---------------------------------------------------------- IIIb tests

def eccentric_annulus(n_out, n_in, e_hole):
    outer = generate_oriented_circle(n_out, 1.0, (0.0, 0.0), "cpu",
                                     DT)
    inner = generate_oriented_circle(n_in, 0.5, (e_hole, 0.0),
                                     "cpu", DT)
    v = OrientedPointCloudVarifold(
        positions=torch.cat([outer.positions, inner.positions]),
        angles=torch.cat([outer.angles, inner.angles + math.pi]))
    a = np.concatenate([np.full(n_out, 2 * math.pi / n_out),
                        np.full(n_in, 2 * math.pi * 0.5 / n_in)])
    return v, a, outer.angles.numpy(), inner.angles.numpy()


def phase_iiib_transfer(report):
    """IIIb-1/2: extended resolved block vs grid reference on the
    eccentric-annulus family; C_pred = 2(Da)^T S (Db) computed
    directly (no scalar ratio); per-mode discrete neutralization."""
    from src.torch.transport.grid_wasserstein import (
        GridMetricConfig, GridWassersteinMetric)
    from src.torch.transport.phase_grid import PhaseGridConfig

    n_out, n_in = 256, 128
    vol_pairs = []
    rows = []
    for e_hole in (0.05, 0.10, 0.15, 0.20):
        v, a, th_out, th_in = eccentric_annulus(n_out, n_in, e_hole)
        m_t = torch.tensor(a, dtype=DT)
        q_t = torch.ones(v.n_points, dtype=DT)
        vol = math.pi * (1.0 - 0.25)
        e_out = np.concatenate([np.ones(n_out), np.zeros(n_in)])
        e_in = np.concatenate([np.zeros(n_out), np.ones(n_in)])
        modes = {
            "u0": e_out - 2.0 * e_in,
            "out_k1": np.concatenate([np.cos(th_out),
                                      np.zeros(n_in)]),
            "out_k2": np.concatenate([np.cos(2 * th_out),
                                      np.zeros(n_in)]),
            "in_k1": np.concatenate([np.zeros(n_out),
                                     np.cos(th_in)]),
        }
        pairs = [("u0", "out_k1"), ("u0", "out_k2"),
                 ("u0", "in_k1"), ("out_k1", "in_k1")]

        def make_gm(grid):
            cfg = GridMetricConfig(phase=PhaseGridConfig(
                grid_shape=(grid, grid), fill_epsilon=0.04,
                support_threshold=3e-3))
            gm = GridWassersteinMetric(cfg)
            gm.setup_for_step(v, m_t, q_t, vol)
            return gm

        def neutralize(gm, u):
            zeros = torch.zeros(v.n_points, dtype=DT)

            def res(w):
                drho = gm.flux(torch.tensor(w, dtype=DT), zeros)
                return float(gm.poisson.compatibility_residuals(
                    drho)[0])

            r_u = res(u)
            r_e = res(e_in)
            return u - (r_u / r_e) * e_in

        def gm_Q(gm, u):
            s = torch.tensor(u, dtype=DT)
            return 2.0 * float(gm(s, torch.zeros(v.n_points,
                                                 dtype=DT), 1.0))

        # grid reference (both resolutions, per-mode neutralized)
        ref = {}
        for grid in (512, 768):
            gm = make_gm(grid)
            mo = {k: neutralize(gm, u) for k, u in modes.items()}
            Qs = {k: gm_Q(gm, u) for k, u in mo.items()}
            Cs = {}
            for (ka, kb) in pairs:
                Cs[f"{ka}*{kb}"] = (gm_Q(gm, mo[ka] + mo[kb])
                                    - Qs[ka] - Qs[kb])
            ref[grid] = dict(Q=Qs, C=Cs)

        # hybrid v3 (extended resolved block, bulk direct sum)
        op = build_operator_v3(v, a, m_raw=a, panel=a.copy())
        info = op["infos"][0]           # single bulk here
        P_ch = info["P_ch"]
        S_B = info["S_B"]
        D_B = info["D_B"]
        MB = info["Mphi_res"]
        # resolved coefficients of each mode: a_res = MB^T ... the
        # coefficient of r = alpha*u in the M-orthonormal basis is
        # phi^T r; restricted to resolved idx via MB pseudo-relation:
        # a_res_r = phi_r^T (alpha u) = (Mphi_r)^T u
        coef = {k: MB.T @ modes[k] for k in modes}
        for (ka, kb) in pairs:
            aC, bC = coef[ka], coef[kb]
            C_SL = float(2.0 * aC @ (S_B @ bC))
            C_pred = float(2.0 * (D_B @ aC) @ (S_B @ (D_B @ bC)))
            ra = a * modes[ka]
            rb = a * modes[kb]
            C_hyb = float((ra + rb) @ (op["Q"] @ (ra + rb))
                          - ra @ (op["Q"] @ ra)
                          - rb @ (op["Q"] @ rb))
            C_ref = ref[768]["C"][f"{ka}*{kb}"]
            Qsum = ref[768]["Q"][ka] + ref[768]["Q"][kb]
            big_cross = abs(C_ref) >= 0.05 * Qsum
            if big_cross:
                ok = (abs(C_hyb - C_ref) / abs(C_ref) < 0.05
                      and np.sign(C_hyb) == np.sign(C_ref))
            else:
                ok = abs(C_hyb - C_ref) < 0.0025 * Qsum
            rows.append(dict(
                e_hole=e_hole, pair=f"{ka}*{kb}",
                C_ref_512=ref[512]["C"][f"{ka}*{kb}"],
                C_ref_768=C_ref, C_SL=C_SL, C_pred=C_pred,
                C_hyb=C_hyb, Q_sum=Qsum,
                big_cross=bool(big_cross), ok=bool(ok)))
            print(f"IIIb-2 e={e_hole:.2f} {ka}*{kb:8s} "
                  f"C_ref={C_ref:+.4f} C_SL={C_SL:+.4f} "
                  f"C_pred={C_pred:+.4f} C_hyb={C_hyb:+.4f} "
                  f"{'OK' if ok else 'FAIL'}", flush=True)
    report["iiib2_transfer"] = rows
    report["iiib2_pass"] = bool(all(r["ok"] for r in rows))


def phase_iiib_d1(report):
    """IIIb-3: two-disk neutral cross (bulk direct sum check) +
    operator-norm pin."""
    n_per = 96
    v = make_two_disks(n_per, 0.5, ((-1.0, 0.0), (1.0, 0.0)))
    a = np.full(v.n_points, 2 * math.pi * 0.5 / n_per)
    op = build_operator_v3(v, a, m_raw=a, panel=a.copy())
    th = v.angles.numpy()
    u1 = np.where(np.arange(v.n_points) < n_per, np.cos(th), 0.0)
    u2 = np.where(np.arange(v.n_points) >= n_per, np.cos(th), 0.0)
    r1, r2 = a * u1, a * u2
    Q = op["Q"]
    C12 = float((r1 + r2) @ (Q @ (r1 + r2)) - r1 @ (Q @ r1)
                - r2 @ (Q @ r2))
    eta = abs(C12) / (float(r1 @ (Q @ r1)) + float(r2 @ (Q @ r2)))
    # operator-norm pin: off-bulk blocks are structurally zero
    sel1 = op["bulk_row"] == 0
    off = Q[np.ix_(sel1, ~sel1)]
    report["iiib3_d1"] = dict(
        C12=C12, eta=eta, off_block_norm=float(
            np.linalg.norm(off, 2)),
        pass_eta=bool(eta < 1e-3),
        pass_offblock=bool(np.linalg.norm(off, 2) < 1e-14))
    print(f"IIIb-3 D1: eta={eta:.2e} off-block norm="
          f"{np.linalg.norm(off, 2):.2e}", flush=True)


def phase_iiib_regression(report):
    """IIIb-3: full regression gate with the v3 operator."""
    gate = {}
    # circle Fourier k=1..10
    n = 256
    Xc = generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DT)
    ac = weights_oracle_circle(n, 1.0)
    opc = build_operator_v3(Xc, ac, m_raw=ac, panel=ac.copy())
    mx = 0.0
    for k in range(1, 11):
        _, _, u = circle_mode_pair(n, k, 1e-2)
        r = ac * u
        val = float(r @ (opc["Q"] @ r)) / 1e-4
        mx = max(mx, abs(math.log(val / (math.pi / k))))
    gate["circle_maxlog"] = mx
    # annulus radial (direct + UOT)
    Xa, Ya, ua, geo = annulus_pair(128, 1e-2)
    aA = weights_oracle_annulus(128, geo["n_in"], R_PLUS, R_MINUS)
    panelA = panel_lengths_annulus(128, geo["n_in"], R_PLUS,
                                   R_MINUS)
    opA = build_operator_v3(Xa, aA, m_raw=aA, panel=panelA)
    rA = aA * ua
    annv = float(rA @ (opA["Q"] @ rA)) / 1e-4
    gate["annulus_direct"] = annv
    gate["annulus_rel_err"] = abs(annv / (2 * math.pi
                                          * math.log(2)) - 1)
    # U0 with two-sided constraints (one config)
    aY = weights_oracle_annulus(128, geo["n_in"], geo["rp_e"],
                                geo["rm_e"])
    opY = build_operator_v3(Ya, aY, m_raw=aY,
                            panel=panel_lengths_annulus(
                                128, geo["n_in"], geo["rp_e"],
                                geo["rm_e"]))
    diff = (Ya.positions.numpy()[None, :, :]
            - Xa.positions.numpy()[:, None, :])
    nbar = (Xa.normals.numpy()[:, None, :]
            + Ya.normals.numpy()[None, :, :])
    nbar /= np.maximum(np.linalg.norm(nbar, axis=2, keepdims=True),
                       1e-300)
    Dm = (diff * nbar).sum(2)
    adm = np.eye(Xa.n_points, dtype=bool)
    _, dec = solve_uot_cw2(aA, aY, Dm, adm, opA["Q"], opY["Q"],
                           1e-2, bulk_row_X=opA["bulk_row"],
                           bulk_row_Y=opY["bulk_row"])
    gate["U0_J_kin_over_eps2"] = dec["J_kin"] / 1e-4
    # D0 with new metrics
    d0_rows = []
    for eps in (1e-2, 5e-3):
        R = 0.5
        R1 = R + eps
        R2 = math.sqrt(2 * R * R - R1 * R1)
        X = make_two_disks(96, R, ((-1.0, 0.0), (1.0, 0.0)))
        Y = OrientedPointCloudVarifold(
            positions=torch.cat([
                generate_oriented_circle(96, R1, (-1.0, 0.0),
                                         "cpu", DT).positions,
                generate_oriented_circle(96, R2, (1.0, 0.0),
                                         "cpu", DT).positions]),
            angles=X.angles.clone())
        aX = np.full(X.n_points, 2 * math.pi * R / 96)
        aY2 = np.concatenate([np.full(96, 2 * math.pi * R1 / 96),
                              np.full(96, 2 * math.pi * R2 / 96)])
        oX = build_operator_v3(X, aX, m_raw=aX, panel=aX.copy())
        oY = build_operator_v3(Y, aY2, m_raw=aY2, panel=aY2.copy())
        diff = (Y.positions.numpy()[None, :, :]
                - X.positions.numpy()[:, None, :])
        nb = (X.normals.numpy()[:, None, :]
              + Y.normals.numpy()[None, :, :])
        nb /= np.maximum(np.linalg.norm(nb, axis=2, keepdims=True),
                         1e-300)
        Dm2 = (diff * nb).sum(2)
        _, decd = solve_uot_cw2(aX, aY2, Dm2,
                                np.eye(X.n_points, dtype=bool),
                                oX["Q"], oY["Q"], eps,
                                bulk_row_X=oX["bulk_row"],
                                bulk_row_Y=oY["bulk_row"])
        d0_rows.append(dict(eps=eps,
                            J_total_over_eps2=decd["J_total"]
                            / eps ** 2,
                            J_total_over_J0=decd["J_total_over_J0"],
                            g_abs_X=decd["g_abs_X"],
                            transported=decd[
                                "transported_mass_ratio"]))
    gate["D0"] = d0_rows
    gate["D0_scaling_ratio"] = (d0_rows[1]["J_total_over_eps2"]
                                / d0_rows[0]["J_total_over_eps2"])
    # resampling (k=2, hybrid v3)
    vals = {}
    for nn in (128, 181, 256):
        Xn = generate_oriented_circle(nn, 1.0, (0.0, 0.0), "cpu",
                                      DT)
        an = weights_oracle_circle(nn, 1.0)
        on = build_operator_v3(Xn, an, m_raw=an, panel=an.copy())
        _, _, un = circle_mode_pair(nn, 2, 1e-2)
        rn = an * un
        vals[nn] = float(rn @ (on["Q"] @ rn)) / 1e-4
    varr = np.array(list(vals.values()))
    gate["resampling_spread"] = float((varr.max() - varr.min())
                                      / varr.mean())
    # PSD
    ev = np.linalg.eigvalsh(opc["Q"])
    gate["psd_min_over_norm"] = float(ev.min()
                                      / max(abs(ev).max(), 1e-300))
    report["iiib3_gate"] = gate
    print("IIIb-3 gate:", {k: (round(v, 4) if isinstance(v, float)
                               else v)
                           for k, v in gate.items()
                           if not isinstance(v, list)}, flush=True)


def phase_iiib_q0floor(report):
    """IIIb-4: attribute the 10% wall floor (sigma FIXED at 0.1)."""
    sigma = 0.1
    rbar = 0.394
    rows = []
    for s in (1, 2, 4):
        for counts, name in (((90 * s, 141 * s), "unequal"),
                             ((128 * s, 128 * s), "equal")):
            for phase_off in (0.0, 0.5):
                n1, n2 = counts
                th1 = 2 * math.pi * (np.arange(n1) + 0.0) / n1
                th2 = 2 * math.pi * (np.arange(n2) + phase_off) / n2
                pos = np.concatenate([
                    rbar * np.stack([np.cos(th1), np.sin(th1)], 1),
                    rbar * np.stack([np.cos(th2), np.sin(th2)], 1)])
                ang = np.concatenate([th1, th2 + math.pi])
                v = OrientedPointCloudVarifold(
                    positions=torch.tensor(pos, dtype=DT),
                    angles=torch.tensor(ang, dtype=DT))
                # oracle weights
                mo = np.concatenate([
                    np.full(n1, 2 * math.pi * rbar / n1),
                    np.full(n2, 2 * math.pi * rbar / n2)])
                q_or = compute_coherence(
                    v, torch.tensor(mo, dtype=DT), sigma,
                    "wendland_c2").numpy()
                # production weights
                delta, tau = compute_recommended_params(v.positions)
                mp = compute_masses(v.positions, delta, tau,
                                    "wendland_c2")
                q_pr = compute_coherence(v, mp, sigma,
                                         "wendland_c2").numpy()
                rows.append(dict(s=s, name=name,
                                 phase_off=phase_off,
                                 q_oracle_mean=float(q_or.mean()),
                                 q_prod_mean=float(q_pr.mean())))
                print(f"IIIb-4 s={s} {name:8s} off={phase_off}: "
                      f"q_oracle={q_or.mean():.4f} "
                      f"q_prod={q_pr.mean():.4f}", flush=True)
    report["iiib4_q0floor"] = rows


def phase_iiib():
    report = {}
    phase_iiib_d1(report)
    phase_iiib_regression(report)
    phase_iiib_transfer(report)
    phase_iiib_q0floor(report)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "phase3b.json"
    out.write_text(json.dumps(report, indent=1, default=lambda o:
                              float(o) if np.isscalar(o) else None))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
