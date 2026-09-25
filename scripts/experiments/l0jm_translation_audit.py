"""L0J-M0 (reviewer 2026-08-21): translation-force audit of the four
coherence arms -- NO code changes, measurement only.

(a) RIGID zero pin: delta x = 1_ell a, delta theta = 0. The loopwise
    0K mass depends only on same-loop pairwise distances and normals,
    so a rigid translation of one whole loop leaves every mass
    invariant and P_X^q(Y) is EXACTLY constant for any frozen q.
(b) SOLVER normal-graph translation mode: the mode the dynamics can
    actually take -- s = 1_ell a . n projected into the admissible
    parametrization (s = c Q y, dtheta = c AB s), so the measured
    force is the one the MM step feels (mass responds to normals too).
(c) Mobility-closed drift prediction: on the admissible translation
    modes, Gram G_kl = W-form(xi_k, xi_l) and f_k = DP[xi_k] give
    alpha* = -G^dagger f and the predicted per-step barycenter drift,
    compared against the MEASURED L0J-2b drift rate.
(d) Negative pin: the arclength-affine profile r0 + r1 s solves to
    r1 ~ 0 in this symmetric configuration (the reviewer's refutation
    of the M1-arclength proposal).

Usage:
    uv run python scripts/experiments/l0jm_translation_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import build_config  # noqa: E402
from l0j_locality_audit import _V, arms_q  # noqa: E402
from quotient_switch_stability import production_args  # noqa: E402
from two_ellipses_benchmark import resolve_m, signed_area_order  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)

DT = torch.float64
ARMS = ("F", "W", "C", "S")


def sharp(m, q):
    return float((m * q).sum())


def P_at(pos, ang, lbl, delta, tau, q):
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    return sharp(resolve_m(pos, nor, delta, tau), q)


def audit_state(tag, step_key, grid=768):
    ck = torch.load(f"results/two_ellipses/two_ellipses{tag}_states.pt",
                    weights_only=True)
    step = max(ck) if step_key == "last" else step_key
    st0 = ck[step]
    pos, ang = st0["positions"], st0["angles"]
    n = pos.shape[0] // 2
    lbl = torch.zeros(pos.shape[0], dtype=torch.long)
    lbl[n:] = 1
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    delta, tau = map(float, compute_recommended_params(pos))
    m = resolve_m(pos, nor, delta, tau)
    qs, res_cc = arms_q(pos, nor, m, lbl)
    out = dict(tag=tag, step=int(step))
    h = float(m.median())
    t_fd = 1e-4 * h

    # ---- (a) rigid zero pin --------------------------------------
    rigid = {}
    for li, sel in (("loop0", lbl == 0), ("loop1", lbl == 1)):
        d = torch.zeros_like(pos)
        d[sel, 0] = 1.0
        row = {}
        for a in ARMS:
            Pp = P_at(pos + t_fd * d, ang, lbl, delta, tau, qs[a])
            Pm = P_at(pos - t_fd * d, ang, lbl, delta, tau, qs[a])
            row[a] = (Pp - Pm) / (2 * t_fd)
        rigid[li] = row
    out["F_rigid"] = rigid

    # ---- solver parametrization (admissible modes) ---------------
    from src.torch.solver.mm_step import MMStepper
    cfg = build_config(production_args(
        grid=grid, redist_monotone=True,
        q_mode="contact_complex_renormalized"), delta, tau)
    st = MMStepper(cfg)
    st._setup_step(OrientedPointCloudVarifold(positions=pos.clone(),
                                              angles=ang.clone()))
    param = st.param
    c = param.coherence
    Q = param.Q
    CQ = c.unsqueeze(1) * Q                       # s = (cQ) y

    def admissible(s_target):
        y = torch.linalg.lstsq(CQ, s_target.unsqueeze(1)).solution
        s_hat = (CQ @ y).flatten()
        dth = param.AB_solve @ s_hat
        dth = dth * c
        return s_hat, dth

    modes = []
    for li, sel in ((0, lbl == 0), (1, lbl == 1)):
        for k in (0, 1):
            s_t = torch.zeros(pos.shape[0], dtype=DT)
            s_t[sel] = nor[sel, k]
            modes.append((admissible(s_t), li, k))
    # ---- (b) normal-graph translation force ----------------------
    F_ng = {a: [] for a in ARMS}
    proj_frac = []
    for (s_hat, dth), l0, k in modes:
        proj_frac.append(float(s_hat.norm()))
        dpos = s_hat.unsqueeze(1) * nor
        for a in ARMS:
            Pp = P_at(pos + t_fd * dpos, ang + t_fd * dth, lbl,
                      delta, tau, qs[a])
            Pm = P_at(pos - t_fd * dpos, ang - t_fd * dth, lbl,
                      delta, tau, qs[a])
            F_ng[a].append((Pp - Pm) / (2 * t_fd))
    out["F_ng"] = F_ng
    out["mode_norms"] = proj_frac

    # ---- (c) Gram + drift prediction (arm C) ---------------------
    gm = st.grid_wasserstein
    K = len(modes)
    G = torch.zeros(K, K, dtype=DT)

    def qform(v, w):
        with torch.no_grad():
            drho = gm.flux(v, w)
            b = gm.poisson.project_componentwise_zero_mean(drho)
            phi = gm.poisson.solve(b)[0]
            return (1.0 / cfg.time_step) * float(
                gm.poisson.inner(b, phi))          # = d2W (W quad)
    singles = []
    for (s_hat, dth), _, _ in modes:
        singles.append(qform(s_hat, dth))
    for i in range(K):
        for j in range(i, K):
            if i == j:
                G[i, i] = singles[i]
            else:
                (si, di), (sj, dj) = modes[i][0], modes[j][0]
                G[i, j] = G[j, i] = 0.5 * (
                    qform(si + sj, di + dj) - singles[i] - singles[j])
    f = torch.tensor(F_ng["C"], dtype=DT)
    alpha = -torch.linalg.pinv(G) @ f
    # barycenter response of each mode (x-component per loop)
    A0 = (signed_area_order(pos[lbl == 0]),
          signed_area_order(pos[lbl == 1]))
    db = torch.zeros(K, 2, dtype=DT)               # loops x (dbar_x)
    for kk, ((s_hat, dth), _, _) in enumerate(modes):
        for li, sel in ((0, lbl == 0), (1, lbl == 1)):
            db[kk, li] = float((m[sel] * s_hat[sel]
                                * nor[sel, 0]).sum()) / A0[li]
    pred = (alpha.unsqueeze(1) * db).sum(0)
    out["alpha_star"] = [float(x) for x in alpha]
    out["G_diag"] = [float(x) for x in G.diag()]
    out["dbar_pred_per_step"] = [float(x) for x in pred]
    return out


def affine_r1(tag):
    """(d) negative pin: solve the arclength-affine M0+M1 system on
    the contact side and report r1 (expected ~0 by symmetry)."""
    from src.torch.perimeter.contact_complex import certify_loop_orders
    ck = torch.load(f"results/two_ellipses/two_ellipses{tag}_states.pt",
                    weights_only=True)
    st0 = ck[max(ck)]
    pos, ang = st0["positions"], st0["angles"]
    n = pos.shape[0] // 2
    lbl = torch.zeros(pos.shape[0], dtype=torch.long)
    lbl[n:] = 1
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    delta, tau = map(float, compute_recommended_params(pos))
    m = resolve_m(pos, nor, delta, tau)
    qs, res_cc = arms_q(pos, nor, m, lbl)
    side = res_cc["cx"]["complexes"][0].sides[0]
    sel = side.particles
    orders = certify_loop_orders(pos, lbl)
    od = orders[side.loop]
    D12 = torch.cdist(pos[lbl == 0], pos[lbl == 1])
    gpp = D12.min(dim=1).values
    i_c = sel[int(torch.argmin(
        torch.cdist(pos[sel], pos[lbl == 1]).min(dim=1).values))]
    s_all, L = od["arclen"], od["length"]
    s = s_all[od["rank_of"][sel]] - s_all[od["rank_of"][i_c]]
    s = torch.where(s > L / 2, s - L, s)
    s = torch.where(s < -L / 2, s + L, s)
    mI, qsI, qfI = m[sel], qs["S"][sel], qs["F"][sel]
    A = torch.stack([
        torch.stack([(mI * qsI).sum(), (mI * s * qsI).sum()]),
        torch.stack([(mI * s * qsI).sum(), (mI * s * s * qsI).sum()])])
    b = torch.stack([(mI * qfI).sum(), (mI * s * qfI).sum()])
    r0, r1 = torch.linalg.solve(A, b)
    return dict(r0=float(r0), r1=float(r1),
                r1_scaled=float(r1) * float(s.abs().max()))


def main():
    torch.set_default_dtype(DT)
    rep = {}
    for tag, key in (("_l0_768", 1425), ("_l0j2_768", "last")):
        r = audit_state(tag, key)
        rep[f"{tag}@{r['step']}"] = r
        print(f"== {tag} @ {r['step']}")
        print("  F_rigid loop0:", {a: f"{v:.2e}" for a, v in
                                   r["F_rigid"]["loop0"].items()})
        for a in ARMS:
            print(f"  F_ng[{a}]:", ["%.3e" % x for x in r["F_ng"][a]])
        print("  alpha*:", ["%.3e" % x for x in r["alpha_star"]],
              " dbar_pred/step:", ["%.3e" % x for x in
                                   r["dbar_pred_per_step"]])
    neg = affine_r1("_l0j2_768")
    rep["affine_negative_pin"] = neg
    print("affine r0/r1:", neg)
    Path("results/reports/l0jm_translation_audit.json").write_text(
        json.dumps(rep, indent=1, default=str))
    print("wrote results/reports/l0jm_translation_audit.json")


if __name__ == "__main__":
    main()
