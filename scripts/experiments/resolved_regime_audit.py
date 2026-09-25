"""Resolved-regime audit v2 for the two-ellipses merger target
(closure patch 2; reviewer-approved design).

Decides which static configurations may be called "resolved separated
dynamics" -- not by N alone but by dimensionless ratios and a
superposition test. The pair frame and the two isolated single-ellipse
frames share IDENTICAL positions/normals (the generators only
concatenate), so the per-particle normal-displacement space s is an
exactly common coordinate space; only the derived quantities (KDE
masses, coherence, BEM operator) differ.

Measurements per case (n_per, gap g, perimeter_sigma, angle_sigma):

1. two-tier validity -- `solve_valid` is a QUANTITATIVE rule
   (finite outputs, objective decreased, relative gradient <= 1e-6,
   r_comp <= 1e-8, max|s|/h_same_q10 <= 0.5). The optimizer's
   `converged` boolean and message are recorded separately and do NOT
   gate validity (trust-ncg routinely reports converged=False at
   relgrad ~ 1e-7 on numerically fine solves).
2. D_super in refinement-compatible norms with a COUPLING-INDEPENDENT
   reference weight m_ref = m_iso1 (+) m_iso2:
       D^(m_ref)  weighted L2,   D^(inf)  max-norm,
   absolute numerators/denominators stored; the pair-weighted L2 is a
   secondary diagnostic only. Control: relative deviation of m_pair
   from m_ref (must vanish when the sheets are far apart).
3. permutation-invariant facing geometry (labels are AUDIT-ONLY oracle
   information, never used by production): facing set = points whose
   normal-compatible (n_i . n_j < -0.5) cross-sheet nearest distance is
   in the lowest 10%; h scales from same-sheet kNN radii (k=3), NEVER
   from array adjacency.
4. q-channel linearized decomposition at y=0 with a common reference
   projection P_ref (componentwise mean-zero wrt m_ref):
       d_E    = -K_pair^{-1} P_ref (g_pair - g_iso)       (energy side)
       d_M    = -(K_pair^{-1} - K_blk^{-1}) P_ref g_iso    (metric side)
       d_full = -K_pair^{-1} P_ref g_pair + K_blk^{-1} P_ref g_iso
   with the additive identity d_full = d_E + d_M checked to CG
   tolerance. These are LINEARIZED quantities; the full nonlinear
   one-step D_super is reported separately and no additivity is claimed
   for it. K action is matrix-free (autograd HVP + projected CG).
5. shadow rank telemetry: the adaptive rank machine is run on the same
   frame and its decision recorded; it never influences D_super (which
   uses the certified oracle rank).

Sweeps
------
  --mode g-sweep      g in {0.10,...,0.03} (+far control), sigma fixed
  --mode sigma-sweep  perimeter_sigma swept, angle_sigma FIXED
  --mode angle-sweep  angle_sigma swept, perimeter_sigma FIXED

Usage
-----
    uv run python scripts/experiments/resolved_regime_audit.py \
        --mode g-sweep --n-per 256 512 --out results/resolved_regime
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
)
from src.torch.shapes.generator import (
    generate_oriented_ellipse,
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_step import MMStepper
from src.torch.transport.bem_wasserstein import AmbiguousComponentRankError

sys.path.insert(0, str(Path(__file__).parent))
from p1_production_comparison import make_cfg  # noqa: E402

DT = 1e-5
DTYPE = torch.float64
A_EL, B_EL = 0.4, 1.0
TAU_STEP = 0.5
RELGRAD_TOL = 1e-6
RCOMP_TOL = 1e-8


def pair_frames(n_per, g):
    cx = A_EL + g / 2.0
    v = generate_oriented_two_ellipses(
        n_per, a1=A_EL, b1=B_EL, center1=(-cx, 0.0),
        a2=A_EL, b2=B_EL, center2=(cx, 0.0), device="cpu", dtype=DTYPE)
    v1 = generate_oriented_ellipse(n_per, A_EL, B_EL, (-cx, 0.0),
                                   "cpu", DTYPE)
    v2 = generate_oriented_ellipse(n_per, A_EL, B_EL, (cx, 0.0),
                                   "cpu", DTYPE)
    labels = torch.cat([torch.zeros(n_per, dtype=torch.long),
                        torch.ones(n_per, dtype=torch.long)])
    # particle correspondence: pair storage IS iso1 (+) iso2
    assert torch.equal(v.positions,
                       torch.cat([v1.positions, v2.positions]))
    assert torch.equal(v.angles, torch.cat([v1.angles, v2.angles]))
    return v, (v1, v2), labels


# --------------------------------------------------------------------
# permutation-invariant audit geometry (labels = audit-only oracle)
# --------------------------------------------------------------------

def facing_stats(positions, normals, labels, k=3, eta_n=0.5, frac=0.10):
    """Facing set from normal-compatible cross-sheet nearest distances
    (lowest `frac` quantile); h scales from same-sheet kNN radii. No
    array order is used anywhere."""
    out = {}
    # d_face in STORAGE order (mask-assigned), so every downstream mask
    # aligns with h_same regardless of particle permutation
    d_face = torch.empty(positions.shape[0], dtype=positions.dtype)
    for c in (0, 1):
        A = positions[labels == c]
        nA = normals[labels == c]
        B = positions[labels != c]
        nB = normals[labels != c]
        d = torch.cdist(A, B)
        compat = (nA @ nB.T) < -eta_n
        d = d.masked_fill(~compat, float("inf"))
        d_face[labels == c] = d.min(dim=1).values
    finite = torch.isfinite(d_face)
    if finite.sum() < 4:
        return dict(n_facing=0)
    thresh = torch.quantile(d_face[finite], frac)
    facing = finite & (d_face <= thresh)

    h_same = torch.empty_like(d_face)
    for c in (0, 1):
        A = positions[labels == c]
        dAA = torch.cdist(A, A)
        dAA.fill_diagonal_(float("inf"))
        knn = dAA.sort(dim=1).values[:, k - 1]      # k-NN radius
        h_same[labels == c] = knn
    hf = h_same[facing]
    out.update(
        n_facing=int(facing.sum()),
        gap_facing_min=float(d_face[finite].min()),
        h_same_min=float(h_same.min()),
        h_same_q10=float(torch.quantile(h_same, 0.10)),
        h_same_median=float(h_same.median()),
        h_face_min=float(hf.min()),
        h_face_median=float(hf.median()),
    )
    return out


def d_super_norms(s_pair, s_iso, m_ref, m_pair):
    ds = s_pair - s_iso
    num_w = float(torch.sqrt((m_ref * ds ** 2).sum()))
    den_w = float(torch.sqrt((m_ref * s_iso ** 2).sum()))
    num_i = float(ds.abs().max())
    den_i = float(s_iso.abs().max())
    num_p = float(torch.sqrt((m_pair * ds ** 2).sum()))
    den_p = float(torch.sqrt((m_pair * s_iso ** 2).sum()))
    small = den_w < 1e-14 or den_i < 1e-14
    return dict(
        d_super_mref=(num_w / den_w if not small else None),
        d_super_inf=(num_i / den_i if not small else None),
        d_super_pairweight=(num_p / den_p if den_p > 1e-14 else None),
        num_mref=num_w, den_mref=den_w, num_inf=num_i, den_inf=den_i,
        denominator_too_small=small,
        m_pair_vs_mref_reldev=float(
            (m_pair - m_ref).abs().max() / m_ref.abs().max()),
    )


# --------------------------------------------------------------------
# solvers / one-step
# --------------------------------------------------------------------

def _cfg(rank, delta, tau, sigma_p, sigma_a):
    cfg = make_cfg("C3", delta, tau, rank)
    cfg.time_step = DT
    cfg.perimeter_sigma = sigma_p
    cfg.angle_sigma = sigma_a
    return cfg


def one_step(v, rank, delta, tau, sigma_p, sigma_a, h_step_scale):
    st = MMStepper(_cfg(rank, delta, tau, sigma_p, sigma_a))
    r = st.step(v)
    s = r.displacements.detach()
    finite = bool(torch.isfinite(s).all()
                  and math.isfinite(r.objective))
    max_disp = float(s.abs().max())
    solve_valid = bool(
        finite and r.objective_decreased
        and (r.relative_gradient_norm or 1.0) <= RELGRAD_TOL
        and (r.r_comp is None or r.r_comp <= RCOMP_TOL)
        and max_disp / h_step_scale <= TAU_STEP)
    tele = dict(
        solve_valid=solve_valid,
        optimizer_converged=bool(r.converged),
        optimizer_message=r.optimizer_message,
        relative_gradient_norm=r.relative_gradient_norm,
        objective_decreased=bool(r.objective_decreased),
        r_comp=r.r_comp, n_iter=r.n_iter,
        max_disp=max_disp,
        max_disp_over_h_same_q10=max_disp / h_step_scale)
    return s, st, tele


def shadow_adaptive_rank(v, delta, tau, sigma_p, sigma_a):
    """Adaptive rank machine on the same frame -- shadow telemetry
    only; never used for D_super."""
    cfg = _cfg(2, delta, tau, sigma_p, sigma_a)
    cfg.bem_rank_mode = "adaptive_state_machine"
    cfg.bem_component_rank = None
    st = MMStepper(cfg)
    try:
        st._setup_step(v)
        dec = st._last_rank_decision
        return dict(shadow_rank=dec.rank if dec else None,
                    shadow_action=dec.action if dec else None,
                    shadow_status=dec.evidence_status if dec else None)
    except AmbiguousComponentRankError as e:
        return dict(shadow_rank=None, shadow_action="raised",
                    shadow_status=str(e)[:120])


# --------------------------------------------------------------------
# linearized q-channel decomposition (common s-space, common P_ref)
# --------------------------------------------------------------------

def _wlin_closure(stepper):
    """W_lin(s) through the frame's OWN angle map and coherence -- the
    exact quadratic the production step minimizes, as a function of the
    common s coordinates."""
    q = stepper.fixed_coherence
    AB = stepper.param.AB_solve
    bem = stepper.bem_wasserstein
    h = stepper.config.time_step

    def w(s):
        dth = q * (AB @ s)
        return bem(displacements=s, delta_angles=dth, time_step=h)
    return w


def _perimeter_grad(stepper, v):
    """grad_s of the frozen perimeter P(s) = sum m(x + s n) q_frozen."""
    x, n = v.positions, v.normals
    q = stepper.fixed_coherence
    s = torch.zeros(x.shape[0], dtype=DTYPE, requires_grad=True)

    m = compute_masses(x + s[:, None] * n,
                       stepper._mass_delta_for_kde,
                       stepper._mass_tau_for_kde,
                       stepper.config.mass_kernel)
    P = (m * q).sum()
    g, = torch.autograd.grad(P, s)
    return g.detach()


def _hvp_closure(wfun):
    def hvp(vv):
        s = torch.zeros(vv.shape[0], dtype=DTYPE, requires_grad=True)
        W = wfun(s)
        g, = torch.autograd.grad(W, s, create_graph=True)
        Hv, = torch.autograd.grad(g, s, grad_outputs=vv)
        return Hv.detach()
    return hvp


def _proj_ref(m_ref, labels):
    """Componentwise mean-zero projection wrt the REFERENCE weights --
    the common oracle admissible space (never pair-q-dependent)."""
    vecs = []
    for c in (0, 1):
        w = torch.where(labels == c, m_ref,
                        torch.zeros_like(m_ref))
        vecs.append(w / w.norm())
    def P(s):
        out = s
        for w in vecs:
            out = out - (w @ out) * w
        return out
    return P


def _projected_cg(hvp, P, rhs, tol=1e-12, max_iter=400):
    """Solve P K P d = P rhs on the reference subspace."""
    b = P(rhs)
    d = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = r @ r
    b_norm = math.sqrt(float(rs)) + 1e-300
    for _ in range(max_iter):
        Kp = P(hvp(P(p)))
        alpha = rs / (p @ Kp)
        d = d + alpha * p
        r = r - alpha * Kp
        rs_new = r @ r
        if math.sqrt(float(rs_new)) / b_norm < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return d, math.sqrt(float(rs)) / b_norm


def linearized_decomposition(st_pair, st1, st2, v, v1, v2,
                             m_ref, labels):
    n1 = v1.positions.shape[0]
    g_pair = _perimeter_grad(st_pair, v)
    g_iso = torch.cat([_perimeter_grad(st1, v1),
                       _perimeter_grad(st2, v2)])

    K_pair = _hvp_closure(_wlin_closure(st_pair))
    w1 = _wlin_closure(st1)
    w2 = _wlin_closure(st2)

    def K_blk(s):
        return torch.cat([_hvp_closure(w1)(s[:n1]),
                          _hvp_closure(w2)(s[n1:])])

    P = _proj_ref(m_ref, labels)
    d_pair, res1 = _projected_cg(K_pair, P, -g_pair)
    d_iso, res2 = _projected_cg(K_blk, P, -g_iso)
    d_mix, res3 = _projected_cg(K_pair, P, -g_iso)

    d_full = d_pair - d_iso                       # -Kp^-1 g_p + Kb^-1 g_i
    d_E = d_pair - d_mix                          # -Kp^-1 (g_p - g_i)
    d_M = d_mix - d_iso                           # -(Kp^-1 - Kb^-1) g_i
    add_err = float((d_full - (d_E + d_M)).norm()
                    / (d_full.norm() + 1e-300))
    den = float(torch.sqrt((m_ref * d_iso ** 2).sum())) + 1e-300

    def wn(x):
        return float(torch.sqrt((m_ref * x ** 2).sum()))
    return dict(
        linearized_energy_side=wn(d_E) / den,
        linearized_metric_side=wn(d_M) / den,
        linearized_full_defect=wn(d_full) / den,
        additivity_residual=add_err,
        cg_residuals=[res1, res2, res3])


# --------------------------------------------------------------------

def run_case(n_per, g, sigma_p, sigma_a, do_linearized=True):
    v, (v1, v2), labels = pair_frames(n_per, g)
    delta, tau = compute_recommended_params(v.positions)
    delta, tau = float(delta), float(tau)

    geo = facing_stats(v.positions, v.normals, labels)
    if geo.get("n_facing", 0) < 4:
        return dict(n_per=n_per, gap=g, sigma=sigma_p,
                    angle_sigma=sigma_a, valid=False,
                    invalid_reason="facing set too small", **geo)
    h_scale = geo["h_same_q10"]

    s_pair, st_pair, tp = one_step(v, 2, delta, tau, sigma_p, sigma_a,
                                   h_scale)
    s1, st1, t1 = one_step(v1, 1, delta, tau, sigma_p, sigma_a, h_scale)
    s2, st2, t2 = one_step(v2, 1, delta, tau, sigma_p, sigma_a, h_scale)
    s_iso = torch.cat([s1, s2])

    m_ref = torch.cat([st1.fixed_masses, st2.fixed_masses]).detach()
    m_pair = st_pair.fixed_masses.detach()
    norms = d_super_norms(s_pair, s_iso, m_ref, m_pair)

    q_pair = st_pair.fixed_coherence.detach()
    q_iso = torch.cat([st1.fixed_coherence,
                       st2.fixed_coherence]).detach()
    dq = q_pair - q_iso
    quants = (0.0, 0.5, 0.9, 1.0)

    def qq(x):
        return [float(torch.quantile(x, torch.tensor(qv, dtype=DTYPE)))
                for qv in quants]

    all_valid = tp["solve_valid"] and t1["solve_valid"] \
        and t2["solve_valid"]
    row = dict(
        n_per=n_per, gap=g, sigma=sigma_p, angle_sigma=sigma_a,
        delta=delta, tau=tau,
        eps_bem=st_pair.bem_wasserstein._epsilon_used,
        valid=bool(all_valid and not norms["denominator_too_small"]),
        solve_pair=tp, solve_iso1=t1, solve_iso2=t2,
        **geo, **norms,
        ratios=dict(
            h_face_over_delta=geo["h_face_median"] / delta,
            h_face_over_sigma=geo["h_face_median"] / sigma_p,
            delta_over_gap=delta / g,
            sigma_over_gap=sigma_p / g,
            eps_over_gap=(st_pair.bem_wasserstein._epsilon_used or 0)
            / g,
            max_disp_over_gap=tp["max_disp"] / g),
        q_facing=dict(q_pair=qq(q_pair), q_iso=qq(q_iso),
                      q_diff=qq(dq), q_diff_absmax=float(
                          dq.abs().max())),
        shadow=shadow_adaptive_rank(v, delta, tau, sigma_p, sigma_a),
    )
    if do_linearized:
        row["linearized"] = linearized_decomposition(
            st_pair, st1, st2, v, v1, v2, m_ref, labels)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="g-sweep",
                    choices=["g-sweep", "sigma-sweep", "angle-sweep"])
    ap.add_argument("--n-per", type=int, nargs="+", default=[256])
    ap.add_argument("--gaps", type=float, nargs="+",
                    default=[0.10, 0.08, 0.06, 0.05, 0.04, 0.03])
    ap.add_argument("--control-gap", type=float, default=0.8)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.05, 0.1, 0.125, 0.15, 0.2])
    ap.add_argument("--sigma-p", type=float, default=0.1,
                    help="perimeter_sigma for g-sweep mode (Phase II->IV "
                         "bridge: confirm the INDEPENDENTLY chosen sigma)")
    ap.add_argument("--sigma-a", type=float, default=0.1,
                    help="angle_sigma for g-sweep mode")
    ap.add_argument("--no-linearized", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("results/resolved_regime"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.mode == "g-sweep":
        cases = [(n, g, args.sigma_p, args.sigma_a) for n in args.n_per
                 for g in ([args.control_gap] + args.gaps)]
    elif args.mode == "sigma-sweep":
        cases = [(n, 0.1, sp, 0.1) for n in args.n_per
                 for sp in args.sigmas]
    else:
        cases = [(n, 0.1, 0.1, sa) for n in args.n_per
                 for sa in args.sigmas]

    for n_per, g, sp, sa in cases:
        row = run_case(n_per, g, sp, sa,
                       do_linearized=not args.no_linearized)
        rows.append(row)
        lin = row.get("linearized", {})
        print(f"n={n_per} g={g:.2f} sp={sp:.3f} sa={sa:.3f} "
              f"valid={row.get('valid')} "
              f"D_mref={row.get('d_super_mref')!r:12.12s} "
              f"D_inf={row.get('d_super_inf')!r:12.12s} "
              f"dE={lin.get('linearized_energy_side', float('nan')):.2e} "
              f"dM={lin.get('linearized_metric_side', float('nan')):.2e} "
              f"add={lin.get('additivity_residual', float('nan')):.1e}",
              flush=True)

    path = args.out / f"audit_{args.mode}.json"
    path.write_text(json.dumps(dict(mode=args.mode, rows=rows),
                               indent=1, default=str))
    print(f"raw results -> {path}", flush=True)


if __name__ == "__main__":
    main()
