"""D2b-0: shadow time-level audit of the redistribution rules on
common post-MM annulus geometries (q-dip regime).

Nothing is applied to any trajectory. At exact-ODE annulus snapshots
chosen so the inner-ring coherence minimum hits prescribed targets
(q_min ~ 0.9 / 0.5 / 0.2 -- which on the annulus only occur DEEP inside
the carrier closure layer; R_-/delta is reported next to every row and
support validity is explicitly flagged), ONE corrected MM step produces
the post-MM candidate. On that common geometry the three redistribution
rules run in shadow (production budget: 10 subiterations), separating:

    legacy vs frozen    ONLY the q staleness across the MM step
                        (operator level pinned them bitwise-equal)
    frozen vs substep   the subiteration refresh bundle

Recorded per snapshot:
    ||q_pre - q_postMM||_inf,  mass-weighted mean |dq|   (staleness)
    ||x_legacy - x_frozen||_inf, ||x_frozen - x_substep||_inf
    per-rule normalized drift (dA/A, dP/P, dH/diam vs the post-MM input)
    substep raw-m vs visible-qm curvature difference (the Algorithm-3
    pairing discriminator that was invisible at q ~ 1)

Usage
-----
    uv run python scripts/experiments/d2b0_timelevel_shadow.py \
        --out results/d2b_redist
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
)
from src.torch.shapes.generator import (
    annulus_point_counts,
    generate_oriented_annulus,
)
from src.torch.solver.mm_step import MMStepper
from src.torch.solver.redistribution_rules import redistribute_with_rule
from src.torch.transport import compute_coherence

sys.path.insert(0, str(Path(__file__).parent))
from d1a_mm_shadow import make_config  # noqa: E402
from d2a_redistribution_calibration import (  # noqa: E402
    curve_metrics,
    sampled_polyline_hausdorff,
)

DT_TENSOR = torch.float64
SIGMA = 0.1
DT = 2e-5
N_OUTER = 128
CSQ = 1.0 - 0.5 ** 2          # conserved R_+^2 - R_-^2
Q_TARGETS = (0.9, 0.5, 0.2)
# production redistribution budget (MMConfig defaults)
N_SUB, LAM, TOL_R, RATIO = 10, 0.01, 1e-4, 0.05


def snapshot(Rm: float, warp: float = 0.0):
    """Exact concentric circles; warp != 0 makes the SAMPLING
    nonuniform (spacing ratio ~2) while the geometry stays the exact
    annulus -- the parametrization perturbation redistribution is FOR.
    Normal convention matches the generator (inner normals into the
    hole, both rings stored CCW)."""
    n_out, n_in = annulus_point_counts(N_OUTER, 1.0, 0.5)
    Rp = math.sqrt(CSQ + Rm * Rm)
    if warp == 0.0:
        v = generate_oriented_annulus(n_out, Rp, Rm, (0.0, 0.0), "cpu",
                                      DT_TENSOR, n_inner=n_in)
        return v, n_out, n_in

    def ring(n, R, flip):
        u = (torch.arange(n, dtype=DT_TENSOR) + 0.5) / n
        t = 2 * math.pi * u + warp * torch.sin(2 * math.pi * u)
        P = R * torch.stack([torch.cos(t), torch.sin(t)], 1)
        ang = t + (math.pi if flip else 0.0)
        return P, ang

    Po, ao = ring(n_out, Rp, False)
    Pi, ai = ring(n_in, Rm, True)
    v = OrientedPointCloudVarifold(
        positions=torch.cat([Po, Pi]), angles=torch.cat([ao, ai]))
    return v, n_out, n_in


def q_of(v, delta, tau):
    m = compute_masses(v.positions, delta, tau)
    return m, compute_coherence(v, m, SIGMA, "wendland_c2")


def pick_radii(delta, tau):
    """Scan R_- and pick the radius whose inner-ring q_min is closest
    to each target."""
    grid = [0.40, 0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08,
            0.06, 0.05, 0.04, 0.03, 0.02]
    rows = []
    for Rm in grid:
        v, n_out, _ = snapshot(Rm)
        _, q = q_of(v, delta, tau)
        rows.append((Rm, q[n_out:].min().item()))
    picks = {}
    for tgt in Q_TARGETS:
        Rm, qm = min(rows, key=lambda r: abs(r[1] - tgt))
        picks[tgt] = dict(R_in=Rm, q_min=qm)
    return rows, picks


def drift(P0, P1, n_out):
    """PER-LOOP metrics (review-proofing: whole-cloud polygon metrics
    on an annulus mix the two loops and the storage-order jump chords
    -- meaningless). Reported per loop, then max over loops."""
    out = {}
    for name, sl in (("outer", slice(0, n_out)),
                     ("inner", slice(n_out, None))):
        A0, A1 = P0[sl], P1[sl]
        b, a = curve_metrics(A0), curve_metrics(A1)
        diam = torch.cdist(A0, A0).max().item()
        out[name] = dict(
            dA_rel=abs(a["area"] - b["area"]) / abs(b["area"]),
            dP_rel=abs(a["perimeter"] - b["perimeter"]) / b["perimeter"],
            dH_over_diam=sampled_polyline_hausdorff(A0, A1) / diam,
            cv_l_before=b["cv_l"], cv_l_after=a["cv_l"])
    out["max_dP_rel"] = max(out["outer"]["dP_rel"],
                            out["inner"]["dP_rel"])
    out["max_dH_over_diam"] = max(out["outer"]["dH_over_diam"],
                                  out["inner"]["dH_over_diam"])
    return out


def _loop_h(P, n_out):
    hs = {}
    for name, sl in (("outer", slice(0, n_out)),
                     ("inner", slice(n_out, None))):
        Pl = P[sl]
        segs = (Pl.roll(-1, 0) - Pl).norm(dim=1)
        hs[name] = dict(h_min=segs.min().item(),
                        h_median=segs.median().item())
    return hs


def _classify(P, ang, n_out, delta, tau, provenance):
    """Review D2b-0.1 item 5/6: POSITION-only validity and ORIENTATION
    consistency are separate columns. Intermediate legacy checkpoints
    intentionally carry frozen angles, so an orientation failure there
    means 'the frozen-angle oriented representation became
    inconsistent', not necessarily 'the position polygon degenerated'."""
    from src.torch.diagnostics import classify_support
    v = OrientedPointCloudVarifold(positions=P, angles=ang)
    loops = [torch.arange(n_out), torch.arange(n_out, P.shape[0])]
    segs = (P.roll(-1, 0) - P).norm(dim=1)
    try:
        rep = classify_support(v, loops, delta=delta, sigma=SIGMA,
                               h=segs.median().item(), mass_tau=tau,
                               provenance=provenance)
        crossings = rep.self_intersections + rep.cross_intersections
        pos_status = ("self_intersecting" if crossings > 0
                      else ("open" if rep.open_endpoints > 0
                            else "valid_positions"))
        return dict(geometry_status=rep.geometry_status, state=rep.state,
                    position_geometry_status=pos_status,
                    orientation_status=("consistent"
                                        if rep.orientation_consistent
                                        else "inconsistent"),
                    self_intersections=rep.self_intersections,
                    orientation_consistent=rep.orientation_consistent)
    except Exception as e:
        return dict(geometry_status="unknown",
                    position_geometry_status="unknown",
                    orientation_status="unknown",
                    error=f"{type(e).__name__}: {e}")


def _shadow_block(P0, ang0, q_stale, delta, tau, n_out, tag) -> dict:
    """The four shadow rules + validity classification on ONE common
    geometry (pre-MM exact snapshot or post-MM candidate)."""
    kw = dict(delta=delta, kernel="wendland_c2", n_iters=N_SUB,
              step_size=LAM, tol=TOL_R, max_disp_ratio=RATIO,
              mass_tau=tau, delta_redist=0.5 * delta)
    loops = [torch.arange(n_out), torch.arange(n_out, P0.shape[0])]
    res = {}
    res["legacy_hybrid"] = redistribute_with_rule(
        P0, ang0, "legacy_hybrid", coherence=q_stale, loops=loops, **kw)
    res["post_mm_frozen"] = redistribute_with_rule(
        P0, ang0, "post_mm_frozen", sigma=SIGMA, loops=loops, **kw)
    res["substep_refreshed"] = redistribute_with_rule(
        P0, ang0, "substep_refreshed", sigma=SIGMA, loops=loops, **kw)
    res["substep_visible_qm"] = redistribute_with_rule(
        P0, ang0, "substep_refreshed", sigma=SIGMA,
        curvature_visible_qm=True, loops=loops, **kw)

    out = dict(tag=tag)
    hs = _loop_h(P0, n_out)
    Rm_eff = P0[n_out:].norm(dim=1).mean().item()
    for name, (P1, A1, info) in res.items():
        d = drift(P0, P1, n_out)
        d["validity"] = _classify(P1, A1, n_out, delta, tau,
                                  f"shadow_{tag}")
        tr = info["trace"]
        if tr:
            d["clip_active_iters"] = sum(t["clip_active"] for t in tr)
            d["by_loop_first_iter"] = tr[0].get("by_loop")
            d["argmax_loop_first_iter"] = tr[0].get("argmax_loop")
        out[f"drift_{name}"] = d
    # legacy: first invalid subiteration (checkpoint every substep)
    Pl_, Al_, info_l = redistribute_with_rule(
        P0, ang0, "legacy_hybrid", coherence=q_stale,
        checkpoint_every=1, **kw)
    first_bad = None
    for it, Pc, Ac in info_l["checkpoints"]:
        c = _classify(Pc, Ac, n_out, delta, tau, "shadow_scan")
        if c["geometry_status"] != "valid_closed":
            first_bad = dict(iteration=it, **c)
            break
    out["legacy_first_invalid_subiteration"] = first_bad

    x = {k: v[0] for k, v in res.items()}
    dlf = (x["legacy_hybrid"] - x["post_mm_frozen"]).norm(dim=1).max() \
        .item()
    out["dx_legacy_vs_frozen_inf"] = dlf
    out["dx_legacy_vs_frozen_over_R_in"] = dlf / Rm_eff
    out["dx_legacy_vs_frozen_over_h_min_inner"] = \
        dlf / hs["inner"]["h_min"]
    out["dx_legacy_vs_frozen_over_h_med_inner"] = \
        dlf / hs["inner"]["h_median"]
    out["dx_frozen_vs_substep_inf"] = (
        (x["post_mm_frozen"] - x["substep_refreshed"])
        .norm(dim=1).max().item())
    out["dx_substep_rawm_vs_visibleqm_inf"] = (
        (x["substep_refreshed"] - x["substep_visible_qm"])
        .norm(dim=1).max().item())
    a = {k: v[1] for k, v in res.items()}
    out["dang_substep_rawm_vs_visibleqm_inf"] = (
        (a["substep_refreshed"] - a["substep_visible_qm"])
        .abs().max().item())

    # counterfactual controls (cause attribution, review D2b-0.1 item
    # 3): small-step (clip inactive) and pointwise local-h cap
    out["controls"] = {}
    for ctl, extra in (
            ("small_step_lam0.001", dict(step_size=0.001)),
            ("local_knn_cap", dict(cap_mode="local_knn"))):
        row = {}
        for rule, rkw in (("legacy_hybrid", dict(coherence=q_stale)),
                          ("substep_refreshed", dict(sigma=SIGMA))):
            kw_c = dict(kw, **extra)
            P1, A1, info = redistribute_with_rule(
                P0, ang0, rule, **rkw, **kw_c)
            d = drift(P0, P1, n_out)
            d["clip_active_iters"] = sum(
                t["clip_active"] for t in info["trace"])
            row[rule] = d
        out["controls"][ctl] = row
    return out


def audit_snapshot(Rm: float, q_min_uniform: float, delta, tau,
                   warp: float = 0.35) -> dict:
    v, n_out, n_in = snapshot(Rm, warp=warp)
    m_pre_kde, q_warped_pre = q_of(v, delta, tau)
    hs_pre = _loop_h(v.positions, n_out)
    stepper = MMStepper(make_config(DT, delta, tau))
    try:
        r = stepper.step(v)
    except Exception as e:
        return dict(R_in=Rm, mm_step_error=f"{type(e).__name__}: {e}")
    post = r.varifold
    q_pre = stepper.fixed_coherence
    m_post, q_post = q_of(post, delta, tau)
    eps_bem = getattr(stepper.bem_wasserstein, "_epsilon_used", None)
    max_s = r.displacements.abs().max().item()

    def _stale(sl):
        dq = (q_post - q_pre).abs()[sl]
        m = m_post[sl]
        return dict(inf=dq.max().item(),
                    mass_weighted=((m * dq).sum() / m.sum()).item())

    out = dict(
        R_in=Rm, R_in_over_delta=Rm / delta,
        R_in_over_sigma=Rm / SIGMA,
        support_note=("inside carrier closure layer (R < 1.2 delta): "
                      "ALGORITHMIC audit data, not a resolved-regime "
                      "statement" if Rm < 1.2 * delta else "resolved"),
        # review item 1: uniform vs warped q are DIFFERENT quantities --
        # the sampling nonuniformity itself amplifies the q-dip
        q_min_uniform=q_min_uniform,
        q_min_warped_pre=q_warped_pre[n_out:].min().item(),
        q_min_warped_post=q_post[n_out:].min().item(),
        reparam_q_bias_inner=abs(
            q_warped_pre[n_out:].min().item() - q_min_uniform),
        # review item 4: MM step validity on THIS snapshot
        eta_step=max_s / Rm,
        max_s_over_h_inner=max_s / hs_pre["inner"]["h_min"],
        eps_bem=eps_bem,
        R_in_over_eps_bem=(Rm / eps_bem if eps_bem else None),
        eps_bem_over_h_inner=(eps_bem / hs_pre["inner"]["h_min"]
                              if eps_bem else None),
        support_pre=_classify(v.positions, v.angles, n_out, delta, tau,
                              "pre_mm"),
        support_post=_classify(post.positions, post.angles, n_out,
                               delta, tau, "post_mm"),
        relgrad=r.relative_gradient_norm,
        objective_decreased=bool(r.objective_decreased),
        # review item 2: per-loop staleness (the global mass-weighted
        # mean was dominated by the outer ring)
        staleness_inner=_stale(slice(n_out, None)),
        staleness_outer=_stale(slice(0, n_out)),
        h_inner=hs_pre["inner"], h_outer=hs_pre["outer"],
        delta_redist_over_R_in=(0.5 * delta) / Rm,
        max_disp_over_h_min_inner=(RATIO * 0.5 * delta)
        / hs_pre["inner"]["h_min"],
    )
    # review item 4: shadow on the exact PRE-MM snapshot separates MM
    # invalidity from operator failure
    out["shadow_pre_mm"] = _shadow_block(
        v.positions, v.angles, q_warped_pre, delta, tau, n_out,
        "pre_mm_exact")
    out["shadow_post_mm"] = _shadow_block(
        post.positions, post.angles, q_pre, delta, tau, n_out,
        "post_mm_candidate")
    return out


def q_stats_full(v, delta, tau, n_out):
    """Review item 1: quantiles + mass-weighted mean + argmin, not the
    minimum alone (the minimum is one-point-sensitive)."""
    m, q = q_of(v, delta, tau)
    qi, mi = q[n_out:], m[n_out:]
    k = int(qi.argmin().item())
    return dict(
        quantiles=[qi.quantile(x).item() for x in
                   (0.0, 0.1, 0.5, 0.9, 1.0)],
        mass_weighted_mean=((mi * qi).sum() / mi.sum()).item(),
        argmin_local_index=k,
        argmin_theta=math.atan2(v.positions[n_out + k, 1].item(),
                                v.positions[n_out + k, 0].item()))


def reparam_bias_refinement() -> dict:
    """Review item 1 + D2b-1 correction 3: the uniform-to-warped
    coherence bias under n_outer refinement, in TWO distinct modes:

      production_coupled    each N uses compute_recommended_params(N)
                            -- N and the bandwidths move together (what
                            production actually does);
      fixed_delta_sigma     the N=128 delta/tau are reused at N=256 --
                            pure finite-sampling quadrature bias at
                            FIXED physical scales.

    The first run of this function conflated the two (docstring said
    'same PHYSICAL scale' while recomputing the bandwidths per N)."""
    global N_OUTER
    out = {}
    saved = N_OUTER
    N_OUTER = 128
    v0, _, _ = snapshot(0.5)
    delta128, tau128 = compute_recommended_params(v0.positions)
    for mode in ("production_coupled", "fixed_delta_sigma"):
        for n in (128, 256):
            N_OUTER = n
            if mode == "production_coupled":
                v0, _, _ = snapshot(0.5)
                delta, tau = compute_recommended_params(v0.positions)
            else:
                delta, tau = delta128, tau128
            for Rm in (0.06, 0.03, 0.02):
                vu, n_out, _ = snapshot(Rm, warp=0.0)
                vw, _, _ = snapshot(Rm, warp=0.35)
                su = q_stats_full(vu, delta, tau, n_out)
                sw = q_stats_full(vw, delta, tau, n_out)
                out[f"{mode}_n{n}_R{Rm}"] = dict(
                    delta=delta, uniform=su, warped=sw,
                    bias_min=abs(sw["quantiles"][0]
                                 - su["quantiles"][0]),
                    bias_median=abs(sw["quantiles"][2]
                                    - su["quantiles"][2]),
                    bias_mass_weighted=abs(sw["mass_weighted_mean"]
                                           - su["mass_weighted_mean"]))
    N_OUTER = saved
    return out


def cause_separation(Rm: float, delta, tau) -> dict:
    """Review item 4: separate the fixed-tangent axis from the global
    bandwidth axis on the post-MM geometry.

    Axis A: delta_redist/R sweep at MATCHED first-substep max update
            (lambda renormalized per bandwidth so the local step scale
            is identical; q fixed for every run).
    Axis B: tangent policy {fixed (legacy), geometric refresh only,
            full substep} under IDENTICAL fixed q -- 'geometric refresh
            only' rebuilds tangents from the current polygon every
            subiteration with NO angle/curvature machinery.
    """
    from src.torch.solver.redistribute import (
        _compute_density_log_gradient)
    v, n_out, n_in = snapshot(Rm, warp=0.35)
    stepper = MMStepper(make_config(DT, delta, tau))
    r = stepper.step(v)
    post = r.varifold
    q_fix = stepper.fixed_coherence
    P0, ang0 = post.positions, post.angles
    hs = _loop_h(P0, n_out)
    u_target = 0.25 * hs["inner"]["h_min"]   # matched local step scale

    def _lam_for(d_red, u=None):
        g, _ = _compute_density_log_gradient(P0, d_red, "wendland_c2")
        t = torch.stack([-torch.sin(ang0), torch.cos(ang0)], 1)
        w = ((g * t).sum(1, keepdim=True) * t).norm(dim=1).max().item()
        return (u if u is not None else u_target) / w if w > 0 else 0.0

    out = dict(R_in=Rm, u_target=u_target)
    loops = [torch.arange(n_out), torch.arange(n_out, P0.shape[0])]
    base = dict(delta=delta, kernel="wendland_c2", n_iters=N_SUB,
                tol=TOL_R, max_disp_ratio=1e9,   # cap disabled: the
                # matched-lambda normalization IS the step control
                mass_tau=tau, q_override="fixed", coherence=q_fix,
                loops=loops)   # loop-aware geometric tangents (review:
    # the first axis-B run used a global roll over the concatenated
    # loops -- four points got cross-loop neighbours; retracted)

    out["axis_A_bandwidth"] = {}
    for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
        d_red = ratio * Rm
        lam = _lam_for(d_red)
        P1, _, _ = redistribute_with_rule(
            P0, ang0, "legacy_hybrid", delta_redist=d_red,
            step_size=lam, **base)
        d = drift(P0, P1, n_out)
        out["axis_A_bandwidth"][f"ratio{ratio}"] = dict(
            lam=lam, inner_dP_rel=d["inner"]["dP_rel"],
            inner_dH=d["inner"]["dH_over_diam"])

    # axis C (review item 5): the destruction onset as a DIMENSIONLESS
    # local step scale u/h_min at fixed bandwidth, fixed q, fixed
    # (loop-aware-diagnosed) tangent policy = legacy fixed tangents
    out["axis_C_step_scale"] = {}
    d_red_c = 0.5 * delta
    for uh in (0.025, 0.05, 0.1, 0.25, 0.5, 1.0):
        u = uh * hs["inner"]["h_min"]
        lam_c = _lam_for(d_red_c, u)
        P1, _, _ = redistribute_with_rule(
            P0, ang0, "legacy_hybrid", delta_redist=d_red_c,
            step_size=lam_c, **base)
        d = drift(P0, P1, n_out)
        out["axis_C_step_scale"][f"u_over_h{uh}"] = dict(
            inner_dP_rel=d["inner"]["dP_rel"],
            inner_dH=d["inner"]["dH_over_diam"])

    out["axis_B_tangent"] = {}
    d_red = 0.5 * delta
    lam = _lam_for(d_red)
    for label, rule, extra in (
            ("fixed_tangent", "legacy_hybrid", {}),
            # production-compatible permutation-invariant refresh; the
            # PCA bandwidth must be LOCAL (O(h)) -- at delta_redist the
            # neighbourhood covers the whole sub-bandwidth ring and the
            # estimate degrades below fixed tangents (see
            # axis_B_pca_bandwidth in the JSON)
            ("local_pca_refresh", "legacy_hybrid",
             dict(tangent_source="local_pca",
                  pca_delta_tan=3 * hs["inner"]["h_min"])),
            # DIAGNOSTIC ONLY: consumes the generator's cyclic order
            ("ordered_loop_oracle", "legacy_hybrid",
             dict(tangent_source="ordered_loop_oracle")),
            ("full_substep", "substep_refreshed", {})):
        P1, _, _ = redistribute_with_rule(
            P0, ang0, rule, delta_redist=d_red, step_size=lam,
            **base, **extra)
        d = drift(P0, P1, n_out)
        out["axis_B_tangent"][label] = dict(
            inner_dP_rel=d["inner"]["dP_rel"],
            inner_dH=d["inner"]["dH_over_diam"],
            inner_cv_post=d["inner"]["cv_l_after"])
    return out


def velocity_saturation_check(Rm: float, delta, tau) -> dict:
    """Review item 6 (partial): one-step eta_step under real q vs
    unit q. The exact-arclength-carrier leg is NOT run here, so the
    reduced velocity stays 'consistent with carrier/coherence
    saturation (F8)', not confirmed."""
    out = {}
    for label, unit in (("real_q", False), ("unit_q", True)):
        v, n_out, _ = snapshot(Rm, warp=0.35)
        cfg = make_config(DT, delta, tau)
        cfg.use_unit_coherence = unit
        st = MMStepper(cfg)
        r = st.step(v)
        out[label] = dict(
            max_s=r.displacements.abs().max().item(),
            eta_step=r.displacements.abs().max().item() / Rm)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d2b_redist"))
    ap.add_argument("--extras-only", action="store_true",
                    help="run only the D2b-0.1 follow-up diagnostics "
                         "and merge them into the existing JSON")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.extras_only:
        path = args.out / "d2b0_timelevel_shadow.json"
        out = json.loads(path.read_text())
        v0, _, _ = snapshot(0.5)
        delta, tau = compute_recommended_params(v0.positions)
        out["reparam_bias_refinement"] = reparam_bias_refinement()
        print("reparam bias (min-q, mass-weighted):")
        for k, d in out["reparam_bias_refinement"].items():
            print(f"  {k:12s} bias_min={d['bias_min']:.3f} "
                  f"bias_med={d['bias_median']:.4f} "
                  f"bias_mw={d['bias_mass_weighted']:.4f}")
        cs = cause_separation(0.03, delta, tau)
        out["cause_separation_R0.03"] = cs
        print(f"axis A (bandwidth sweep, matched step "
              f"u={cs['u_target']:.2e}):")
        for k, d in cs["axis_A_bandwidth"].items():
            print(f"  {k:10s} lam={d['lam']:.2e} "
                  f"inner dP/P={d['inner_dP_rel']:.3e}")
        print("axis B (tangent policy, fixed q, same lam):")
        for k, d in cs["axis_B_tangent"].items():
            print(f"  {k:24s} inner dP/P={d['inner_dP_rel']:.3e} "
                  f"cv_post={d['inner_cv_post']:.3f}")
        vs = {f"R{Rm}": velocity_saturation_check(Rm, delta, tau)
              for Rm in (0.06, 0.03)}
        out["velocity_saturation_check"] = vs
        for k, d in vs.items():
            print(f"velocity check {k}: real_q eta="
                  f"{d['real_q']['eta_step']:.3f} unit_q eta="
                  f"{d['unit_q']['eta_step']:.3f}")
        path.write_text(json.dumps(out, indent=1))
        print(f"raw results -> {path}")
        return

    v0, _, _ = snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    scan, picks = pick_radii(delta, tau)
    out = dict(meta=dict(n_outer=N_OUTER, dt=DT, sigma=SIGMA,
                         delta=delta, tau=tau, n_sub=N_SUB,
                         q_min_scan=scan),
               picks={str(k): v for k, v in picks.items()},
               snapshots={})
    print("q_min(R_-) scan: " +
          "  ".join(f"{r:.2f}:{q:.3f}" for r, q in scan))
    for tgt, p in picks.items():
        print(f"\n=== unwarped-scan target q_min ~ {tgt}: "
              f"R_-={p['R_in']} (uniform q_min {p['q_min']:.3f}) ===")
        s = audit_snapshot(p["R_in"], p["q_min"], delta, tau)
        out["snapshots"][str(tgt)] = s
        if "mm_step_error" in s:
            print(f"  MM step FAILED: {s['mm_step_error']}")
            continue
        print(f"  R/delta={s['R_in_over_delta']:.2f}  q_min: uniform "
              f"{s['q_min_uniform']:.3f} / warped-pre "
              f"{s['q_min_warped_pre']:.3f} / warped-post "
              f"{s['q_min_warped_post']:.3f}")
        print(f"  MM validity: eta_step={s['eta_step']:.3f} "
              f"max|s|/h_in={s['max_s_over_h_inner']:.2f} "
              f"R/eps_bem={s['R_in_over_eps_bem']:.2f} "
              f"support_post={s['support_post']['geometry_status']}")
        print(f"  staleness inner inf={s['staleness_inner']['inf']:.3e} "
              f"outer inf={s['staleness_outer']['inf']:.3e}   "
              f"d_redist/R={s['delta_redist_over_R_in']:.2f} "
              f"max_disp/h_in={s['max_disp_over_h_min_inner']:.2f}")
        for tag in ("shadow_pre_mm", "shadow_post_mm"):
            b = s[tag]
            print(f"  [{b['tag']}]")
            print(f"    |x_leg-x_frz|={b['dx_legacy_vs_frozen_inf']:.2e}"
                  f" (/h_min_in="
                  f"{b['dx_legacy_vs_frozen_over_h_min_inner']:.2f}, /R="
                  f"{b['dx_legacy_vs_frozen_over_R_in']:.3f})  "
                  f"|x_frz-x_sub|={b['dx_frozen_vs_substep_inf']:.2e}  "
                  f"dang_qm={b['dang_substep_rawm_vs_visibleqm_inf']:.1e}")
            for name in ("legacy_hybrid", "substep_refreshed"):
                d = b[f"drift_{name}"]
                print(f"    {name:18s} max dP/P={d['max_dP_rel']:.3e} "
                      f"inner cv {d['inner']['cv_l_before']:.3f}->"
                      f"{d['inner']['cv_l_after']:.3f} "
                      f"validity={d['validity']['geometry_status']} "
                      f"clip_iters={d.get('clip_active_iters')}")
            fb = b["legacy_first_invalid_subiteration"]
            print(f"    legacy first invalid subiter: "
                  f"{fb if fb else 'none (stayed valid_closed)'}")
            for ctl, row in b["controls"].items():
                lg, sb = row["legacy_hybrid"], row["substep_refreshed"]
                print(f"    control {ctl:18s} legacy dP/P="
                      f"{lg['max_dP_rel']:.2e} (clip "
                      f"{lg['clip_active_iters']}) | substep dP/P="
                      f"{sb['max_dP_rel']:.2e}", flush=True)

    path = args.out / "d2b0_timelevel_shadow.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nraw results -> {path}")


if __name__ == "__main__":
    main()
