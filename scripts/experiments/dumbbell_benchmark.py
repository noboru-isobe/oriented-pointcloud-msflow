"""Phase P dynamic driver: dumbbell reconnaissance and linear rate
pins under the production MM stack (observation only, no events).

Config: WB (self_renormalized; a single loop has q^WB = q^full),
order-free current_centered first-moment rows, Phase P grid domain
box (-3,3)^2 with H = 1.5 x grid_base so dx and eps_fill/dx are
bit-identical to the production 768/1024 grids.

Modes:
  --fixture capsule|cassini      reconnaissance run (P1-3 pilots)
  --seed-mode varicose|sinuous   P1-2 rate pin: seed eps cos(kx) on
      the straight neck walls with a smooth cutoff vanishing near
      the blends; a uniform normal compensation zeroes the linearized
      area change (k = 0 / area / translation projection); per-step
      amplitude telemetry = cos(kx)-projection of the wall deviation.

Telemetry per step: order-free w_point / w_fit (neck_telemetry),
lobe areas A_L/A_R + drainage flux, area/moment conservation
(current form), P, max_s; rate mode adds amp_varicose/amp_sinuous.
P1-3 live gates (reviewer-approved protocol, P0-c' thresholds 5%):
  gate_q     = |1 - qbar_neck|  (coherence on the neck ROI |x| <= L/4)
  gate_P     = |P_sigma / P_geom - 1|  (P_sigma = sum m q recomputed on
               the committed cloud; P_geom = ordered-polygon perimeter
               -- an ordered-shadow quantity, validity flagged by
               consecutive-gap check order_ok)
  gate_A     = |A_grid / A - 1|  (phase_grid_volume of the step's grid
               snapshot vs the current-form area)

R0 (STOP 2 verdict): --h-ref threads the point spacing through the
generator, both neck estimators, ROI/order guards and w* =
max(3h, 0.75 sigma) (computed). Added telemetry: state-functional
strip estimator (w_strip, x_waist, x_plateau, ambiguity), dynamic
CENV data P_eff_l/r(t) with ell_l/r, neck control-volume area A_N
(|x - x_plateau| < ell_N, plus A_N0 at the fixed centre) and the
symmetric split budget delta_split = P_poly - 4 sqrt(pi A/2).

Stops: conservation sanity |dA|/A > 5e-3, w_fit or w_strip < w*
(admissibility floor, pre-registered), any live gate > 5% on 5
consecutive steps (admissibility-window exit, pre-registered
thresholds), self-intersection (every 5 steps), --steps cap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import two_ellipses_benchmark as teb  # noqa: E402
from neck_telemetry import (  # noqa: E402
    delta_split, lobe_areas, neck_control_area, neck_width,
    neck_width_strips, p_eff_dynamic,
)
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.shapes.dumbbell import (  # noqa: E402
    build_capsule_dumbbell,
    build_cassini,
)
from src.torch.solver.arc_splice import _has_self_intersection  # noqa
from src.torch.solver.mm_solver import MMSolver  # noqa: E402
from src.torch.transport.bem_wasserstein import compute_coherence  # noqa

H_REF = teb.H_REF
SIGMA = teb.SIGMA
EPS_FILL = 0.04
A0 = math.pi * 0.8
def w_star(h: float, sigma: float = SIGMA) -> float:
    """Pre-registered admissibility floor w* = max(3h, 0.75 sigma)
    (P0-c'); computed, not hard-coded, so an h refinement keeps the
    semantics (the sigma term dominates for h <= 0.025)."""
    return max(3.0 * h, 0.75 * sigma)
OUT = Path("results/pinchoff")


def phase_p_cfg(grid_base: int, dt: float, delta, tau,
                redistribute=True):
    from src.torch.transport.grid_wasserstein import GridMetricConfig
    from src.torch.transport.phase_grid import PhaseGridConfig
    cfg = teb.build_config(teb.production_args(
        grid=grid_base, redist_monotone=True, quotient_mode="off",
        no_redistribute=not redistribute), delta, tau)
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(int(grid_base * 1.5), int(grid_base * 1.5)),
        box_min=(-3.0, -3.0), box_max=(3.0, 3.0),
        fill_epsilon=EPS_FILL,
        support_threshold=3e-3, projection_rel_tol=5e-2),
        compatibility_components="conservative_sweep")
    cfg.time_step = dt
    cfg.grid_bulk_first_moment_rows = True
    cfg.grid_bulk_first_moment_rows_form = "current_centered"
    return cfg


def seed_mode(pos, ang, w, L_str, k, eps, mode, blend=0.0,
              comp="none"):
    """P1b-c0 geometrically consistent seed: perturb the straight
    neck walls as graphs y -> y + d_s(x) with d_s = eps cos(kx) chi(x)
    (sinuous) / -sign(y) eps cos(kx) chi(x) (varicose), where chi is a
    C^2 compactly supported quintic taper (chi = chi' = chi'' = 0 at
    the support edge x1 = 0.35 L -- MS takes curvature as Dirichlet
    data, so at least C^2 contact with the unperturbed wall is
    required; the earlier exp(-(2x/L)^8) factor combined with a
    Boolean mask at 0.35 L left an O(eps) hard edge, chi(x1) = 0.944).

    Normals are set EXACTLY to the perturbed graph's outward normals:
    n_s = (-s d_s', s)/sqrt(1 + d_s'^2) for the wall at y = s w/2,
    i.e. dtheta_s = atan(d_s') with the SAME sign on both walls
    (reviewer P1b-c audit: the earlier dtheta = -n_y d' flipped the
    upper-wall sign, an O(eps) branch-parity defect -- sinuous
    positions carried varicose-parity normals and vice versa).

    eps may be NEGATIVE: the centered pair (+eps, -eps) gives the
    tangent response (amp_+ - amp_-)/2, cancelling the background
    and all even-order nonlinearity.

    comp="none" (default): no area compensation -- the centered
    difference removes the even background and the seed's own O(eps)
    area content is part of the tangent being measured. comp="cap"
    reproduces the earlier lobe-cap compensation (reviewer flagged:
    cap displacement forces the neck through the nonlocal harmonic
    response, so it is NOT innocuous; kept only for controls).
    Returns (positions, angles)."""
    x, y = pos[:, 0], pos[:, 1]
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    # C^2 taper: plateau for |x| <= x0, quintic smoothstep
    # S(u) = 10u^3 - 15u^4 + 6u^5 down to 0 at x1 (S' = S'' = 0 at
    # both ends), identically 0 beyond.
    x1 = 0.7 * L_str / 2
    x0 = 0.5 * x1
    u = ((x1 - x.abs()) / (x1 - x0)).clamp(0.0, 1.0)
    chi = u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)
    dchi = 30.0 * (u * (1.0 - u)) ** 2 * \
        (-torch.sign(x) / (x1 - x0))
    on_wall = (x.abs() < x1) & \
        (y.abs() < w / 2 + 2 * H_REF) & (nor[:, 1].abs() > 0.9)
    C = torch.cos(k * x) * chi
    dC = -k * torch.sin(k * x) * chi + torch.cos(k * x) * dchi
    if mode == "varicose":
        sgn_wall = -torch.sign(y)               # both walls inward
    else:                                       # sinuous
        sgn_wall = torch.ones_like(y)
    dy = eps * sgn_wall * C
    ddy_dx = eps * sgn_wall * dC
    wall_f = on_wall.to(pos.dtype)
    pos2 = pos.clone()
    pos2[:, 1] = pos[:, 1] + dy * wall_f
    ang2 = ang + torch.atan(ddy_dx) * wall_f
    if comp == "cap":
        m = (pos.roll(-1, 0) - pos.roll(1, 0)).norm(dim=1) / 2
        dA = float((m * dy * wall_f * nor[:, 1]).sum())
        on_cap = x.abs() > L_str / 2 + blend
        c = -dA / float(m[on_cap].sum())
        pos2 = pos2 + c * nor * on_cap[:, None].to(pos.dtype)
    return pos2, ang2


def mode_amplitude(pos, w0, L_str, k):
    """cos(kx)-projection of the wall deviation over the straight
    region (varicose amp = symmetric inward component, sinuous amp =
    common transverse component)."""
    x, y = pos[:, 0], pos[:, 1]
    sel_t = (x.abs() < 0.7 * L_str / 2) & (y > 0) & (y < w0)
    sel_b = (x.abs() < 0.7 * L_str / 2) & (y < 0) & (y > -w0)
    if int(sel_t.sum()) < 5 or int(sel_b.sum()) < 5:
        return None, None

    def proj(sel, sign):
        dev = sign * y[sel] - w0 / 2
        cc = torch.cos(k * x[sel])
        return float((dev * cc).sum() / (cc * cc).sum())
    a_top = proj(sel_t, +1.0)
    a_bot = proj(sel_b, -1.0)
    # varicose: walls move oppositely in y => dev_top = dev_bot = -amp
    amp_v = -(a_top + a_bot) / 2
    # sinuous: dev_top = +amp, dev_bot = -amp
    amp_s = (a_top - a_bot) / 2
    return amp_v, amp_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="capsule",
                    choices=["capsule", "cassini"])
    ap.add_argument("--R-l", type=float, default=0.55)
    ap.add_argument("--R-r", type=float, default=0.55)
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--w", type=float, default=0.29)
    ap.add_argument("--cassini-ab", type=float, default=0.998)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--grid", type=int, default=768,
                    help="grid_base (actual H = 1.5x, box (-3,3))")
    ap.add_argument("--seed-mode", default=None,
                    choices=[None, "varicose", "sinuous"])
    ap.add_argument("--seed-k", type=float, default=1.8)
    ap.add_argument("--seed-eps", type=float, default=0.004,
                    help="may be negative (centered-pair minus arm)")
    ap.add_argument("--seed-comp", default="none",
                    choices=["none", "cap"])
    ap.add_argument("--no-redistribute", action="store_true")
    ap.add_argument("--no-area-norm", action="store_true",
                    help="rate-pin fixtures: keep raw dimensions "
                         "(the linear test needs no A0 comparability "
                         "and normalization can push long necks out "
                         "of the box)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--h-ref", type=float, default=H_REF,
                    help="point spacing: generator sampling, neck "
                         "estimators, ROI/order guards, w* (R0)")
    ap.add_argument("--stop-events", action="store_true",
                    help="P1-4 long-neck persistence test: stop at "
                         "the FIRST of healing event / budget closure "
                         "/ deep-thinning checkpoint (2 w*) / w* "
                         "(reviewer STOP 3 protocol)")
    ap.add_argument("--gate-tol", type=float, default=0.05,
                    help="live admissibility gates (P0-c' thresholds)")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    h_ref = args.h_ref
    W_STAR = w_star(h_ref)
    if args.fixture == "capsule":
        c = build_capsule_dumbbell(
            R_l=args.R_l, R_r=args.R_r, L=args.L, w=args.w, h=h_ref,
            area_target=None if args.no_area_norm else A0)
        L_str = c["L"]
        fix_meta = dict(kind="capsule", R_l=args.R_l, R_r=args.R_r,
                        L=c["L"], w=c["w"], blend=c["blend"],
                        p_eff_left=c["p_eff_left"],
                        p_eff_right=c["p_eff_right"],
                        scale=c["scale"])
    else:
        c = build_cassini(a=args.cassini_ab, b=1.0, h=h_ref,
                          area_target=None if args.no_area_norm
                          else A0)
        L_str = c["w"] * 2          # ROI scale only
        fix_meta = dict(kind="cassini", ab=args.cassini_ab, w=c["w"],
                        scale=c["scale"])
    fix_meta.update(area=c["area"], perimeter=c["perimeter"])
    pos, ang = c["positions"], c["angles"]
    w0 = c["w"]
    # neck control volume half-length for A_N (fixed per run): the
    # straight-neck half length (capsule) / neck half-width (cassini)
    ELL_N = L_str / 2
    if args.seed_mode:
        pos, ang = seed_mode(pos, ang, w0, L_str, args.seed_k,
                             args.seed_eps, args.seed_mode,
                             blend=float(c.get("blend", 0.0) or 0.0),
                             comp=args.seed_comp)
    v0 = OrientedPointCloudVarifold(positions=pos.clone(),
                                    angles=ang.clone())
    delta, tau = map(float, compute_recommended_params(pos))
    cfg = phase_p_cfg(args.grid, args.dt, delta, tau,
                      redistribute=not args.no_redistribute)
    A_init = None
    ev_state = {"prev_P": None}
    gate_fail = []
    series, states = [], {}
    t0 = time.time()
    out_json = OUT / f"dumbbell{args.tag}.json"
    meta = dict(fixture=fix_meta, dt=args.dt, grid_base=args.grid,
                grid_actual=int(args.grid * 1.5),
                box=[-3.0, 3.0], dx=6.0 / (args.grid * 1.5),
                eps_fill_over_dx=EPS_FILL / (6.0 / (args.grid * 1.5)),
                w_star=W_STAR, seed=dict(mode=args.seed_mode,
                                         k=args.seed_k,
                                         eps=args.seed_eps,
                                         comp=args.seed_comp,
                                         geometric=True,
                                         version="c0"),
                redistribute=not args.no_redistribute,
                gate_tol=args.gate_tol, sigma=SIGMA, h_ref=h_ref,
                ell_N=ELL_N, telemetry_version="r0",
                stop_reason=None, error=None)

    def dump(err=None):
        meta["error"] = err
        tmp = out_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(meta=meta, series=series),
                                  indent=1, default=str))
        tmp.replace(out_json)
        torch.save(states, OUT / f"dumbbell{args.tag}_states.pt")

    def cb(step, res):
        nonlocal A_init
        vv = (res.committed_varifold
              if res.committed_varifold is not None else res.varifold)
        p = vv.positions
        nor = vv.normals
        m = teb.resolve_m(p, nor, delta, tau)
        A_c = float(0.5 * (m * (p * nor).sum(-1)).sum())
        if A_init is None:
            A_init = A_c
        nw = neck_width(p, nor, m, h_ref)
        ns = neck_width_strips(p, nor, h_ref)
        x_c = ns.get("x_plateau", 0.0) if ns.get("status") == "ok" \
            else 0.0
        w_now = ns.get("w_strip") if ns.get("status") == "ok" \
            else nw.get("w_fit")
        pe = (p_eff_dynamic(p, nor, h_ref, w_now, x_c)
              if w_now is not None else {})
        la = lobe_areas(p, nor, m)
        amp_v, amp_s = mode_amplitude(p, w0, L_str, args.seed_k)
        # --- live gates (P1-3 protocol) ---
        q = compute_coherence(vv, m, SIGMA, "wendland_c2")
        neck_roi = p[:, 0].abs() <= max(L_str / 4, 2 * h_ref)
        qbar = float(q[neck_roi].mean()) if bool(neck_roi.any()) \
            else float("nan")
        P_sigma = float((m * q).sum())
        seg = (p.roll(-1, 0) - p).norm(dim=1)
        order_ok = bool(float(seg.max()) <= 3.0 * h_ref)
        P_poly = float(seg.sum())
        sn = getattr(sv._stepper, "_last_grid_snapshot", None)
        A_grid = (float(sn.phase_grid_volume)
                  if sn is not None else float("nan"))
        gates = dict(gate_q=abs(1.0 - qbar),
                     gate_P=abs(P_sigma / P_poly - 1.0),
                     gate_A=abs(A_grid / A_c - 1.0))
        rec = dict(step=step, t=(step + 1) * args.dt,
                   A=A_c, dA_rel=(A_c - A_init) / A_init,
                   P=float(res.perimeter), W=float(res.wasserstein),
                   n_iter=int(res.n_iter),
                   max_s=float(res.displacements.abs().max()),
                   w_status=nw.get("status"),
                   w_point=nw.get("w_point"), w_fit=nw.get("w_fit"),
                   A_L=la["A_L"], A_R=la["A_R"],
                   amp_varicose=amp_v, amp_sinuous=amp_s,
                   qbar_neck=qbar, P_sigma=P_sigma, P_poly=P_poly,
                   P_mass=float(m.sum()), order_ok=order_ok,
                   A_grid=A_grid,
                   w_strip=ns.get("w_strip"), x_waist=ns.get("x_waist"),
                   x_plateau=ns.get("x_plateau"),
                   strip_status=ns.get("status"),
                   n_strips=ns.get("n_strips"),
                   strip_runner_up=ns.get("runner_up"),
                   A_N=neck_control_area(p, nor, m, ELL_N, x_c),
                   A_N0=neck_control_area(p, nor, m, ELL_N, 0.0),
                   delta_split=delta_split(P_poly, A_c),
                   eps_sym=abs(la["A_L"] - la["A_R"]) / A_c,
                   **pe, **gates)
        series.append(rec)
        bad = [k for k, v in gates.items()
               if not (v <= args.gate_tol)]
        gate_fail.append(bad)
        gate_fail[:] = gate_fail[-5:]
        if step % 25 == 0:
            states[step] = dict(positions=p.clone(),
                                angles=vv.angles.clone())
            print(f"step {step:4d}: w_fit "
                  f"{nw.get('w_fit') if nw.get('w_fit') is None else round(nw['w_fit'], 5)}"
                  f" w_strip {ns.get('w_strip') if ns.get('w_strip') is None else round(ns['w_strip'], 5)}"
                  f"@{ns.get('x_plateau') if ns.get('x_plateau') is None else round(ns['x_plateau'], 3)}"
                  f" Peff {pe.get('p_eff_l') if pe.get('p_eff_l') is None else round(pe['p_eff_l'], 2)}/"
                  f"{pe.get('p_eff_r') if pe.get('p_eff_r') is None else round(pe['p_eff_r'], 2)}"
                  f" dsplit {rec['delta_split']:.3f}"
                  f" dA {rec['dA_rel']:+.2e} gates q {gates['gate_q']:.3f}"
                  f" P {gates['gate_P']:.3f} A {gates['gate_A']:.3f}"
                  f" amp_v "
                  f"{amp_v if amp_v is None else f'{amp_v:+.2e}'} "
                  f"amp_s {amp_s if amp_s is None else f'{amp_s:+.2e}'}"
                  f" ({(time.time() - t0) / (step + 1):.1f}s/step)",
                  flush=True)
            dump()
        stop = None
        if args.stop_events and len(series) > 1:
            T_MON = 2e-3
            tail = [r for r in series
                    if r["t"] > rec["t"] - T_MON
                    and r.get("w_strip") is not None]
            if len(tail) >= 20 and rec["t"] > 2 * T_MON:
                import numpy as _np
                tt = _np.array([r["t"] for r in tail])
                wv = _np.array([r["w_strip"] for r in tail])
                av = _np.array([r["A_N"] for r in tail])
                wdot = float(_np.polyfit(tt, wv, 1)[0])
                J_N = -0.5 * float(_np.polyfit(tt, av, 1)[0])
                pes = [0.5 * (r["p_eff_l"] + r["p_eff_r"])
                       for r in tail
                       if r.get("p_eff_l") is not None
                       and r.get("p_eff_r") is not None]
                pe_bar = sum(pes) / len(pes) if pes else None
                ev_state["heal"] = (ev_state.get("heal", 0) + 1) if (
                    pe_bar is not None and pe_bar < 2.0
                    and J_N <= 0.0 and wdot >= 0.0) else 0
                dsn = rec["delta_split"] < 0
                if dsn and ev_state.get("cross_P") is None:
                    ev_state["cross_P"] = rec["P_poly"]
                    ev_state["eps_P_up"] = 0.0
                if ev_state.get("cross_P") is not None:
                    ev_state["eps_P_up"] = max(
                        ev_state["eps_P_up"],
                        rec["P_poly"] - ev_state["prev_P"])
                ev_state["budget"] = (ev_state.get("budget", 0) + 1) \
                    if dsn else 0
                if ev_state["heal"] >= int(T_MON / args.dt):
                    stop_ev = (f"healing event: sustained P_eff "
                               f"{pe_bar:.2f} < 2, J_N {J_N:+.3f} <= 0"
                               f", wdot {wdot:+.3f} >= 0 over T_mon")
                elif (ev_state["budget"] >= int(T_MON / args.dt)
                        and rec["eps_sym"] <= 1e-3
                        and ev_state["eps_P_up"] <= 1e-6):
                    stop_ev = (f"budget closure: delta_split < 0 over "
                               f"T_mon, eps_sym {rec['eps_sym']:.1e}, "
                               f"eps_P_up {ev_state['eps_P_up']:.1e}")
                elif (rec["w_strip"] is not None
                        and rec["w_strip"] <= 2 * W_STAR
                        and pe_bar is not None and pe_bar > 2.0
                        and J_N > 0.0 and rec["delta_split"] > 0):
                    stop_ev = (f"deep-thinning checkpoint: w_strip "
                               f"{rec['w_strip']:.4f} <= 2 w* with "
                               f"P_eff {pe_bar:.2f} > 2, J_N "
                               f"{J_N:+.3f} > 0, delta_split "
                               f"{rec['delta_split']:.3f} > 0")
                else:
                    stop_ev = None
                if stop_ev:
                    meta["stop_reason"] = stop_ev
                    states[step] = dict(positions=p.clone(),
                                        angles=vv.angles.clone())
                    print(f"== stop at {step}: {stop_ev} ==",
                          flush=True)
                    dump()
                    return True
            ev_state["prev_P"] = rec["P_poly"]
        if abs(rec["dA_rel"]) > 5e-3:
            stop = f"area sanity {rec['dA_rel']:.2e}"
        elif nw.get("w_fit") is not None and nw["w_fit"] < W_STAR:
            stop = (f"admissibility floor: w_fit {nw['w_fit']:.4f} < "
                    f"w* {W_STAR:.4f} (resolution stop, pre-registered)")
        elif ns.get("status") == "ok" and ns["w_strip"] < W_STAR:
            stop = (f"admissibility floor: w_strip {ns['w_strip']:.4f} "
                    f"< w* {W_STAR:.4f} (resolution stop, pre-registered)")
        elif len(gate_fail) == 5 and all(gate_fail):
            stop = (f"live gate exit ({sorted(set(sum(gate_fail, [])))}"
                    f" > {args.gate_tol} on 5 consecutive steps, "
                    "pre-registered admissibility window)")
        elif step % 5 == 0 and _has_self_intersection(p):
            stop = "self-intersection"
        if stop:
            meta["stop_reason"] = stop
            states[step] = dict(positions=p.clone(),
                                angles=vv.angles.clone())
            print(f"== stop at {step}: {stop} ==", flush=True)
            dump()
            return True
        return False

    sv = MMSolver(cfg)
    err = None
    try:
        hist = sv.solve(v0, args.steps, callback=cb)
        if getattr(hist, "stop_message", None):
            err = (f"{getattr(hist, 'stop_exception_type', '?')}: "
                   f"{hist.stop_message}")
    except BaseException as ex:                   # noqa: BLE001
        err = f"{type(ex).__name__}: {ex}"
        print(f"    [FAIL-CLOSED] {err}", flush=True)
        if isinstance(ex, (KeyboardInterrupt, SystemExit)):
            dump(err)
            raise
    dump(err)
    print(f"done {len(series)} steps; stop {meta['stop_reason']}; "
          f"err {err}", flush=True)


if __name__ == "__main__":
    main()
