"""P1-3 R1: refinement comparison for the mod persistent-thinning
candidate (reviewer STOP 2 verdict §3-§4, §9) + healing controls.

Arms: baseline (h0, 768; run p13_mod, R0 telemetry reconstructed
from the stored states every 25 steps), h0/2 (768; p13r_mod_h2),
H = 1024 (p13r_mod_H1024).

Per arm:
  * T_mon windows of w_strip (falls back to w_fit for the baseline
    JSON, which predates the strip estimator; the two agree to 1e-4
    on the symmetric neck), self-consistency: wdot < 0 on all
    post-startup windows and last/middle |wdot| ratio >= 0.5.
  * wdot(w): per-window (mean w, slope) -> piecewise-linear
    interpolant; cross-arm comparison on the COMMON width interval
    I_w = intersection of the post-startup window ranges:
        ||wdot_ref - wdot_768||_inf(I_w) /
        max(||wdot_768||_inf(I_w), v_floor) <= 0.25
    (v_floor = the baseline quasi-steady floor).
  * telemetry: P_eff_l/r(t), delta_split(t), J_N = -(1/2) dA_N/dt
    per window, x_plateau drift.
Healing controls: cassini / low at H = 1024 vs 768 (sign pattern of
the window slopes must be reproduced).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fit_p13 import T_MON, windows  # noqa: E402

R = Path("results/pinchoff")
H0 = 2.0 * math.pi / 256.0


def series(tag, key_pref=("w_strip", "w_fit")):
    d = json.load(open(R / f"dumbbell_{tag}.json"))
    s = d["series"]
    key = next(k for k in key_pref if s[0].get(k) is not None)
    ok = [r for r in s if r.get(key) is not None]
    t = np.array([r["t"] for r in ok])
    w = np.array([r[key] for r in ok])
    return d, t, w, key


def shadow_from_states(tag, h, ell_N):
    """R0 telemetry for a pre-R0 run from its stored states."""
    import two_ellipses_benchmark as teb
    from neck_telemetry import (neck_control_area, neck_width_strips,
                                p_eff_dynamic)
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.oriented_varifold.mass import (
        compute_recommended_params)
    st = torch.load(R / f"dumbbell_{tag}_states.pt")
    dt = json.load(open(R / f"dumbbell_{tag}.json"))["meta"]["dt"]
    rows = []
    for k in sorted(st):
        p = st[k]["positions"]
        nor = OrientedPointCloudVarifold(
            positions=p, angles=st[k]["angles"]).normals
        delta, tau = map(float, compute_recommended_params(p))
        m = teb.resolve_m(p, nor, delta, tau)
        ns = neck_width_strips(p, nor, h)
        xc = ns.get("x_plateau", 0.0)
        pe = p_eff_dynamic(p, nor, h, ns["w_strip"], xc)
        rows.append(dict(step=k, t=(k + 1) * dt, w_strip=ns["w_strip"],
                         x_plateau=xc, strip_status=ns["status"],
                         A_N=neck_control_area(p, nor, m, ell_N, xc),
                         **pe))
    return rows


def arm_summary(name, d, t, w, tele_rows):
    ww = windows(t, w)
    post = ww[1:]
    sl = np.array([x["slope"] for x in post])
    n3 = max(1, len(post) // 3)
    mid = float(np.abs(sl[n3:2 * n3]).mean())
    last = float(np.abs(sl[-n3:]).mean())
    ses = np.array([x["slope_se"] for x in post])
    floor = max(0.01, 2 * float(np.median(ses)))
    self_ok = bool(np.all(sl < 0) and last >= 0.5 * mid)
    # telemetry windows
    tt = np.array([r["t"] for r in tele_rows])
    AN = np.array([r["A_N"] for r in tele_rows])
    pel = np.array([np.nan if r.get("p_eff_l") is None else r["p_eff_l"]
                    for r in tele_rows])
    per = np.array([np.nan if r.get("p_eff_r") is None else r["p_eff_r"]
                    for r in tele_rows])
    xp = np.array([r["x_plateau"] for r in tele_rows])
    tele = []
    for x in ww:
        lo, hi = x["k"] * T_MON, (x["k"] + 1) * T_MON
        m = (tt > lo) & (tt <= hi + 1e-12)
        if m.sum() >= 2:
            c = np.polyfit(tt[m], AN[m], 1)
            J_N = -0.5 * float(c[0])
        else:
            J_N = None
        tele.append(dict(k=x["k"], J_N=J_N,
                         p_eff_l=(float(np.nanmean(pel[m]))
                                  if m.any() else None),
                         p_eff_r=(float(np.nanmean(per[m]))
                                  if m.any() else None),
                         x_plateau=(float(xp[m].mean())
                                    if m.any() else None)))
    ds = [r.get("delta_split") for r in d["series"]
          if r.get("delta_split") is not None]
    if not ds:   # pre-R0 JSON: reconstruct from P_poly and A
        ds = [r["P_poly"] - 4 * math.sqrt(math.pi * r["A"] / 2)
              for r in d["series"]]
    gates = {g: float(max(r.get(g, 0.0) for r in d["series"]))
             for g in ("gate_q", "gate_P", "gate_A")}
    return dict(name=name, n_steps=len(d["series"]),
                stop=d["meta"]["stop_reason"], err=d["meta"]["error"],
                h_ref=d["meta"].get("h_ref", H0),
                grid_actual=d["meta"]["grid_actual"],
                N=None, w0=float(w[0]), w_end=float(w[-1]),
                windows=ww, tele=tele, floor=floor,
                mid_abs=mid, last_abs=last, self_consistent=self_ok,
                delta_split_0=float(ds[0]), delta_split_end=float(ds[-1]),
                gate_max=gates)


def wdot_of_w(summary):
    post = summary["windows"][1:]
    wv = np.array([x["mean"] for x in post])
    sv = np.array([x["slope"] for x in post])
    o = np.argsort(wv)
    return wv[o], sv[o]


def compare(base, ref):
    wb, sb = wdot_of_w(base)
    wr, sr = wdot_of_w(ref)
    lo, hi = max(wb.min(), wr.min()), min(wb.max(), wr.max())
    if hi <= lo:
        return dict(status="no_common_interval")
    grid = np.linspace(lo, hi, 50)
    fb = np.interp(grid, wb, sb)
    fr = np.interp(grid, wr, sr)
    num = float(np.abs(fr - fb).max())
    den = max(float(np.abs(fb).max()), base["floor"])
    return dict(status="ok", I_w=[float(lo), float(hi)],
                linf_rel=num / den, linf_abs=num,
                pass_25=bool(num / den <= 0.25))


def print_arm(a):
    print(f"[{a['name']}] steps={a['n_steps']} h={a['h_ref']:.5f} "
          f"H={a['grid_actual']} w {a['w0']:.4f}->{a['w_end']:.4f} "
          f"stop={a['stop']} err={a['err']} gates="
          f"{ {k: round(v, 4) for k, v in a['gate_max'].items()} }")
    print(f"   delta_split {a['delta_split_0']:.3f} -> "
          f"{a['delta_split_end']:.3f}; floor={a['floor']:.4f}; "
          f"|wdot| mid/last = {a['mid_abs']:.3f}/{a['last_abs']:.3f} "
          f"(ratio {a['last_abs'] / a['mid_abs']:.2f}); "
          f"self-consistent thinning: {a['self_consistent']}")
    print("   win  <w>     wdot    se     J_N      Peff_l  Peff_r  x_plat")
    for x, te in zip(a["windows"], a["tele"]):
        f = lambda v, s: "   n/a" if v is None else f"{v:{s}}"
        print(f"   {x['k']:3d} {x['mean']:.4f} {x['slope']:+.3f} "
              f"{x['slope_se']:.3f} {f(te['J_N'], '+.4f')} "
              f"{f(te['p_eff_l'], '6.2f')} {f(te['p_eff_r'], '6.2f')} "
              f"{f(te['x_plateau'], '+.3f')}")


def main():
    out = dict(T_mon=T_MON, arms={}, comparisons={}, controls={})
    # --- mod arms ---
    arms = {}
    d, t, w, key = series("p13_mod")
    ell_N = d["meta"]["fixture"]["L"] / 2
    tele = shadow_from_states("p13_mod", H0, ell_N)
    arms["base_768"] = arm_summary("base_768 (shadow tele)", d, t, w, tele)
    for tag, nm in (("p13r_mod_h2", "h/2_768"),
                    ("p13r_mod_H1024", "h0_1024")):
        if (R / f"dumbbell_{tag}.json").exists():
            d, t, w, key = series(tag)
            arms[nm] = arm_summary(nm, d, t, w, d["series"])
    for a in arms.values():
        print_arm(a)
    for nm in ("h/2_768", "h0_1024"):
        if nm in arms:
            c = compare(arms["base_768"], arms[nm])
            out["comparisons"][nm] = c
            print(f"compare base_768 vs {nm}: {c}")
    out["arms"] = arms
    # --- healing controls ---
    for base, ref in (("p13_cassini", "p13r_cassini_H1024"),
                      ("p13_low", "p13r_low_H1024")):
        if not (R / f"dumbbell_{ref}.json").exists():
            continue
        _, tb, wb, _ = series(base, ("w_fit",))
        _, tr, wr, _ = series(ref)
        sb = [x["slope"] for x in windows(tb, wb)[1:]]
        sr = [x["slope"] for x in windows(tr, wr)[1:]]
        signs_same = bool(np.all(np.sign(sb) == np.sign(sr)))
        out["controls"][ref] = dict(slopes_768=sb, slopes_1024=sr,
                                    sign_pattern_reproduced=signs_same,
                                    w_end_768=float(wb[-1]),
                                    w_end_1024=float(wr[-1]))
        print(f"control {ref}: sign pattern reproduced={signs_same}; "
              f"w_end 768 {wb[-1]:.4f} vs 1024 {wr[-1]:.4f}")
    # --- asym re-run with the amended estimator ---
    if (R / "dumbbell_p13r_asym.json").exists():
        d, t, w, key = series("p13r_asym")
        a = arm_summary("asym_r0 (strip estimator)", d, t, w, d["series"])
        amb = sum(1 for r in d["series"]
                  if r.get("strip_status") == "ambiguous")
        a["n_ambiguous"] = amb
        print_arm(a)
        print(f"   ambiguous steps: {amb}")
        out["asym"] = a
    (R / "p13r_fit_summary.json").write_text(
        json.dumps(out, indent=1, default=str))
    print("wrote", R / "p13r_fit_summary.json")


if __name__ == "__main__":
    main()
