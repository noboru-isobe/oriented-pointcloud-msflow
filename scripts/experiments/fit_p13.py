"""P1-3 pilot classification (pre-registered, docs/pinchoff_study.md
"事前登録判定") on the T_mon stage.

Per pilot JSON (results/pinchoff/dumbbell_p13_<name>.json):
  * w_fit(t) split into fixed physical windows T_mon = 200 dt_ref =
    2e-3; per window wdot_fit = linear-regression slope, plus the
    window mean w.  First window = startup (reported, excluded from
    the sign count).
  * live-gate maxima over the run (gate_q, gate_P, gate_A), area
    drift, stop reason.
  * drainage J_drain = -dA_L/dt per window (order-free lobe areas).
  * classification (four classes, thresholds recorded here):
      healing:      wdot > 0 in every post-startup window
      persistent-thinning candidate: wdot < 0 in every post-startup
                    window AND |wdot| in the last third >= 0.5 x the
                    middle third (rate not weakening); refinement
                    (h/2, H=1024) still required before "persistent"
      quasi-steady: max |wdot| < FLOOR (numerical floor = 2 x the
                    median per-window slope standard error, min 0.01)
      mixed:        anything else (reported as such)
  * 1D lubrication comparison (capsule pilots, symmetric lift):
    wdot_1D(w) = (2 c_mob H_ell^2 / ell^4) d_tau hhat_min evaluated
    at hhat_min = w / (2 H_ell) via the lifted CENV solve (scale of
    the area normalization applied; c_mob = C_MOB from p1bc).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

R = Path("results/pinchoff")
T_MON = 2e-3
C_MOB = json.load(open(R / "p1bc_fit_summary.json"))["c_mob"]
PILOTS = [
    ("cassini", None),
    ("low", dict(R=0.3, L=0.3)),
    ("mod", dict(R=0.55, L=1.0)),
    ("asym", dict(R=0.4, R_r=0.8, L=1.0)),
]


def windows(t, y):
    out = []
    k = 0
    while True:
        lo, hi = k * T_MON, (k + 1) * T_MON
        m = (t > lo) & (t <= hi + 1e-12)
        if m.sum() < 5:
            break
        c = np.polyfit(t[m], y[m], 1)
        r = y[m] - np.polyval(c, t[m])
        n = int(m.sum())
        se = math.sqrt((r @ r) / (n - 2) / ((t[m] - t[m].mean()) ** 2).sum())
        out.append(dict(k=k, t_mid=0.5 * (lo + hi),
                        mean=float(y[m].mean()), slope=float(c[0]),
                        slope_se=float(se), n=n))
        k += 1
    return out


def lub_1d(R, L, scale, w_series):
    """1D lifted prediction wdot_1D at the 2D w values (symmetric)."""
    from lift_profiles_1d import lift
    from lubrication1d import solve_neck
    from src.torch.shapes.dumbbell import CapsuleProfile
    prof = CapsuleProfile(R, R, L, 0.29, None)
    ell, H_ell, h0, p_eff = lift(prof)
    ell, H_ell = ell * scale, H_ell * scale
    r = solve_neck(P=p_eff, t_end=3.0, dt=1e-5, h0=h0)
    tau, hm = np.array(r["t"]), np.array(r["min_h"])
    dh = np.gradient(hm, tau)
    fac = 2 * C_MOB * H_ell ** 2 / ell ** 4
    out = []
    for w in w_series:
        hh = w / (2 * H_ell)
        # hmin is monotone in tau (both classes); interpolate d_tau h
        # only within the range the 1D solution actually visits
        # (no extrapolation above the lifted start)
        if hm.min() <= hh <= hm.max():
            i = int(np.argmin(np.abs(hm - hh)))
            out.append(fac * dh[i])
        else:
            out.append(None)
    return dict(P_eff=p_eff, ell=ell, H_ell=H_ell, status=r["status"],
                tau_end=float(tau[-1]), hmin_end=float(hm[-1]),
                hhat0=float(hm[0]), wdot_1D=out,
                # time at which the 1D solution reaches the numerical
                # floor hhat = 1e-3 (NOT an estimate of T*)
                t_floor_2D=(float(tau[-1]) * ell ** 4 / (C_MOB * H_ell)
                            if r["status"] == "pinch_floor" else None))


def main():
    res = dict(T_mon=T_MON, c_mob=C_MOB, pilots={})
    for name, cap in PILOTS:
        f = R / f"dumbbell_p13_{name}.json"
        if not f.exists():
            print(f"[{name}] missing")
            continue
        d = json.load(open(f))
        s = [r for r in d["series"] if r.get("w_fit") is not None]
        t = np.array([r["t"] for r in s])
        w = np.array([r["w_fit"] for r in s])
        AL = np.array([r["A_L"] for r in s])
        ww = windows(t, w)
        wa = windows(t, AL)
        for a, b in zip(ww, wa):
            a["J_drain"] = -b["slope"]
        post = ww[1:]
        slopes = np.array([x["slope"] for x in post])
        # numerical floor from the per-window fit standard errors
        # (reviewer STOP 2 §8: never use the physical window-to-window
        # rate change as noise)
        ses = np.array([x["slope_se"] for x in post])
        floor = max(0.01, 2 * float(np.median(ses))) if len(ses) else 0.01
        n3 = max(1, len(post) // 3)
        mid = np.abs(slopes[n3:2 * n3]).mean() if len(post) >= 3 else np.nan
        last = np.abs(slopes[-n3:]).mean()
        if len(post) == 0:
            cls = "too_short"
        elif np.all(slopes > 0):
            cls = "healing"
        elif np.all(slopes < 0) and last >= 0.5 * mid:
            cls = "persistent-thinning candidate (refinement pending)"
        elif np.abs(slopes).max() < floor:
            cls = "quasi-steady"
        else:
            cls = "mixed"
        gates = {g: float(max(r.get(g, 0.0) for r in d["series"]))
                 for g in ("gate_q", "gate_P", "gate_A")}
        rec = dict(fixture=d["meta"]["fixture"], n_steps=len(d["series"]),
                   t_end=float(d["series"][-1]["t"]),
                   stop=d["meta"]["stop_reason"], err=d["meta"]["error"],
                   w0=float(w[0]), w_end=float(w[-1]),
                   dA_rel_max=float(max(abs(r["dA_rel"])
                                        for r in d["series"])),
                   gate_max=gates, windows=ww, floor=floor,
                   classification=cls)
        print(f"[{name}] {d['meta']['fixture'].get('kind')} steps="
              f"{len(d['series'])} t_end={rec['t_end']:.4f} w "
              f"{w[0]:.4f}->{w[-1]:.4f} stop={rec['stop']} "
              f"err={rec['err']} gates={ {k: round(v, 4) for k, v in gates.items()} }")
        print("   win  t_mid    <w>     wdot     J_drain")
        for x in ww:
            print(f"   {x['k']:3d} {x['t_mid']:.4f} {x['mean']:.4f} "
                  f"{x['slope']:+.4f} {x['J_drain']:+.4f}")
        print(f"   floor={floor:.4f}  => {cls}")
        if cap is not None and "R_r" not in cap:
            sc = d["meta"]["fixture"]["scale"]
            l1 = lub_1d(cap["R"], cap["L"], sc, [x["mean"] for x in ww])
            rec["lub1d"] = l1
            print(f"   1D lift: P_eff={l1['P_eff']:.2f} status="
                  f"{l1['status']} hhat0={l1['hhat0']:.3f} "
                  f"t_floor_2D(hhat=1e-3)={l1['t_floor_2D']}")
            for x, v in zip(ww, l1["wdot_1D"]):
                print(f"     win {x['k']:3d} w={x['mean']:.4f} "
                      f"wdot_2D={x['slope']:+.4f} wdot_1D="
                      f"{'n/a' if v is None else f'{v:+.4f}'}")
        res["pilots"][name] = rec
    (R / "p13_fit_summary.json").write_text(
        json.dumps(res, indent=1, default=str))
    print("wrote", R / "p13_fit_summary.json")


if __name__ == "__main__":
    main()
