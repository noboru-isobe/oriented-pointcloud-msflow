"""P1b-2: amplitude/grid separation fits (reviewer program).

Centered-pair tangent response delta(t) = (amp_+ - amp_-)/2 per pair,
log-linear lambda over the pre-registered window scan. Products:
  1. eps ladder at grid 768, varicose k=1.8:
     eps in {0.002, 0.004 (Stage A), 0.008, 0.016} -> lambda(eps),
     linear eps->0 extrapolation (odd nonlinearity enters at eps^2
     in the centered difference, so fit lambda vs eps^2).
  2. grid comparison at k=1.8: 768 (Stage A, eps=0.004) vs
     1024 fixed physical eps=0.004 vs 1024 fixed eps/dx (eps=0.003).
  3. sinuous k=1.8 control: 768 (Stage A) vs 1024.
Each lambda is compared to the MS local symbol
lambda_MS = c_mob k^3 {tanh,coth}(k w/2) with the Stage A sinuous
calibration c_mob = 1.249.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

R = Path("results/pinchoff")
W0 = 0.29
# c_mob is NOT hard-coded: it is calibrated in main() from the s18
# pair (last window) and recorded once in the summary JSON
# (reviewer P1b provenance requirement).
C_MOB = None
WINDOWS = [(0.001, 0.004), (0.001, 0.006), (0.002, 0.008),
           (0.002, 0.010)]


def load(tag, ch):
    d = json.load(open(R / f"dumbbell_{tag}.json"))
    t = np.array([r["t"] for r in d["series"]])
    a = np.array([r[ch] for r in d["series"]])
    return t, a


def tangent(pair, ch):
    tp, ap = load(pair + "p", ch)
    tm, am = load(pair + "m", ch)
    n = min(len(tp), len(tm))
    assert np.allclose(tp[:n], tm[:n])
    return tp[:n], (ap[:n] - am[:n]) / 2


def lam_windows(t, x):
    out = []
    for lo, hi in WINDOWS:
        m = (t >= lo) & (t <= hi) & (x > 0)
        if m.sum() < 5:
            out.append(None)
            continue
        c = np.polyfit(t[m], np.log(x[m]), 1)
        resid = float(np.std(np.log(x[m]) - np.polyval(c, t[m])))
        out.append((-float(c[0]), resid))
    return out


def ms_rate(k, mode):
    f = math.tanh if mode == "varicose" else lambda z: 1 / math.tanh(z)
    return C_MOB * k ** 3 * f(k * W0 / 2)


def report(name, pair, mode, k, eps):
    ch = "amp_varicose" if mode == "varicose" else "amp_sinuous"
    t, dl = tangent(pair, ch)
    sgn = 1.0 if dl[np.abs(t - 0.001).argmin()] > 0 else -1.0
    lams = lam_windows(t, sgn * dl)
    ok = [v for v in lams if v is not None]
    lam = ok[-1][0] if ok else float("nan")
    pred = ms_rate(k, mode)
    print(f"{name:22s} eps={eps:+.4f} lam_windows="
          + " ".join("None" if v is None else f"{v[0]:7.2f}"
                     for v in lams)
          + f"  resid={ok[-1][1]:.4f}" if ok else "", end="")
    print(f"  MS={pred:.2f}  ratio={lam / pred:.2f}"
          f"  log-ratio={math.log(lam / pred):+.3f}")
    return dict(name=name, pair=pair, mode=mode, k=k, eps=eps,
                lam_windows=[None if v is None else
                             [round(v[0], 3), round(v[1], 5)]
                             for v in lams],
                lam=round(lam, 3), ms=round(pred, 3),
                log_ratio=round(math.log(lam / pred), 4))


def calibrate_c_mob():
    """c_mob from the s18 pair, last window (single provenance)."""
    global C_MOB
    t, dl = tangent("p1b_s18", "amp_sinuous")
    sgn = 1.0 if dl[np.abs(t - 0.001).argmin()] > 0 else -1.0
    lam = lam_windows(t, sgn * dl)[-1][0]
    C_MOB = lam / (1.8 ** 3 / math.tanh(1.8 * W0 / 2))
    print(f"c_mob = {C_MOB:.4f} (calibrated from p1b_s18, "
          f"last window, lam = {lam:.2f})")


def main():
    rows = []
    calibrate_c_mob()
    print("== eps ladder, grid 768, varicose k=1.8 ==")
    ladder = [
        ("e002 (768)", "p1b2_v18_e002", 0.002),
        ("e004 StageA (768)", "p1b_v18", 0.004),
        ("e008 (768)", "p1b2_v18_e008", 0.008),
        ("e016 (768)", "p1b2_v18_e016", 0.016),
    ]
    lam_eps = []
    for name, pair, eps in ladder:
        r = report(name, pair, "varicose", 1.8, eps)
        rows.append(r)
        lam_eps.append((eps, r["lam"]))
    e2 = np.array([e * e for e, _ in lam_eps])
    lv = np.array([v for _, v in lam_eps])
    c = np.polyfit(e2, lv, 1)
    lam0 = float(c[1])
    print(f"eps->0 extrapolation (lambda vs eps^2): "
          f"lam(0) = {lam0:.2f}, slope = {c[0]:.3g} "
          f"(MS = {ms_rate(1.8, 'varicose'):.2f}, "
          f"log-ratio = {math.log(lam0 / ms_rate(1.8, 'varicose')):+.3f})")

    print("\n== grid comparison, varicose k=1.8 ==")
    rows.append(report("g768  eps  0.004", "p1b_v18", "varicose",
                       1.8, 0.004))
    rows.append(report("g1024 eps  0.004", "p1b2_g1024_v18",
                       "varicose", 1.8, 0.004))
    rows.append(report("g1024 eps/dx fix", "p1b2_g1024_v18f",
                       "varicose", 1.8, 0.003))

    print("\n== sinuous k=1.8 control ==")
    rows.append(report("g768  sinuous", "p1b_s18", "sinuous", 1.8,
                       0.004))
    rows.append(report("g1024 sinuous", "p1b2_g1024_s18", "sinuous",
                       1.8, 0.004))

    out = dict(c_mob=C_MOB, w0=W0, windows=WINDOWS, rows=rows,
               eps_extrapolation=dict(lam0=round(lam0, 3),
                                      slope=float(c[0])))
    (R / "p1b2_fit_summary.json").write_text(
        json.dumps(out, indent=1))
    print("\nwrote", R / "p1b2_fit_summary.json")


if __name__ == "__main__":
    main()
