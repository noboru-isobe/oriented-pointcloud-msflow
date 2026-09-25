"""P1b-c1 verdicts, in the reviewer's prescribed order:

1. Smoke gate: |lam_new - lam_old| / lam_old vs 5% for v18 and s18
   (corrected C^2 seed + exact normals vs the P1b-1 seed).
2. Three-wavenumber judge against the WINDOWED strip prediction
   A(t) = <g_k, exp(-c_mob t M) f_k> for the corrected seed
   (pin_p1bc_seed.py, results/pinchoff/p1bc_seed_pins.json) -- the
   seed is not an eigenfunction, so the pure-mode symbol is not the
   correct comparison (reviewer P1b-c0). c_mob is recalibrated on
   the corrected s18 against its own windowed prediction (single
   provenance, recorded in the summary).
CORRIGENDUM (reviewer, P1b-c RETURN): the fitted rate on a fixed
window is NOT proportional to c_mob (the windowed seed is a mixture
of eigenmodes: Lambda_I(c) = c Lambda_{cI}(1), the window moves).
c_mob is therefore obtained by a 1-D root solve (Brent) of
Lambda_s18(c) = lam_obs(s18) with the strip semigroup re-evaluated
from the FFT at every c, and all pairs/windows are re-evaluated at
c*. Pure-mode log-ratios are printed for reference only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
from pin_p1bc_seed import new_seed_window, strip_prediction  # noqa: E402

R = Path("results/pinchoff")
W0 = 0.29
C_MOB_PINS = 1.3022  # value the pins-file strip predictions used
WINDOWS = [(0.001, 0.004), (0.001, 0.006), (0.002, 0.008),
           (0.002, 0.010)]
PAIRS = [("v18", "varicose", 1.8), ("v22", "varicose", 2.2),
         ("v27", "varicose", 2.7), ("s18", "sinuous", 1.8),
         ("s27", "sinuous", 2.7)]


def load(tag, ch):
    d = json.load(open(R / f"dumbbell_{tag}.json"))
    t = np.array([r["t"] for r in d["series"]])
    a = np.array([r[ch] for r in d["series"]])
    return t, a


def lam_last(prefix, name, mode):
    ch = "amp_varicose" if mode == "varicose" else "amp_sinuous"
    tp, ap = load(f"{prefix}_{name}p", ch)
    tm, am = load(f"{prefix}_{name}m", ch)
    n = min(len(tp), len(tm))
    t, dl = tp[:n], (ap[:n] - am[:n]) / 2
    sgn = 1.0 if dl[np.abs(t - 0.001).argmin()] > 0 else -1.0
    out = []
    for lo, hi in WINDOWS:
        m = (t >= lo) & (t <= hi) & (sgn * dl > 0)
        c = np.polyfit(t[m], np.log(sgn * dl[m]), 1)
        out.append(-float(c[0]))
    return out


def main():
    pins = json.load(open(R / "p1bc_seed_pins.json"))
    res = dict(windows=WINDOWS, rows=[])

    print("== 1. smoke gate (corrected vs P1b-1 seed, last window,"
          " 5%) ==")
    gate = {}
    for name, mode, k in PAIRS:
        new = lam_last("p1bc", name, mode)
        old = lam_last("p1b", name, mode)
        rel = abs(new[-1] - old[-1]) / old[-1]
        gate[name] = (new, old, rel)
        flag = "CHANGED" if rel > 0.05 else "unchanged"
        star = " <- gate" if name in ("v18", "s18") else ""
        print(f"{name}: lam_new={new[-1]:7.2f}  lam_old="
              f"{old[-1]:7.2f}  |d|/old={rel:6.1%}  {flag}{star}")

    print("\n== 2. windowed-strip judge (corrected seed) ==")
    # calibrate c_mob on corrected s18 (last window) by root solve:
    # Lambda_s18(c) = lam_obs, semigroup re-evaluated at every c
    lam_s18 = gate["s18"][0][-1]

    def resid(c):
        return strip_prediction(1.8, "sinuous", new_seed_window,
                                c)[-1] - lam_s18
    c_mob = brentq(resid, 0.5, 2.0, xtol=1e-6)
    lin = C_MOB_PINS * lam_s18 / \
        pins["strip_multiplier"]["sinuous_k1.8"]["new"][-1]
    print(f"c_mob* = {c_mob:.5f} (Brent root of Lambda_s18(c) = "
          f"{lam_s18:.3f}; naive linear rescale would give {lin:.4f}"
          " -- invalid, see docstring)")
    logr = {}
    for name, mode, k in PAIRS:
        new = gate[name][0]
        pred = strip_prediction(k, mode, new_seed_window, c_mob)
        z = k * W0 / 2
        pure = c_mob * k ** 3 * (math.tanh(z) if mode == "varicose"
                                 else 1 / math.tanh(z))
        lr = math.log(new[-1] / pred[-1])
        lr_w = [math.log(a / b) for a, b in zip(new, pred)]
        logr[name] = lr
        print(f"{name}: lam={new[-1]:7.2f}  strip_pred="
              f"{pred[-1]:7.2f}  log-ratio={lr:+.3f} "
              f"(windows: " + " ".join(f"{v:+.3f}" for v in lr_w)
              + f")  [pure-mode {pure:6.2f}, ref log-ratio "
              f"{math.log(new[-1] / pure):+.3f}]")
        res["rows"].append(dict(
            pair=name, mode=mode, k=k,
            lam_windows=[round(v, 3) for v in new],
            lam_old_windows=[round(v, 3) for v in gate[name][1]],
            gate_rel=round(gate[name][2], 4),
            strip_pred_windows=[round(v, 3) for v in pred],
            log_ratio_windows=[round(v, 4) for v in lr_w],
            log_ratio=round(lr, 4),
            pure_mode=round(pure, 3)))
    judge = max(abs(v) for n, v in logr.items() if n != "s18")
    print(f"\njudge max |log-ratio| over pairs != s18 "
          f"(calibrator): {judge:.3f}")
    res["c_mob"] = round(c_mob, 5)
    res["c_mob_calibration_rule"] = (
        "Brent root of Lambda_s18(c) = lam_obs(s18, corrected seed, "
        "last window [0.002,0.010]); Lambda = fitted log-slope of the "
        "frozen-width infinite-strip semigroup A(t)=<g,exp(-c t M) f> "
        "re-evaluated by FFT at each c (no linear rescaling)")
    res["judge_max_abs_log_ratio_windows"] = [
        round(max(abs(r["log_ratio_windows"][i]) for r in res["rows"]
                  if r["pair"] != "s18"), 4) for i in range(4)]
    res["judge_max_abs_log_ratio"] = round(judge, 4)
    (R / "p1bc_fit_summary.json").write_text(
        json.dumps(res, indent=1))
    print("wrote", R / "p1bc_fit_summary.json")


if __name__ == "__main__":
    main()
