"""P1-2 rate-pin analysis: mobility-calibrated cross-k prediction.

Per run: load dumbbell_<tag>.json, take the seeded-channel amplitude
series, subtract the unseeded BASELINE run's same-channel series
(the background neck evolution of the fixture leaks into the cos(kx)
projection; the baseline measures exactly that leak), and fit
log |amp_corr| linearly over the pre-registered early window
[t_skip, t_fit] (skip the first steps: seed relaxation transient).

Calibration: c_mob = lambda_num(k0) / (k0^3 tanh(k0 w/2)) from the
designated varicose k0; prediction lambda_pred(k) =
c_mob k^3 tanh(k w/2) (varicose) or coth (sinuous); judge quantity
J = max_{k != k0} |log(lambda_num / lambda_pred)|.

Delta-t consistency: the dt/2 run of k0 must reproduce lambda within
the pre-registered tolerance before any window-closure claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

OUT = Path("results/pinchoff")


def load_series(tag, channel):
    d = json.load(open(OUT / f"dumbbell_{tag}.json"))
    s = d["series"]
    t = np.array([r["t"] for r in s], dtype=float)
    a = np.array([r[channel] for r in s], dtype=float)
    return d["meta"], t, a


def fit_lambda(t, a, t_skip, t_fit):
    sel = (t >= t_skip) & (t <= t_fit) & (a > 0)
    if sel.sum() < 10:
        return None
    c = np.polyfit(t[sel], np.log(a[sel]), 1)
    resid = float(np.sqrt(np.mean(
        (np.log(a[sel]) - np.polyval(c, t[sel])) ** 2)))
    return dict(lam=float(-c[0]), amp0=float(math.exp(c[1])),
                resid=resid, n=int(sel.sum()),
                window=[float(t_skip), float(t_fit)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="p12_base")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="tag:mode:k triplets, first = calibration k0")
    ap.add_argument("--w", type=float, default=None,
                    help="neck width for tanh/coth arg (default: "
                         "baseline w_fit averaged over the window)")
    ap.add_argument("--t-skip", type=float, default=0.002)
    ap.add_argument("--t-fit", type=float, default=0.010)
    args = ap.parse_args()
    _, tb, base_v = load_series(args.baseline, "amp_varicose")
    _, _, base_s = load_series(args.baseline, "amp_sinuous")
    db = json.load(open(OUT / f"dumbbell_{args.baseline}.json"))
    wf = np.array([r["w_fit"] for r in db["series"]], dtype=float)
    selw = (tb >= args.t_skip) & (tb <= args.t_fit)
    w = args.w if args.w else float(wf[selw].mean())
    rows = []
    for spec in args.runs:
        tag, mode, kstr = spec.split(":")
        k = float(kstr)
        ch = "amp_varicose" if mode == "varicose" else "amp_sinuous"
        meta, t, a = load_series(tag, ch)
        # baseline subtraction interpolated IN TIME (a dt/2 run is
        # sampled twice as densely; index alignment was a bug)
        base = np.interp(t, tb,
                         base_v if mode == "varicose" else base_s)
        corr = a - base
        sgn = np.sign(corr[0])
        fit = fit_lambda(t, sgn * corr, args.t_skip, args.t_fit)
        raw = fit_lambda(t, np.sign(a[0]) * a,
                         args.t_skip, args.t_fit)
        shape = (math.tanh(k * w / 2) if mode == "varicose"
                 else 1.0 / math.tanh(k * w / 2))
        rows.append(dict(tag=tag, mode=mode, k=k, dt=meta["dt"],
                         shape=k ** 3 * shape, fit=fit, fit_raw=raw))
    k0 = rows[0]
    c_mob = k0["fit"]["lam"] / k0["shape"]
    for r in rows:
        r["lam_pred"] = c_mob * r["shape"]
        r["log_ratio"] = (math.log(r["fit"]["lam"] / r["lam_pred"])
                          if r["fit"] and r["fit"]["lam"] > 0
                          else None)
    judge = max(abs(r["log_ratio"]) for r in rows[1:]
                if r["log_ratio"] is not None and r["dt"] == k0["dt"]
                and r["mode"] == "varicose")
    res = dict(w_used=w, c_mob=c_mob, judge_varicose=judge,
               window=[args.t_skip, args.t_fit], rows=rows)
    print(json.dumps(res, indent=1))
    (OUT / "rate_pin_fits.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
