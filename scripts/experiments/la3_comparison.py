"""L-A3 three-configuration comparison in dimensionless geometry
(reviewer hardening 10, eq 10.1): at t_act -- g/sigma, g/h, |I| both
sides, L_side/sigma, r_side, Delta_birth, P^full - P^self; and at
matched g/sigma post-activation -- area / first moment / barycenter /
window length / tempo / refresh. Reads the three L-A3 run JSONs.
"""

import json
import sys

SIGMA = 0.1
RUNS = [("768 axis", "results/two_ellipses/two_ellipses_la3_768.json"),
        ("1024 axis", "results/two_ellipses/two_ellipses_la3_1024.json"),
        ("768 rot22.5", "results/two_ellipses/two_ellipses_la3_768rot.json")]


def row_at(series, step):
    return next(r for r in series if r["step"] == step)


def main():
    data = {}
    for name, path in RUNS:
        try:
            data[name] = json.load(open(path))
        except FileNotFoundError:
            print(f"[missing] {path}")
    out = {}
    for name, d in data.items():
        m = d["meta"]
        if not m.get("act_events"):
            print(f"[{name}] not activated yet -- skipped")
            continue
        ev = m["act_events"][0]
        s = d["series"]
        act = row_at(s, m["activated_at"])
        rec = dict(
            t_act=m["activated_at"],
            gap_over_sigma=ev["activation_gap_over_sigma"],
            gap_over_h=act["g_min_over_h"],
            sides_I=[x[1] for x in ev["source_sides"]],
            neutrality_total=ev["neutrality"]["total"],
            far_exact=ev["neutrality"]["far_cc_max"],
            dX_arms_over_h=ev["dX_arms_over_h"],
        )
        cc = [r for r in s if r["step"] > m["activated_at"]]
        rec.update(
            cc_steps=len(cc),
            refresh_max=max((r.get("refresh_pos") or 0) for r in cc),
            deaths=any(any((r.get("side_deaths") or {}).values())
                       for r in cc),
            births=[(r["step"] - m["activated_at"], r["delta_birth"])
                    for r in cc if r.get("delta_birth") is not None],
        )
        # L_side/sigma and r_side at t_act (from first CC telemetry)
        first_cc = next(r for r in cc if r.get("cc_sides"))
        rec["r_side_first"] = [round(x["r"], 5)
                               for x in first_cc["cc_sides"]]
        last = cc[-1]
        rec.update(
            final_step=last["step"],
            final_g_over_sigma=last["g_min"] / SIGMA,
            final_g_over_h=last["g_min_over_h"],
            final_sides=[(x["loop"], x["n"], round(x["r"], 5))
                         for x in last["cc_sides"]],
            final_t=last["t"],
            dA_max=max(max(abs(x) for x in r["dA_rel"]) for r in s),
            dbar_max=max(max(r["dbar"]) for r in s),
            window_at=next((r["step"] for r in cc
                            if r.get("n_W", (0, 0))[0] >= 2
                            and r["n_W"][1] >= 2), None),
        )
        # matched-depth window quantities (final state)
        rec["final_L_W_over_h"] = last.get("L_W_over_h")
        rec["final_n_W"] = last.get("n_W")
        out[name] = rec
    print(json.dumps(out, indent=1, default=str))
    # cross-config spreads at t_act and matched depth
    if len(out) >= 2:
        names = list(out)
        for k in ("gap_over_sigma", "final_g_over_sigma", "final_t",
                  "dA_max", "dbar_max"):
            vals = [out[n][k] for n in names]
            lo, hi = min(vals), max(vals)
            spread = (hi - lo) / abs(hi) if hi else 0.0
            print(f"spread {k}: {spread:.2e}  ({lo:.6g} .. {hi:.6g})")
    ok = all(r["refresh_max"] == 0 and not r["deaths"]
             and all(b[1] <= 1e-11 for b in r["births"])
             for r in out.values())
    print("standing certificates all clean:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
