"""0L-B0 analysis: contact-layer compatibility/gauge causal decomposition.

Reads exact_merger_0lb0_{768,1024}_{pre,A,B,C,D,E}.json. For every
cell (all started from the SAME committed pre-event state):
  one-step (rec[0]): MM e_int gain, S_disk, R_disk, objective
     decrease, n_iter, converged, max_disp/h_wall, W_grid, P,
     row actions G v / C_bulk v / C_rel v, eps_row(_composed),
     constraint singular values, relation base/used, gauge_hold,
     gamma_cross, max_projection_rel
  window: post-event 15-step median/max MM gain, S_disk first/last/
     max, R_disk range, V_geom disk/ann range, flux max, dv_mm pump
     signature (sign pattern of dv_mm_disk vs dv_mm_ann), stop step
Pre-registered clean criterion: post-event MM gain median inside the
pre-event envelope (pre run median * 3) AND no S_disk e-fold (last/
first < 3) AND conservation green (R_disk drift < 1e-5, flux 0).
Cell C additionally: max_projection_rel <= eps_proj,* where
eps_proj,* = max(pre-run proj_rel_shadow) + 1e-6.
"""

import json
from pathlib import Path

OUT = Path("results/exact_merger")


def load(tag):
    p = OUT / f"exact_merger_0lb0_{tag}.json"
    if not p.exists():
        return None
    return json.load(open(p))["series"]


def mm_gains(s, s_prev_last=None):
    out = []
    prev = s_prev_last
    for r in s:
        if prev is not None and r.get("e2_postMM") is not None \
                and prev.get("e2") is not None:
            out.append(r["e2_postMM"] - prev["e2"])
        prev = r
    return out


def med(x):
    x = sorted(x)
    return x[len(x) // 2] if x else None


def analyze(H):
    pre = load(f"{H}_pre")
    if pre is None:
        print(f"-- no pre run for {H}")
        return None
    pre_gains = mm_gains(pre)
    pre_med = med(pre_gains)
    proj_star = max((r.get("proj_rel_shadow") or 0.0) for r in pre) + 1e-6
    hw = pre[-1]["h_wall"]
    print(f"== H={H}: pre {pre[0]['step']}-{pre[-1]['step']}  MM gain "
          f"median {pre_med:+.2e} max {max(pre_gains):+.2e}  "
          f"eps_proj* {proj_star:.3e}  rel(last) "
          f"{pre[-1]['partition_relation']}")
    rows = {"pre": dict(mm_gain_median=pre_med, eps_proj_star=proj_star)}
    for cell in "ABCDE":
        s = load(f"{H}_{cell}")
        if not s:
            print(f"  [{cell}] missing")
            continue
        r0 = s[0]
        gains = mm_gains(s, pre[-1])
        S = [r["S_disk"] for r in s]
        Rd = [r["R_disk"] for r in s]
        Vd = [r["vol_geom_disk"] for r in s]
        Va = [r["vol_geom_ann"] for r in s]
        flux = max(max(abs(x) for x in r["bulk_flux_residuals"])
                   if r.get("bulk_flux_residuals") else 0.0 for r in s)
        pump = [(r.get("dv_mm_disk"), r.get("dv_mm_ann")) for r in s
                if r.get("dv_mm_disk") is not None]
        # pump signature: disk gains what annulus loses (opposite signs)
        n_pump = sum(1 for d, a in pump if d * a < 0
                     and abs(d) > 1e-8 and abs(a) > 1e-8)
        ct = r0.get("contact_rows_telemetry") or {}
        rec = dict(
            first_step=r0["step"], last_step=s[-1]["step"], n=len(s),
            one_step=dict(
                mm_gain=gains[0] if gains else None,
                S_disk=r0["S_disk"], R_disk=r0["R_disk"],
                obj_decrease=(r0.get("objective_initial") or 0)
                - r0["objective"],
                n_iter=r0["n_iter"], converged=r0["converged"],
                max_disp_over_hwall=r0["max_disp"] / hw,
                W=r0["wasserstein"], P=r0["perimeter"],
                relation=r0.get("partition_relation"),
                relation_base=r0.get("partition_relation_base"),
                gauge_hold=r0.get("gauge_hold_fired"),
                gamma_cross=r0.get("gamma_cross"),
                n_compat=(r0.get("n_compat_base"),
                          r0.get("n_compat_used")),
                eps_row=r0.get("eps_row"),
                eps_row_composed=r0.get("eps_row_composed"),
                r_stack=r0.get("r_stack"),
                sv=r0.get("constraint_sv"),
                Gv=ct.get("G"), Cbulk_v=ct.get("C_bulk"),
                Crel_v=ct.get("C_rel"),
                max_projection_rel=r0.get("max_projection_rel"),
            ),
            window=dict(
                mm_gain_median=med(gains), mm_gain_max=max(gains),
                S_first=S[0], S_last=S[-1], S_max=max(S),
                R_disk_range=(min(Rd), max(Rd)),
                Vd_range=(min(Vd), max(Vd)), Va_range=(min(Va), max(Va)),
                flux_max=flux, n_pump_steps=n_pump,
                max_projection_rel=max((r.get("max_projection_rel") or 0)
                                       for r in s),
                gamma_cross_max=max((r.get("gamma_cross") or 0)
                                    for r in s),
                e2_last=s[-1]["e2"],
            ),
        )
        w, o = rec["window"], rec["one_step"]
        clean = (w["mm_gain_median"] <= 3 * max(pre_med, 1e-6)
                 and w["S_last"] / max(w["S_first"], 1e-30) < 3
                 and (w["R_disk_range"][1] - w["R_disk_range"][0]) < 1e-5
                 and w["flux_max"] < 1e-12)
        if cell == "C":
            clean = clean and w["max_projection_rel"] <= proj_star
        rec["clean"] = bool(clean)
        rows[cell] = rec
        print(f"  [{cell}] {rec['first_step']}-{rec['last_step']}  "
              f"1st: mm {o['mm_gain']:+.2e} it {o['n_iter']} conv "
              f"{o['converged']} dObj {o['obj_decrease']:.2e} rel "
              f"{o['relation']}/{o['relation_base']} eps_row "
              f"{o['eps_row']} r_stack {o['r_stack']} gh {o['gauge_hold']}"
              f" gc {o['gamma_cross']}  |  win: mm med "
              f"{w['mm_gain_median']:+.2e} max {w['mm_gain_max']:+.2e} "
              f"S {w['S_first']:.1e}->{w['S_last']:.1e} R_disk "
              f"[{w['R_disk_range'][0]:.6f},{w['R_disk_range'][1]:.6f}] "
              f"flux {w['flux_max']:.1e} pump {w['n_pump_steps']} "
              f"proj {w['max_projection_rel']:.2e} e2 {w['e2_last']:.4f}"
              f"  -> {'CLEAN' if clean else 'fail'}", flush=True)
    return rows


def main():
    report = {}
    for H in (768, 1024):
        r = analyze(H)
        if r:
            report[str(H)] = r
    outp = Path("results/reports/phase3c0l_b0.json")
    outp.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
