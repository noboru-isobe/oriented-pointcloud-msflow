"""0L-theta2 arm comparison over the exact_merger_th2_* runs.

Per arm: stop step + last-step cause context, R_disk / V_geom_disk
drift over the window, S_disk trajectory (the E3 stop channel),
e_int stage decomposition (MM gain = e2_postMM[k] - e2[k-1];
redistribution gain = e2[k] - e2_postMM[k]), gap progression,
conservation (bulk flux residual max), component count. Acceptance
(pre-registered, theta2a): loopwise arms must (i) keep every E3
conservation guard green, (ii) not regress R_disk / V_geom
constancy, (iii) remove or reduce the 10x MM e_int jump at
gap ~ 0.5 sigma_angle that stopped E3, (iv) reach at least as
deep as the union control from the same state.
"""

import json
from pathlib import Path

OUT = Path("results/exact_merger")
ARMS = [
    ("1300", "U"), ("1300", "Lm"), ("1300", "Lmc"),
    ("1500", "U"), ("1500", "Lv"), ("1500", "Lm"), ("1500", "Lmc"),
    ("1525", "U"), ("1525", "Lm"), ("1525", "Lmc"),
]


def main():
    rows = []
    for start, arm in ARMS:
        p = OUT / f"exact_merger_th2_{start}_{arm}.json"
        if not p.exists():
            print(f"-- missing {p.name}")
            continue
        d = json.load(open(p))
        s = d["series"]
        first, last = s[0], s[-1]
        rd = [r["R_disk"] for r in s if r.get("R_disk") is not None]
        vg = [r.get("vol_geom_disk") for r in s
              if r.get("vol_geom_disk") is not None]
        sd = [r.get("S_disk") for r in s
              if r.get("S_disk") is not None]
        e2 = [r.get("e2") for r in s]
        e2p = [r.get("e2_postMM") for r in s]
        mm_gain = [e2p[k] - e2[k - 1] for k in range(1, len(s))
                   if e2p[k] is not None and e2[k - 1] is not None]
        rd_gain = [e2[k] - e2p[k] for k in range(1, len(s))
                   if e2p[k] is not None and e2[k] is not None]
        flux = [max(abs(x) for x in r["bulk_flux_residuals"])
                if r.get("bulk_flux_residuals") else 0.0 for r in s]
        gaps = [r.get("gap") for r in s if r.get("gap") is not None]
        row = dict(
            start=start, arm=arm,
            first_step=first["step"], last_step=last["step"],
            n_recorded=len(s),
            merged_at=d["meta"].get("merged_at_step"),
            error=d["meta"].get("error"),
            R_disk_range=(min(rd), max(rd)) if rd else None,
            Vgeom_range=(min(vg), max(vg)) if vg else None,
            S_disk_first_last=(sd[0], sd[-1], max(sd))
            if sd else None,
            mm_gain_max=max(mm_gain) if mm_gain else None,
            mm_gain_median=(sorted(mm_gain)[len(mm_gain) // 2]
                            if mm_gain else None),
            redist_gain_absmax=(max(abs(x) for x in rd_gain)
                                if rd_gain else None),
            e2_last=e2[-1], e2_max=max(x for x in e2
                                       if x is not None),
            flux_max=max(flux),
            gap_first_last=(gaps[0], gaps[-1]) if gaps else None,
            n_components_last=last.get("n_components"),
        )
        rows.append(row)
        print(f"[{start}/{arm:4s}] steps {row['first_step']}-"
              f"{row['last_step']} ({row['n_recorded']})  "
              f"R_disk [{row['R_disk_range'][0]:.7f},"
              f"{row['R_disk_range'][1]:.7f}]  "
              f"S_disk {row['S_disk_first_last'][0]:.2e}->"
              f"{row['S_disk_first_last'][1]:.2e} "
              f"(max {row['S_disk_first_last'][2]:.2e})  "
              f"e2 last {row['e2_last']:.4f} "
              f"max {row['e2_max']:.4f}  "
              f"mm_gain med {row['mm_gain_median']:.1e} "
              f"max {row['mm_gain_max']:.1e}  "
              f"redist |max| {row['redist_gain_absmax']:.1e}  "
              f"flux {row['flux_max']:.1e}  "
              f"gap {row['gap_first_last'][0]:.4f}->"
              f"{row['gap_first_last'][1]:.4f}  "
              f"C={row['n_components_last']}", flush=True)
    outp = Path("results/reports/phase3c0l_theta2.json")
    outp.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
