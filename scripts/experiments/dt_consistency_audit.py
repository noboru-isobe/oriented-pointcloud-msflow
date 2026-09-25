"""First-moment rows Delta-t consistency audit (reviewer 2026-08-23,
section 5 of the L-A closure verdict): H=768, WB, rows on, fixed
physical time t = 0.002, (dt, N) = (1e-5, 200) vs (5e-6, 400).
Measures eq (5.1) dbar_dt / dbar_{dt/2} and compares dA, dM, g_min,
P, and shape modes; rows-off controls at both dt for reference.

Branches (pre-registered): ratio ~ 2 -> first-order-constraint
consistency error, quantifiable O(dt); no reduction -> defect in the
discrete rows implementation or row-space coupling.
"""

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import two_ellipses_benchmark as teb  # noqa: E402

R = Path("results/two_ellipses")
# series row at t = 0.002 exactly: step index N-1 (t = (step+1) dt);
# states are saved every 25 steps only, so the mode/position
# comparison uses the nearest saved checkpoint (175 / 350, 1 step of
# dt/2 apart -- flagged in the output)
CASES = {
    "on_dt":   ("two_ellipses_dtaud_on_dt", 199, 175),
    "on_dt2":  ("two_ellipses_dtaud_on_dt2", 399, 350),
    "off_dt":  ("two_ellipses_l0_768", 199, 175),
    "off_dt2": ("two_ellipses_dtaud_off_dt2", 399, 350),
}
BAR0 = [(-0.45, 0.0), (0.45, 0.0)]


def modes(pos, k_max=8):
    """Radial Fourier modes about the centroid (|c_k| / R_mean)."""
    c = pos.mean(dim=0)
    d = pos - c
    r = d.norm(dim=1)
    th = torch.atan2(d[:, 1], d[:, 0])
    out = {}
    for k in range(2, k_max + 1):
        ck = ((r * torch.exp(-1j * k * th)).sum() / r.shape[0])
        out[k] = float(ck.abs() / r.mean())
    return out


def main():
    torch.set_default_dtype(torch.float64)
    res = {}
    states = {}
    for name, (tag, step, cstep) in CASES.items():
        js = R / f"{tag}.json"
        pt = R / f"{tag}_states.pt"
        if not js.exists():
            print(f"[missing] {js}")
            continue
        d = json.load(open(js))
        row = next((r for r in d["series"] if r["step"] == step), None)
        ck = torch.load(pt, weights_only=True)
        if row is None or cstep not in ck:
            print(f"[{name}] step {step} not reached yet")
            continue
        st = ck[cstep]
        pos = st["positions"]
        n = pos.shape[0] // 2
        rec = dict(step=step, t=row["t"], ckpt_step=cstep,
                   ckpt_t=(cstep + 1) * (row["t"] / (step + 1)),
                   dbar_x=[row["bar"][j][0] - BAR0[j][0]
                           for j in (0, 1)],
                   dbar_y=[row["bar"][j][1] - BAR0[j][1]
                           for j in (0, 1)],
                   dbar=row["dbar"], dA_rel=row["dA_rel"],
                   g_min=row["g_min"], P_frozen=row["P_frozen"],
                   W=row["W"], n_iter=row["n_iter"],
                   max_s=row["max_s"])
        rec["M"] = [teb.polygon_moments_order(pos[:n]),
                    teb.polygon_moments_order(pos[n:])]
        rec["A"] = [teb.signed_area_order(pos[:n]),
                    teb.signed_area_order(pos[n:])]
        rec["modes_loop0"] = modes(pos[:n])
        states[name] = pos
        res[name] = rec
    # eq (5.1)
    out = dict(cases=res)
    if "on_dt" in res and "on_dt2" in res:
        r1 = [abs(x) for x in res["on_dt"]["dbar_x"]]
        r2 = [abs(x) for x in res["on_dt2"]["dbar_x"]]
        out["ratio_on_dbar_x"] = [a / b for a, b in zip(r1, r2)]
        out["dX_inf_over_h_on"] = float(
            (states["on_dt"] - states["on_dt2"]).abs().max()
            / teb.H_REF)
    if "off_dt" in res and "off_dt2" in res:
        r1 = [abs(x) for x in res["off_dt"]["dbar_x"]]
        r2 = [abs(x) for x in res["off_dt2"]["dbar_x"]]
        out["ratio_off_dbar_x"] = [a / b for a, b in zip(r1, r2)]
        out["dX_inf_over_h_off"] = float(
            (states["off_dt"] - states["off_dt2"]).abs().max()
            / teb.H_REF)
    if "on_dt" in res and "off_dt" in res:
        out["dX_inf_over_h_on_vs_off_dt"] = float(
            (states["on_dt"] - states["off_dt"]).abs().max()
            / teb.H_REF)
    if "on_dt2" in res and "off_dt2" in res:
        out["dX_inf_over_h_on_vs_off_dt2"] = float(
            (states["on_dt2"] - states["off_dt2"]).abs().max()
            / teb.H_REF)
    (R / "dt_consistency_audit.json").write_text(
        json.dumps(out, indent=1, default=str))
    for name, rec in res.items():
        print(f"{name:8s} t={rec['t']:.4f} dbar_x={rec['dbar_x'][0]:+.3e}"
              f"/{rec['dbar_x'][1]:+.3e} dA={rec['dA_rel'][0]:+.3e} "
              f"g={rec['g_min']:.5f} P={rec['P_frozen']:.6f} "
              f"modes2..4={[round(rec['modes_loop0'][k], 6) for k in (2, 3, 4)]}")
    for k in ("ratio_on_dbar_x", "ratio_off_dbar_x", "dX_inf_over_h_on",
              "dX_inf_over_h_off", "dX_inf_over_h_on_vs_off_dt",
              "dX_inf_over_h_on_vs_off_dt2"):
        if k in out:
            print(f"{k}: {out[k]}")


if __name__ == "__main__":
    main()
