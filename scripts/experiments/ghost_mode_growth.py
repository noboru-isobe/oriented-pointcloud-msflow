"""E5 / 1024-continuation: ghost-pair m=2 mode decomposition and
growth-rate comparison at MATCHED ghost geometry (reviewer 2026-08-18).

For every recorded state (25-step checkpoints + switch state) of the
768 t=0 run (E5) and the 1024 State-III continuation, compute per
ghost loop (old disk / old hole) the complex m=2 coefficient
    z2 = sum m (r - rbar) e^{-2 i theta} / sum m
and the common / relative combinations
    z2+ = (z2_d + z2_h) / 2   (common-shape mode),
    z2- = (z2_h - z2_d) / 2   (varicose / gap-modulation mode),
plus the matching coordinates (R_d, R_h, gap, q_wall, eps_cur2 from
the JSON series). Growth rate lambda2 = d/dt log|z2+| over the window
where the two runs' ghost geometry matches (same R_d within 1e-3).
Prediction if the growth is the 768 O(h^2) grid residue:
lambda2(1024)/lambda2(768) ~ (768/1024)^2 = 0.5625 or no growth at
all; intrinsic ghost instability: comparable positive rates.

Usage:
    uv run python scripts/experiments/ghost_mode_growth.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import OUT  # noqa: E402

DT = torch.float64


def z_m(pos, m_order, sel):
    x = pos[sel]
    c = x.mean(0)
    d = x - c
    r = d.norm(dim=1)
    th = torch.atan2(d[:, 1], d[:, 0])
    dr = r - r.mean()
    re = float((dr * torch.cos(m_order * th)).mean() * 2)
    im = float(-(dr * torch.sin(m_order * th)).mean() * 2)
    return complex(re, im), float(r.mean())


def series_modes(tag, steps_min):
    ck = torch.load(OUT / f"exact_merger{tag}_states.pt", weights_only=True)
    js = {r["step"]: r for r in json.load(
        open(OUT / f"exact_merger{tag}.json"))["series"]}
    rows = []
    for k in sorted(ck):
        if k < steps_min:
            continue
        pos = ck[k]["positions"]
        r = pos.norm(dim=1)
        # split the ghost pair by radius inside the wall
        wall = r < 0.6
        rw = r[wall]
        thr = 0.5 * (float(rw.min()) + float(rw.max()))
        disk = wall & (r < thr)
        hole = wall & (r >= thr)
        zd, Rd = z_m(pos, 2, disk)
        zh, Rh = z_m(pos, 2, hole)
        z4d, _ = z_m(pos, 4, disk)
        z4h, _ = z_m(pos, 4, hole)
        rec = js.get(k, {})
        cc = rec.get("contact_cert_committed") or rec.get(
            "contact_cert_source")
        cur = (cc[0]["metrics"]["eps_cur2_max"]
               if isinstance(cc, list) and cc else None)
        rows.append(dict(
            step=k, R_d=Rd, R_h=Rh, gap=Rh - Rd,
            q_wall=rec.get("q_wall_mean"), eps_cur2=cur,
            std_disk=rec.get("std_disk"), S_disk=rec.get("S_disk"),
            z2_d=(zd.real, zd.imag), z2_h=(zh.real, zh.imag),
            A2_plus=abs((zd + zh) / 2), A2_minus=abs((zh - zd) / 2),
            phi2_plus_deg=math.degrees(
                math.atan2(-((zd + zh) / 2).imag, ((zd + zh) / 2).real)
                / 2.0) % 180.0,
            A4_d=abs(z4d), A4_h=abs(z4h)))
    return rows


def growth(rows, key="A2_plus"):
    out = []
    for a, b in zip(rows[:-1], rows[1:]):
        if a[key] > 0 and b[key] > 0 and b["step"] > a["step"]:
            out.append(dict(step_a=a["step"], step_b=b["step"],
                            R_d=a["R_d"],
                            lam=math.log(b[key] / a[key])
                            / (b["step"] - a["step"])))
    return out


def main():
    torch.set_default_dtype(DT)
    rep = {}
    for tag, lo in (("_endgame5", 1900), ("_b2r_1024c", 2025),
                    ("_b2r_1024", 1900)):
        p = OUT / f"exact_merger{tag}_states.pt"
        if not p.exists():
            print(f"-- {tag} missing")
            continue
        rows = series_modes(tag, lo)
        rep[tag] = dict(rows=rows, growth=growth(rows))
        print(f"== {tag}")
        for r in rows:
            print(f"  {r['step']}: R_d {r['R_d']:.4f} R_h {r['R_h']:.4f} "
                  f"gap {r['gap']:.4f} q_wall {r['q_wall']} cur2 "
                  f"{r['eps_cur2']} | A2+ {r['A2_plus']:.2e} A2- "
                  f"{r['A2_minus']:.2e} phi2+ {r['phi2_plus_deg']:.0f} | "
                  f"A4 d/h {r['A4_d']:.1e}/{r['A4_h']:.1e} | std_disk "
                  f"{r['std_disk']}")
        for g in rep[tag]["growth"]:
            print(f"    lambda2+ [{g['step_a']}-{g['step_b']}] R_d "
                  f"{g['R_d']:.4f}: {g['lam']:+.4f}/step")
    # matched comparison: pair windows by R_d
    if "_endgame5" in rep and "_b2r_1024c" in rep:
        g768 = rep["_endgame5"]["growth"]
        g1024 = rep["_b2r_1024c"]["growth"]
        print("== matched R_d comparison (|dR_d| < 2e-3)")
        for a in g768:
            for b in g1024:
                if abs(a["R_d"] - b["R_d"]) < 2e-3:
                    ratio = (b["lam"] / a["lam"]) if a["lam"] != 0 else None
                    print(f"  R_d {a['R_d']:.4f}: lam768 {a['lam']:+.4f} "
                          f"lam1024 {b['lam']:+.4f} ratio {ratio}")
    Path("results/reports/endgame5_ghost_modes.json").write_text(
        json.dumps(rep, indent=1, default=str))
    print("wrote results/reports/endgame5_ghost_modes.json")


if __name__ == "__main__":
    main()
