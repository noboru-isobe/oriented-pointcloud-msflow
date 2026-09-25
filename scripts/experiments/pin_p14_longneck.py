"""P1-4 long-neck persistence test: static pre-check (reviewer STOP 3
protocol section 5). R = 0.55, L = 2.0, area-normalized to A0 =
0.8 pi. Pins: P_eff(0) > 4 via the DYNAMIC estimator on the
area-normalized cloud (vs generator), w0 > w*, delta_split(0) > 0,
box clearance, N/blend resolution; lifted 1D solve for the planning
timescale (floor time, NOT a T* estimate)."""
import json, math, sys
from pathlib import Path
sys.path.insert(0, "scripts/experiments"); sys.path.insert(0, ".")
import torch, numpy as np
torch.set_default_dtype(torch.float64)
import two_ellipses_benchmark as teb
from neck_telemetry import (neck_width_strips, p_eff_dynamic,
                            delta_split, neck_control_area)
from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.shapes.dumbbell import build_capsule_dumbbell, CapsuleProfile
from lift_profiles_1d import lift
from lubrication1d import solve_neck

H = teb.H_REF; A0 = math.pi * 0.8
c = build_capsule_dumbbell(R_l=0.55, R_r=0.55, L=2.0, w=0.29, h=H,
                           area_target=A0)
pos, ang = c["positions"], c["angles"]
nor = OrientedPointCloudVarifold(positions=pos, angles=ang).normals
ns = neck_width_strips(pos, nor, H)
pe = p_eff_dynamic(pos, nor, H, ns["w_strip"], ns["x_plateau"])
ds = delta_split(c["perimeter"], c["area"])
sc = c["scale"]
prof = CapsuleProfile(0.55, 0.55, 2.0, 0.29, None)
ell, H_ell, h0, p_eff_gen = lift(prof)
r = solve_neck(P=p_eff_gen, t_end=3.0, dt=1e-5, h0=h0)
c_mob = json.load(open("results/pinchoff/p1bc_fit_summary.json"))["c_mob"]
t_floor = (r["t"][-1] * (ell * sc) ** 4 / (c_mob * H_ell * sc)
           if r["status"] == "pinch_floor" else None)
out = dict(
    fixture=dict(R=0.55, L=2.0, w_gen=0.29, scale=sc,
                 L_scaled=c["L"], w0=c["w"], area=c["area"],
                 perimeter=c["perimeter"], N=int(pos.shape[0]),
                 blend=c["blend"]),
    pins=dict(
        p_eff_dyn_l=pe["p_eff_l"], p_eff_dyn_r=pe["p_eff_r"],
        p_eff_generator=p_eff_gen,
        p_eff_gt_4=bool(min(pe["p_eff_l"], pe["p_eff_r"]) > 4),
        w0=ns["w_strip"], w_star=0.075,
        w0_gt_wstar=bool(ns["w_strip"] > 0.075),
        delta_split0=ds, delta_split_pos=bool(ds > 0),
        x_extent=[float(pos[:, 0].min()), float(pos[:, 0].max())],
        clearance=float(3.0 - pos.abs().max()),
        clearance_ok=bool(3.0 - float(pos.abs().max()) > 0.3),
        spacing_cv=float(((pos.roll(-1, 0) - pos).norm(dim=1).std()
                          / (pos.roll(-1, 0) - pos).norm(dim=1).mean()))),
    lift_1d=dict(P_eff=p_eff_gen, status=r["status"],
                 hhat0=float(r["min_h"][0]), tau_end=float(r["t"][-1]),
                 t_floor_2D=t_floor,
                 note="floor time hhat=1e-3, NOT a T* estimate"))
Path("results/pinchoff/p14_longneck_pins.json").write_text(
    json.dumps(out, indent=1))
for k, v in out["pins"].items(): print(f"{k}: {v}")
print("lift:", out["lift_1d"])
