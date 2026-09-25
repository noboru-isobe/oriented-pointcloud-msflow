"""0I: sigma-N static refinement of the contact-layer shape drift.

Reviewer-approved protocol (2026-08-14). Decides whether the
sigma-energy's layer shape interaction is harmless in the joint limit:

State family (physical-path type, eq 1): from the organic 1400 state
extract R_d* (disk radius) and A_ann* (annulus area); for each cell
    R_d = R_d*,   R_h = R_d* + delta*sigma,
    R_o = sqrt(R_h^2 + A_ann*/pi),
so disk and annulus areas are IDENTICAL across cells and only the gap
changes (matches the pre-contact MS trajectory; the symmetric R+-d/2
family is kept as a secondary path-independence control).

Seed (eq 2): primary A2/sigma = 0.0554 (organic ratio); controls
A2 = 0 and A2/sigma = 0.0277 (linearity). Area-exact perturbation
(eq 3): r(t) = R0 + A2 cos(2(t-phi)), R0 = sqrt(R_d*^2 - A2^2/2),
with POLAR-GRAPH normals n = (r e_r - r' e_t)/|.| (the force is a
normal-pattern x q interaction -- radial normals would be wrong).

Cells: matched ladder (sigma, N_out) = (0.10,256), (0.075,342),
(0.05,512) with grid ~ 512*(0.1/sigma) (dx/sigma const) and
tau/sigma const (eq 4); over-resolved particle control (0.075,684);
grid control (0.05, grid 1536); tau/2 control. Sampling-phase
control: phi in {phi_1400, phi_1400 + dtheta_disk/2}; cells whose
sign flips with phase are classified quadrature-sensitive.

Per (cell, delta in {0.45,0.30,0.20,0.10}, phase): ONE production MM
step (frozen-q objective, incidence rows on), measuring
v2 = dA2/tau and v_d = d(gap)/tau on the COMMITTED state (production
semantics; the MM-minimizer values are recorded too). Cumulative
drift over the layer crossing (eq 5): dA2/dd = v2/|v_d| integrated
over d, reporting Delta_signed, Delta_TV and the max excursion A_exc.
Acceptance (eqs 6,7): A_exc/R_d* decreasing in sigma;
sup_sigma A_exc/h_wall < 1; A2(delta) < d(delta)/2 along the way.

Usage:
    uv run python scripts/experiments/sigma_n_refinement.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import OUT, masses_for  # noqa: E402
from shape_mode_audit import disk_sheet, fourier  # noqa: E402

from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.shapes.generator import (  # noqa: E402
    generate_oriented_annulus,
    generate_oriented_circle,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402

DT = torch.float64
DELTAS = (0.45, 0.30, 0.20, 0.10)
SEED_RATIO = 0.0554          # A2 / sigma, organic 1400 value
R1, R2, R3 = 0.35, 0.55, 1.0

CELLS = [
    dict(name="m10", sigma=0.10, n_out=256, grid=512, tau_f=1.0),
    dict(name="m075", sigma=0.075, n_out=342, grid=684, tau_f=1.0),
    dict(name="m05", sigma=0.05, n_out=512, grid=1024, tau_f=1.0),
    dict(name="or075", sigma=0.075, n_out=684, grid=684, tau_f=1.0),
    dict(name="gc05", sigma=0.05, n_out=512, grid=1536, tau_f=1.0),
    dict(name="tau075", sigma=0.075, n_out=342, grid=684, tau_f=0.5),
]

# 0K-C: mass estimator under test (settable via --mass-estimator)
MASS_EST = "oriented_kde"


def organic_reference():
    ck = torch.load(OUT / "exact_merger_0f_oriented_states.pt",
                    weights_only=True)
    st = ck[1400]
    v = OrientedPointCloudVarifold(positions=st["positions"],
                                   angles=st["angles"])
    v0d, v0t = map(float, compute_recommended_params(
        build_cloud_n(256).positions))
    m = masses_for(MASS_EST, v.positions, v.normals, v0d, v0t)
    r = v.positions.norm(dim=1)
    wall = r < 0.6
    outw = (v.positions * v.normals).sum(1) > 0
    R_d = float((m[wall & outw] * r[wall & outw]).sum()
                / m[wall & outw].sum())
    hole = wall & ~outw
    R_h = float((m[hole] * r[hole]).sum() / m[hole].sum())
    R_o = float((m[~wall] * r[~wall]).sum() / m[~wall].sum())
    A_ann = math.pi * (R_o ** 2 - R_h ** 2)
    _, w, rad, th = disk_sheet(v, m)
    _, four = fourier(w, rad, th)
    a2, b2, _ = four[2]
    phi = 0.5 * math.atan2(b2, a2)
    return R_d, A_ann, phi


def build_cloud_n(n_out):
    """t=0 cloud at resolution n_out (build_cloud parameterized)."""
    spacing = 2.0 * math.pi * R3 / n_out
    n_disk = max(3, round(2.0 * math.pi * R1 / spacing))
    disk = generate_oriented_circle(n_disk, R1, (0.0, 0.0), "cpu", DT)
    ann = generate_oriented_annulus(n_out, R3, R2, (0.0, 0.0),
                                    "cpu", DT)
    return OrientedPointCloudVarifold(
        positions=torch.cat([disk.positions, ann.positions]),
        angles=torch.cat([disk.angles, ann.angles]))


def synthetic_cell_state(R_d_star, A_ann, sigma, delta, n_out,
                         a2, phi, k=2, isolated=False):
    """Eq (1)+(3): disk sheet with area-exact k-mode (polar normals),
    hole sheet at R_d* + delta*sigma (inward), outer ring preserving
    the annulus area. Wall counts scale as the organic 90:141.
    isolated=True returns the perturbed disk sheet ALONE (the 0J-C
    isolated-self control: same intrinsic geometry, no interaction)."""
    n_d = max(8, round(90 * n_out / 256))
    n_h = max(8, round(141 * n_out / 256))
    r0 = math.sqrt(R_d_star ** 2 - 0.5 * a2 ** 2)
    t_d = 2 * math.pi * torch.arange(n_d, dtype=DT) / n_d
    r_t = r0 + a2 * torch.cos(k * (t_d - phi))
    rp_t = -k * a2 * torch.sin(k * (t_d - phi))
    pos_d = torch.stack([r_t * t_d.cos(), r_t * t_d.sin()], 1)
    e_r = torch.stack([t_d.cos(), t_d.sin()], 1)
    e_t = torch.stack([-t_d.sin(), t_d.cos()], 1)
    n_vec = r_t[:, None] * e_r - rp_t[:, None] * e_t
    n_vec = n_vec / n_vec.norm(dim=1, keepdim=True)
    ang_d = torch.atan2(n_vec[:, 1], n_vec[:, 0])

    if isolated:
        return OrientedPointCloudVarifold(positions=pos_d,
                                          angles=ang_d), n_d
    R_h = R_d_star + delta * sigma
    R_o = math.sqrt(R_h ** 2 + A_ann / math.pi)
    hole = generate_oriented_circle(n_h, R_h, (0.0, 0.0), "cpu", DT)
    outer = generate_oriented_circle(n_out, R_o, (0.0, 0.0),
                                     "cpu", DT)
    return OrientedPointCloudVarifold(
        positions=torch.cat([pos_d, hole.positions, outer.positions]),
        angles=torch.cat([ang_d, hole.angles + math.pi,
                          outer.angles])), n_d


def cell_config(cell, delta_kde, tau_kde, target0, q_mode="full"):
    import sys as _s
    _s.path.insert(0, str(Path(__file__).parent))
    from p1_production_comparison import make_cfg

    from src.torch.transport.grid_wasserstein import GridMetricConfig
    from src.torch.transport.phase_grid import PhaseGridConfig

    cfg = make_cfg("C3", delta_kde, tau_kde, 2)
    sigma = cell["sigma"]
    cfg.time_step = 1e-5 * (sigma / 0.1) * cell["tau_f"]
    cfg.perimeter_sigma = sigma
    cfg.angle_sigma = sigma
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(cell["grid"], cell["grid"]),
        fill_epsilon=0.4 * sigma,
        support_threshold=3e-3, projection_rel_tol=5e-2),
        compatibility_components="conservative_sweep")
    cfg.redistribute = True
    cfg.remove_dead_points = False
    cfg.enforce_support_gates = False
    cfg.grid_volume_target_mode = "initial_fixed"
    cfg.grid_phase_support_projection = False
    cfg.grid_bulk_rows_mode = "augment"
    cfg.mass_estimator = MASS_EST
    cfg.perimeter_q_mode = q_mode
    return cfg


def measure(v, delta_kde, tau_kde):
    m = masses_for(MASS_EST, v.positions, v.normals,
                   delta_kde, tau_kde)
    r = v.positions.norm(dim=1)
    wall = r < 0.6
    outw = (v.positions * v.normals).sum(1) > 0
    hole = wall & ~outw
    R_d = float((m[wall & outw] * r[wall & outw]).sum()
                / m[wall & outw].sum())
    R_h = float((m[hole] * r[hole]).sum() / m[hole].sum())
    _, w, rad, th = disk_sheet(v, m)
    _, four = fourier(w, rad, th)
    return dict(A2=four[2][2], gap=R_h - R_d,
                h_wall=float(m[wall].median()))


def one_step(cell, R_d_star, A_ann, phi_off, delta, a2, kde, target0,
             q_mode="full"):
    delta_kde, tau_kde = kde
    v, n_d = synthetic_cell_state(R_d_star, A_ann, cell["sigma"],
                                  delta, cell["n_out"], a2, phi_off)
    cfg = cell_config(cell, delta_kde, tau_kde, target0,
                      q_mode=q_mode)
    stp = MMStepper(cfg)
    stp._grid_target_volume_initial = target0
    before = measure(v, delta_kde, tau_kde)
    res = stp.step(v)
    vv = (res.committed_varifold if res.committed_varifold is not None
          else res.varifold)
    after = measure(vv, delta_kde, tau_kde)
    mm_after = measure(res.varifold, delta_kde, tau_kde)
    tau = cfg.time_step
    return dict(
        delta=delta, A2_before=before["A2"], gap_before=before["gap"],
        h_wall=before["h_wall"],
        v2=(after["A2"] - before["A2"]) / tau,
        vd=(after["gap"] - before["gap"]) / tau,
        v2_mm=(mm_after["A2"] - before["A2"]) / tau,
        eps_flux=(max(abs(x) for x in res.bulk_flux_residuals)
                  if res.bulk_flux_residuals else None),
        converged=bool(res.converged), tau=tau)


def excursion(rows, sigma):
    """Eq (5): integrate dA2/dd = v2/|vd| over d (trapezoid on the
    delta grid, descending), return signed / TV / max excursion."""
    rows = sorted(rows, key=lambda r: -r["delta"])
    ds = [r["delta"] * sigma for r in rows]
    f = [r["v2"] / abs(r["vd"]) if r["vd"] else 0.0 for r in rows]
    cum, cum_tv, path = 0.0, 0.0, [0.0]
    for i in range(1, len(rows)):
        dd = ds[i - 1] - ds[i]
        cum += 0.5 * (f[i - 1] + f[i]) * dd
        cum_tv += 0.5 * (abs(f[i - 1]) + abs(f[i])) * dd
        path.append(cum)
    return dict(signed=cum, tv=cum_tv,
                exc=max(abs(x) for x in path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--q-mode", default="full",
                    choices=("full", "loop_mean", "self_renormalized"))
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--mass-estimator", default="oriented_kde",
                    choices=("oriented_kde", "loopwise_oriented_kde"))
    args = ap.parse_args()
    global MASS_EST
    MASS_EST = args.mass_estimator

    R_d_star, A_ann, phi = organic_reference()
    print(f"organic reference: R_d*={R_d_star:.4f} A_ann*={A_ann:.4f} "
          f"phi={phi:.4f}", flush=True)
    report = dict(R_d_star=R_d_star, A_ann=A_ann, phi=phi, cells={})

    cells = (CELLS[:1] if args.smoke
             else CELLS[:3] if args.q_mode != "full" else CELLS)
    deltas = DELTAS[:1] if args.smoke else DELTAS
    for cell in cells:
        v0 = build_cloud_n(cell["n_out"])
        kde = tuple(map(float,
                        compute_recommended_params(v0.positions)))
        m0 = masses_for(MASS_EST, v0.positions, v0.normals,
                        *kde)
        target0 = float(
            0.5 * (m0 * (v0.positions * v0.normals).sum(-1)).sum())
        a2 = SEED_RATIO * cell["sigma"]
        n_d = max(8, round(90 * cell["n_out"] / 256))
        dphi = math.pi / n_d          # half the disk sampling step
        crec = dict(cell=cell, runs=[])
        for phase_name, phase in (("base", phi), ("half", phi + dphi)):
            rows = []
            for delta in deltas:
                try:
                    row = one_step(cell, R_d_star, A_ann, phase,
                                   delta, a2, kde, target0,
                                   q_mode=args.q_mode)
                    row["phase"] = phase_name
                    rows.append(row)
                    print(f"[{cell['name']}/{phase_name}] d/s={delta}"
                          f" v2={row['v2']:+.3e} vd={row['vd']:+.3e}"
                          f" eps={row['eps_flux']:.1e}", flush=True)
                except Exception as exc:      # noqa: BLE001
                    print(f"[{cell['name']}/{phase_name}] d/s={delta}"
                          f" ERROR {type(exc).__name__}: {exc}",
                          flush=True)
                    rows.append(dict(delta=delta, phase=phase_name,
                                     error=str(exc)))
            ok = [r for r in rows if "v2" in r]
            summ = (excursion(ok, cell["sigma"])
                    if len(ok) >= 2 else None)
            crec["runs"].append(dict(phase=phase_name, rows=rows,
                                     excursion=summ))
            if summ and ok:
                hw = ok[0]["h_wall"]
                print(f"  == {cell['name']}/{phase_name}: "
                      f"A_exc={summ['exc']:.3e} "
                      f"exc/R={summ['exc'] / R_d_star:.3e} "
                      f"exc/h_wall={summ['exc'] / hw:.3f} "
                      f"signed={summ['signed']:+.3e} "
                      f"tv={summ['tv']:.3e}", flush=True)
        # seed controls on matched-ladder cells only (delta subset)
        if not args.smoke and cell["name"].startswith("m"):
            for tag, a2c in (("zero", 0.0),
                             ("half", 0.5 * a2)):
                rows = []
                for delta in (0.45, 0.20):
                    try:
                        row = one_step(cell, R_d_star, A_ann, phi,
                                       delta, a2c, kde, target0,
                                       q_mode=args.q_mode)
                        row["phase"] = f"seed_{tag}"
                        rows.append(row)
                        print(f"[{cell['name']}/seed_{tag}] "
                              f"d/s={delta} v2={row['v2']:+.3e}",
                              flush=True)
                    except Exception as exc:      # noqa: BLE001
                        rows.append(dict(delta=delta, error=str(exc)))
                crec["runs"].append(dict(phase=f"seed_{tag}",
                                         rows=rows, excursion=None))
        report["cells"][cell["name"]] = crec
        Path(f"results/reports/phase3c0i_refinement"
             f"{args.out_tag}.json").write_text(
            json.dumps(report, indent=1, default=float))
    print("wrote results/reports/phase3c0i_refinement.json")


if __name__ == "__main__":
    main()
