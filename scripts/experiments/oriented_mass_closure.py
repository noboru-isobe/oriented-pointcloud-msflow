"""Phase IIIc-0: antipodal-sheet-aware mass estimator closure.

Root cause fixed here (IIIb-4): position-only KDE counts coincident
antiparallel sheets as one support, so per-sheet totals scale with
point counts and the coherence keeps a count-ratio residual
|N- - N+|/(N- + N+) (= 51/231 = 0.2208, measured 0.225). The
oriented estimator weights the density by omega(n_i . n_j) = (1+s)/2
and calibrates (delta, tau) by the SAME kNN rule applied to
co-oriented neighbors. Scope: antipodal-sheet separation only --
NOT a general multiplicity estimator (co-oriented coincident sheets
out of scope); global scalar bandwidth only (per-point modes are
rejected explicitly in mass.py).

Protocol (reviewer-approved): M0 sheet closure with cutoff audit /
imperfect-antiparallel sweep / clean-shape non-regression + angle
noise (flower as separate stress test) / checkpoint-1200 estimator
ablation in two arms (phase-only static, phase+flux one-step).

Usage:
    uv run python scripts/experiments/oriented_mass_closure.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    chi_tau,
    compute_kde_density,
    compute_kde_density_oriented,
    compute_masses,
    compute_masses_oriented,
    compute_recommended_params,
    compute_recommended_params_oriented,
)
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_flower,
)
from src.torch.transport.bem_wasserstein import compute_coherence

DT = torch.float64
OUT = Path("results/reports")
SIGMA = 0.1


def wall_fixture(n1, n2, rbar=0.394, offset=0.0, normal_tilt=0.0):
    th1 = 2 * math.pi * np.arange(n1) / n1
    th2 = 2 * math.pi * (np.arange(n2) + offset) / n2
    pos = np.concatenate([
        rbar * np.stack([np.cos(th1), np.sin(th1)], 1),
        rbar * np.stack([np.cos(th2), np.sin(th2)], 1)])
    ang = np.concatenate([th1, th2 + math.pi - normal_tilt])
    return OrientedPointCloudVarifold(
        positions=torch.tensor(pos, dtype=DT),
        angles=torch.tensor(ang, dtype=DT)), n1


def estimators(v):
    """Per-estimator consistent (delta, tau) + masses + cutoff audit."""
    out = {}
    dp, tp = compute_recommended_params(v.positions)
    dp = float(dp)
    m = compute_masses(v.positions, dp, tp, "wendland_c2")
    th = compute_kde_density(v.positions, dp, "wendland_c2")
    out["old"] = dict(m=m, delta=dp, tau=float(tp),
                      n_cutoff=int((chi_tau(th, tp) < 1).sum()),
                      min_th_over_tau=float((th / tp).min()))
    do, to = compute_recommended_params_oriented(v.positions,
                                                 v.normals)
    mo = compute_masses_oriented(v.positions, v.normals, do, to,
                                 "wendland_c2")
    tho = compute_kde_density_oriented(v.positions, v.normals, do,
                                       "wendland_c2")
    out["oriented"] = dict(m=mo, delta=do, tau=to,
                           n_cutoff=int((chi_tau(tho, to) < 1).sum()),
                           min_th_over_tau=float((tho / to).min()))
    return out


def m0_sheet_closure(report):
    rows = []
    rbar = 0.394
    L = 2 * math.pi * rbar
    for s in (1, 2, 4):
        for counts, name in (((90 * s, 141 * s), "unequal"),
                             ((128 * s, 128 * s), "equal")):
            for off in (0.0, 0.5):
                v, n1 = wall_fixture(*counts, offset=off)
                est = estimators(v)
                # oracle
                mo = np.concatenate([
                    np.full(counts[0], L / counts[0]),
                    np.full(counts[1], L / counts[1])])
                est["oracle"] = dict(
                    m=torch.tensor(mo, dtype=DT), delta=None,
                    tau=None, n_cutoff=0, min_th_over_tau=None)
                for key, e in est.items():
                    m = e["m"]
                    q = compute_coherence(v, m, SIGMA,
                                          "wendland_c2")
                    M1 = float(m[:n1].sum())
                    M2 = float(m[n1:].sum())
                    rows.append(dict(
                        s=s, config=name, offset=off, estimator=key,
                        e_sheet=max(abs(M1 - L), abs(M2 - L)) / L,
                        e_q=float((m * q).sum() / m.sum()),
                        n_cutoff=e["n_cutoff"],
                        min_th_over_tau=e["min_th_over_tau"],
                        delta=e["delta"], tau=e["tau"]))
    report["m0"] = rows
    for r in rows:
        if r["offset"] == 0.0:
            print(f"M0 s={r['s']} {r['config']:8s} {r['estimator']:9s}"
                  f" e_sheet={r['e_sheet']:.4f} e_q={r['e_q']:.4f}"
                  f" cutoff={r['n_cutoff']}", flush=True)
    # acceptance: e_sheet monotone in s for oriented (unequal, off=0)
    es = [r["e_sheet"] for r in rows
          if r["estimator"] == "oriented" and r["config"] == "unequal"
          and r["offset"] == 0.0]
    report["m0_accept"] = dict(
        oriented_e_sheet_by_s=es,
        monotone_nonincreasing=bool(all(
            es[i + 1] <= es[i] * 1.05 for i in range(len(es) - 1))),
        max_e_q_oriented=max(r["e_q"] for r in rows
                             if r["estimator"] == "oriented"),
        any_cutoff_oriented=any(r["n_cutoff"] > 0 for r in rows
                                if r["estimator"] == "oriented"))
    print("M0 accept:", report["m0_accept"], flush=True)


def m0_imperfect_antiparallel(report):
    """n_i . n_j = -cos(gamma): omega residual (1-cos gamma)/2."""
    rows = []
    for ndot in (-1.0, -0.95, -0.9, -0.8):
        # angle between normals = acos(ndot); tilt from exact
        # antiparallel (pi) is pi - acos(ndot)
        tilt = math.pi - math.acos(max(-1.0, min(1.0, ndot)))
        v, n1 = wall_fixture(90, 141, normal_tilt=tilt)
        est = estimators(v)
        for key in ("old", "oriented"):
            m = est[key]["m"]
            q = compute_coherence(v, m, SIGMA, "wendland_c2")
            M1 = float(m[:n1].sum())
            M2 = float(m[n1:].sum())
            L = 2 * math.pi * 0.394
            rows.append(dict(ndot=ndot, estimator=key,
                             e_sheet=max(abs(M1 - L),
                                         abs(M2 - L)) / L,
                             e_q=float((m * q).sum() / m.sum()),
                             n_cutoff=est[key]["n_cutoff"]))
        print(f"M0-tilt ndot={ndot:+.2f}: "
              + " ".join(f"{r['estimator']}:e_q={r['e_q']:.4f}"
                         for r in rows[-2:]), flush=True)
    report["m0_tilt"] = rows


def clean_shapes(report):
    fixtures = {
        "circle": generate_oriented_circle(256, 1.0, (0.0, 0.0),
                                           "cpu", DT),
        "ellipse": generate_oriented_ellipse(256, 1.2, 0.8,
                                             (0.0, 0.0), "cpu", DT),
        "annulus": generate_oriented_annulus(192, 1.2, 0.5,
                                             (0.0, 0.0), "cpu", DT),
    }
    two = generate_oriented_circle(96, 0.5, (-1.0, 0.0), "cpu", DT)
    two2 = generate_oriented_circle(96, 0.5, (1.0, 0.0), "cpu", DT)
    fixtures["two_disks"] = OrientedPointCloudVarifold(
        positions=torch.cat([two.positions, two2.positions]),
        angles=torch.cat([two.angles, two2.angles]))
    fixtures["flower_STRESS"] = generate_oriented_flower(
        384, device="cpu", dtype=DT)
    rows = []
    for name, v in fixtures.items():
        est = estimators(v)
        rec = dict(shape=name)
        for key in ("old", "oriented"):
            m = est[key]["m"]
            q = compute_coherence(v, m, SIGMA, "wendland_c2")
            P = float((m * q).sum())
            V = float(0.5 * (m * (v.positions
                                  * v.normals).sum(1)).sum())
            rec[key] = dict(sum_m=float(m.sum()), P=P, V=V,
                            q_min=float(q.min()),
                            q_mean=float(q.mean()),
                            n_cutoff=est[key]["n_cutoff"])
        rec["e_P"] = abs(rec["oriented"]["P"] - rec["old"]["P"]) \
            / abs(rec["old"]["P"])
        rec["e_V"] = abs(rec["oriented"]["V"] - rec["old"]["V"]) \
            / max(abs(rec["old"]["V"]), 1e-12)
        rows.append(rec)
        print(f"clean {name:14s} e_P={rec['e_P']:.2e} "
              f"e_V={rec['e_V']:.2e} cutoff(or)="
              f"{rec['oriented']['n_cutoff']}", flush=True)
    report["clean_shapes"] = rows


def angle_noise(report):
    rng = np.random.default_rng(7)
    rows = []
    for name, v0 in (("circle", generate_oriented_circle(
            256, 1.0, (0.0, 0.0), "cpu", DT)),
                     ("ellipse", generate_oriented_ellipse(
                         256, 1.2, 0.8, (0.0, 0.0), "cpu", DT))):
        for eps_th in (0.0, 0.01, 0.05, 0.1):
            xi = rng.standard_normal(v0.n_points)
            v = OrientedPointCloudVarifold(
                positions=v0.positions.clone(),
                angles=v0.angles + eps_th * torch.tensor(xi,
                                                         dtype=DT))
            est = estimators(v)
            rec = dict(shape=name, eps_theta=eps_th)
            for key in ("old", "oriented"):
                m = est[key]["m"]
                q = compute_coherence(v, m, SIGMA, "wendland_c2")
                rec[key] = dict(sum_m=float(m.sum()),
                                P=float((m * q).sum()),
                                q_mean=float(q.mean()),
                                n_cutoff=est[key]["n_cutoff"])
            rows.append(rec)
    report["angle_noise"] = rows
    for r in rows:
        if r["shape"] == "circle":
            print(f"noise eps={r['eps_theta']:.2f}: "
                  f"sum_m old={r['old']['sum_m']:.4f} "
                  f"or={r['oriented']['sum_m']:.4f} "
                  f"P old={r['old']['P']:.4f} "
                  f"or={r['oriented']['P']:.4f}", flush=True)


def checkpoint_ablation(report):
    """IIIc-0C: estimator-only ablation on checkpoint 1200.
    Arm 1 (static, phase channel): CurrentToPhase projection with
    old vs oriented masses. Arm 2 (one-step, both channels):
    MMConfig.mass_estimator flag."""
    from src.torch.transport.phase_grid import (CurrentToPhase,
                                                PhaseGridConfig)

    ck = torch.load("results/exact_merger/"
                    "exact_merger_control_states.pt",
                    weights_only=True)
    st = ck[1200]
    V = OrientedPointCloudVarifold(positions=st["positions"],
                                   angles=st["angles"])
    wall = (V.positions.norm(dim=1) < 0.6).numpy()
    est = estimators(V)
    cfg = PhaseGridConfig(grid_shape=(512, 512), fill_epsilon=0.04,
                          support_threshold=3e-3)
    vol = math.pi * (1.0 ** 2 - 0.55 ** 2) + math.pi * 0.35 ** 2
    arm1 = {}
    for key in ("old", "oriented"):
        m = est[key]["m"]
        q = compute_coherence(V, m, SIGMA, "wendland_c2")
        alpha = (m * q).numpy()
        pg = CurrentToPhase(cfg).reconstruct(V, m, vol)
        arm1[key] = dict(
            alpha_wall_frac=float(alpha[wall].sum() / alpha.sum()),
            q_wall_mean=float(q.numpy()[wall].mean()),
            projection_l2_relative=pg.projection_l2_relative,
            rho_raw_max=float(pg.rho_raw.max()),
            n_cutoff=est[key]["n_cutoff"])
        print(f"ckpt arm1 {key:9s} alpha_wall="
              f"{arm1[key]['alpha_wall_frac']:.4f} "
              f"q_wall={arm1[key]['q_wall_mean']:.4f} "
              f"proj={arm1[key]['projection_l2_relative']:.3e} "
              f"raw_max={arm1[key]['rho_raw_max']:.3f}", flush=True)
    report["checkpoint_arm1_static"] = arm1

    # arm 2: one grid step, both channels via the solver flag
    from p1_production_comparison import make_cfg
    from src.torch.solver.mm_step import MMStepper
    from src.torch.transport.grid_wasserstein import GridMetricConfig

    arm2 = {}
    for key in ("kde", "oriented_kde"):
        if key == "kde":
            d, t = compute_recommended_params(V.positions)
            d = float(d)
        else:
            d, t = compute_recommended_params_oriented(V.positions,
                                                       V.normals)
        c = make_cfg("C3", d, t, 2)
        c.bem_solver_mode = "legacy_global_projection"
        c.metric_backend = "grid_poisson"
        c.grid_metric = GridMetricConfig(
            phase=PhaseGridConfig(grid_shape=(512, 512),
                                  fill_epsilon=0.04,
                                  support_threshold=3e-3,
                                  projection_rel_tol=5e-2),
            compatibility_components="conservative_sweep")
        c.grid_volume_target_mode = "current"
        c.mass_estimator = key
        stepper = MMStepper(c)
        try:
            res = stepper.step(V)
            snap = res.grid_setup_snapshot
            arm2[key] = dict(
                objective=float(res.objective),
                perimeter=float(res.perimeter),
                wasserstein=float(res.wasserstein),
                n_iter=int(res.n_iter),
                converged=bool(res.converged),
                projection=snap.projection_l2_relative
                if snap else None,
                max_disp=float(res.displacements.abs().max()))
        except Exception as exc:      # noqa: BLE001 record outcome
            arm2[key] = dict(error=f"{type(exc).__name__}: {exc}")
        print(f"ckpt arm2 {key:12s}: {arm2[key]}", flush=True)
    report["checkpoint_arm2_onestep"] = arm2


def main():
    report = {}
    m0_sheet_closure(report)
    m0_imperfect_antiparallel(report)
    clean_shapes(report)
    angle_noise(report)
    checkpoint_ablation(report)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "phase3c0.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
