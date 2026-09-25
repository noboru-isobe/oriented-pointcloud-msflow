"""0G-0a: growing-shape-mode extraction + full-objective curvature.

The 0F shape stop measured a volume-neutral radial diffusion of the
disk sheet (S_disk +0.5%/step). Before any gauge/preconditioner is
designed, this audit answers, with the PRODUCTION objective:

1. WHICH mode grows: M-weighted Fourier coefficients of r(theta) on
   the disk sheet at steps 1350/1375/1400 (M = diag(m~), so the
   numbers are point-count independent).
2. Is the grown mode a near-null direction of the full MM objective
   J = P_hat + W_grid? Central-difference curvature at t in
   {1e-3, 5e-4, 2.5e-4}, decomposed into the perimeter part and the
   (analytically exact) metric part 2 gm(xi)/||xi||_M^2.

Parity pin (reviewer-mandated): the manual assembly J_manual(s) is
verified against the stepper's optimizer objective J_production(y)
to 1e-12 relative before any curvature is quoted.

Pre-registered branch: H_J < 0 -> STOP (no gauge; discrete shape
stability must be revisited). Dominant growing k = 1 -> STOP (rigid
translation on a circle; loop-mean-preserving designs are wrong).
H_J ~ 0 or small positive -> proceed to 0G-0p.

Usage:
    uv run python scripts/experiments/shape_mode_audit.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import (  # noqa: E402
    DT_STEP,
    OUT,
    build_cloud,
    masses_for,
)
from incidence_rows_onestep import make_config  # noqa: E402

from src.torch.math_utils.angles import wrap_angles  # noqa: E402
from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402

DT = torch.float64
KMAX = 8


def disk_sheet(v, m):
    r = v.positions.norm(dim=1)
    mask = (r < 0.6) & ((v.positions * v.normals).sum(1) > 0)
    w = m[mask]
    c = (w[:, None] * v.positions[mask]).sum(0) / w.sum()
    d = v.positions[mask] - c[None, :]
    rad = d.norm(dim=1)
    th = torch.atan2(d[:, 1], d[:, 0])
    return mask, w, rad, th


def fourier(w, rad, th):
    rbar = float((w * rad).sum() / w.sum())
    dr = rad - rbar
    out = {}
    for k in range(1, KMAX + 1):
        a = float(2.0 * (w * dr * torch.cos(k * th)).sum() / w.sum())
        b = float(2.0 * (w * dr * torch.sin(k * th)).sum() / w.sum())
        out[k] = (a, b, math.hypot(a, b))
    return rbar, out


def manual_objective(stepper, s, dth):
    """J_manual(s, dtheta): reproduces mm_step.objective (P with
    re-estimated masses x frozen q, plus the grid quadratic)."""
    p = stepper.param
    pos = p.prev_positions + s[:, None] * p.prev_normals
    ang = wrap_angles(p.prev_angles + dth)
    va = OrientedPointCloudVarifold(positions=pos, angles=ang)
    masses = stepper._masses_for(va.positions, va.normals)
    P = (masses * stepper.fixed_coherence).sum()
    W = stepper.wasserstein_metric(s, dth,
                                   stepper.config.time_step)
    return P + W, P, W


def main():
    ck = torch.load(OUT / "exact_merger_0f_oriented_states.pt",
                    weights_only=True)
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    m0 = masses_for("oriented_kde", v0.positions, v0.normals,
                    delta, tau)
    target0 = float(
        0.5 * (m0 * (v0.positions * v0.normals).sum(-1)).sum())

    # -------- 1. Fourier growth table --------
    report = {"fourier": {}}
    per_state = {}
    for step in (1350, 1375, 1400):
        st = ck[step]
        v = OrientedPointCloudVarifold(positions=st["positions"],
                                       angles=st["angles"])
        m = masses_for("oriented_kde", v.positions, v.normals,
                       delta, tau)
        mask, w, rad, th = disk_sheet(v, m)
        rbar, four = fourier(w, rad, th)
        per_state[step] = (v, m, mask, th, four)
        report["fourier"][step] = {k: dict(a=a, b=b, amp=amp)
                                   for k, (a, b, amp) in four.items()}
        print(f"step {step}: rbar={rbar:.4f} amps=" + " ".join(
            f"k{k}:{four[k][2]:.2e}" for k in range(1, KMAX + 1)),
            flush=True)
    f0, f1 = per_state[1350][4], per_state[1400][4]
    growth = {k: (f1[k][2] / f0[k][2] if f0[k][2] > 0 else None)
              for k in f1}
    report["growth_1350_1400"] = growth
    grown = {k: g for k, g in growth.items() if g and g > 1.0}
    k_dom = (max(grown, key=lambda k: f1[k][2] * (growth[k] - 1))
             if grown else max(f1, key=lambda k: f1[k][2]))
    report["k_dominant"] = k_dom
    print(f"growth ratios: " + " ".join(
        f"k{k}:{g:.2f}" for k, g in growth.items() if g), flush=True)
    print(f"dominant growing mode: k = {k_dom}", flush=True)

    # -------- 2. stepper at 1400 + parity pin --------
    st = ck[1400]
    v = OrientedPointCloudVarifold(positions=st["positions"],
                                   angles=st["angles"])
    m = masses_for("oriented_kde", v.positions, v.normals, delta, tau)
    cfg = make_config("augment", delta, tau, target0)
    cfg.mass_estimator = "oriented_kde"
    stepper = MMStepper(cfg)
    stepper._grid_target_volume_initial = target0
    stepper._setup_step(v)
    p = stepper.param
    N = v.positions.shape[0]

    torch.manual_seed(0)
    y = 1e-4 * torch.randn(p.n_params, dtype=DT)
    s_t, th_t = p.unpack_params(y)
    Jp = float(stepper.objective(y))
    Jm, _, _ = manual_objective(stepper, s_t, th_t)
    parity = abs(float(Jm) - Jp) / max(1.0, abs(Jp))
    report["parity_rel"] = parity
    print(f"parity pin: |J_manual - J_production|/max(1,|J|) = "
          f"{parity:.2e}", flush=True)
    assert parity < 1e-12, "parity pin FAILED -- do not quote curvature"

    # -------- 3. curvature of modes --------
    mask, w, rad, th = disk_sheet(v, m)
    # admissible projection: project xi onto span(Q) (the composed
    # constraint kernel; q_suppress is False so pre-scale == s), which
    # makes the fail-closed grid forward accept the probe exactly
    Q = p.Q

    def project_admissible(xi):
        return Q @ (Q.T @ xi)

    def curvature(xi, label):
        xi = project_admissible(xi)
        nrm2 = float((m * xi * xi).sum())          # ||xi||_M^2
        if nrm2 <= 0:
            return None
        dth = p.AB_solve @ xi
        gmv = float(stepper.wasserstein_metric(xi, dth,
                                               cfg.time_step))
        J0, P0, _ = manual_objective(stepper,
                                     torch.zeros(N, dtype=DT),
                                     torch.zeros(N, dtype=DT))
        rowsout = {"metric_term": 2.0 * gmv / nrm2, "t": {}}
        for t in (1e-3, 5e-4, 2.5e-4):
            Jp_, Pp_, _ = manual_objective(stepper, t * xi, t * dth)
            Jm_, Pm_, _ = manual_objective(stepper, -t * xi, -t * dth)
            HJ = float((Jp_ + Jm_ - 2 * J0) / (t * t * nrm2))
            HP = float((Pp_ + Pm_ - 2 * P0) / (t * t * nrm2))
            rowsout["t"][t] = dict(H_J=HJ, H_P=HP)
        print(f"curvature[{label}]: metric={rowsout['metric_term']:.4e} "
              + " ".join(f"t={t:g}: H_J={d['H_J']:+.4e} "
                         f"H_P={d['H_P']:+.4e}"
                         for t, d in rowsout["t"].items()),
              flush=True)
        return rowsout

    report["curvature"] = {}
    idx = torch.nonzero(mask).flatten()
    for k in sorted({k_dom, 2, 3, 4, 6}):
        a, b, amp = per_state[1400][4][k]
        if amp == 0:
            continue
        xi = torch.zeros(N, dtype=DT)
        xi[idx] = (a * torch.cos(k * th) + b * torch.sin(k * th)) / amp
        report["curvature"][f"k{k}"] = curvature(xi, f"k{k}")
    # reference: wall common translation (both sheets, e_x . n)
    wallm = v.positions.norm(dim=1) < 0.6
    xi_tr = torch.zeros(N, dtype=DT)
    xi_tr[wallm] = v.normals[wallm, 0]
    report["curvature"]["wall_translation"] = curvature(
        xi_tr, "wall_translation")

    out = Path("results/reports/phase3c0g_audit.json")
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
