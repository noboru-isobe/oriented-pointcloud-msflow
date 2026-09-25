"""Notch geometry diagnosis for the arm-4+5 v4 death (step 989).

Verdict to establish (pre-registered rules in the plan):
  A -- physical, representation-limited: the notch arms approach
       smoothly through eps at physical velocity; pairs sit on
       contiguous index runs across the notch; no folds.
  B -- numerical buckling: scattered pairs, displacement spikes,
       tangent reversals along the (verified) curve order.

Analysis only: no solver/spec/gate changes. Index order is used as a
DIAGNOSTIC tool (the algorithm is permutation-equivariant, pinned by
test_permutation_invariance) and is verified before use.

Usage:
  python diag_notch_v4.py phase1          # static geometry @900 (~1 min)
  python diag_notch_v4.py capture         # instrumented replica 900->death
  python diag_notch_v4.py analyze         # post-hoc verdict + figures
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
)
from src.torch.shapes.generator import generate_oriented_two_ellipses
from src.torch.transport.bem_wasserstein import compute_coherence

DT = torch.float64
EPS = 0.04
N_PER = 256
STATES = ROOT / "results/grid_metric/pair_contact_otto_adv_fade_states.pt"
OUTDIR = ROOT / "results/grid_metric/notch_diag"
CAPTURE = OUTDIR / "capture_states.pt"


def _setup():
    v0 = generate_oriented_two_ellipses(
        N_PER, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu", dtype=DT)
    delta, tau = compute_recommended_params(v0.positions)
    from p1_production_comparison import make_cfg
    cfg = make_cfg("C3", delta, tau, 2)
    return v0, delta, tau, cfg


def _load(step: int) -> OrientedPointCloudVarifold:
    ck = torch.load(STATES, weights_only=True)
    st = ck[step]
    return OrientedPointCloudVarifold(positions=st["positions"],
                                      angles=st["angles"])


def _verify_curve_order(pos):
    """Array order == curve order iff consecutive gaps are uniform."""
    rep = {}
    ok = True
    for name, sl in (("ellipse1", slice(0, N_PER)),
                     ("ellipse2", slice(N_PER, 2 * N_PER))):
        p = pos[sl]
        gaps = (p.roll(-1, 0) - p).norm(dim=1)
        ratio = float(gaps.max() / gaps.median())
        rep[name] = {"max_gap": float(gaps.max()),
                     "median_gap": float(gaps.median()),
                     "max_over_median": ratio}
        # one large gap is allowed per loop ONLY if the loop is open
        # (it never is here); tolerate moderate stretching near the
        # notch but flag anything pathological
        ok = ok and ratio < 8.0
    return ok, rep


def _pairs(pos, nrm):
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    return (d < EPS) & ((nrm @ nrm.T) < -0.8)


def _contiguous_runs(idx_sorted):
    runs, start, prev = [], None, None
    for i in idx_sorted:
        if start is None:
            start = prev = i
            continue
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    if start is not None:
        runs.append((start, prev))
    return runs


def _curvature(pos, sl):
    """Discrete signed curvature along the (verified) index order."""
    p = pos[sl]
    a, b, c = p.roll(1, 0), p, p.roll(-1, 0)
    ab, cb = b - a, c - b
    cross = ab[:, 0] * cb[:, 1] - ab[:, 1] * cb[:, 0]
    la, lc = ab.norm(dim=1), cb.norm(dim=1)
    chord = (c - a).norm(dim=1)
    kappa = 2.0 * cross / (la * lc * chord).clamp_min(1e-300)
    return kappa


def phase1():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    v0, delta, tau, cfg = _setup()
    v = _load(900)
    pos, nrm = v.positions, v.normals
    ok, order_rep = _verify_curve_order(pos)
    print("curve-order check:", json.dumps(order_rep, indent=1))
    if not ok:
        print("!! order broken -- fall back to order estimation needed")
        return

    m = compute_masses(pos, delta, tau, cfg.mass_kernel)
    q = compute_coherence(v, m, cfg.perimeter_sigma,
                          cfg.perimeter_kernel, backend=cfg.backend)
    P = _pairs(pos, nrm)
    paired = P.any(dim=1)
    idx = torch.nonzero(paired).flatten().tolist()
    runs = _contiguous_runs(idx)
    print(f"paired marks: {len(idx)}; contiguous runs: "
          f"{[(a, b, b - a + 1) for a, b in runs]}")
    print("q of paired marks: "
          f"min={float(q[paired].min()):.2f} "
          f"med={float(q[paired].median()):.2f} "
          f"max={float(q[paired].max()):.2f}")
    # where are they?
    cx = pos[paired]
    print(f"paired centroid clusters: x in "
          f"[{float(cx[:,0].min()):+.3f},{float(cx[:,0].max()):+.3f}], "
          f"|y| in [{float(cx[:,1].abs().min()):.3f},"
          f"{float(cx[:,1].abs().max()):.3f}]")
    # notch geometry: concave curvature maxima per loop
    rep = {"order": order_rep, "n_paired": len(idx),
           "runs": [(int(a), int(b)) for a, b in runs]}
    for name, sl in (("ellipse1", slice(0, N_PER)),
                     ("ellipse2", slice(N_PER, 2 * N_PER))):
        kappa = _curvature(pos, sl)
        # outward normals + counterclockwise order -> convex boundary
        # has one sign; the notch is the strongest OPPOSITE-sign spot
        sign = torch.sign(kappa.median())
        concave = -sign * kappa
        top = torch.topk(concave, 6)
        locs = [(int(i) + sl.start,
                 round(float(pos[sl][i, 0]), 3),
                 round(float(pos[sl][i, 1]), 3),
                 round(float(concave[i]), 1)) for i in top.indices]
        r_min = 1.0 / max(float(top.values.max()), 1e-12)
        print(f"{name}: strongest concave kappa={float(top.values.max()):.1f}"
              f" (radius {r_min:.4f} = {r_min/EPS:.2f} eps) at {locs[:3]}")
        rep[name] = {"max_concave_kappa": float(top.values.max()),
                     "min_radius_over_eps": r_min / EPS,
                     "top_concave": locs}
    # arm separation: min cross-lobe distance among paired marks
    left = paired & (torch.arange(2 * N_PER) < N_PER)
    right = paired & (torch.arange(2 * N_PER) >= N_PER)
    if bool(left.any()) and bool(right.any()):
        d_arm = float(torch.cdist(pos[left], pos[right]).min())
        print(f"min cross-lobe paired distance @900: {d_arm:.4f} "
              f"= {d_arm/EPS:.2f} eps")
        rep["min_cross_lobe_distance_over_eps"] = d_arm / EPS
    (OUTDIR / "phase1.json").write_text(json.dumps(rep, indent=1))
    print("written:", OUTDIR / "phase1.json")


def capture():
    """Instrumented deterministic replica of v4 (resume 900)."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.mm_step import MMStepper
    from src.torch.transport.grid_wasserstein import GridMetricConfig
    from src.torch.transport.phase_grid import PhaseGridConfig

    v0, delta, tau, cfg = _setup()
    cfg.time_step = 1e-5
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(512, 512), fill_epsilon=EPS,
        support_threshold=3e-3, projection_rel_tol=5e-2),
        compatibility_components="conservative_sweep")
    cfg.redistribute = True
    cfg.remove_dead_points = False
    cfg.enforce_support_gates = False
    cfg.grid_volume_target_mode = "initial_fixed"
    cfg.grid_phase_support_projection = False
    cfg.grid_fossil_advection = True
    cfg.grid_carrier_amplitude_fade = True

    m0 = compute_masses(v0.positions, delta, tau, cfg.mass_kernel)
    target0 = float(0.5 * (m0 * (v0.positions
                                 * v0.normals).sum(-1)).sum())
    _orig = MMStepper.__init__

    def _seed(s, *a, **k):
        _orig(s, *a, **k)
        s._grid_target_volume_initial = target0
    MMStepper.__init__ = _seed

    v_start = _load(900)
    solver = MMSolver(cfg)
    states = {900: dict(positions=v_start.positions.clone(),
                        angles=v_start.angles.clone())}

    def cb(step, r):
        vv = (r.committed_varifold if r.committed_varifold is not None
              else r.varifold)
        states[901 + step] = dict(positions=vv.positions.clone(),
                                  angles=vv.angles.clone())
        if (901 + step) % 10 == 0:
            torch.save(states, CAPTURE)
            print(f"  captured through {901 + step}", flush=True)
        return False

    err = None
    try:
        solver.solve(v_start, 3100, callback=cb)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    torch.save(states, CAPTURE)
    last = max(states)
    print(f"capture ended at step {last}; err={err}")
    (OUTDIR / "capture_meta.json").write_text(json.dumps(
        dict(last_step=last, error=err), indent=1))


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = torch.load(CAPTURE, weights_only=True)
    meta = json.loads((OUTDIR / "capture_meta.json").read_text())
    steps = sorted(states)
    print(f"captured steps {steps[0]}..{steps[-1]}; death: "
          f"{meta['error']}")
    v0, delta, tau, cfg = _setup()

    rep = {"death": meta, "series": []}
    # (1) arm separation + (2) displacement attribution + (3) folds
    prev_pos = None
    for s in steps:
        pos = states[s]["positions"]
        ang = states[s]["angles"]
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        nrm = v.normals
        P = _pairs(pos, nrm)
        paired = P.any(dim=1)
        left = paired & (torch.arange(pos.shape[0]) < N_PER)
        right = paired & (torch.arange(pos.shape[0]) >= N_PER)
        d_arm = (float(torch.cdist(pos[left], pos[right]).min())
                 if bool(left.any()) and bool(right.any()) else None)
        # tangent monotonicity (fold detection) along each loop
        folds = 0
        for sl in (slice(0, N_PER), slice(N_PER, 2 * N_PER)):
            p = pos[sl]
            t = p.roll(-1, 0) - p
            t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-300)
            dots = (t * t.roll(-1, 0)).sum(dim=1)
            folds += int((dots < 0.0).sum())   # >90deg turn per point
        rec = {"step": s, "n_paired": int(paired.sum()),
               "d_arm": d_arm, "folds": folds}
        if prev_pos is not None:
            disp = (pos - prev_pos).norm(dim=1)
            if bool(paired.any()) and bool((~paired).any()):
                rec["disp_paired_max"] = float(disp[paired].max())
                rec["disp_unpaired_p95"] = float(
                    disp[~paired].quantile(0.95))
        rep["series"].append(rec)
        prev_pos = pos
    (OUTDIR / "notch_diag.json").write_text(json.dumps(rep, indent=1))

    # (4) boundary snapshots
    picks = [s for s in (900, 930, 950, 965, 975, steps[-1])
             if s in states]
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4.2))
    for ax, s in zip(axes, picks):
        pos = states[s]["positions"]
        v = OrientedPointCloudVarifold(positions=pos,
                                       angles=states[s]["angles"])
        paired = _pairs(pos, v.normals).any(dim=1)
        for sl, col in ((slice(0, N_PER), "tab:blue"),
                        (slice(N_PER, 2 * N_PER), "tab:green")):
            p = pos[sl]
            ax.plot(torch.cat([p[:, 0], p[:1, 0]]),
                    torch.cat([p[:, 1], p[:1, 1]]),
                    "-", lw=0.7, color=col, alpha=0.7)
        ax.plot(pos[paired, 0], pos[paired, 1], ".", ms=3,
                color="crimson")
        ax.set_title(f"step {s} (paired={int(paired.sum())})")
        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.4, 1.4)
    fig.tight_layout()
    fig.savefig(OUTDIR / "boundary_evolution.png", dpi=140)
    print("written:", OUTDIR / "notch_diag.json",
          OUTDIR / "boundary_evolution.png")

    # verdict summary to stdout
    tail = [r for r in rep["series"] if r["step"] >= 950]
    print("\nstep  n_paired  d_arm/eps  folds  disp_paired_max/p95")
    for r in tail[::5] + tail[-3:]:
        da = f"{r['d_arm']/EPS:.2f}" if r["d_arm"] else "--"
        ratio = ""
        if "disp_paired_max" in r and r.get("disp_unpaired_p95"):
            ratio = f"{r['disp_paired_max']/max(r['disp_unpaired_p95'],1e-300):.1f}x"
        print(f"{r['step']:5d} {r['n_paired']:8d} {da:>9} "
              f"{r['folds']:5d}  {ratio}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    {"phase1": phase1, "capture": capture, "analyze": analyze}[mode]()
