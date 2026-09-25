"""Production-pipeline ablation: the five paper cells rebuilt on the
audited solver stack (MMSolver + D1b dead-point rules + D2
redistribution + support timeline + calibrated C3 scales).

Motivation (2026-08-07): the original Section 6.2.3 ablation ran on
the legacy example pipeline (scripts/examples/run_two_ellipses_batch:
own loop, ad-hoc deletion at threshold 0.15, pre-speedup W_lin,
trust-ncg at gtol 1e-8 -- ~6 s/step). Under the corrected angle map
the legacy deletion recipe destabilizes the run at the first removal
epoch (cells 4/5 blow up at step ~510 before contact), so the legacy
rerun documents "what happened to the old figures" but cannot carry
the ablation's scientific claims. This driver reruns the SAME
five-cell design on the production stack (~0.5-0.9 s/step, every
ingredient audited):

    cell1_none        q==1, no redistribution, no deletion
    cell2_coh         coherence on
    cell3_coh_redist  + redistribution (D2 production rule)
    cell4_full        + dead-point removal (production rule/threshold,
                      NOT the legacy 0.15 recipe)
    cell5_full_no_coh full with q==1

Paper pair geometry (n_per=128, N=256, gap 0.1 -- the p12
convention; the legacy ablation's n_per=64 puts the C3 carrier
bandwidth ABOVE the gap, delta ~ 0.145 > 0.1, violating the pinned
carrier-range law and blowing up within ~23 steps -- measured
before this correction), dt=1e-5 (measured
stable), 5000 steps = t=0.05 crossing the superposition contact
prediction t* ~ 0.023. Each cell runs to completion or to its
fail-closed stop; the stop step/stage IS data (the scientific window
of that configuration). BIE backend with the production C3A rank
state machine, gates enforced -- this ablation measures the
production system's behavior, unlike the ungated legacy pipeline.

Usage:
    python scripts/experiments/ablation_production.py --cell cell4_full
    python scripts/experiments/ablation_production.py          # all 5

Output: results/ablation_production/<cell>.json (series + stop record
+ provenance) and <cell>_states.pt (positions/angles every 100 steps).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import torch  # noqa: E402

from src.torch.diagnostics import SupportTimeline  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.shapes.generator import (  # noqa: E402
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_solver import MMSolver  # noqa: E402
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64
DT_STEP = 1e-5
N_STEPS = 5000
N_PER = 128
OUT = ROOT / "results" / "ablation_production"

CELLS = {
    "cell1_none": dict(use_unit_coherence=True, redistribute=False,
                       remove_dead_points=False),
    "cell2_coh": dict(use_unit_coherence=False, redistribute=False,
                      remove_dead_points=False),
    "cell3_coh_redist": dict(use_unit_coherence=False,
                             redistribute=True,
                             remove_dead_points=False),
    "cell4_full": dict(use_unit_coherence=False, redistribute=True,
                       remove_dead_points=True),
    "cell5_full_no_coh": dict(use_unit_coherence=True,
                              redistribute=True,
                              remove_dead_points=True),
}


def run_cell(name: str) -> dict:
    v0 = generate_oriented_two_ellipses(
        N_PER, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu", dtype=DT)
    n1 = N_PER
    delta, tau = compute_recommended_params(v0.positions)
    cfg = make_cfg("C3", delta, tau, 2)
    cfg.time_step = DT_STEP
    # oracle rank C=2 (paper-precontact convention): at N=128 the
    # first-frame spectrum is weak_gap and the adaptive machine
    # correctly refuses to initialise; this ablation studies the
    # post-processing ingredients, not rank inference. Post-contact
    # the oracle becomes wrong and the run fail-closes -- that stop
    # is the end of the configuration's scientific window.
    cfg.bem_rank_mode = "oracle"
    cfg.bem_component_rank = 2
    # p12 convention: the gated main line -- the metric-independent
    # support gate ends the scientific window at first unresolved
    # contact instead of letting the run continue into garbage (the
    # first VM pass ran UNGATED by omission: every cell blew up past
    # contact, P -> 0, positions scattering to |x| ~ 100; kept as the
    # ungated stress record, results/ablation_production_ungated/)
    cfg.enforce_support_gates = True
    for k, val in CELLS[name].items():
        setattr(cfg, k, val)

    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(n1), torch.arange(n1, 2 * n1)],
        store_pointwise_fields=False)

    series = []
    states = {}
    t_prev = [time.perf_counter()]

    def cb(step, result):
        now = time.perf_counter()
        v = result.varifold
        pos = v.positions
        left = pos[:, 0] < 0
        gap = (float(torch.cdist(pos[left], pos[~left]).min())
               if left.any() and (~left).any() else float("nan"))
        m, qm = result.masses, result.effective_masses
        series.append(dict(
            step=step, wall_s=now - t_prev[0], n_points=v.n_points,
            perimeter=result.perimeter,
            q_min=float((qm / m.clamp_min(1e-300)).min()),
            gap_lr=gap,
            relgrad=result.relative_gradient_norm,
            objective_decreased=bool(result.objective_decreased),
            rank_status=result.rank_status,
            rank_action=result.rank_action,
        ))
        t_prev[0] = now
        if step % 100 == 0 or step == N_STEPS - 1:
            states[step] = dict(positions=pos.clone(),
                                angles=v.angles.clone())
        if step % 250 == 0:
            r = series[-1]
            print(f"[{name}] step {step:4d}: N={r['n_points']} "
                  f"P={r['perimeter']:.4f} gap={gap:.4f} "
                  f"q_min={r['q_min']:.3f} ({r['wall_s']:.2f}s)",
                  flush=True)
        return False

    t0 = time.time()
    err, stop = None, None
    try:
        solver.solve(v0, N_STEPS, callback=cb)
    except Exception as e:  # noqa: BLE001 -- the stop IS data
        err = f"{type(e).__name__}: {e}"
        stop = "exception"
    OUT.mkdir(exist_ok=True)
    torch.save(states, OUT / f"{name}_states.pt")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    rec = dict(
        cell=name, knobs=CELLS[name],
        dt=DT_STEP, n_steps=N_STEPS, n_per=N_PER,
        completed_steps=len(series), stop=stop, error=err,
        wall_total_s=time.time() - t0, git_sha=sha,
        torch_version=torch.__version__,
        mm_config={k: (val if isinstance(val, (int, float, str, bool,
                                              type(None)))
                       else str(val))
                   for k, val in dataclasses.asdict(cfg).items()},
        series=series,
    )
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(f"[{name}] DONE {len(series)}/{N_STEPS} steps, "
          f"stop={stop} err={(err or '')[:80]} "
          f"wall={(time.time() - t0) / 60:.1f}min", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", choices=list(CELLS), default=None)
    args = ap.parse_args()
    for name in ([args.cell] if args.cell else list(CELLS)):
        run_cell(name)


if __name__ == "__main__":
    main()
