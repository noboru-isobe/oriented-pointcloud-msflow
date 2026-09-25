"""D0: golden regression of the legacy production pipeline (D0.1 rev).

Freezes what `legacy_hybrid + legacy_lagged` actually does TODAY. The
harness is FAIL-CLOSED (review D0.1): a golden is written only after the
fixture passes its event assertions, contains no NaNs and no timeline
errors, completed every step, and a second run reproduced it bitwise;
verification failures exit nonzero. `--mode locked` fails on any bitwise
drift; `--mode semantic` fails only when the tolerant layer breaks
(baseline moved -- every 0D comparison void).

Fixtures (G1 is genuinely NONUNIFORM so ordinary redistribution has a
real effect to pin; the uniform circle's tangential density gradient is
zero by symmetry):

    G0  bunch-free circle, no post-processing        (nothing may fire)
    G1  bunched circle, redistribution only           (must move points)
    G2  bunched circle, deletion only                 (must remove)
    G3  bunched circle, deletion + post-removal redistribution
                                                      (all three events)

The artifact stores, per fixture: the committed trajectory, ALL FIVE
time-level payloads per step (positions/angles/IDs/keep masks/used and
current m,q/scales/flags/support state), the legacy deletion score
e_i = m_used q_used and its threshold at every deletion evaluation, the
three volume tracks ALIGNED on t_index = 0..N_STEPS, and a
self-description block (schema, full config asdict, torch version,
dtype, git commit). A human-diffable JSON summary sits next to each .pt.

Usage
-----
    uv run python scripts/experiments/d0_golden_regression.py \
        --out results/d0_golden [--mode locked|semantic]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from src.torch.diagnostics import SupportTimeline
from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.solver.mm_solver import MMSolver
from src.torch.solver.mm_step import MMConfig

DT = torch.float64
N_BASE = 48
N_STEPS = 4
SCHEMA_VERSION = 2
DEAD_RHO = 0.3

FIXTURES = {
    "G0": dict(redistribute=False, remove_dead_points=False, extra=0),
    "G1": dict(redistribute=True, remove_dead_points=False, extra=12),
    "G2": dict(redistribute=False, remove_dead_points=True, extra=12),
    "G3": dict(redistribute=True, remove_dead_points=True, extra=12,
               after_removal=3),
}


def make_initial(extra: int):
    th = (torch.arange(N_BASE, dtype=DT) + 0.5) / N_BASE * 2 * math.pi
    if extra:
        th_e = th[10] + (torch.arange(extra, dtype=DT) + 1) / (extra + 1) \
            * (2 * math.pi / N_BASE)
        th = torch.cat([th[:11], th_e, th[11:]])
    return OrientedPointCloudVarifold(
        positions=torch.stack([torch.cos(th), torch.sin(th)], 1),
        angles=th.clone())


def golden_config(fx: dict) -> MMConfig:
    return MMConfig(
        time_step=1e-5,
        bem_trace_side="legacy_exterior",
        bem_solver_mode="legacy_global_projection",
        bem_epsilon_mode="sigma",
        bem_operator_measure="visible",
        wlin_use_coherence_velocity=True,
        mass_bandwidth_type="global",
        mass_delta_adaptive=False,
        redistribution_rule="legacy_hybrid",
        dead_point_rule="legacy_lagged",
        use_unit_coherence=False,
        perimeter_kernel="wendland_c2",
        mass_kernel="wendland_c2",
        backend="naive",
        optimizer_method="trust-ncg", optimizer_tol=1e-8,
        optimizer_max_iter=200,
        redistribute=fx["redistribute"],
        redistribute_n_iters=3,
        redistribute_n_iters_after_removal=fx.get("after_removal", 0),
        remove_dead_points=fx["remove_dead_points"],
        dead_point_threshold=DEAD_RHO,
    )


def _record_payload(r):
    return dict(
        step=r.step, level=r.level,
        positions=r.positions, angles=r.angles,
        particle_ids=list(r.particle_ids),
        keep_mask=r.keep_mask,
        m_used=r.m_used, q_used=r.q_used,
        m_current=r.m_current,
        q_current=r.q_current_diagnostic,
        delta=r.delta_used, tau=r.tau_used, sigma=r.sigma_used,
        eps_bem=r.eps_bem_used,
        flags=(r.stage_enabled, r.stage_executed, r.geometry_changed,
               r.committed, r.n_removed),
        state=(r.report.state if r.report else None),
        error=r.error,
    )


def run_fixture(name: str, fx: dict) -> dict:
    v0 = make_initial(fx["extra"])
    N0 = v0.n_points
    cfg = golden_config(fx)
    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(N0)])
    hist = solver.solve(v0, N_STEPS)
    tl = solver.support_timeline

    committed = [r for r in tl.records if r.level == "post_deletion_final"]

    def vol_current_of(positions, angles, masses):
        vf = OrientedPointCloudVarifold(positions=positions, angles=angles)
        return (0.5 * ((positions * vf.normals).sum(-1) * masses)
                .sum()).item()

    # aligned volume tracks: index 0 = initial state
    pre0 = [r for r in tl.records
            if r.step == 0 and r.level == "pre_mm"][0]
    vol_current = [vol_current_of(pre0.positions, pre0.angles,
                                  pre0.m_current)]
    vol_geom = [sum(pre0.report.areas_geom) if pre0.report else
                float("nan")]
    for r in committed:
        vol_geom.append(sum(r.report.areas_geom) if r.report
                        else float("nan"))
        vol_current.append(vol_current_of(r.positions, r.angles,
                                          r.m_current)
                           if r.m_current is not None else float("nan"))

    # legacy deletion score + threshold at every deletion evaluation
    deletion_scores = []
    for r in tl.records:
        if r.level == "post_deletion_raw" and r.stage_executed:
            e = r.mq_used
            deletion_scores.append(dict(
                step=r.step, e_legacy=e,
                threshold=(DEAD_RHO * e.max()).item(),
                keep_mask=r.keep_mask, n_removed=r.n_removed))

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        git = "unknown"

    return dict(
        schema_version=SCHEMA_VERSION,
        name=name,
        config=dataclasses.asdict(cfg),
        torch_version=torch.__version__,
        dtype=str(DT), git_commit=git,
        n_steps_requested=N_STEPS,
        n_steps_completed=hist.n_completed,
        positions=[vf.positions for vf in hist._varifolds],
        angles=[vf.angles for vf in hist._varifolds],
        perimeters=hist.perimeters,
        wassersteins=hist.wassersteins,
        objectives=hist.objectives,
        converged=hist.converged,
        t_index=list(range(N_STEPS + 1)),
        volume_legacy_lagged=hist.volumes,
        volume_current=torch.tensor(vol_current, dtype=DT),
        volume_geometric=torch.tensor(vol_geom, dtype=DT),
        timeline_records=[_record_payload(r) for r in tl.records],
        deletion_scores=deletion_scores,
        particle_ids=[list(r.particle_ids) for r in committed],
        keep_masks=[r.keep_mask for r in tl.records
                    if r.level == "post_deletion_raw"],
        n_removed=[r.n_removed for r in tl.records
                   if r.level == "post_deletion_raw"],
        levels=[(r.step, r.level, r.stage_enabled, r.stage_executed,
                 r.geometry_changed, r.committed) for r in tl.records],
        support_states=[(r.step, r.level,
                         r.report.state if r.report else None)
                        for r in tl.records],
        scales=[(r.level, r.delta_used, r.tau_used, r.sigma_used,
                 r.eps_bem_used) for r in tl.records if r.step == 0],
    )


def validate_fixture(name: str, d: dict):
    """Fixture-specific event assertions -- a golden that did not run its
    events pins nothing. Raises on any violation."""
    assert d["n_steps_completed"] == N_STEPS, \
        f"{name}: only {d['n_steps_completed']}/{N_STEPS} steps"
    for k in ("perimeters", "wassersteins", "objectives",
              "volume_legacy_lagged", "volume_current",
              "volume_geometric"):
        assert torch.isfinite(d[k]).all(), f"{name}: NaN in {k}"
    for rec in d["timeline_records"]:
        assert rec["error"] is None, \
            f"{name}: timeline error at {rec['step']}/{rec['level']}: " \
            f"{rec['error']}"
    assert len(d["positions"]) == N_STEPS + 1
    assert (len(d["volume_current"]) == len(d["volume_geometric"])
            == len(d["volume_legacy_lagged"]) == N_STEPS + 1)

    lv = {(r[0], r[1]): r for r in d["levels"]}
    redist_moved = any(r[1] == "post_redistribution" and r[4]
                       for r in d["levels"])
    removed_total = sum(d["n_removed"])
    final_exec = any(r[1] == "post_deletion_final" and r[3]
                     for r in d["levels"])
    if name == "G0":
        assert not redist_moved and removed_total == 0 and not final_exec
    elif name == "G1":
        assert redist_moved, "G1 redistribution had no effect"
        # nonuniform sampling: the move must be substantive, not roundoff
        p0 = d["timeline_records"][1]["positions"]      # post_mm step 0
        pr = [r for r in d["timeline_records"]
              if r["step"] == 0 and r["level"] == "post_redistribution"
              ][0]["positions"]
        assert (p0 - pr).abs().max() > 1e-8, "G1 move is roundoff-level"
        assert removed_total == 0
    elif name == "G2":
        assert removed_total > 0 and not redist_moved and not final_exec
    elif name == "G3":
        assert removed_total > 0 and redist_moved and final_exec


def compare(golden: dict, fresh: dict, exact: bool = True) -> list:
    bad = []

    def eq_t(a, b, key, tol):
        if a is None and b is None:
            return
        if (a is None) != (b is None):
            bad.append(f"{key}: presence mismatch")
            return
        ok = torch.equal(a, b) if exact else \
            torch.allclose(a, b, rtol=0, atol=tol)
        if not ok:
            bad.append(f"{key}: max diff "
                       f"{(a - b).abs().max().item():.2e}")

    # config: every key the golden PINNED must match; fields ADDED to
    # MMConfig after the freeze (with behavior-preserving defaults) are
    # allowed but reported by name -- a changed pinned value still
    # fails. Trajectory payloads below remain the bitwise authority.
    for ck, cv in golden["config"].items():
        if ck not in fresh["config"] or fresh["config"][ck] != cv:
            bad.append(f"config[{ck}]: mismatch")
    for k in ("schema_version", "n_steps_completed",
              "t_index", "particle_ids", "keep_masks", "n_removed",
              "levels", "support_states", "scales"):
        if golden[k] != fresh[k]:
            bad.append(f"{k}: mismatch")
    if not torch.equal(golden["converged"], fresh["converged"]):
        bad.append("converged: mismatch")
    for k in ("perimeters", "wassersteins", "objectives",
              "volume_legacy_lagged", "volume_current",
              "volume_geometric"):
        eq_t(golden[k], fresh[k], k, 1e-10)
    if len(golden["positions"]) != len(fresh["positions"]) or \
            len(golden["angles"]) != len(fresh["angles"]):
        bad.append("history length mismatch")
    else:
        for i, (a, b) in enumerate(zip(golden["positions"],
                                       fresh["positions"])):
            eq_t(a, b, f"positions[{i}]", 1e-12)
        for i, (a, b) in enumerate(zip(golden["angles"],
                                       fresh["angles"])):
            eq_t(a, b, f"angles[{i}]", 1e-12)
    if len(golden["timeline_records"]) != len(fresh["timeline_records"]):
        bad.append("timeline length mismatch")
    else:
        for gr, fr in zip(golden["timeline_records"],
                          fresh["timeline_records"]):
            key = f"tl[{gr['step']}/{gr['level']}]"
            if (gr["step"] != fr["step"] or gr["level"] != fr["level"]
                    or gr["flags"] != fr["flags"]
                    or gr["state"] != fr["state"]
                    or gr["particle_ids"] != fr["particle_ids"]
                    or gr["keep_mask"] != fr["keep_mask"]
                    or (gr["error"] is None) != (fr["error"] is None)):
                bad.append(f"{key}: discrete mismatch")
            for fkey in ("positions", "angles", "m_used", "q_used",
                         "m_current", "q_current"):
                eq_t(gr[fkey], fr[fkey], f"{key}.{fkey}", 1e-12)
            # per-record scale history IS baseline (sigma and eps_BEM
            # are geometry-dependent under the legacy modes)
            for skey in ("delta", "tau", "sigma", "eps_bem"):
                ga, fa = gr[skey], fr[skey]
                if (ga is None) != (fa is None):
                    bad.append(f"{key}.{skey}: presence mismatch")
                elif ga is not None:
                    if exact:
                        if ga != fa:
                            bad.append(f"{key}.{skey}: {ga} != {fa}")
                    elif abs(ga - fa) > 1e-10:
                        bad.append(f"{key}.{skey}: |diff| "
                                   f"{abs(ga - fa):.2e}")
    if len(golden["deletion_scores"]) != len(fresh["deletion_scores"]):
        bad.append("deletion_scores length mismatch")
    else:
        for gs, fs in zip(golden["deletion_scores"],
                          fresh["deletion_scores"]):
            if (gs["step"] != fs["step"]
                    or gs["n_removed"] != fs["n_removed"]
                    or gs["keep_mask"] != fs["keep_mask"]):
                bad.append(f"deletion[{gs['step']}]: discrete mismatch")
            eq_t(gs["e_legacy"], fs["e_legacy"],
                 f"e_legacy[{gs['step']}]", 1e-12)
            if exact:
                if gs["threshold"] != fs["threshold"]:
                    bad.append(f"threshold[{gs['step']}]: mismatch")
            elif abs(gs["threshold"] - fs["threshold"]) > 1e-12:
                bad.append(f"threshold[{gs['step']}]: drift")
    return bad


def summary_json(d: dict) -> dict:
    return dict(
        schema_version=d["schema_version"], name=d["name"],
        git_commit=d["git_commit"], torch=d["torch_version"],
        n_steps=d["n_steps_completed"], n_removed=d["n_removed"],
        support_states=d["support_states"],
        volume_legacy_lagged=d["volume_legacy_lagged"].tolist(),
        volume_current=d["volume_current"].tolist(),
        volume_geometric=d["volume_geometric"].tolist(),
        config={k: v for k, v in d["config"].items()
                if not k.startswith("_")},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d0_golden"))
    ap.add_argument("--mode", choices=("locked", "semantic"),
                    default="locked")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0

    for name, fx in FIXTURES.items():
        path = args.out / f"{name}.pt"
        fresh = run_fixture(name, fx)
        try:
            validate_fixture(name, fresh)
        except AssertionError as e:
            print(f"{name}: VALIDATION FAILED: {e}")
            failures += 1
            continue
        if path.exists():
            golden = torch.load(path, weights_only=False)
            bad = compare(golden, fresh, exact=True)
            if not bad:
                print(f"{name}: bitwise reproduction PASS")
                continue
            sem = compare(golden, fresh, exact=False)
            if args.mode == "locked":
                print(f"{name}: BITWISE FAIL ({bad[:3]}...)")
                failures += 1
            elif sem:
                print(f"{name}: SEMANTIC FAIL -- baseline moved: {sem}")
                failures += 1
            else:
                print(f"{name}: bitwise drift, semantic PASS "
                      f"(environment change)")
        else:
            again = run_fixture(name, fx)
            bad = compare(fresh, again, exact=True)
            if bad:
                print(f"{name}: NOT writing golden -- second run differs:"
                      f" {bad}")
                failures += 1
                continue
            torch.save(fresh, path)
            (args.out / f"{name}_summary.json").write_text(
                json.dumps(summary_json(fresh), indent=1))
            print(f"{name}: golden written (events validated, "
                  f"self-reproduction bitwise PASS, "
                  f"removed={fresh['n_removed']})")

    if failures:
        sys.exit(f"{failures} golden failure(s)")


if __name__ == "__main__":
    main()
