"""0C-5d-1: pre-contact dynamic audit -- M and Q up to the first non-clean
rank evidence, plus the continuum control that outranks any oracle run.

INDEPENDENT-COMPONENT CONTROL. Before contact, the one-phase harmonic
problems are componentwise independent: the two-component evolution should
be the superposition of each component evolved ALONE. So each ellipse is
also evolved in isolation and the concatenated isolated trajectories are
compared per step against the pair run (max position / circular-angle
differences, per-component areas and centroids -- the continuum flow
conserves each component's area and, by the curvature-vector identity, its
centroid). This separates three scenarios:

    - the gap changes because each component relaxes toward its own round
      shape (superposition holds to discretisation level);
    - the pair drifts from the isolated runs: finite-N global-BIE
      cross-block coupling, quantified here;
    - the corrected pair never approaches contact at all -- true for THIS
      AUDIT CONFIGURATION (major axes on the line of centers, gap 0.6),
      whose rounding retracts the facing tips.

SCOPE CORRECTION (0C-5d-1.1a, after user pushback; identification fixed
again at review): the audit configuration is NOT the published
two-ellipse experiment. The PAPER experiment is the minor-axes-facing
family at gap 0.1 (manuscript section 6.2.2, batch default, and the
archived run's parameters all agree: centers +-0.45); centers +-0.6 /
gap 0.4 are only the generator default. Both have negative round-limit
gaps (-0.365 paper, -0.065 generator default) -- independent relaxation
alone predicts contact, with the paper configuration's predicted contact
time inside its original computation window T = 0.05. So: the legacy
metric's EXCHANGE dynamics are spurious (Gate A), and the audit
configuration shows no fusion under the corrected metric; but the paper
geometry has a GENUINE PRE-CONTACT CONTACT MECHANISM. Not yet
established: that the corrected point-cloud scheme reproduces the merger
through and after contact (pair runs, rank transition, 0D/0E).

Stated prediction (before measurement): from this configuration the gap
GROWS -- both ellipses have their major axes along the line of centers, so
rounding retracts the facing tips while areas and centroids stay put.
Measured (first version): confirmed, 0.600 -> 0.614; and the small
ellipse's null level drifts across the fixed confidence threshold at step
~10 while the geometry stays cleanly separated (a resolution-margin
event, not topology).

RANK POLICY (0C-5d-1.1): the hysteretic state machine now runs INSIDE the
stepper (bem_rank_mode = 'state_machine'), so same-rank weak evidence
HOLDS the working rank and the evolution genuinely continues -- the
confidence-decay event above must not stop a run. Hard stops only:
no_null_block, failure to initialise, reject_artificial. No post-contact
continuation either way: rank hysteresis cannot reconstruct the phase
support (0D/0E own that).

Continuation diagnostic: the row space of the particle-level constraint
C_u = U_m^T W^{1/2} D_vel (G - R Amap) D_s between consecutive steps
(particle IDs fixed, no deletion) -- unlike the weighted endpoint bases
U0, whose principal angles mix in the weight change.

Usage
-----
    uv run python scripts/experiments/spectral_precontact_0c5d1.py \
        --out results/spectral_0c5d1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.torch.solver.mm_step import MMStepper
from src.torch.transport.bem_wasserstein import AmbiguousComponentRankError
from src.torch.transport.rank_state import (
    max_angle_span_in_span_deg,
)

sys.path.insert(0, str(Path(__file__).parent))
from spectral_flux_measure_0c5c import (  # noqa: E402
    cat, component_slices, config, ellipse, flux_activity, flux_leakage,
)
from spectral_rank_auto_0c5b import circular_angle_diff  # noqa: E402

DT = 1e-4          # 10x the one-step DT: visible motion, |s| ~ 1.5e-3 << h
N_STEPS = 40
GAP0 = 0.6


def parts_pair(h: float = 0.05):
    return [ellipse(0.7, 0.4, -0.7 - GAP0 / 2, h=h),
            ellipse(1.2, 0.8, 1.2 + GAP0 / 2, h=h)]


def make_config(variant: str, rank_mode: str = "auto",
                rank: int | None = None):
    cfg = config(variant)
    cfg.time_step = DT
    cfg.bem_rank_mode = rank_mode
    cfg.bem_component_rank = rank
    return cfg


def comp_stats(v, masses, slices):
    """Per-component conservation diagnostics, two clearly separated kinds.

    GEOMETRIC (the continuum control): ordered-polygon area and polygon
    first moments by Green's formula on the post-step vertices -- valid
    pre-contact because particle IDs and the generator's curve ordering
    are fixed. These are the discrete versions of |E_alpha| and the PHASE
    centroid (1/|E_alpha|) int_E x dx, both conserved by the continuum
    one-phase flow.

    CARRIER-LAGGED (kept under distinct names): divergence area and
    boundary-carrier mean with `masses` = MMStepResult.masses, which is
    the PRE-step fixed_masses -- a lagged estimator quantity, not a
    geometric conservation statement (and sum m x / sum m is a boundary
    centroid, a different object from the phase centroid even with
    current masses)."""
    out = []
    for i, j in slices:
        P = v.positions[i:j]
        X, Y = P[:, 0], P[:, 1]
        Xn, Yn = X.roll(-1), Y.roll(-1)
        c = X * Yn - Xn * Y
        A = 0.5 * c.sum()
        pos, m = P, masses[i:j]
        n = v.normals[i:j]
        out.append(dict(
            area_geom=A.item(),
            centroid_geom=[(((X + Xn) * c).sum() / (6 * A)).item(),
                           (((Y + Yn) * c).sum() / (6 * A)).item()],
            area_carrier_lagged=(0.5 * ((pos * n).sum(-1) * m).sum()).item(),
            boundary_mean_carrier_lagged=(
                (pos * m[:, None]).sum(0) / m.sum()).tolist(),
        ))
    return out


def rowspace_basis(C_full):
    q, _ = torch.linalg.qr(C_full.T)
    return q[:, : C_full.shape[0]]


def evolve_pair(variant: str, n_steps: int = N_STEPS,
                gap_min_override: float | None = None) -> dict:
    """0C-5d-1.1: the rank state machine now runs INSIDE the stepper
    (bem_rank_mode='state_machine'), so same-rank weak evidence holds the
    working rank and the evolution genuinely continues -- the earlier
    version's exception-catch loop re-evaluated the same geometry without
    advancing it. Hard stops (no_null_block, reject_artificial) still
    raise and are recorded as terminal frames.

    gap_min_override (2026-08-06, C_2D angle-map fix): the natural
    confidence-decay episode this fixture originally exhibited (winning
    gap ratio 62 -> <30 crossing bem_rank_gap_min=30 within ~10 steps)
    was largely DRIVEN by the pre-fix angle map -- the under-rotating
    normals rotted the small component's null block. Post-fix the ratio
    declines only 61.9 -> 60.5 in 14 steps: rank evidence is genuinely
    more stable now. Raising the evidence confidence bar after step 0
    (once the machine has initialised on a clean frame) re-creates
    sustained same-rank weak_gap evidence on the REAL evolution, which
    is what the hold-and-continue pins need to exercise."""
    parts = parts_pair()
    v = cat(*parts)
    slices = component_slices(parts)
    n1 = parts[0].n_points
    cfg = make_config(variant, "state_machine", None)
    stepper = MMStepper(cfg)
    frames, snapshots = [], []
    prev_rows = None
    for step_idx in range(n_steps):
        try:
            r = stepper.step(v)
            if step_idx == 0 and gap_min_override is not None:
                stepper.bem_wasserstein.rank_gap_min = gap_min_override
        except AmbiguousComponentRankError as e:
            frames.append(dict(step=step_idx, hard_stop=str(e),
                               status=(e.evidence.status
                                       if e.evidence else None)))
            break
        m, qm, s = r.masses, r.effective_masses, r.displacements
        rows = rowspace_basis(stepper._spectral_C_full)
        cu_angle = (max_angle_span_in_span_deg(rows, prev_rows)
                    if prev_rows is not None else None)
        prev_rows = rows
        g = torch.cdist(r.varifold.positions[:n1],
                        r.varifold.positions[n1:]).min().item()
        frames.append(dict(
            step=step_idx, t=(step_idx + 1) * DT,
            C=r.detected_rank, rank_status=r.rank_status,
            rank_action=r.rank_action, rank_pending=r.rank_pending,
            rank_gap_ratio=r.rank_gap_ratio,
            rank_abs_level=r.rank_abs_level,
            cu_rowspace_angle_deg=cu_angle,
            gap=g, q_min=stepper.fixed_coherence.min().item(),
            max_s_over_gap=(s.abs().max().item() / g),
            max_s_over_dt=(s.abs().max() / DT).item(),
            leakage_m=flux_leakage(s, m, slices),
            leakage_qm=flux_leakage(s, qm, slices),
            activity_m=flux_activity(s, m, slices),
            activity_qm=flux_activity(s, qm, slices),
            components=comp_stats(r.varifold, m, slices),
            objective_decreased=bool(r.objective_decreased),
        ))
        snapshots.append(r.varifold)
        v = r.varifold
    return dict(variant=variant, frames=frames, snapshots=snapshots, n1=n1)


def evolve_isolated(variant: str) -> list:
    """Each component alone, same dt and metric, ORACLE rank 1: a single
    closed curve has C = 1 by definition, and these runs are superposition
    references, not rank tests. (Measured reason to bypass auto detection
    here: the small ellipse alone at h = 0.05 sits at gap ratio 29.6,
    a hair under gap_min = 30 -- a datapoint for the threshold audit, not
    an ambiguity of the control.) Returns per-step concatenated
    superposition references."""
    trajs = []
    for p in parts_pair():
        v, snaps = p, []
        stepper = MMStepper(make_config(variant, "oracle", 1))
        for _ in range(N_STEPS):
            v = stepper.step(v).varifold
            snaps.append(v)
        trajs.append(snaps)
    return [cat(a, b) for a, b in zip(*trajs)]


def _evolve_simple(parts, variant: str, n_steps: int, dt: float,
                   oracle_rank: int, blockdiag_cut: int | None = None):
    """Bare evolution at oracle rank; optionally with the oracle
    cross-component blocks of BOTH S and K* zeroed (same geometry, same
    masses/coherence/angle map -- the only change is the BIE coupling)."""
    import src.torch.transport.bem_wasserstein as bw
    from unittest.mock import patch

    orig = bw.build_bem_matrices_point

    def blocked(pos, nor, w, eps):
        S, K_star = orig(pos, nor, w, eps)
        cut = blockdiag_cut
        S = S.clone()
        K_star = K_star.clone()
        S[:cut, cut:] = 0.0
        S[cut:, :cut] = 0.0
        K_star[:cut, cut:] = 0.0
        K_star[cut:, :cut] = 0.0
        return S, K_star

    cfg = make_config(variant, "oracle", oracle_rank)
    cfg.time_step = dt
    stepper = MMStepper(cfg)
    v = cat(*parts) if len(parts) > 1 else parts[0]
    ctx = (patch.object(bw, "build_bem_matrices_point", blocked)
           if blockdiag_cut is not None else _null_ctx())
    snaps, diags = [], []
    with ctx:
        for _ in range(n_steps):
            r = stepper.step(v)
            v = r.varifold
            snaps.append(v)
            # keep the 0A final-point diagnostics -- the longest control
            # must not be the one run that discards them
            diags.append(dict(
                objective_decreased=bool(r.objective_decreased),
                relative_gradient_norm=r.relative_gradient_norm,
                objective_drop=r.objective_initial - r.objective,
                n_iter=int(r.n_iter),
                max_s=r.displacements.abs().max().item(),
            ))
    return snaps, diags


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def attribution_control(variant: str = "M", n_steps: int = 10) -> list:
    """Item 2 of the review: pair-vs-isolated defect attribution.

    Same pair geometry, three solves at oracle rank: full global BIE,
    block-diagonal BIE (oracle cross blocks of S and K* zeroed -- identical
    masses, coherence, angle map), and the concatenated isolated runs.
    dx(full, blk) isolates the BIE cross-block coupling; dx(blk, iso) is
    everything else (estimator/self-coupling differences of separate
    steppers). Plus N-refinement at fixed physical time T = n_steps * DT:
    the full-vs-isolated defect must DECREASE with h before it may be
    called a finite-N cross-block error."""
    rows = []
    for h in (0.05, 0.025):
        parts = parts_pair(h)
        n1 = parts[0].n_points
        K = 3
        full, _ = _evolve_simple(parts, variant, n_steps, DT, 2)
        blk, _ = _evolve_simple(parts, variant, n_steps, DT, 2,
                                blockdiag_cut=n1 * K)
        iso = [cat(a, b) for a, b in zip(
            _evolve_simple([parts[0]], variant, n_steps, DT, 1)[0],
            _evolve_simple([parts[1]], variant, n_steps, DT, 1)[0])]
        d = lambda a, b: (a.positions - b.positions).abs().max().item()
        rows.append(dict(
            h=h, N=cat(*parts).n_points, T=n_steps * DT,
            dx_full_iso=d(full[-1], iso[-1]),
            dx_full_blk=d(full[-1], blk[-1]),
            dx_blk_iso=d(blk[-1], iso[-1]),
        ))
    return rows


def polygon_stats(v, slices):
    """Geometric-only per-component stats: ordered-polygon area, phase
    centroid (Green's formula), and radial coefficient of variation about
    the PHASE centroid (this is a noncircularity measure, NOT the
    mathematical ellipse eccentricity)."""
    out = []
    for i, j in slices:
        P = v.positions[i:j]
        X, Y = P[:, 0], P[:, 1]
        Xn, Yn = X.roll(-1), Y.roll(-1)
        c = X * Yn - Xn * Y
        A = 0.5 * c.sum()
        gx = ((X + Xn) * c).sum() / (6 * A)
        gy = ((Y + Yn) * c).sum() / (6 * A)
        r = (P - torch.stack([gx, gy])).norm(dim=1)
        out.append(dict(area_geom=A.item(),
                        centroid_geom=[gx.item(), gy.item()],
                        radial_cv=(r.std() / r.mean()).item()))
    return out


def long_run_control(variant: str = "M", n_steps: int = 400,
                     dt: float = 5e-4) -> dict:
    """Item 6: oracle C = 2 control until the ellipses are nearly round.
    The continuum FIXED-CENTROID round-circle estimate for this family is
    2.5 - sqrt(0.28) - sqrt(0.96) ~ 0.991 > 0, but the discrete run
    accumulates area and centroid drift, so the drift-adjusted limit
    |c2 - c1| - sqrt(A1/pi) - sqrt(A2/pi) from the FINAL polygon data is
    reported alongside it -- the trajectory should be compared with that,
    not with 0.991 alone. Records the gap trajectory, per-component
    radial_cv (noncircularity about the phase centroid), optimizer
    diagnostics, and the final superposition defect."""
    parts = parts_pair()
    n1 = parts[0].n_points
    slices = component_slices(parts)
    pair, diags = _evolve_simple(parts, variant, n_steps, dt, 2)
    iso = [cat(a, b) for a, b in zip(
        _evolve_simple([parts[0]], variant, n_steps, dt, 1)[0],
        _evolve_simple([parts[1]], variant, n_steps, dt, 1)[0])]

    traj = []
    for k in (list(range(0, n_steps, 20)) + [n_steps - 1]):
        v = pair[k]
        traj.append(dict(
            step=k, t=(k + 1) * dt,
            gap=torch.cdist(v.positions[:n1],
                            v.positions[n1:]).min().item(),
            radial_cv=[s["radial_cv"] for s in polygon_stats(v, slices)],
        ))
    final = polygon_stats(pair[-1], slices)
    import math as _m
    drift_adjusted_limit = (
        abs(final[1]["centroid_geom"][0] - final[0]["centroid_geom"][0])
        - _m.sqrt(abs(final[0]["area_geom"]) / _m.pi)
        - _m.sqrt(abs(final[1]["area_geom"]) / _m.pi))
    return dict(
        variant=variant, dt=dt, n_steps=n_steps, traj=traj,
        final_dx_iso=(pair[-1].positions
                      - iso[-1].positions).abs().max().item(),
        final_components=final,
        optimizer=dict(
            all_objective_decreased=all(d["objective_decreased"]
                                        for d in diags),
            worst_relative_gradient=max(d["relative_gradient_norm"]
                                        for d in diags),
            max_step_over_spacing=max(d["max_s"] for d in diags) / 0.05,
            max_n_iter=max(d["n_iter"] for d in diags)),
        limit_gap_fixed_centroid=2.5 - (0.7 * 0.4) ** 0.5
        - (1.2 * 0.8) ** 0.5,
        limit_gap_drift_adjusted=drift_adjusted_limit)


def genuine_contact_prediction(n_steps: int = 400, dt: float = 5e-4,
                               gaps=(0.1, 0.2, 0.4)) -> dict:
    """Genuine PRE-CONTACT contact mechanism of the minor-axes-facing
    two-ellipse family (a = 0.4, b = 1.0 at +-(0.4 + g0/2)).

    CONFIGURATION IDENTIFICATION (review-corrected): the PAPER experiment
    is gap 0.1 (manuscript section 6.2.2, batch default, and the archived
    run's parameters in results/two_ellipses_data.json all agree ->
    centers +-0.45). Centers +-0.6 are the GENERATOR default, gap 0.4.
    One isolated corrected trajectory (oracle C = 1; the pair is
    symmetric, and an isolated ellipse does not know the gap) serves
    every g0 at once: reconstructed pair gap g(t) = (0.8 + g0) - 2
    x_max(t). Round-limit gaps: (0.8 + g0) - 2 sqrt(0.4) = -0.365 (paper
    g0 = 0.1) / -0.265 (g0 = 0.2) / -0.065 (generator g0 = 0.4) -- ALL
    negative; the paper configuration has by far the largest contact
    reserve, and its predicted contact time falls inside the original
    computation window T = 0.05.

    Claim discipline: this establishes a genuine pre-contact CONTACT
    MECHANISM (independent componentwise relaxation drives the facing
    tips outward). It does NOT establish that the corrected point-cloud
    scheme reproduces the merger through and after contact -- that waits
    for the pair runs, rank transition, and 0D/0E support validation.
    NOTE dt = 5e-4 accumulates polygon-area drift (~+0.4% by t = 0.02,
    ~+7% by t = 0.2); the qualitative prediction is robust, quantitative
    contact timing needs the refinement study."""
    import math as _m
    from src.torch.shapes.generator import generate_oriented_ellipse

    peri = _m.pi * (3 * 1.4 - _m.sqrt((3 * 0.4 + 1.0) * (0.4 + 3 * 1.0)))
    n = round(peri / 0.05)
    v = generate_oriented_ellipse(n, 0.4, 1.0, (0.0, 0.0), "cpu",
                                  torch.float64, "arc_length")
    cfg = make_config("M", "oracle", 1)
    cfg.time_step = dt
    stepper = MMStepper(cfg)
    traj = []
    for k in range(n_steps):
        r = stepper.step(v)
        v = r.varifold
        if k % 20 == 0 or k == n_steps - 1:
            X, Y = v.positions[:, 0], v.positions[:, 1]
            A = 0.5 * (X * Y.roll(-1) - X.roll(-1) * Y).sum().item()
            xm = X.max().item()
            traj.append(dict(
                step=k, t=(k + 1) * dt, x_max=xm, area_geom=A,
                reconstructed_gap={f"g0={g}": (0.8 + g) - 2 * xm
                                   for g in gaps}))
    contact_t = {}
    for g in gaps:
        key = f"g0={g}"
        prev = None
        hit = None
        for fr in traj:
            cur = (fr["t"], fr["reconstructed_gap"][key])
            if prev is not None and prev[1] > 0 >= cur[1]:
                t0, g0v = prev
                t1, g1v = cur
                hit = t0 + (t1 - t0) * g0v / (g0v - g1v)
                break
            prev = cur
        contact_t[key] = hit
    return dict(
        family="minor-axes-facing: a=0.4 b=1.0, centers +-(0.4+g0/2)",
        paper_gap=0.1, generator_default_gap=0.4,
        artifact_parameters_gap=0.1,
        artifact_source="results/two_ellipses_data.json parameters.gap",
        round_limit_gap={f"g0={g}": (0.8 + g) - 2 * _m.sqrt(0.4)
                         for g in gaps},
        predicted_contact_time=contact_t,
        dt=dt, traj=traj)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--long-steps", type=int, default=400)
    args = ap.parse_args()
    out = {}

    for variant in ("M", "Q"):
        print(f"=== variant {variant}: pair + isolated controls "
              f"(dt = {DT}, cap {N_STEPS} steps, gap0 = {GAP0}) ===",
              flush=True)
        pair = evolve_pair(variant)
        iso = evolve_isolated(variant)
        clean = [f for f in pair["frames"] if "hard_stop" not in f]
        for f, snap, ref in zip(clean, pair["snapshots"], iso):
            f["pair_vs_isolated_dx"] = (
                snap.positions - ref.positions).abs().max().item()
            f["pair_vs_isolated_dtheta"] = circular_angle_diff(
                snap.angles, ref.angles).abs().max().item()
        for f in clean[:3] + clean[9:10] + clean[19:20] + clean[-1:]:
            cu = ("--" if f["cu_rowspace_angle_deg"] is None
                  else f"{f['cu_rowspace_angle_deg']:.3f}")
            print(f"  step {f['step']:>3}: gap={f['gap']:.4f} "
                  f"C={f['C']}({f['rank_status']}) "
                  f"gap_ratio={f['rank_gap_ratio']:.0f} Cu-angle={cu} "
                  f"dx_iso={f['pair_vs_isolated_dx']:.2e} "
                  f"dth_iso={f['pair_vs_isolated_dtheta']:.2e}")
        g0, g1 = clean[0]["gap"], clean[-1]["gap"]
        a_first = [c["area_geom"] for c in clean[0]["components"]]
        a_last = [c["area_geom"] for c in clean[-1]["components"]]
        cen_first = [c["centroid_geom"] for c in clean[0]["components"]]
        cen_last = [c["centroid_geom"] for c in clean[-1]["components"]]
        cen_drift = max(abs(a - b) for p0, p1 in zip(cen_first, cen_last)
                        for a, b in zip(p0, p1))
        stops = [f for f in pair["frames"] if "hard_stop" in f]
        print(f"  gap: {GAP0:.3f} -> {g0:.4f} (step 0) -> {g1:.4f} "
              f"(step {clean[-1]['step']}): "
              f"{'GROWS' if g1 > g0 else 'shrinks'}")
        print(f"  areas: {a_first} -> {a_last}")
        print(f"  max centroid drift: {cen_drift:.2e}")
        print(f"  worst pair-vs-isolated dx: "
              f"{max(f['pair_vs_isolated_dx'] for f in clean):.2e}, "
              f"evolved frames: {len(clean)}, hard stops: {len(stops)}",
              flush=True)
        pair.pop("snapshots")
        out[variant] = pair

    print("\n=== attribution: full vs block-diagonal BIE vs isolated "
          "(oracle rank, T = 1e-3) ===")
    attr = attribution_control("M")
    for r in attr:
        print(f"  h={r['h']}: N={r['N']} "
              f"dx(full,iso)={r['dx_full_iso']:.2e} "
              f"dx(full,blk)={r['dx_full_blk']:.2e} "
              f"dx(blk,iso)={r['dx_blk_iso']:.2e}", flush=True)

    print("\n=== long-run oracle C=2 control (dt = 5e-4) ===")
    lr = long_run_control("M", args.long_steps)
    for t in lr["traj"]:
        print(f"  step {t['step']:>4} t={t['t']:.3f}: gap={t['gap']:.4f} "
              f"radial_cv=({t['radial_cv'][0]:.4f}, "
              f"{t['radial_cv'][1]:.4f})", flush=True)
    print(f"  limit gap, continuum fixed-centroid: "
          f"{lr['limit_gap_fixed_centroid']:.3f}; drift-adjusted from "
          f"final polygon data: {lr['limit_gap_drift_adjusted']:.3f}")
    print(f"  optimizer: {lr['optimizer']}")
    print(f"  final superposition defect dx = {lr['final_dx_iso']:.2e}")

    print("\n=== genuine pre-contact mechanism: minor-axes-facing family "
          "(paper gap 0.1, generator default 0.4) ===")
    gc = genuine_contact_prediction(args.long_steps)
    print(f"  round-limit gaps: {gc['round_limit_gap']}")
    for t in gc["traj"][::4] + [gc["traj"][-1]]:
        rg = t["reconstructed_gap"]
        print(f"  t={t['t']:.3f}: gap(paper 0.1)={rg['g0=0.1']:+.4f} "
              f"gap(0.4)={rg['g0=0.4']:+.4f} area={t['area_geom']:.4f}",
              flush=True)
    print(f"  predicted contact times: {gc['predicted_contact_time']}")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "spectral_precontact_0c5d1.json"
        path.write_text(json.dumps(dict(
            meta=dict(dt=DT, n_steps=N_STEPS, gap0=GAP0,
                      rank_mode="state_machine",
                      operator_measure="carrier",
                      rank_null_tol=0.05, rank_gap_min=30.0,
                      prediction=("gap grows; weak_gap held by the state "
                                  "machine; run completes the cap")),
            attribution=attr,
            long_run={k: v for k, v in lr.items()},
            genuine_contact_prediction=gc,
            **out), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
