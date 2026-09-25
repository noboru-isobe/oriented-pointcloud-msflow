"""0C-5a: the spectral componentwise metric inside the actual MM step.

First integration of the 0C-3 machinery into MMStepper, in the narrowest
scope the review fixed: q = 1, C = 2 supplied by the oracle
(bem_component_rank), u = m, separated unequal disks, no redistribution, no
deletion. Rank detection, hysteresis and the m-vs-qm comparison are all
later stages.

The physical statement under test (F2): two separated disks are EXACTLY
stationary under the one-phase flow -- each component's volume is conserved
individually, so no exchange is admissible. The three solver modes then
tell the whole story in one table:

    spectral_bordered + interior   componentwise fluxes ~ 0 (machine-level
                                   compatibility by construction), residual
                                   motion at discretisation level, halving
                                   with h;
    legacy + exterior (production) the exchange mode runs at O(1) velocity:
                                   the small disk shrinks and the large one
                                   grows at |flux|/dt ~ 2, independent of N
                                   -- Gate A's algebraic defect measured
                                   dynamically;
    legacy + interior              exchange suppressed by the near-null
                                   direction's diverging cost but not
                                   removed: regularisation-dependent drift,
                                   halving with N.

Residual attribution (measured; `attribution` in the JSON). An earlier
attribution to KDE cross-component coupling was wrong: the mass kernel
(wendland_c2) has compact support of radius delta = 0.249 (logged per run
as `mass_delta`), well below every tested gap, and indeed |grad P_fr(0)| is
identical to all printed digits across gaps -- the perimeter estimator sees
the other disk not at all. The measured separation instead lands on the
first candidate: the frozen perimeter gradient at s = 0 is componentwise
compatible to MACHINE precision (oracle projection ~1e-14), while its
projection onto the spectral compatible space is ~2-4e-3 and grows as the
disks approach, tracking the projector distance between the two subspaces
(2.4e-3 at gap 4 -> 5.2e-3 at gap 1). The residual motion is therefore the
finite-N difference between the BEM left near-nullspace and the component
indicators. U0 comes from A = 1/2 I + K* (the single-layer log operator S
never enters rank detection). The `kstar_block_diag` diagnostic zeroes the
oracle cross-component K* blocks (labels used for diagnostics only) and
settles the split causally: with the cross blocks removed the
U0-vs-indicator angle collapses to a gap-INDEPENDENT 1.0688 deg (the
per-component discretisation baseline), while the full operator adds a
gap-dependent increment (+0.009 / +0.021 / +0.041 deg at gaps 4 / 2 / 1).
So the gap dependence IS the discrete K* cross-block coupling -- decaying
with distance, as K* does -- while the bulk of the tilt is per-component
quadrature error, refined away under N-refinement.

Usage
-----
    uv run python scripts/experiments/spectral_evolution_0c5a.py \
        --out results/spectral_0c5a
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.shapes.generator import generate_oriented_circle
from src.torch.solver.mm_step import MMConfig, MMStepper

DTYPE = torch.float64
DT = 1e-5
R1, R2 = 0.5, 1.0

MODES = {
    "spectral/interior": dict(bem_solver_mode="spectral_bordered",
                              bem_trace_side="interior",
                              bem_component_rank=2),
    "legacy/exterior": dict(bem_solver_mode="legacy_global_projection",
                            bem_trace_side="legacy_exterior"),
    "legacy/interior": dict(bem_solver_mode="legacy_global_projection",
                            bem_trace_side="interior"),
}


def two_disks(gap: float, h: float):
    parts = []
    for R, cx in ((R1, -(R1 + gap / 2)), (R2, R2 + gap / 2)):
        n = max(16, round(2 * math.pi * R / h))
        parts.append(generate_oriented_circle(n, R, (cx, 0.0), "cpu", DTYPE))
    v = OrientedPointCloudVarifold(
        positions=torch.cat([p.positions for p in parts]),
        angles=torch.cat([p.angles for p in parts]))
    return v, parts[0].n_points


def config(mode_kwargs, rank=None) -> MMConfig:
    return MMConfig(time_step=DT, use_unit_coherence=True,
                    bem_epsilon_mode="carrier_segment_length",
                    angle_sigma=0.1,
                    optimizer_method="trust-ncg", optimizer_tol=1e-8,
                    optimizer_max_iter=300, **mode_kwargs)


def one_step(mode: str, h: float, gap: float) -> dict:
    v, n1 = two_disks(gap, h)
    stepper = MMStepper(config(MODES[mode]))
    r = stepper.step(v)
    s, m = r.displacements, r.masses
    return dict(
        mode=mode, h=h, gap=gap, N=v.n_points,
        mass_delta=stepper.mass_delta,
        max_s_over_dt=(s.abs().max() / DT).item(),
        flux1_over_dt=((s[:n1] * m[:n1]).sum() / DT).item(),
        flux2_over_dt=((s[n1:] * m[n1:]).sum() / DT).item(),
        r_comp=r.r_comp,
        relative_gradient_norm=r.relative_gradient_norm,
        objective_decreased=bool(r.objective_decreased),
        converged=bool(r.converged), n_iter=int(r.n_iter),
    )


def attribute_residual(h: float, gap: float) -> dict:
    """Separate the sources of the spectral-mode residual motion BEFORE any
    optimization: project the frozen perimeter gradient at s = 0 onto the
    oracle-compatible space (kernel of the two component mass-sum rows) and
    onto the spectral-compatible space actually used by the optimizer.

        oracle ~ 0, spectral > 0   -> finite-resolution difference between
                                      the left nullspace and the indicators
        both > 0                   -> discrete perimeter stationarity error
        both ~ 0 (step nonzero)    -> BEM quadratic form / optimizer
    """
    from src.torch.oriented_varifold.mass import compute_masses

    v, n1 = two_disks(gap, h)
    stepper = MMStepper(config(MODES["spectral/interior"]))
    stepper._setup_step(v)

    s = torch.zeros(v.n_points, dtype=DTYPE, requires_grad=True)
    pos = (stepper.param.prev_positions
           + s[:, None] * stepper.param.prev_normals)
    masses = compute_masses(pos, stepper._mass_delta_for_kde,
                            stepper._mass_tau_for_kde,
                            stepper.config.mass_kernel)
    P = (masses * stepper.fixed_coherence).sum()
    g = torch.autograd.grad(P, s)[0]

    m = stepper.fixed_masses
    G = torch.zeros(2, v.n_points, dtype=DTYPE)
    G[0, :n1] = m[:n1]
    G[1, n1:] = m[n1:]
    _, _, Vh = torch.linalg.svd(G, full_matrices=True)
    Q_oracle = Vh[2:].T
    Q_spec = stepper.param.Q
    # operator-norm distance between the two projectors: how far the
    # spectral compatible space sits from the oracle one
    subspace_dist = torch.linalg.matrix_norm(
        Q_spec @ Q_spec.T - Q_oracle @ Q_oracle.T, ord=2).item()
    return dict(
        h=h, gap=gap, N=v.n_points, mass_delta=stepper.mass_delta,
        grad_norm=g.norm().item(),
        proj_oracle=(Q_oracle.T @ g).norm().item(),
        proj_spectral=(Q_spec.T @ g).norm().item(),
        subspace_dist=subspace_dist,
    )


def oracle_single_disk(R: float, h: float) -> float:
    """Discretisation-level relaxation of one isolated disk under the same
    spectral machinery (C = 1): the baseline the two-disk residual motion is
    compared against."""
    n = max(16, round(2 * math.pi * R / h))
    v = generate_oriented_circle(n, R, (0.0, 0.0), "cpu", DTYPE)
    kwargs = dict(bem_solver_mode="spectral_bordered",
                  bem_trace_side="interior", bem_component_rank=1)
    r = MMStepper(config(kwargs)).step(v)
    return (r.displacements.abs().max() / DT).item()


def kstar_block_diag(h: float, gap: float) -> dict:
    """Causal check for the U0 tilt: zero the oracle cross-component K*
    blocks (labels used for diagnostics ONLY) and measure the principal
    angle between the left near-null space and the weighted component
    indicators W^{1/2} 1_alpha, for the full vs block-diagonal operator.
    If the cross-block coupling causes the tilt, the block-diagonal angle
    must collapse toward zero."""
    v, n1 = two_disks(gap, h)
    stepper = MMStepper(config(MODES["spectral/interior"]))
    stepper._setup_step(v)
    bem = stepper.bem_wasserstein
    K_star, w, K = bem._K_star_cache, bem._endpoint_weights_cache, bem._K
    NK = w.numel()
    sqrt_w = w.sqrt()
    eye = torch.eye(NK, dtype=DTYPE)

    def u0_of(kstar):
        A_w = (sqrt_w[:, None] * (0.5 * eye + kstar)) / sqrt_w[None, :]
        U, _, _ = torch.linalg.svd(A_w)
        return U[:, -2:]

    cut = n1 * K
    K_bd = K_star.clone()
    K_bd[:cut, cut:] = 0.0
    K_bd[cut:, :cut] = 0.0

    ind = torch.zeros(NK, 2, dtype=DTYPE)
    ind[:cut, 0] = sqrt_w[:cut]
    ind[cut:, 1] = sqrt_w[cut:]
    ind, _ = torch.linalg.qr(ind)

    def max_angle_deg(U0):
        cos = torch.linalg.svdvals(ind.T @ U0)
        return float(torch.rad2deg(torch.acos(cos.min().clamp(-1, 1))))

    return dict(h=h, gap=gap,
                angle_full_deg=max_angle_deg(u0_of(K_star)),
                angle_blockdiag_deg=max_angle_deg(u0_of(K_bd)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = []

    print("=== three modes x two resolutions (gap = 2.0) ===")
    hdr = (f"{'mode':>19} {'h':>6} {'N':>4} {'max|s|/dt':>10} "
           f"{'flux1/dt':>10} {'flux2/dt':>10} {'r_comp':>9} {'relgrad':>9}")
    print(hdr)
    for h in (0.05, 0.025):
        for mode in MODES:
            r = one_step(mode, h, 2.0)
            rows.append(r)
            rc = "--" if r["r_comp"] is None else f"{r['r_comp']:.1e}"
            print(f"{mode:>19} {h:>6} {r['N']:>4} {r['max_s_over_dt']:>10.3e} "
                  f"{r['flux1_over_dt']:>10.2e} {r['flux2_over_dt']:>10.2e} "
                  f"{rc:>9} {r['relative_gradient_norm']:>9.1e}", flush=True)
        print()

    print("=== spectral gap sweep (h = 0.05): fluxes must stay ~0; the "
          "residual |s| may vary with gap (source under attribution) ===")
    for gap in (1.0, 2.0, 4.0):
        r = one_step("spectral/interior", 0.05, gap)
        rows.append(r)
        print(f"  gap={gap}: max|s|/dt={r['max_s_over_dt']:.3e}  "
              f"fluxes/dt=({r['flux1_over_dt']:.1e}, "
              f"{r['flux2_over_dt']:.1e})  delta={r['mass_delta']:.3f}")

    print("=== oracle baseline: isolated disks, same machinery, C = 1 ===")
    oracle = {f"R{R}": oracle_single_disk(R, 0.05) for R in (R1, R2)}
    for k, val in oracle.items():
        print(f"  {k}: max|s|/dt = {val:.3e}")

    print("=== residual attribution: |Q^T grad P_fr(0)| before any "
          "optimization ===")
    attribution = []
    for gap in (1.0, 2.0, 4.0):
        a = attribute_residual(0.05, gap)
        attribution.append(a)
        print(f"  gap={gap}: |g|={a['grad_norm']:.3e}  "
              f"oracle-proj={a['proj_oracle']:.3e}  "
              f"spectral-proj={a['proj_spectral']:.3e}  "
              f"subspace-dist={a['subspace_dist']:.3e}  "
              f"delta={a['mass_delta']:.3f}")

    print("=== causal check: U0-vs-indicator angle, full vs "
          "block-diagonal K* ===")
    blockdiag = []
    for gap in (1.0, 2.0, 4.0):
        b = kstar_block_diag(0.05, gap)
        blockdiag.append(b)
        print(f"  gap={gap}: full={b['angle_full_deg']:.4f} deg  "
              f"block-diag={b['angle_blockdiag_deg']:.4f} deg")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "spectral_evolution_0c5a.json"
        path.write_text(json.dumps(dict(
            meta=dict(dt=DT, R1=R1, R2=R2, modes=list(MODES),
                      q="identically 1", u="m", rank="oracle-supplied"),
            oracle_single_disk=oracle, rows=rows,
            attribution=attribution, kstar_blockdiag=blockdiag), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
