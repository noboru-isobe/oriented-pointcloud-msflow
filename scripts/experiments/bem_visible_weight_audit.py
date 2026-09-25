"""0C-2b: where does coherence q belong in the BEM metric?

0C-2a established that in the clean q = 1 regime the weighted left
near-nullspace of the interior trace reproduces the componentwise
compatibility space. This audit asks the modelling question that regime
hides: the production solver feeds q into the metric in THREE places at once
(endpoint offsets ~ q_i m_i, quadrature weights ~ q_i m_i / K, and the
endpoint velocity ~ q_i v_geom), so even if its spectral nullspace behaves,
the compatibility it enforces on the *geometric* displacement is
approximately

    sum_{i in E_alpha} q_i^2 m_i v_i = 0        (visible, q^2 m - flux)

whereas the physical one-phase condition is

    sum_{i in E_alpha} m_i v_i = 0              (physical, m - flux).

For q constant on each component the two row spaces coincide (q_alpha^2
factors out), so the discrepancy only appears when q varies WITHIN a
component -- precisely the hidden-boundary situation the estimator is built
around. The idealised prediction, made before measuring, is

    theta_alpha = arccos( <m, q^2 m>_alpha / (||m||_alpha ||q^2 m||_alpha) )

per phase component; e.g. a quarter-arc dip to q0 = 0.1 on a component gives
~30 degrees. The audit measures the actual spectral version of that angle
and the two leakage norms, alongside the operator-degeneracy metrics.

Operator modes, same geometry and same q profile:

  I   carrier    endpoint lengths & quadrature from m       (physical control)
  II  visible    endpoint lengths & quadrature from q*m,
                 velocity map D_q                            (production)
  III reweight   carrier operator A_m, visible spectral W_q  (DIAGNOSTIC ONLY:
                 separates the W dynamic-range effect from the operator
                 rebuild; not a meaningful discretisation by itself)

epsilon_BEM is fixed from the carrier median segment length in every mode, so
q-induced changes in the diagonal regularisation are not mixed into the
comparison.

Usage
-----
    uv run python scripts/experiments/bem_visible_weight_audit.py \
        --out results/spectral_0c2b
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.transport.bem_wasserstein import BEMWasserstein, compute_coherence

import sys

sys.path.insert(0, str(Path(__file__).parent))
from bem_spectral_audit import (  # noqa: E402
    GEOMETRIES,
    build_component,
    principal_angles_deg,
)

DTYPE = torch.float64
EPS_SCALE = 0.1
K_ENDPOINTS = 3


# --------------------------------------------------------------------------
# geometry with BOUNDARY-component labels (the audit-script build_geometry
# only exposes phase labels; synthetic q profiles need to target one ring)
# --------------------------------------------------------------------------

def build_geometry_full(name: str, spacing: float):
    spec = GEOMETRIES[name]
    pos, ang, wts, phase_l, bcomp_l = [], [], [], [], []
    for b_idx, (kind, params, center, inward, phase) in enumerate(
            spec["components"]):
        p, a, w = build_component(kind, params, center, inward, spacing)
        pos.append(p)
        ang.append(a)
        wts.append(w)
        phase_l.append(torch.full((len(w),), phase, dtype=torch.long))
        bcomp_l.append(torch.full((len(w),), b_idx, dtype=torch.long))
    varifold = OrientedPointCloudVarifold(
        positions=torch.cat(pos), angles=torch.cat(ang))
    return (varifold, torch.cat(wts), torch.cat(phase_l), torch.cat(bcomp_l),
            len(spec["components"]), int(torch.cat(phase_l).max().item()) + 1)


# --------------------------------------------------------------------------
# q profiles
# --------------------------------------------------------------------------

def q_synthetic_dip(bcomp: torch.Tensor, target_bcomp: int, q0: float,
                    frac: float = 0.25, ramp: float = 0.05) -> torch.Tensor:
    """q = 1 everywhere except a cosine-tapered dip to q0 on an arc covering
    `frac` of one boundary component. Index position stands in for arc
    position, which is accurate because components are sampled uniformly in
    arc length."""
    q = torch.ones(len(bcomp), dtype=DTYPE)
    idx = torch.nonzero(bcomp == target_bcomp).squeeze(1)
    n = len(idx)
    t = torch.arange(n, dtype=DTYPE) / n            # arc position in [0,1)
    lo, hi = 0.5 - frac / 2, 0.5 + frac / 2
    bump = torch.zeros(n, dtype=DTYPE)
    core = (t >= lo + ramp) & (t <= hi - ramp)
    bump[core] = 1.0
    up = (t >= lo) & (t < lo + ramp)
    bump[up] = 0.5 * (1 - torch.cos(math.pi * (t[up] - lo) / ramp))
    dn = (t > hi - ramp) & (t <= hi)
    bump[dn] = 0.5 * (1 - torch.cos(math.pi * (hi - t[dn]) / ramp))
    q[idx] = 1.0 - (1.0 - q0) * bump
    return q


def q_real(varifold, masses, sigma_coh: float) -> torch.Tensor:
    return compute_coherence(varifold, masses, sigma_coh).clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# operator modes
# --------------------------------------------------------------------------

def _setup_bem(varifold, lengths, eps_target: float):
    """Interior-trace BEM with an EXPLICIT epsilon.

    BEMWasserstein derives eps = EPS_SCALE * sigma / c_sigma when sigma is
    given, so passing sigma = eps_target * c_sigma / EPS_SCALE pins the
    regularisation regardless of what `lengths` median to. (The clean flag
    for this is the 0A scale API, still pending; this is the documented
    workaround.)
    """
    bem = BEMWasserstein(method="point", epsilon_scale=EPS_SCALE,
                         n_endpoints=K_ENDPOINTS, trace_side="interior")
    bem.setup_for_step(varifold.positions, varifold.normals, lengths,
                       sigma=eps_target * 3.0 / EPS_SCALE, c_sigma=3.0)
    return bem


def decomposition(varifold, m, q, phases, C, mode: str, eps_target: float):
    """Spectral data for one operator mode. Returns endpoint-level tensors."""
    if mode == "carrier":
        lengths, w_spec_particle, D_particle = m, m, torch.ones_like(m)
    elif mode == "visible":
        lengths, w_spec_particle, D_particle = q * m, q * m, q
    elif mode == "reweight":            # DIAGNOSTIC ONLY
        lengths, w_spec_particle, D_particle = m, q * m, q
    else:
        raise ValueError(mode)

    bem = _setup_bem(varifold, lengths, eps_target)
    op = bem.endpoint_operator()
    A = op.trace_operator
    K = op.n_endpoints

    w_end = (w_spec_particle / K).repeat_interleave(K)
    sqrt_w = w_end.sqrt()
    A_weighted = (sqrt_w[:, None] * A) / sqrt_w[None, :]

    U, S_desc, Vh = torch.linalg.svd(A_weighted)
    sigma = S_desc.flip(0)
    U0 = U[:, -C:]
    V0 = Vh[-C:, :].T

    lab_end = phases.repeat_interleave(K)
    B = torch.zeros(len(w_end), C, dtype=DTYPE)
    B[torch.arange(len(w_end)), lab_end] = 1.0
    QB = sqrt_w[:, None] * B
    QB = QB / QB.norm(dim=0, keepdim=True)

    kmax = min(8, len(sigma) - 1)
    detected_C = int(torch.argmax(sigma[1:kmax + 1] / sigma[:kmax]).item()) + 1

    return dict(op=op, sigma=sigma, U0=U0, V0=V0, QB=QB,
                sqrt_w=sqrt_w, D_end=D_particle.repeat_interleave(K),
                detected_C=detected_C, operator_norm=S_desc[0].item(),
                eps=op.epsilon)


# --------------------------------------------------------------------------
# physical vs visible compatibility on the DISPLACEMENT space
# --------------------------------------------------------------------------

def row_space(Cmat: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the row space (columns of the returned matrix)."""
    q, _ = torch.linalg.qr(Cmat.T)
    return q


def kernel_basis(Cmat: torch.Tensor) -> torch.Tensor:
    _, s, Vh = torch.linalg.svd(Cmat, full_matrices=True)
    rank = int((s > s.max() * 1e-12).sum())
    return Vh[rank:].T.contiguous()


def compat_comparison(d: dict, m, q, phases, C, N: int) -> dict:
    """Constraint the mode actually imposes on geometric displacements vs the
    physical componentwise m-flux, plus the idealised q^2 m prediction."""
    K = d["op"].n_endpoints

    # physical: C_phys[alpha, i] = m_i on phase alpha
    C_phys = torch.zeros(C, N, dtype=DTYPE)
    C_phys[phases, torch.arange(N)] = m

    # spectral constraint of this mode on particle displacements:
    # U0^T W^{1/2} D G, aggregated back to particles
    vec = d["sqrt_w"] * d["D_end"]
    C_vis = (d["U0"].T * vec[None, :]).reshape(C, N, K).sum(-1)

    # idealised prediction if U0 were exactly the visible indicator space:
    # rows q^2 m per phase (visible weight q m times velocity factor q)
    C_ideal = torch.zeros(C, N, dtype=DTYPE)
    C_ideal[phases, torch.arange(N)] = q * q * m

    ang_meas = principal_angles_deg(row_space(C_phys), row_space(C_vis))
    ang_pred = principal_angles_deg(row_space(C_phys), row_space(C_ideal))
    ang_ideal_vs_meas = principal_angles_deg(row_space(C_ideal),
                                             row_space(C_vis))

    # leakage with unit-norm rows: how much of one admissible space violates
    # the other's constraints
    Cp = C_phys / C_phys.norm(dim=1, keepdim=True)
    Cv = C_vis / C_vis.norm(dim=1, keepdim=True)
    leak_phys_into_vis = torch.linalg.matrix_norm(
        Cv @ kernel_basis(C_phys), ord=2).item()
    leak_vis_into_phys = torch.linalg.matrix_norm(
        Cp @ kernel_basis(C_vis), ord=2).item()

    return dict(
        theta_compat_meas_deg=ang_meas.max().item(),
        theta_compat_pred_deg=ang_pred.max().item(),
        theta_ideal_vs_meas_deg=ang_ideal_vs_meas.max().item(),
        leak_phys_into_vis=leak_phys_into_vis,
        leak_vis_into_phys=leak_vis_into_phys,
    )


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

SWEEP_A_GEOMS = {
    # geometry -> (boundary component carrying the synthetic dip)
    "two_unequal_ellipses": 0,
    "eccentric_annulus": 1,       # dip on the inner ring
    "flower": 0,
}
Q_LEVELS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
MODES = ["carrier", "visible", "reweight"]


def run_case(name, spacing, q, phases, C, varifold, m, mode, eps):
    d = decomposition(varifold, m, q, phases, C, mode, eps)
    comp = compat_comparison(d, m, q, phases, C, len(m))
    qm = (q * m)
    row = dict(
        geometry=name, mode=mode, spacing=spacing, C=C,
        n_endpoints=len(d["sqrt_w"]),
        q_min=q.min().item(), q_q10=q.quantile(0.1).item(),
        q_med=q.median().item(), q_max=q.max().item(),
        weight_dynamic_range=(qm.max() / qm.min()).item(),
        min_visible_len_over_eps=(qm.min() / eps).item(),
        sigma_C=d["sigma"][C - 1].item(),
        sigma_Cp1=d["sigma"][C].item(),
        gap=(d["sigma"][C - 1] / d["sigma"][C]).item(),
        detected_C=d["detected_C"],
        theta_indicator_deg=principal_angles_deg(
            d["U0"], d["QB"]).max().item(),
        **comp,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spacing", type=float, default=0.03)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    hdr = (f"{'geometry':>22} {'mode':>9} {'q0/prof':>8} {'C':>2} {'Cdet':>4} "
           f"{'gap':>9} {'th_ind':>8} {'th_cmp':>8} {'th_prd':>8} "
           f"{'leakP>V':>8} {'wdyn':>9}")

    print("=== Sweep A: synthetic cosine dips (topology fixed, q controlled) ===")
    print(hdr)
    for name, dip_bc in SWEEP_A_GEOMS.items():
        varifold, m, phases, bcomp, M, C = build_geometry_full(
            name, args.spacing)
        eps = EPS_SCALE * m.median().item()      # carrier-based, q-independent
        for q0 in Q_LEVELS:
            q = q_synthetic_dip(bcomp, dip_bc, q0)
            for mode in MODES:
                r = run_case(name, args.spacing, q, phases, C,
                             varifold, m, mode, eps)
                r.update(profile=f"dip{q0}", q0=q0, dip_bcomp=dip_bc)
                rows.append(r)
                print(f"{name:>22} {mode:>9} {q0:>8} {C:>2} "
                      f"{r['detected_C']:>4} {r['gap']:>9.2e} "
                      f"{r['theta_indicator_deg']:>7.2f}° "
                      f"{r['theta_compat_meas_deg']:>7.2f}° "
                      f"{r['theta_compat_pred_deg']:>7.2f}° "
                      f"{r['leak_phys_into_vis']:>8.3f} "
                      f"{r['weight_dynamic_range']:>9.1e}", flush=True)
        print()

    print("=== Sweep B: real coherence, shrinking inner ring (h=0.01, "
          "sigma_coh=0.15) ===")
    print(hdr)
    for R_in in (0.30, 0.15, 0.08, 0.04):
        name = f"annulus_Rin{R_in}"
        GEOMETRIES[name] = dict(tier="primary", components=[
            ("circle", dict(R=1.0), (0.0, 0.0), False, 0),
            ("circle", dict(R=R_in), (0.0, 0.0), True, 0)])
        varifold, m, phases, bcomp, M, C = build_geometry_full(name, 0.01)
        eps = EPS_SCALE * m.median().item()
        q = q_real(varifold, m, sigma_coh=0.15)
        for mode in ("carrier", "visible"):
            r = run_case(name, 0.01, q, phases, C, varifold, m, mode, eps)
            r.update(profile="real", R_in=R_in)
            rows.append(r)
            print(f"{name:>22} {mode:>9} {'real':>8} {C:>2} "
                  f"{r['detected_C']:>4} {r['gap']:>9.2e} "
                  f"{r['theta_indicator_deg']:>7.2f}° "
                  f"{r['theta_compat_meas_deg']:>7.2f}° "
                  f"{r['theta_compat_pred_deg']:>7.2f}° "
                  f"{r['leak_phys_into_vis']:>8.3f} "
                  f"{r['weight_dynamic_range']:>9.1e}", flush=True)
        del GEOMETRIES[name]
    print()

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "bem_visible_weight_audit.json"
        path.write_text(json.dumps(dict(
            meta=dict(spacing=args.spacing, epsilon_scale=EPS_SCALE,
                      n_endpoints=K_ENDPOINTS, trace_side="interior",
                      q_levels=Q_LEVELS, modes=MODES,
                      dip=dict(frac=0.25, ramp=0.05)),
            rows=rows), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
