"""0C-2c: separate the topology sensor from the flux measure.

0C-2b showed the production bundling (q in panel geometry, quadrature, AND
endpoint velocity at once) enforces ~q^2 m compatibility and can destroy the
near-null rank structure under moderate q-degeneracy, while the q-free
carrier operator keeps both. But "carrier wins numerically" must not decide
the modelling question: m is the scalar carrier and counts hidden
multiplicity, whereas q m is the natural finite-scale candidate for the
current mass |D chi_E| that the manuscript's energy is actually built on.

This audit therefore separates two questions the production code currently
fuses:

  (i)  which operator DETECTS C and carries the left/right nullspaces
       (topology sensor);
  (ii) which measure the compatibility condition preserves (flux measure).

Terminology (review correction): the observed rank damage is not a mere
numerical quadrature error. Substituting q m for the geometric arc length m
as the INTEGRATION MEASURE of the layer potential assembles a genuinely
different weighted operator, whose Neumann nullspace need not encode the
phase components at all -- "operator-measure substitution", not "quadrature
degeneration". Relatedly, cond(bordered) = O(1) is NOT a correctness
criterion: a bordered system built on the wrong rank C is typically still
well-conditioned.

Five modes, same geometry, same q, epsilon_BEM fixed from the carrier scale:

           panel     quadrature   velocity     ideal compatibility
    M0      m           m            1              m - flux      (carrier)
    M1      m           q m          1              q m - flux    (current)
    M2      q m         q m          1              q m - flux
    M3      m           q m          q              q^2 m - flux
    M4      q m         q m          q              q^2 m - flux  (production)

M1 vs M2 isolates PANEL COLLAPSE (endpoint offsets shrinking with q m);
M1 vs M3 isolates the EXTRA q in the endpoint velocity; M3 vs M4 closes the
square. The leading hypothesis -- topology sensor = carrier operator, metric
quadrature = q m, no extra velocity q (i.e. M1) -- is a hypothesis to be
tested here, not a conclusion.

Also included: the actual angle pullback. The production endpoint velocity is
v_ik = q_i (s_i - r_ik * Dalpha_i) / dt with Dalpha = q * (A_angle^+ B_angle) s,
so before any "exactly q^2 m" claim the constraint is rebuilt with that map
and compared to the Delta-alpha = 0 version and to the ideal rows; symmetric
endpoint offsets (sum_k r_ik = 0) make q^2 m the leading term, but the
numerical left mode is only approximately an indicator.

Usage
-----
    uv run python scripts/experiments/bem_metric_policy_audit.py \
        --out results/spectral_0c2c
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.perimeter.angle_constraint import compute_angle_constraint_matrices
from src.torch.perimeter.coherence_perimeter import compute_recommended_sigma
from src.torch.transport.bem_wasserstein import build_bem_matrices_point

import sys

sys.path.insert(0, str(Path(__file__).parent))
from bem_spectral_audit import GEOMETRIES, principal_angles_deg  # noqa: E402
from bem_visible_weight_audit import (  # noqa: E402
    build_geometry_full,
    q_real,
    q_synthetic_dip,
)

DTYPE = torch.float64
EPS_SCALE = 0.1
K_ENDPOINTS = 3

MODES = {
    #        panel   quad    velocity
    "M0": ("m",     "m",    "one"),
    "M1": ("m",     "qm",   "one"),
    "M2": ("qm",    "qm",   "one"),
    "M3": ("m",     "qm",   "q"),
    "M4": ("qm",    "qm",   "q"),
}
IDEAL_OF_MODE = {"M0": "m", "M1": "qm", "M2": "qm", "M3": "q2m", "M4": "q2m"}


# --------------------------------------------------------------------------
# operator assembly with panel geometry decoupled from quadrature
# --------------------------------------------------------------------------

def assemble_operator(varifold, panel_lengths, quad_masses, eps):
    """Interior-trace endpoint operator with independent panel/quadrature.

    The production BEMWasserstein bundles both into one `masses` argument;
    this assembly (same kernels, same K, same collocation rule) is what lets
    M1 and M3 exist at all. Returns endpoint-level tensors.
    """
    nor = varifold.normals
    tang = torch.stack([-nor[:, 1], nor[:, 0]], 1)
    K = K_ENDPOINTS
    r_vals = torch.linspace(-0.5, 0.5, K, dtype=DTYPE)
    r_phys = r_vals[None, :] * panel_lengths[:, None]              # (N, K)
    P = (varifold.positions[:, None, :]
         + r_phys[:, :, None] * tang[:, None, :]).reshape(-1, 2)
    Nn = nor[:, None, :].expand(-1, K, -1).reshape(-1, 2)
    w_end = (quad_masses[:, None] / K).expand(-1, K).reshape(-1)
    S, K_star = build_bem_matrices_point(P, Nn, w_end, eps)
    A = 0.5 * torch.eye(len(w_end), dtype=DTYPE) + K_star
    return dict(A=A, S=S, w_end=w_end, r_phys=r_phys, K=K,
                positions_end=P)


def mode_decomposition(varifold, m, q, phases, C, mode: str, eps):
    panel_sel, quad_sel, vel_sel = MODES[mode]
    panel = m if panel_sel == "m" else q * m
    quad = m if quad_sel == "m" else q * m
    vel = torch.ones_like(m) if vel_sel == "one" else q

    op = assemble_operator(varifold, panel, quad, eps)
    w = op["w_end"]
    sqrt_w = w.sqrt()
    A_weighted = (sqrt_w[:, None] * op["A"]) / sqrt_w[None, :]
    U, S_desc, Vh = torch.linalg.svd(A_weighted)
    sigma = S_desc.flip(0)
    U0 = U[:, -C:]
    V0 = Vh[-C:, :].T

    K = op["K"]
    lab_end = phases.repeat_interleave(K)
    B = torch.zeros(len(w), C, dtype=DTYPE)
    B[torch.arange(len(w)), lab_end] = 1.0
    QB = sqrt_w[:, None] * B
    QB = QB / QB.norm(dim=0, keepdim=True)

    kmax = min(8, len(sigma) - 1)
    detected_C = int(torch.argmax(sigma[1:kmax + 1] / sigma[:kmax]).item()) + 1

    # bordered conditioning (the system 0C-3 would solve in this metric)
    NK = len(w)
    Msys = torch.zeros(NK + C, NK + C, dtype=DTYPE)
    Msys[:NK, :NK] = A_weighted
    Msys[:NK, NK:] = U0
    Msys[NK:, :NK] = V0.T
    sv = torch.linalg.svdvals(Msys)
    cond_bordered = (sv.max() / sv.min()).item()

    return dict(op=op, U0=U0, V0=V0, QB=QB, sigma=sigma, sqrt_w=sqrt_w,
                vel=vel, panel=panel, quad=quad, detected_C=detected_C,
                cond_bordered=cond_bordered)


# --------------------------------------------------------------------------
# constraint spaces
# --------------------------------------------------------------------------

def ideal_constraint(u: torch.Tensor, phases, C, N) -> torch.Tensor:
    """Rows u_i 1{i in phase alpha}: the componentwise u-flux functional."""
    Cmat = torch.zeros(C, N, dtype=DTYPE)
    Cmat[phases, torch.arange(N)] = u
    return Cmat


def measured_constraint(d: dict, phases, C, N) -> torch.Tensor:
    """U0^T W^{1/2} D_vel G, aggregated to particle displacements
    (Delta alpha = 0 version)."""
    K = d["op"]["K"]
    vec = d["sqrt_w"] * d["vel"].repeat_interleave(K)
    return (d["U0"].T * vec[None, :]).reshape(C, N, K).sum(-1)


def row_space(Cmat):
    qm, _ = torch.linalg.qr(Cmat.T)
    return qm


def kernel_basis(Cmat):
    _, s, Vh = torch.linalg.svd(Cmat, full_matrices=True)
    rank = int((s > s.max() * 1e-12).sum())
    return Vh[rank:].T.contiguous()


def angle_between(Ca, Cb) -> float:
    return principal_angles_deg(row_space(Ca), row_space(Cb)).max().item()


def leakage(Ca, Cb) -> float:
    """||C_a-normalised applied to ker C_b||: how much of b's admissible
    space violates a's constraints."""
    Cn = Ca / Ca.norm(dim=1, keepdim=True)
    return torch.linalg.matrix_norm(Cn @ kernel_basis(Cb), ord=2).item()


# --------------------------------------------------------------------------
# the actual angle pullback (production velocity map)
# --------------------------------------------------------------------------

def full_pullback_constraint(d: dict, varifold, m, q, phases, C, N):
    """C_full = U0^T W^{1/2} D_vel (G - R_end A_curl) with the production
    angle map Delta alpha = q * (A_angle^+ B_angle) s."""
    sigma_ang = compute_recommended_sigma(varifold.positions)
    nor = varifold.normals
    tang = torch.stack([-nor[:, 1], nor[:, 0]], 1)
    A_ang, B_ang = compute_angle_constraint_matrices(
        varifold.positions, tang, m, q, sigma_ang)
    Amap = q[:, None] * torch.linalg.lstsq(A_ang, B_ang).solution   # (N, N)

    K = d["op"]["K"]
    vec = (d["sqrt_w"] * d["vel"].repeat_interleave(K))              # (NK,)
    X = d["U0"].T * vec[None, :]                                     # (C, NK)
    # G part: aggregate endpoint columns per particle
    C_G = X.reshape(C, N, K).sum(-1)
    # R_end A part: row (i,k) contributes -r_phys[i,k] * Amap[i, :]
    r_flat = d["op"]["r_phys"].reshape(-1)                           # (NK,)
    coeff = (X * r_flat[None, :]).reshape(C, N, K).sum(-1)           # (C, N)
    C_full = C_G - coeff @ Amap
    return C_full, C_G


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

def audit(name, spacing, q, phases, C, varifold, m, eps, tag) -> list[dict]:
    N = len(m)
    ideals = {u: ideal_constraint(w, phases, C, N)
              for u, w in [("m", m), ("qm", q * m), ("q2m", q * q * m)]}
    # pairwise structure of the three candidate constraint spaces themselves
    pair_leak = {f"{a}->{b}": leakage(ideals[a], ideals[b])
                 for a in ideals for b in ideals if a != b}

    rows = []
    for mode in MODES:
        d = mode_decomposition(varifold, m, q, phases, C, mode, eps)
        Cmeas = measured_constraint(d, phases, C, N)
        panel = d["panel"]
        rows.append(dict(
            geometry=name, tag=tag, mode=mode, spacing=spacing, C=C,
            detected_C=d["detected_C"],
            sigma_C=d["sigma"][C - 1].item(),
            sigma_Cp1=d["sigma"][C].item(),
            gap=(d["sigma"][C - 1] / d["sigma"][C]).item(),
            theta_indicator_deg=principal_angles_deg(
                d["U0"], d["QB"]).max().item(),
            theta_to_m=angle_between(Cmeas, ideals["m"]),
            theta_to_qm=angle_between(Cmeas, ideals["qm"]),
            theta_to_q2m=angle_between(Cmeas, ideals["q2m"]),
            ideal_of_mode=IDEAL_OF_MODE[mode],
            weight_dynamic_range=(d["quad"].max() / d["quad"].min()).item(),
            min_panel_over_eps=(panel.min() / eps).item(),
            min_panel_over_h=(panel.min() / spacing).item(),
            cond_bordered=d["cond_bordered"],
            pairwise_ideal_leakage=pair_leak,
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    hdr = (f"{'geometry':>22} {'tag':>9} {'mode':>4} {'Cdet':>4} {'gap':>9} "
           f"{'th_ind':>8} {'->m':>7} {'->qm':>7} {'->q2m':>7} "
           f"{'lmin/eps':>9} {'cond_bord':>10}")

    print("=== Sweep 1: synthetic dips, two_unequal_ellipses, h=0.03 ===")
    print(hdr)
    name = "two_unequal_ellipses"
    varifold, m, phases, bcomp, M, C = build_geometry_full(name, 0.03)
    eps = EPS_SCALE * m.median().item()
    for q0 in (1.0, 0.3, 0.1, 0.01):
        q = q_synthetic_dip(bcomp, 0, q0)
        for r in audit(name, 0.03, q, phases, C, varifold, m, eps,
                       tag=f"dip{q0}"):
            rows.append(r)
            print(f"{r['geometry']:>22} {r['tag']:>9} {r['mode']:>4} "
                  f"{r['detected_C']:>4} {r['gap']:>9.2e} "
                  f"{r['theta_indicator_deg']:>7.2f}° "
                  f"{r['theta_to_m']:>6.2f}° {r['theta_to_qm']:>6.2f}° "
                  f"{r['theta_to_q2m']:>6.2f}° "
                  f"{r['min_panel_over_eps']:>9.2f} "
                  f"{r['cond_bordered']:>10.2e}", flush=True)
        print()

    print("=== Sweep 2: real coherence fields ===")
    print(hdr)
    real_cases = []
    GEOMETRIES["annulus_r005"] = dict(tier="primary", components=[
        ("circle", dict(R=1.0), (0.0, 0.0), False, 0),
        ("circle", dict(R=0.05), (0.0, 0.0), True, 0)])
    real_cases.append(("annulus_r005", 0.02, 0.15))
    GEOMETRIES["near_sheet_ellipses"] = dict(tier="primary", components=[
        ("ellipse", dict(a=0.4, b=1.0), (-0.475, 0.0), False, 0),
        ("ellipse", dict(a=0.4, b=1.0), (0.475, 0.0), False, 1)])
    real_cases.append(("near_sheet_ellipses", 0.02, 0.30))
    for name, h, sigma_coh in real_cases:
        varifold, m, phases, bcomp, M, C = build_geometry_full(name, h)
        eps = EPS_SCALE * m.median().item()
        q = q_real(varifold, m, sigma_coh)
        tag = f"real(qmin={q.min():.2f})"
        for r in audit(name, h, q, phases, C, varifold, m, eps, tag=tag):
            rows.append(r)
            print(f"{r['geometry']:>22} {r['tag']:>9.9} {r['mode']:>4} "
                  f"{r['detected_C']:>4} {r['gap']:>9.2e} "
                  f"{r['theta_indicator_deg']:>7.2f}° "
                  f"{r['theta_to_m']:>6.2f}° {r['theta_to_qm']:>6.2f}° "
                  f"{r['theta_to_q2m']:>6.2f}° "
                  f"{r['min_panel_over_eps']:>9.2f} "
                  f"{r['cond_bordered']:>10.2e}", flush=True)
        print()
        del GEOMETRIES[name]

    print("=== Angle pullback: production map on two_unequal_ellipses, "
          "q0=0.1, M4, h=0.05 ===")
    varifold, m, phases, bcomp, M, C = build_geometry_full(
        "two_unequal_ellipses", 0.05)
    eps = EPS_SCALE * m.median().item()
    q = q_synthetic_dip(bcomp, 0, 0.1)
    d = mode_decomposition(varifold, m, q, phases, C, "M4", eps)
    N = len(m)
    C_full, C_simple = full_pullback_constraint(d, varifold, m, q, phases, C, N)
    ideal_q2m = ideal_constraint(q * q * m, phases, C, N)
    pullback = dict(
        theta_full_vs_simple=angle_between(C_full, C_simple),
        theta_full_vs_q2m=angle_between(C_full, ideal_q2m),
        theta_simple_vs_q2m=angle_between(C_simple, ideal_q2m),
    )
    for k, v in pullback.items():
        print(f"  {k}: {v:.3f}°")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "bem_metric_policy_audit.json"
        path.write_text(json.dumps(dict(
            meta=dict(epsilon_scale=EPS_SCALE, n_endpoints=K_ENDPOINTS,
                      trace_side="interior", modes=MODES,
                      ideal_of_mode=IDEAL_OF_MODE),
            angle_pullback=pullback, rows=rows), indent=2))
        print(f"\nraw results -> {path}")


if __name__ == "__main__":
    main()
