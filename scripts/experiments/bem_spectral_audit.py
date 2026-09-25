"""0C-2a: weighted spectral audit of the interior-trace BEM operator.

Gate A (0C-1) already established that the current global displacement space
is NOT the one-phase tangent space. This audit does not re-litigate that. It
asks the next question: do the left/right near-nullspaces of the interior
trace operator correctly represent the componentwise compatibility conditions
and the density gauge freedom -- i.e. is a label-free spectral construction of
the correct tangent space viable?

Conventions (fixed here, used verbatim by the pytest regressions)
-----------------------------------------------------------------
Everything is at endpoint-collocation level (NK), never particle level.

    A      = (1/2) I + K*                    interior trace operator
    W      = diag(zeta_ik)                   endpoint quadrature weights
    A~     = W^{1/2} A W^{-1/2} = U S V^T    weighted operator, full SVD
    sigma  = singular values in ASCENDING order: sigma_(1) <= sigma_(2) <= ...
    U0,V0  = left/right singular vectors of the C smallest singular values
    L      = W^{-1/2} U0                     weighted LEFT near-nullspace
    R      = W^{-1/2} V0                     RIGHT near-nullspace

with the identities this convention is meant to satisfy:

    A R           ~ 0        (right: density gauge / nonuniqueness)
    L^T W A       ~ 0        (left: componentwise Neumann compatibility)

L and R are NOT interchangeable: A is nonsymmetric, so a single symmetric
projector Q^T A Q cannot handle both. The compatibility functional on Neumann
data v is  L^T W v = 0;  the gauge condition on a density lambda is
R^T W lambda = 0.

Primary test: principal angles between span(U0) and span(W^{1/2} B), where B
is the endpoint-level PHASE-component indicator matrix. Boundary components
belonging to the same phase component (e.g. the outer and inner rings of an
annulus) share ONE column. The right nullspace is deliberately NOT compared
with indicators -- right null vectors are layer densities and need not be
piecewise constant off the circle-symmetric cases.

Recorded per (geometry, resolution):  sigma_C, sigma_{C+1}, their ratio, the
absolute residuals ||A R||_2 and ||L^T W A||_2, ||A~||_2 for scale, all C
principal angles (max is the pass quantity), and the C a max-gap detector
would report. Pass conditions B-a1..B-a5 are evaluated over the whole table,
not per row -- and never on circles/concentric shapes alone, since circle
symmetry is exactly what hid the trace-side error for six months.

A pass here is "the weighted left near-nullspace agrees with the
componentwise compatibility space on separated test geometries". It is NOT
"Gate B passed": near-contact gap behaviour, the C -> C-1 transition at
fusion, rank hysteresis, and subspace continuation are all 0C-4.

Usage
-----
    uv run python scripts/experiments/bem_spectral_audit.py
    uv run python scripts/experiments/bem_spectral_audit.py \
        --levels 0.06 0.03 0.015 --out results/spectral_0c2a
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.shapes.generator import (
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_flower,
)
from src.torch.transport.bem_wasserstein import BEMWasserstein

DTYPE = torch.float64
DEV = "cpu"
EPS_SCALE = 0.1
K_ENDPOINTS = 3


# --------------------------------------------------------------------------
# geometry: boundary components composed into labelled clouds
# --------------------------------------------------------------------------

def _raw_component(kind: str, params: dict, center, n: int, sampling: str):
    if kind == "circle":
        return generate_oriented_circle(n, params["R"], center, DEV, DTYPE)
    if kind == "ellipse":
        return generate_oriented_ellipse(
            n, params["a"], params["b"], center, DEV, DTYPE,
            initial_sampling=sampling)
    if kind == "flower":
        return generate_oriented_flower(
            n, params["petals"], params["r_in"], params["r_out"], center,
            DEV, DTYPE, initial_sampling=sampling)
    raise ValueError(kind)


def _component_length(kind: str, params: dict) -> float:
    """Boundary length from a dense parameter sample (2048 points)."""
    v = _raw_component(kind, params, (0.0, 0.0), 2048, "parameter")
    d = (v.positions.roll(-1, 0) - v.positions).norm(dim=1)
    return d.sum().item()


def build_component(kind: str, params: dict, center, inward: bool,
                    spacing: float):
    """One closed boundary component at target point spacing.

    Non-circles are sampled uniformly in arc length so the spacing is actually
    uniform. `inward=True` flips the normals by pi: the material is outside
    the curve, so its outward normal points into the hole (same convention as
    generate_oriented_annulus).

    Weights are geometric arc lengths from neighbour distances, computed
    WITHIN the component -- rolling across a concatenated cloud would create
    spurious edges between components (the geometric_area bug, not repeated).
    """
    length = _component_length(kind, params)
    n = max(16, int(round(length / spacing)))
    sampling = "parameter" if kind == "circle" else "arc_length"
    v = _raw_component(kind, params, center, n, sampling)
    angles = v.angles + (math.pi if inward else 0.0)
    d = (v.positions.roll(-1, 0) - v.positions).norm(dim=1)
    weights = 0.5 * (d + d.roll(1, 0))
    return v.positions, angles, weights


# Each geometry: list of (kind, params, center, inward, phase_index).
# Boundary components of the same phase component share a phase_index.
GEOMETRIES = {
    # --- sanity tier: circular / concentric (never sufficient on their own) --
    "disk": dict(tier="sanity", components=[
        ("circle", dict(R=1.0), (0.0, 0.0), False, 0)]),
    "concentric_annulus": dict(tier="sanity", components=[
        ("circle", dict(R=1.0), (0.0, 0.0), False, 0),
        ("circle", dict(R=0.5), (0.0, 0.0), True, 0)]),
    "two_disks": dict(tier="sanity", components=[
        ("circle", dict(R=1.0), (-3.0, 0.0), False, 0),
        ("circle", dict(R=1.0), (3.0, 0.0), False, 1)]),
    "three_disks": dict(tier="sanity", components=[
        ("circle", dict(R=0.9), (-3.0, 0.0), False, 0),
        ("circle", dict(R=0.9), (0.0, 2.6), False, 1),
        ("circle", dict(R=0.9), (3.0, 0.0), False, 2)]),
    # --- primary tier: noncircular / nonconcentric ---------------------------
    "ellipse": dict(tier="primary", components=[
        ("ellipse", dict(a=1.0, b=0.5), (0.0, 0.0), False, 0)]),
    "flower": dict(tier="primary", components=[
        ("flower", dict(petals=5, r_in=0.5, r_out=1.0), (0.0, 0.0), False, 0)]),
    "eccentric_annulus": dict(tier="primary", components=[
        ("circle", dict(R=1.0), (0.0, 0.0), False, 0),
        ("circle", dict(R=0.35), (0.30, 0.10), True, 0)]),
    "two_unequal_ellipses": dict(tier="primary", components=[
        ("ellipse", dict(a=1.0, b=0.5), (-1.6, 0.0), False, 0),
        ("ellipse", dict(a=0.55, b=0.35), (1.4, 0.25), False, 1)]),
    "annulus_plus_ellipse": dict(tier="primary", components=[
        ("circle", dict(R=1.0), (0.0, 0.0), False, 0),
        ("circle", dict(R=0.5), (0.0, 0.0), True, 0),
        ("ellipse", dict(a=0.7, b=0.35), (2.9, 0.2), False, 1)]),
    "nonconcentric_two_holes": dict(tier="primary", components=[
        ("circle", dict(R=1.5), (0.0, 0.0), False, 0),
        ("circle", dict(R=0.30), (-0.55, 0.25), True, 0),
        ("ellipse", dict(a=0.35, b=0.18), (0.55, -0.40), True, 0)]),
    "disk_ellipse_flower": dict(tier="primary", components=[
        ("circle", dict(R=0.8), (-2.2, 0.0), False, 0),
        ("ellipse", dict(a=0.9, b=0.45), (0.6, 0.1), False, 1),
        ("flower", dict(petals=5, r_in=0.35, r_out=0.7), (3.2, -0.1), False, 2)]),
}


def build_geometry(name: str, spacing: float):
    spec = GEOMETRIES[name]
    pos, ang, wts, labels = [], [], [], []
    for kind, params, center, inward, phase in spec["components"]:
        p, a, w = build_component(kind, params, center, inward, spacing)
        pos.append(p)
        ang.append(a)
        wts.append(w)
        labels.append(torch.full((len(w),), phase, dtype=torch.long))
    varifold = OrientedPointCloudVarifold(
        positions=torch.cat(pos), angles=torch.cat(ang))
    phases = torch.cat(labels)
    M = len(spec["components"])
    C = int(phases.max().item()) + 1
    return varifold, torch.cat(wts), phases, M, C


# --------------------------------------------------------------------------
# the audit itself
# --------------------------------------------------------------------------

def principal_angles_deg(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Principal angles (degrees, ascending) between the column spans of two
    matrices with orthonormal columns."""
    cos = torch.linalg.svdvals(X.T @ Y).clamp(0.0, 1.0)
    return torch.rad2deg(torch.arccos(cos)).flip(0).sort().values


def spectral_decomposition(name: str, spacing: float) -> dict:
    """Everything the audit and the pytest regressions share.

    Returns tensors, so callers can also examine the subspaces themselves --
    e.g. to verify that the RIGHT nullspace is not the indicator space. On an
    annulus whose OUTER boundary is a circle, the right null density is
    uniform on the outer ring only, at principal angle
    arccos sqrt(L_out / L_total) from the indicator; for a general
    noncircular outer boundary it is the (non-uniform) equilibrium density,
    so only the circular case has this closed form.
    """
    varifold, masses, phases, M, C = build_geometry(name, spacing)

    bem = BEMWasserstein(method="point", epsilon_scale=EPS_SCALE,
                         n_endpoints=K_ENDPOINTS, trace_side="interior")
    bem.setup_for_step(varifold.positions, varifold.normals, masses, sigma=None)
    op = bem.endpoint_operator()
    w = op.weights
    sqrt_w = w.sqrt()

    U, S_desc, Vh = torch.linalg.svd(op.weighted_trace_operator())  # descending
    sigma = S_desc.flip(0)                                          # ascending

    # C smallest singular directions -> the near-nullspaces
    U0 = U[:, -C:]
    V0 = Vh[-C:, :].T
    L = U0 / sqrt_w[:, None]
    R = V0 / sqrt_w[:, None]

    # endpoint-level phase indicators; disjoint supports, so the columns of
    # W^{1/2} B are orthogonal and only need normalising
    lab_end = phases.repeat_interleave(op.n_endpoints)
    B = torch.zeros(len(w), C, dtype=DTYPE)
    B[torch.arange(len(w)), lab_end] = 1.0
    QB = sqrt_w[:, None] * B
    QB = QB / QB.norm(dim=0, keepdim=True)

    return dict(op=op, M=M, C=C, sigma=sigma, U0=U0, V0=V0, L=L, R=R, QB=QB,
                A_weighted=op.weighted_trace_operator(),
                operator_norm=S_desc[0].item())


def audit_case(name: str, spacing: float) -> dict:
    d = spectral_decomposition(name, spacing)
    op, C, sigma = d["op"], d["C"], d["sigma"]
    w, A = op.weights, op.trace_operator

    # Absolute residuals in UNWEIGHTED coordinates. These are convention
    # checks only: given an exact SVD they equal sigma_C dressed with
    # W^{+-1/2} factors, so their apparent h-powers (~h^0.5 right, ~h^1.5
    # left) are largely coordinate scaling, NOT independent convergence
    # orders. Do not quote them as such.
    res_R = torch.linalg.matrix_norm(A @ d["R"], ord=2).item()
    res_L = torch.linalg.matrix_norm(d["L"].T @ (w[:, None] * A), ord=2).item()

    # The primary residuals, in WEIGHTED coordinates, normalised by the
    # operator norm. Both reduce to sigma_C / sigma_max for an exact SVD;
    # computing them from the matrices doubles as an implementation check.
    At = d["A_weighted"]
    res_R_w = (torch.linalg.matrix_norm(At @ d["V0"], ord=2)
               / d["operator_norm"]).item()
    res_L_w = (torch.linalg.matrix_norm(d["U0"].T @ At, ord=2)
               / d["operator_norm"]).item()

    angles_deg = principal_angles_deg(d["U0"], d["QB"])

    # what a max-gap detector would report, among the first 8 modes
    kmax = min(8, len(sigma) - 1)
    ratios = sigma[1:kmax + 1] / sigma[:kmax]
    detected_C = int(torch.argmax(ratios).item()) + 1

    return dict(
        geometry=name, tier=GEOMETRIES[name]["tier"], spacing=spacing,
        M=d["M"], C=C, n_endpoints=len(w),
        epsilon=op.epsilon,
        sigma_low=[s.item() for s in sigma[:C + 3]],
        sigma_C=sigma[C - 1].item(),
        sigma_Cp1=sigma[C].item(),
        gap=(sigma[C - 1] / sigma[C]).item(),
        operator_norm=d["operator_norm"],
        res_R_abs=res_R,
        res_L_abs=res_L,
        res_R_rel_weighted=res_R_w,
        res_L_rel_weighted=res_L_w,
        principal_angles_deg=[a.item() for a in angles_deg],
        theta_max_deg=angles_deg.max().item(),
        detected_C=detected_C,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.06, 0.03, 0.015],
                    help="target point spacings, coarse to fine")
    ap.add_argument("--cases", nargs="+", default=list(GEOMETRIES),
                    choices=list(GEOMETRIES))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    hdr = (f"{'geometry':>24} {'tier':>8} {'h':>7} {'NK':>6} {'M':>2} {'C':>2} "
           f"{'sig_C':>10} {'sig_C+1':>9} {'gap':>10} {'th_max':>8} "
           f"{'||AR||':>10} {'||L^TWA||':>10} {'C_det':>5}")
    print(hdr)
    for name in args.cases:
        for h in args.levels:
            t0 = time.time()
            r = audit_case(name, h)
            r["seconds"] = time.time() - t0
            rows.append(r)
            print(f"{r['geometry']:>24} {r['tier']:>8} {h:>7.4f} "
                  f"{r['n_endpoints']:>6} {r['M']:>2} {r['C']:>2} "
                  f"{r['sigma_C']:>10.2e} {r['sigma_Cp1']:>9.2e} "
                  f"{r['gap']:>10.2e} {r['theta_max_deg']:>7.3f}° "
                  f"{r['res_R_abs']:>10.2e} {r['res_L_abs']:>10.2e} "
                  f"{r['detected_C']:>5}", flush=True)
        print()

    # ---- machine evaluation of B-a1..B-a5 over the table -------------------
    finest = min(args.levels)
    by_geom = {n: sorted((r for r in rows if r["geometry"] == n),
                         key=lambda r: -r["spacing"]) for n in args.cases}
    def gap_trend_ok(seq):
        """The trend form of B-a2: gap shrinking monotonically under
        refinement while sigma_{C+1} stays bounded away from zero."""
        return (all(a["gap"] > b["gap"] for a, b in zip(seq, seq[1:]))
                and all(r["sigma_Cp1"] > 0.1 for r in seq))

    checks = dict(
        # B-a1: detector finds C at the finest level, everywhere
        Ba1_dim=all(seq[-1]["detected_C"] == seq[-1]["C"]
                    for seq in by_geom.values()),
        # DIAGNOSTIC ONLY, kept as the measured counterexample to a
        # geometry-independent ratio threshold: sigma_{C+1} is a geometry
        # constant (~0.49 for disks, ~0.19 for flower-bearing domains), so a
        # fixed gap < 1e-2 at a fixed resolution penalises some shapes even
        # though their gap halves with h exactly like everyone else's.
        Ba2_fixed_ratio_diagnostic=all(seq[-1]["gap"] < 1e-2
                                       for seq in by_geom.values()),
        # B-a2, gate form
        Ba2_asymptotic_trend=all(gap_trend_ok(seq)
                                 for seq in by_geom.values() if len(seq) > 1),
        # B-a3: max principal angle decreases coarse -> fine, everywhere
        Ba3_angle=all(seq[-1]["theta_max_deg"] < seq[0]["theta_max_deg"]
                      or seq[-1]["theta_max_deg"] < 0.5
                      for seq in by_geom.values() if len(seq) > 1),
        # B-a4: absolute null residuals decrease coarse -> fine
        Ba4_res=all(seq[-1]["res_L_abs"] < seq[0]["res_L_abs"]
                    and seq[-1]["res_R_abs"] < seq[0]["res_R_abs"]
                    for seq in by_geom.values() if len(seq) > 1),
        # B-a5, gate form: dimension + trend on the noncircular tier
        Ba5_noncircular_trend=all(
            seq[-1]["detected_C"] == seq[-1]["C"]
            and (len(seq) == 1 or gap_trend_ok(seq))
            for seq in by_geom.values() if seq[-1]["tier"] == "primary"),
    )
    # the machine-readable overall verdict, aligned with the prose: the gate
    # criteria are the trend forms, the fixed ratio stays as a diagnostic
    checks["passed_0C2a"] = (checks["Ba1_dim"]
                             and checks["Ba2_asymptotic_trend"]
                             and checks["Ba3_angle"]
                             and checks["Ba4_res"]
                             and checks["Ba5_noncircular_trend"])
    print("pass conditions over the table "
          f"(finest level h={finest}):")
    for k, v in checks.items():
        tag = "PASS" if v else ("FAIL (diagnostic)" if "diagnostic" in k
                                else "FAIL")
        print(f"  {k}: {tag}")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "bem_spectral_audit.json"
        path.write_text(json.dumps(dict(
            meta=dict(levels=args.levels, epsilon_scale=EPS_SCALE,
                      n_endpoints=K_ENDPOINTS, trace_side="interior",
                      dtype=str(DTYPE)),
            checks=checks, rows=rows), indent=2))
        print(f"\nraw results -> {path}")


if __name__ == "__main__":
    main()
