"""0C-4: can the carrier sensor read the C: 2 -> 1 merger transition?

The go/no-go gate for keeping two-ellipses as a one-phase MS experiment.
The audit deliberately separates what a single two-ellipse sweep would fuse:

  4a CLEAN UNION BOUNDARY -- the actual boundary of B_{R1} u B_{R2}. For a
     resolved positive gap the sensor must read C = 2; for a resolved
     overlap (interior arcs removed) it must read C = 1. Exact tangency is a
     singular boundary and is reported as a transition INTERVAL scaled by
     l_res = max(h, eps_BEM), never as a single pass/fail instant.

     Measured caveat (correcting the 72db828 commit message, which
     overstated "both detectors ... at both h"): at h = 0.015 and overlap
     depth 2 l_res the barely-merged union has a thin neck and its second
     mode still sits at sigma_2/sigma_max = 3.8e-2 < tau = 0.05, so the
     ABSOLUTE detector reports 2 while the max-gap detector reports 1. The
     detectors agree from depth 4 l_res outward on that level (and from
     2 l_res at h = 0.03). Accurate statement: the transition interval,
     defined by detector DISAGREEMENT, is |g| <~ 2-4 l_res; outside it both
     detectors agree with the phase count on every row. The fixed tau shares
     the regime-dependence of any fixed threshold near a pinched neck --
     one more reason the production rank rule combines gap, absolute
     residuals, and continuation rather than any single cut.

  4b RAW TWO-LOOP CARRIER -- both full circles kept through overlap, as the
     production point cloud does before any hidden-point removal. The union
     phase is connected (C = 1) but the scalar carrier still holds two
     closed loops plus hidden arcs. Stated prediction (risk case, from
     review): the carrier operator reads CARRIER topology, not phase
     topology, and stays at C = 2 -- which would be a conditional failure
     ("label-free on valid phase boundaries" survives, "raw cloud alone
     supplies the fused metric" does not, and a current/indicator
     reconstruction becomes a production requirement).

  4c POST-COHERENCE DELETION -- does removing low-q m points from the raw
     overlap cloud actually restore something the sensor reads like the
     clean union boundary? Deleting arcs generally leaves OPEN curves, where
     the closed-boundary nullity statement no longer applies, so this is a
     test of "does deletion recover a valid phase-boundary operator", not
     "does C = 1 happen to be read".

Also validated here, because 0C-4 and the production prototype depend on
it: the label-free compatibility construction

    C_u = U_m^T W_m^{-1/2} W_u G

(carrier left near-null basis U_m reweighted by the target flux measure W_u)
has the same kernel as the oracle componentwise rows B^T W_u G whenever
W_m^{-1/2} U_m = B R with R invertible -- no component labels are ever
recovered. Prediction, stated before measurement: this matches the oracle
space at the indicator-angle level (~1 degree) for BOTH u = m and u = q m,
even for deep q-dips where the q m-operator's own left modes are useless.

Rank reading throughout: never a single ratio. Reported per row are the
ascending sigma_(1..5), the gap ratios rho_k, the absolute null residuals
sigma_(k)/sigma_max, a max-gap detector AND an absolute-threshold detector
(tau = 0.05), and -- on the fixed-N raw family -- the principal angle of the
candidate subspace to the previous sweep step (subspace continuation, the
static stand-in for temporal hysteresis).

Usage
-----
    uv run python scripts/experiments/bem_rank_transition.py \
        --out results/rank_0c4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold

import sys

sys.path.insert(0, str(Path(__file__).parent))
from bem_metric_policy_audit import assemble_operator  # noqa: E402
from bem_spectral_audit import principal_angles_deg  # noqa: E402
from bem_visible_weight_audit import q_real  # noqa: E402

DTYPE = torch.float64
EPS_SCALE = 0.1
R1, R2 = 0.7, 1.0
TAU_NULL = 0.05          # absolute null threshold, relative to sigma_max


# --------------------------------------------------------------------------
# geometry families
# --------------------------------------------------------------------------

def full_circle(R, cx, spacing):
    n = max(12, int(round(2 * math.pi * R / spacing)))
    th = (torch.arange(n, dtype=DTYPE) + 0.5) / n * 2 * math.pi
    pos = torch.stack([cx + R * torch.cos(th), R * torch.sin(th)], 1)
    w = torch.full((n,), 2 * math.pi * R / n, dtype=DTYPE)
    return pos, th, w


def raw_two_loops(d, spacing):
    """Both complete circles regardless of overlap (the production carrier).
    N is fixed by spacing at construction and does not change with d."""
    p1, a1, w1 = full_circle(R1, 0.0, spacing)
    p2, a2, w2 = full_circle(R2, d, spacing)
    varifold = OrientedPointCloudVarifold(
        positions=torch.cat([p1, p2]), angles=torch.cat([a1, a2]))
    loops = torch.cat([torch.zeros(len(w1), dtype=torch.long),
                       torch.ones(len(w2), dtype=torch.long)])
    return varifold, torch.cat([w1, w2]), loops


def union_boundary(d, spacing):
    """The actual boundary of B_R1(0) u B_R2(d), overlap regime
    |R1 - R2| < d < R1 + R2. Each circle contributes one retained arc
    (midpoint on its far side); normals are the circles' own outward
    normals, which coincide with the union's outward normal on the retained
    arcs. The two corner points are genuine corners of the union boundary.
    """
    x = (d * d + R1 * R1 - R2 * R2) / (2 * d)
    y2 = R1 * R1 - x * x
    assert y2 > 0, "no intersection: not in the overlap regime"
    a1 = math.atan2(math.sqrt(y2), x)          # on circle 1, corners at +-a1
    b2 = math.atan2(math.sqrt(y2), x - d)      # on circle 2, corners at +-b2

    ext1 = 2 * math.pi - 2 * a1                # retained arc through theta=pi
    n1 = max(6, int(round(R1 * ext1 / spacing)))
    th1 = a1 + (torch.arange(n1, dtype=DTYPE) + 0.5) / n1 * ext1
    p1 = torch.stack([R1 * torch.cos(th1), R1 * torch.sin(th1)], 1)
    w1 = torch.full((n1,), R1 * ext1 / n1, dtype=DTYPE)

    ext2 = 2 * b2                              # retained arc through phi=0
    n2 = max(6, int(round(R2 * ext2 / spacing)))
    th2 = -b2 + (torch.arange(n2, dtype=DTYPE) + 0.5) / n2 * ext2
    p2 = torch.stack([d + R2 * torch.cos(th2), R2 * torch.sin(th2)], 1)
    w2 = torch.full((n2,), R2 * ext2 / n2, dtype=DTYPE)

    varifold = OrientedPointCloudVarifold(
        positions=torch.cat([p1, p2]), angles=torch.cat([th1, th2]))
    return varifold, torch.cat([w1, w2])


def separated_circles(g, spacing):
    d = R1 + R2 + g
    return raw_two_loops(d, spacing)[:2]


# --------------------------------------------------------------------------
# sensor + rank reading
# --------------------------------------------------------------------------

def sensor(varifold, w):
    """Carrier operator (panel = quadrature = geometric weights)."""
    eps = EPS_SCALE * w.median().item()
    op = assemble_operator(varifold, w, w, eps)
    w_end = op["w_end"]
    sqrt_w = w_end.sqrt()
    At = (sqrt_w[:, None] * op["A"]) / sqrt_w[None, :]
    U, S_desc, Vh = torch.linalg.svd(At)
    return dict(op=op, U=U, S_desc=S_desc, sigma=S_desc.flip(0),
                sqrt_w=sqrt_w, eps=eps)


def rank_report(sigma: torch.Tensor, kmax: int = 4) -> dict:
    smax = sigma[-1]
    rel = (sigma[:kmax + 1] / smax)
    rho = sigma[1:kmax + 1] / sigma[:kmax]
    detected_maxgap = int(torch.argmax(rho).item()) + 1
    detected_abs = int((rel[:kmax] < TAU_NULL).sum().item())
    return dict(
        sigma_low=[s.item() for s in sigma[:5]],
        null_residuals=[r.item() for r in rel[:kmax]],
        gap_ratios=[r.item() for r in rho],
        detected_maxgap=detected_maxgap,
        detected_abs=detected_abs,
    )


def subspace(d: dict, k: int) -> torch.Tensor:
    return d["U"][:, -k:]


# --------------------------------------------------------------------------
# the label-free reweighted constraint C_u (architecture validation)
# --------------------------------------------------------------------------

def reweighted_constraint(dec: dict, u_particle, C, N) -> torch.Tensor:
    """C_u = U_m^T W_m^{-1/2} W_u G, aggregated to particle displacements.
    No component labels touched: the carrier left near-null subspace is
    reweighted by the target measure directly."""
    K = dec["op"]["K"]
    U0 = subspace(dec, C)
    w_u_end = (u_particle / K).repeat_interleave(K)
    vec = w_u_end / dec["sqrt_w"]
    return (U0.T * vec[None, :]).reshape(C, N, K).sum(-1)


def oracle_constraint(u_particle, phases, C, N) -> torch.Tensor:
    Cmat = torch.zeros(C, N, dtype=DTYPE)
    Cmat[phases, torch.arange(N)] = u_particle
    return Cmat


def rowspace_angle(Ca, Cb) -> float:
    qa, _ = torch.linalg.qr(Ca.T)
    qb, _ = torch.linalg.qr(Cb.T)
    return principal_angles_deg(qa, qb).max().item()


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

def sweep_4a(spacing: float) -> list[dict]:
    rows = []
    for g_over_h in (8.0, 4.0, 2.0):
        g = g_over_h * spacing
        varifold, w = separated_circles(g, spacing)
        d = sensor(varifold, w)
        rows.append(dict(family="4a_separated", spacing=spacing,
                         g_over_lres=g_over_h, expected_C=2,
                         n_endpoints=len(d["sqrt_w"]),
                         **rank_report(d["sigma"])))
    for ov_over_h in (2.0, 4.0, 8.0, 13.0, 20.0):
        depth = ov_over_h * spacing
        dist = R1 + R2 - depth
        varifold, w = union_boundary(dist, spacing)
        d = sensor(varifold, w)
        rows.append(dict(family="4a_union", spacing=spacing,
                         g_over_lres=-ov_over_h, expected_C=1,
                         n_endpoints=len(d["sqrt_w"]),
                         **rank_report(d["sigma"])))
    return rows


def sweep_4b(spacing: float) -> list[dict]:
    """Fixed-N raw loops through approach and overlap; also tracks subspace
    continuation between consecutive sweep steps (fixed endpoint count)."""
    rows = []
    prev_U2 = None
    for d_dist in (R1 + R2 + 8 * spacing, R1 + R2 + 2 * spacing,
                   R1 + R2 - 2 * spacing, R1 + R2 - 8 * spacing,
                   R1 + R2 - 0.4, R1 + R2 - 0.8):
        varifold, w, loops = raw_two_loops(d_dist, spacing)
        dec = sensor(varifold, w)
        rep = rank_report(dec["sigma"])
        U2 = subspace(dec, 2)
        cont = (None if prev_U2 is None
                else principal_angles_deg(U2, prev_U2).max().item())
        prev_U2 = U2

        # what does the k=1 candidate align with: the phase constant (all
        # ones) or the per-loop indicators?
        K = dec["op"]["K"]
        lab_end = loops.repeat_interleave(K)
        sw = dec["sqrt_w"]
        B_loops = torch.zeros(len(sw), 2, dtype=DTYPE)
        B_loops[torch.arange(len(sw)), lab_end] = 1.0
        QB_loops = sw[:, None] * B_loops
        QB_loops = QB_loops / QB_loops.norm(dim=0, keepdim=True)
        B_phase = (sw / sw.norm())[:, None]
        rows.append(dict(
            family="4b_raw", spacing=spacing,
            gap_signed=(d_dist - R1 - R2) / spacing,
            phase_C=1 if d_dist < R1 + R2 else 2,
            n_endpoints=len(sw),
            subspace_continuation_deg=cont,
            theta_U2_loops=principal_angles_deg(U2, QB_loops).max().item(),
            theta_U1_phase=principal_angles_deg(
                subspace(dec, 1), B_phase).max().item(),
            **rep))
    return rows


def sweep_4c(spacing: float, sigma_coh: float = 0.2) -> list[dict]:
    rows = []
    for depth in (0.4, 0.8):
        d_dist = R1 + R2 - depth
        varifold, w, loops = raw_two_loops(d_dist, spacing)
        q = q_real(varifold, w, sigma_coh)
        for rho in (0.1, 0.3):
            keep = (q * w) >= rho * (q * w).max()
            sub = OrientedPointCloudVarifold(
                positions=varifold.positions[keep],
                angles=varifold.angles[keep])
            # how much of the hidden (interior) carrier survived?
            p = varifold.positions
            inside_other = torch.where(
                loops == 0,
                (p - torch.tensor([d_dist, 0.0], dtype=DTYPE)).norm(dim=1) < R2,
                p.norm(dim=1) < R1)
            hidden_total = int(inside_other.sum())
            hidden_kept = int((inside_other & keep).sum())
            dec = sensor(sub, w[keep])
            rows.append(dict(
                family="4c_deleted", spacing=spacing, overlap_depth=depth,
                rho_dead=rho, sigma_coh=sigma_coh,
                q_min=q.min().item(),
                n_kept=int(keep.sum()), n_total=len(w),
                hidden_kept=hidden_kept, hidden_total=hidden_total,
                expected_C_clean=1,
                **rank_report(dec["sigma"])))
    return rows


def validate_reweighted_constraint(spacing: float) -> list[dict]:
    """Separated two circles with a synthetic q-dip: the carrier-null
    reweighted C_u must match the oracle componentwise u-rows for both
    u = m and u = q m, with NO labels used on the C_u side."""
    g = 6 * spacing
    varifold, w = separated_circles(g, spacing)
    n1 = int(round(2 * math.pi * R1 / spacing))
    phases = torch.cat([torch.zeros(n1, dtype=torch.long),
                        torch.ones(len(w) - n1, dtype=torch.long)])
    # deep dip on a quarter arc of loop 1
    q = torch.ones(len(w), dtype=DTYPE)
    i0, i1 = int(0.375 * n1), int(0.625 * n1)
    q[i0:i1] = 0.05

    dec = sensor(varifold, w)
    N, C = len(w), 2
    out = []
    for name, u in [("m", w), ("qm", q * w)]:
        Cu = reweighted_constraint(dec, u, C, N)
        Co = oracle_constraint(u, phases, C, N)
        out.append(dict(measure=name, spacing=spacing,
                        theta_Cu_vs_oracle=rowspace_angle(Cu, Co)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spacings", type=float, nargs="+", default=[0.03, 0.015])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    all_rows = []

    print("=== 0C-4a: clean union-boundary family "
          "(expected: C=2 separated, C=1 overlapped) ===")
    print(f"{'family':>14} {'h':>7} {'g/l_res':>8} {'expC':>4} {'det_gap':>7} "
          f"{'det_abs':>7} {'sig_(1..3)/smax':>30}")
    for h in args.spacings:
        for r in sweep_4a(h):
            all_rows.append(r)
            nr = r["null_residuals"]
            print(f"{r['family']:>14} {h:>7.4f} {r['g_over_lres']:>8.1f} "
                  f"{r['expected_C']:>4} {r['detected_maxgap']:>7} "
                  f"{r['detected_abs']:>7} "
                  f"[{nr[0]:.2e} {nr[1]:.2e} {nr[2]:.2e}]", flush=True)
        print()

    print("=== 0C-4b: raw two-loop carrier (phase C=1 after overlap; "
          "risk case: sensor stays 2) ===")
    print(f"{'g/h':>7} {'phaseC':>6} {'det_gap':>7} {'det_abs':>7} "
          f"{'th(U2,loops)':>12} {'th(U1,phase)':>12} {'cont':>7} "
          f"{'sig_(1..3)/smax':>30}")
    for r in sweep_4b(args.spacings[0]):
        all_rows.append(r)
        nr = r["null_residuals"]
        cont = "--" if r["subspace_continuation_deg"] is None \
            else f"{r['subspace_continuation_deg']:.2f}°"
        print(f"{r['gap_signed']:>7.1f} {r['phase_C']:>6} "
              f"{r['detected_maxgap']:>7} {r['detected_abs']:>7} "
              f"{r['theta_U2_loops']:>11.2f}° {r['theta_U1_phase']:>11.2f}° "
              f"{cont:>7} [{nr[0]:.2e} {nr[1]:.2e} {nr[2]:.2e}]", flush=True)
    print()

    print("=== 0C-4c: post-deletion support (does deletion recover a valid "
          "boundary?) ===")
    print(f"{'depth':>6} {'rho':>5} {'q_min':>6} {'kept':>9} {'hidden':>11} "
          f"{'det_gap':>7} {'det_abs':>7} {'sig_(1..3)/smax':>30}")
    for r in sweep_4c(args.spacings[0]):
        all_rows.append(r)
        nr = r["null_residuals"]
        print(f"{r['overlap_depth']:>6.1f} {r['rho_dead']:>5.1f} "
              f"{r['q_min']:>6.2f} "
              f"{r['n_kept']:>4}/{r['n_total']:<4} "
              f"{r['hidden_kept']:>4}/{r['hidden_total']:<4} "
              f"{r['detected_maxgap']:>7} {r['detected_abs']:>7} "
              f"[{nr[0]:.2e} {nr[1]:.2e} {nr[2]:.2e}]", flush=True)
    print()

    print("=== C_u validation: carrier-null reweighted constraint vs oracle "
          "(no labels used) ===")
    cu_rows = validate_reweighted_constraint(args.spacings[0])
    for r in cu_rows:
        all_rows.append(dict(family="Cu_validation", **r))
        print(f"  u = {r['measure']:>3}: angle(C_u, oracle) = "
              f"{r['theta_Cu_vs_oracle']:.3f}°")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "bem_rank_transition.json"
        path.write_text(json.dumps(dict(
            meta=dict(R1=R1, R2=R2, spacings=args.spacings,
                      epsilon_scale=EPS_SCALE, tau_null=TAU_NULL,
                      trace_side="interior", sensor="carrier"),
            rows=all_rows), indent=2))
        print(f"\nraw results -> {path}")


if __name__ == "__main__":
    main()
