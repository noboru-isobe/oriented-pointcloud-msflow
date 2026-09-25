"""L0J-2a (reviewer 2026-08-21): static preflight + WB/CC shadow
activation audit at the three pre-b_finer checkpoints (step 1425).

Preflight (hard, any failure blocks L0J-2b):
  - derived cyclic-order certificate (blocker A machinery),
  - exactly ONE mature contact complex, each side a single cyclic
    interval with >= 2 (preferably >= 3) particles, no satellites,
  - order-preserving pairing (implicit in construction),
  - WB/CC source-energy identity to machine precision:
        P_X^WB(X) = P_X^CC(X) = P^full(X)
    (activation is source-energy neutral).

Shadow activation (B2-analogous, same source, no new thresholds --
readings use the L0 conservation sanity bounds and existing fail-closed
guards): committed one-step under W (self_renormalized) and C
(contact_complex_renormalized); compare component areas, barycenters,
per-arm split residual P#(X1) + W - P#(X), inter-arm |dX|_inf/h and
|dtheta|_inf, and the post-step CC complex certificate.

Usage:
    uv run python scripts/experiments/l0j2_preflight.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import build_config  # noqa: E402
from l0j_locality_audit import _V, arms_q  # noqa: E402
from quotient_switch_stability import production_args, wrap  # noqa: E402
from two_ellipses_benchmark import (  # noqa: E402
    SIGMA,
    centroid_order,
    resolve_m,
    signed_area_order,
)
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)

DT = torch.float64
STATES = (("_l0_768", 768), ("_l0_1024", 1024), ("_l0_768rot", 768))
CKPT = 1425


def one_step(pos, ang, grid, delta, tau, q_mode):
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.mm_step import MMStepper
    cfg = build_config(production_args(grid=grid, redist_monotone=True,
                                       q_mode=q_mode), delta, tau)
    solver = MMSolver(cfg)
    st = MMStepper(cfg)
    v = OrientedPointCloudVarifold(positions=pos.clone(),
                                   angles=ang.clone())
    res, committed = solver._advance_committed(st, v)
    return res, committed, st


def sharp(m, q):
    return float((m * q).sum())


def main():
    torch.set_default_dtype(DT)
    rep = {}
    all_pass = True
    for tag, grid in STATES:
        ck = torch.load(f"results/two_ellipses/two_ellipses{tag}"
                        "_states.pt", weights_only=True)
        st0 = ck[CKPT]
        pos, ang = st0["positions"], st0["angles"]
        n = pos.shape[0] // 2
        lbl = torch.zeros(pos.shape[0], dtype=torch.long)
        lbl[n:] = 1
        nor = torch.stack([ang.cos(), ang.sin()], 1)
        delta, tau = map(float, compute_recommended_params(pos))
        m = resolve_m(pos, nor, delta, tau)
        r = dict(step=CKPT)
        # ---- preflight -------------------------------------------
        qs, res = arms_q(pos, nor, m, lbl)     # raises fail-closed
        cxs = res["cx"]
        r["n_complexes"] = cxs["n_complexes"]
        r["sides"] = [(s.loop, s.n) for c in cxs["complexes"]
                      for s in c.sides]
        pre_ok = (cxs["n_complexes"] == 1
                  and all(s.n >= 2 for c in cxs["complexes"]
                          for s in c.sides))
        r["sides_ge3"] = all(s.n >= 3 for c in cxs["complexes"]
                             for s in c.sides)
        P_full = sharp(m, qs["F"])
        P_wb = sharp(m, qs["W"])
        P_cc = sharp(m, qs["C"])
        r["P_full"] = P_full
        r["dP_wb_vs_full"] = P_wb - P_full
        r["dP_cc_vs_full"] = P_cc - P_full
        r["dP_cc_vs_wb"] = P_cc - P_wb
        energy_ok = (abs(P_cc - P_wb) < 1e-12 * P_full
                     and abs(P_cc - P_full) < 1e-12 * P_full)
        r["preflight_pass"] = bool(pre_ok and energy_ok)
        # ---- shadow activation (same source, two arms) -----------
        arms = {}
        committed = {}
        for arm, mode in (("W", "self_renormalized"),
                          ("C", "contact_complex_renormalized")):
            try:
                res1, com, st = one_step(pos, ang, grid, delta, tau,
                                         mode)
                nor1 = torch.stack([com.angles.cos(),
                                    com.angles.sin()], 1)
                m1 = resolve_m(com.positions, nor1, delta, tau)
                qs1, res_cx1 = arms_q(com.positions, nor1, m1, lbl)
                q_arm1 = qs1[arm]
                A = (signed_area_order(com.positions[:n]),
                     signed_area_order(com.positions[n:]))
                A0 = (signed_area_order(pos[:n]),
                      signed_area_order(pos[n:]))
                arms[arm] = dict(
                    split_residual=sharp(m1, q_arm1)
                    + float(res1.wasserstein)
                    - sharp(m, qs[arm]),
                    dA_rel=[(A[j] - A0[j]) / A0[j] for j in (0, 1)],
                    dbar=[math.hypot(*[a - b for a, b in
                                       zip(centroid_order(
                                           com.positions[sl]),
                                           centroid_order(pos[sl]))])
                          for sl in (slice(0, n), slice(n, None))],
                    n_iter=int(res1.n_iter),
                    post_complexes=res_cx1["cx"]["n_complexes"],
                    post_sides=[(s.loop, s.n)
                                for c in res_cx1["cx"]["complexes"]
                                for s in c.sides],
                    exception=None)
                committed[arm] = com
            except Exception as ex:            # noqa: BLE001
                arms[arm] = dict(
                    exception=f"{type(ex).__name__}: {str(ex)[:120]}")
        r["arms"] = arms
        both = all(a.get("exception") is None for a in arms.values())
        if both:
            h = float(m.median())
            dX = float((committed["W"].positions
                        - committed["C"].positions).abs().max()) / h
            dTh = float(wrap(committed["W"].angles
                             - committed["C"].angles).abs().max())
            r["dX_arms_over_h"] = dX
            r["dtheta_arms_inf"] = dTh
            # readings against the L0 coarse sanity bounds (no new
            # thresholds): conservation must stay in the same class
            cons_ok = all(
                max(abs(x) for x in arms[a]["dA_rel"]) < 1e-3
                and max(arms[a]["dbar"]) < 1e-2 for a in arms)
            post_ok = arms["C"]["post_complexes"] == 1
            r["shadow_pass"] = bool(cons_ok and post_ok)
        else:
            r["shadow_pass"] = False
        r["pass"] = bool(r["preflight_pass"] and r["shadow_pass"])
        all_pass &= r["pass"]
        rep[tag] = r
        print(f"== {tag} @ {CKPT}: preflight "
              f"{'PASS' if r['preflight_pass'] else 'FAIL'} "
              f"(complexes {r['n_complexes']}, sides {r['sides']}, "
              f"dP_cc_vs_wb {r['dP_cc_vs_wb']:.2e}) shadow "
              f"{'PASS' if r['shadow_pass'] else 'FAIL'}")
        if both:
            for a in ("W", "C"):
                d = arms[a]
                print(f"   {a}: split_R {d['split_residual']:+.3e} "
                      f"dA {max(abs(x) for x in d['dA_rel']):.2e} "
                      f"dbar {max(d['dbar']):.2e} post_cx "
                      f"{d['post_complexes']}")
            print(f"   dX_arms/h {r['dX_arms_over_h']:.3e} "
                  f"dtheta_arms {r['dtheta_arms_inf']:.3e}")
        else:
            print("   arms:", {a: arms[a].get("exception")
                               for a in arms})
    rep["all_pass"] = bool(all_pass)
    Path("results/reports/l0j2_preflight.json").write_text(
        json.dumps(rep, indent=1, default=str))
    print(f"ALL {'PASS' if all_pass else 'FAIL'}; wrote "
          "results/reports/l0j2_preflight.json")


if __name__ == "__main__":
    main()
