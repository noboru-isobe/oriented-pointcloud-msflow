"""0L-B1 calibration of the contact certificate thresholds.

Positive family (must certify): coincident antiparallel circles with
equal counts, unequal counts (90/141, 90/200), half-step angular
offset, angle noise 1e-2, (N, 2N) refinement, residual radial offset
{0, 0.25, 0.5, 1} h_ab, rigid rotation/translation, similarity (radius
x0.6).
Negative family (must be rejected, each with its INTENDED channel):
separation {0.25, 0.5, 1, 2} sigma (gap/current), co-oriented
coincident (anti/current), unequal-radius noncoincident
(gap/coverage/mass), local one-point contact (coverage/g90), eccentric
near-contact (gap/coverage), mass ratio 3 (mass), three-loop ambiguity
(uniqueness). E4-768/1024 terminal states = VALIDATION ONLY (must be
uncertified; not used for thresholds).

Threshold rule (pre-registered): upper-bound metric tau = sqrt(p_max *
n_min) over the intended-channel negatives, requiring p_max < n_min;
lower-bound coverage tau = midpoint(positive min, negative max). A
metric whose intended negatives do not separate is DEMOTED to
telemetry (reported, not gated). sigma_c = sigma_perimeter = 0.1,
audited over {0.8, 1, 1.25} sigma.

Usage:
    uv run python scripts/experiments/contact_certificate_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.torch.transport.contact_certificate import (  # noqa: E402
    ContactThresholds,
    evaluate_all_candidates,
    find_contact_candidates,
)

DT = torch.float64
SIGMA = 0.1
# wide, non-binding thresholds for the MEASUREMENT pass (candidate
# search radius is what matters here); the rule then proposes real ones
# coverage radius c_h = 1.5 h_ab (pre-registered: a full-spacing
# residual offset plus a half-step angular offset must still count as
# covered, sqrt(1 + 0.5^2) h < 1.5 h)
C_H = 1.5
MEASURE = ContactThresholds(gap90_over_h_max=1e9, coverage_min=-1.0,
                            anti_max=1e9, mass_max=1e9,
                            current_residual2_max=1e9,
                            coverage_radius_over_h=C_H,
                            search_radius_over_h=8.0)
# separations used for THRESHOLDS (negatives); 0.25 sigma ~ h at this
# sampling (2 pi 0.35 / 90 = 0.0244) coincides with the positive
# 'residual offset = h_ab' geometry -> TRANSITION validation only
TRANSITION_SEP = (0.25,)
NEGATIVE_SEP = (0.5, 1.0, 2.0)


def circle(R, n, center=(0.0, 0.0), inward=False, phase=0.0,
           angle_noise=0.0, seed=0, mass_scale=1.0):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n + phase
    pos = torch.stack([center[0] + R * t.cos(),
                       center[1] + R * t.sin()], 1)
    ang = t + (math.pi if inward else 0.0)
    if angle_noise > 0:
        g = torch.Generator().manual_seed(seed)
        ang = ang + angle_noise * torch.randn(n, generator=g, dtype=DT)
    m = torch.full((n,), mass_scale * 2 * math.pi * R / n, dtype=DT)
    return pos, ang, m


def assemble(parts):
    pos = torch.cat([p for p, _, _ in parts])
    ang = torch.cat([a for _, a, _ in parts])
    m = torch.cat([mm for _, _, mm in parts])
    lbl = torch.cat([torch.full((p.shape[0],), i, dtype=torch.long)
                     for i, (p, _, _) in enumerate(parts)])
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    return pos, nor, m, lbl


def h_of(n_a, R_a, n_b, R_b):
    return max(2 * math.pi * R_a / n_a, 2 * math.pi * R_b / n_b)


def fixtures():
    P, N = [], []
    R = 0.35
    # ---- positive ----
    P.append(("coincident_equal", [circle(R, 141), circle(R, 141, inward=True)]))
    P.append(("unequal_90_141", [circle(R, 90), circle(R, 141, inward=True)]))
    P.append(("unequal_90_200", [circle(R, 90), circle(R, 200, inward=True)]))
    P.append(("half_step_offset", [circle(R, 141),
                                   circle(R, 141, inward=True,
                                          phase=math.pi / 141)]))
    P.append(("angle_noise_1e-2", [circle(R, 141, angle_noise=1e-2, seed=1),
                                   circle(R, 141, inward=True,
                                          angle_noise=1e-2, seed=2)]))
    P.append(("refined_2N", [circle(R, 282), circle(R, 282, inward=True)]))
    for f in (0.25, 0.5, 1.0):
        h = h_of(90, R, 141, R)
        P.append((f"radial_offset_{f}h", [circle(R, 90),
                                          circle(R + f * h, 141,
                                                 inward=True)]))
    P.append(("rigid_moved", [circle(R, 141, center=(0.3, -0.2),
                                     phase=0.7),
                              circle(R, 141, center=(0.3, -0.2),
                                     inward=True, phase=0.7)]))
    P.append(("similar_0.6", [circle(0.6 * R, 141),
                              circle(0.6 * R, 141, inward=True)]))
    # ---- negative (name, parts, intended channels) ----
    for f in TRANSITION_SEP + NEGATIVE_SEP:
        N.append((f"separated_{f}sigma", [circle(R, 90),
                                          circle(R + f * SIGMA, 141,
                                                 inward=True)],
                  ("gap90_over_h", "current", "coverage")
                  if f in NEGATIVE_SEP else ("transition",)))
    N.append(("co_oriented_coincident", [circle(R, 141), circle(R, 141)],
              ("anti", "current")))
    N.append(("unequal_radius", [circle(R, 90),
                                 circle(0.45, 141, center=(0.05, 0.0),
                                        inward=True)],
              ("gap90_over_h", "coverage", "mass")))
    # local contact: eccentric hole (R 0.45, inward) touching the disk on
    # one side only -- coincident over an arc, far apart opposite
    N.append(("local_contact", [circle(R, 90),
                                circle(0.45, 141, center=(0.0999, 0.0),
                                       inward=True)],
              ("coverage", "gap90_over_h")))
    # eccentric near-contact (gap 0.005..0.035 around the loop): at the
    # pre-registered coverage radius 1.5 h = 0.037 nearly every particle
    # counts as covered, so the INTENDED rejection channel is the
    # 90%-gap (coverage is reported, not intended, for this fixture)
    N.append(("eccentric_near", [circle(R, 90),
                                 circle(R + 0.02, 141, center=(0.015, 0.0),
                                        inward=True)],
              ("gap90_over_h",)))
    N.append(("mass_ratio_3", [circle(R, 141),
                               circle(R, 141, inward=True, mass_scale=3.0)],
              ("mass",)))
    N.append(("triple_ambiguity", [circle(R, 141),
                                   circle(R, 141, inward=True),
                                   circle(R, 141, inward=True,
                                          phase=math.pi / 141)],
              ("uniqueness",)))
    return P, N


MASS_MODE = "oracle"     # or "0k": 0K loopwise KDE masses (parity pin)


def masses_0k(pos, nor, ref_pos):
    """0K loopwise oriented KDE masses on the fixture itself (delta, tau
    from the fixture positions) -- B2-0 estimator parity."""
    from src.torch.oriented_varifold.loopwise_mass import (
        resolve_loopwise_oriented_mass_source,
    )
    from src.torch.oriented_varifold.mass import compute_recommended_params
    # production semantics: (delta, tau) are fixed ONCE from a clean
    # cloud whose loops all sit at the coarse spacing h; a bandwidth
    # read on the union of two coincident loops would be dominated by
    # the cross-loop pairing and under-resolve the coarser loop
    from src.torch.oriented_varifold.mass import KERNEL_CONSTANTS
    delta, _ = map(float, compute_recommended_params(ref_pos))
    # tau uses the GLOBAL point count of the cloud the estimator runs on
    # (same formula as compute_recommended_params: 2 k_min / (N C delta))
    tau = 2.0 * 1.0 / (pos.shape[0] * KERNEL_CONSTANTS["wendland_c2"]
                       * delta)
    res = resolve_loopwise_oriented_mass_source(pos, nor, delta, tau)
    return res.m_loop, res.loop_labels_pre


def measure(name, parts, unrestricted=True):
    pos, nor, m, lbl = assemble(parts)
    if MASS_MODE == "0k":
        # bandwidth reference = the coarsest loop of the fixture alone
        coarse = max(range(len(parts)),
                     key=lambda i: float(parts[i][2].median()))
        m, lbl0 = masses_0k(pos, nor, parts[coarse][0])
        from src.torch.transport.incidence import partition_relation
        if partition_relation(lbl0, lbl) != "equal":
            return dict(name=name, candidates=0,
                        note="0K labels != fixture loops")
    certs = evaluate_all_candidates(pos, nor, m, lbl, None, None, SIGMA,
                                    MEASURE, allow_unrestricted=True)
    if not certs:
        cands = find_contact_candidates(pos, m, lbl, None, None, MEASURE,
                                        allow_unrestricted=True)
        return dict(name=name, candidates=0, note="no candidate",
                    raw=[c.__dict__ for c in cands])
    c = certs[0]
    d = c.as_dict()
    d["name"] = name
    d["candidates"] = len(certs)
    return d


def main():
    global MASS_MODE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", default="oracle", choices=("oracle", "0k"))
    MASS_MODE = ap.parse_args().mass
    torch.set_default_dtype(DT)
    P, N = fixtures()
    pos_rows = [measure(n, p) for n, p in P]
    neg_rows = []
    for n, p, ch in N:
        r = measure(n, p)
        r["intended"] = list(ch)
        neg_rows.append(r)

    def val(r, key):
        if r.get("candidates", 0) == 0:
            return None
        if key == "uniqueness":
            return 0.0 if r["candidate"]["ambiguous"] else 1.0
        if key == "current":
            return r["metrics"]["eps_cur2_max"]
        if key == "coverage":
            return r["metrics"]["coverage"]
        if key == "gap90_over_h":
            return r["metrics"]["gap90_over_h"]
        if key == "anti":
            return r["metrics"]["anti"]
        if key == "mass":
            return r["metrics"]["mass_residual"]
        return None

    print("== positive ==")
    for r in pos_rows:
        print(f"  {r['name']:20s} amb {r['candidate']['ambiguous']} "
              f"g90/h {val(r,'gap90_over_h'):.3f} cov {val(r,'coverage'):.3f}"
              f" anti {val(r,'anti'):.2e} mass {val(r,'mass'):.2e} "
              f"cur2 {val(r,'current'):.2e}", flush=True)
    print("== negative ==")
    for r in neg_rows:
        if r.get("candidates", 0) == 0:
            print(f"  {r['name']:20s} NO CANDIDATE (search radius)")
            continue
        print(f"  {r['name']:20s} amb {r['candidate']['ambiguous']} "
              f"g90/h {val(r,'gap90_over_h'):.3f} cov {val(r,'coverage'):.3f}"
              f" anti {val(r,'anti'):.2e} mass {val(r,'mass'):.2e} "
              f"cur2 {val(r,'current'):.2e}  intended {r['intended']}",
              flush=True)

    # threshold rule
    upper = ("gap90_over_h", "anti", "mass", "current")
    proposal, separation = {}, {}
    for key in upper:
        p_max = max(val(r, key) for r in pos_rows)
        negs = [val(r, key) for r in neg_rows
                if key in r["intended"] and val(r, key) is not None]
        n_min = min(negs) if negs else None
        sep = n_min is not None and p_max < n_min
        separation[key] = dict(p_max=p_max, n_min=n_min, separates=sep)
        proposal[key] = math.sqrt(p_max * n_min) if sep and p_max > 0 \
            else (n_min / 3.0 if sep else None)
    p_min = min(val(r, "coverage") for r in pos_rows)
    negs = [val(r, "coverage") for r in neg_rows
            if "coverage" in r["intended"] and val(r, "coverage") is not None]
    n_max = max(negs) if negs else None
    sep = n_max is not None and p_min > n_max
    separation["coverage"] = dict(p_min=p_min, n_max=n_max, separates=sep)
    proposal["coverage"] = 0.5 * (p_min + n_max) if sep else None
    print("== separation / proposal ==")
    for k in separation:
        print(f"  {k:14s} {separation[k]}  -> tau {proposal[k]}")

    # E4 terminal-state validation (must be uncertified)
    e4 = {}
    try:
        from exact_merger_benchmark import OUT, build_cloud
        from src.torch.oriented_varifold import OrientedPointCloudVarifold
        from src.torch.oriented_varifold.loopwise_mass import (
            resolve_loopwise_oriented_mass_source,
        )
        from src.torch.oriented_varifold.mass import (
            compute_recommended_params,
        )
        v0, _ = build_cloud()
        delta, tau = map(float, compute_recommended_params(v0.positions))
        for tag, step in (("_endgame4", 1575), ("_endgame4_h1024", 1575)):
            ck = torch.load(OUT / f"exact_merger{tag}_states.pt",
                            weights_only=True)
            st = ck[max(k for k in ck if k <= step)]
            v = OrientedPointCloudVarifold(positions=st["positions"],
                                           angles=st["angles"])
            res = resolve_loopwise_oriented_mass_source(
                v.positions, v.normals, delta, tau)
            certs = evaluate_all_candidates(
                v.positions, v.normals, res.m_loop, res.loop_labels_pre,
                None, None, SIGMA, MEASURE, allow_unrestricted=True)
            e4[tag] = [c.as_dict() for c in certs]
            for c in certs:
                print(f"  E4{tag} pair {c.candidate.loop_pair} g90/h "
                      f"{c.metrics['gap90_over_h']:.2f} cov "
                      f"{c.metrics['coverage']:.2f} anti "
                      f"{c.metrics['anti']:.2e} mass "
                      f"{c.metrics['mass_residual']:.2e} cur2 "
                      f"{c.metrics['eps_cur2_max']:.3f}")
    except Exception as ex:      # noqa: BLE001
        print("E4 validation skipped:", ex)

    # classification under the pre-registered production thresholds
    from src.torch.transport.contact_certificate import ContactThresholds
    T = ContactThresholds()

    def classify(r):
        if r.get("candidates", 0) == 0:
            return "no-candidate"
        if r["candidate"]["ambiguous"]:
            return "ambiguous"
        ok = (val(r, "gap90_over_h") <= T.gap90_over_h_max
              and val(r, "coverage") >= T.coverage_min
              and val(r, "anti") <= T.anti_max
              and val(r, "mass") <= T.mass_max
              and val(r, "current") <= T.current_residual2_max)
        return "certified" if ok else "rejected"
    out_cls = {r["name"]: classify(r) for r in pos_rows + neg_rows}
    print(f"== classification under production thresholds ({MASS_MODE}) ==")
    for k, v in out_cls.items():
        print(f"  {k:24s} {v}")
    out = dict(mass_mode=MASS_MODE, classification=out_cls, sigma=SIGMA,
               positive=pos_rows, negative=neg_rows,
               separation=separation, proposal=proposal, e4_validation=e4)
    outp = Path(f"results/reports/phase3c0l_b1_calibration"
                f"{'' if MASS_MODE == 'oracle' else '_0k'}.json")
    outp.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
