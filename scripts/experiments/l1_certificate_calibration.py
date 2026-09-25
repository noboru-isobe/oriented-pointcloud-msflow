"""L1-0^geom/raw DESIGN SKELETON (reviewer 2026-08-22: parallel GO
during the L-A validation runs; L1 dynamic quotient stays HOLD until
L-A closure).

Purpose: define the CLEAN FIXTURE FAMILIES from which every L1 window
certificate threshold will be calibrated, and measure the certificate
quantities on them. Pre-registration discipline: thresholds come from
THESE analytic fixtures (+ pre-L1 trajectory states), never from the
L1 phase's own runs. This script performs measurement and
distribution dumps only -- it fixes no thresholds.

Certificate quantities (L1-0^geom):
  pairing      order-preserving mutual-NN pairing between the two
               sheets of an h-scale contact window: pairing fraction
               (coverage), cyclic-order monotonicity of the pairing
               map (no crossing / no order reversal).
  eps_anti     anti-alignment of paired normals: distribution of
               1 + n_a . n_b over pairs (0 = perfectly annihilating).
  parab        window length against the parabolic scale
               L_W ~ sqrt(h / kappa_sum) (dimensionless L_W ratio).

Certificate quantities (L1-0^raw):
  mass_bal     paired raw mass balance sum_a m / sum_b m - 1.
  raw_current  tapered raw boundary current of the window
               || sum m tap (cos, sin)(theta) || relative to the
               untapered window mass; cos^2 taper at the window ends.
               Near-cancellation is the annihilation signature.

Negative fixtures (must FAIL the certificates; they pin the failure
modes, not thresholds): three-arc window (a third sheet inside the
window), same-loop self-contact (horseshoe), order-reversing pairing
(one sheet reversed).

Usage:
    uv run python scripts/experiments/l1_certificate_calibration.py \
        --family geom --n-samples 200 --tag _l1cal_geom
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

DT = torch.float64
OUT = Path("results/two_ellipses/l1_calibration")


# ------------------------------------------------------------------
# fixture families
# ------------------------------------------------------------------

def parabola_sheet(kappa, h, L, x0=0.0, y0=0.0, up=True, phase=0.0,
                   jitter=0.0, gen=None):
    """Arc-length-ish sampled parabola y = y0 +- kappa x^2 / 2 over
    [-L/2, L/2], spacing ~ h, outward normal away from the other
    sheet when up=True is the LOWER sheet (normal +y)."""
    n = max(int(L / h), 4)
    x = torch.linspace(-L / 2, L / 2, n, dtype=DT) + phase * h
    if jitter and gen is not None:
        x = x + jitter * h * (torch.rand(n, generator=gen, dtype=DT)
                              - 0.5)
    s = 1.0 if up else -1.0
    y = y0 + s * kappa * x ** 2 / 2.0
    pos = torch.stack([x + x0, y], 1)
    # tangent (1, s kappa x) -> outward normal for the sheet facing
    # the gap: lower sheet faces +y, upper sheet faces -y
    ty = s * kappa * x
    nrm = torch.stack([-ty, torch.ones_like(ty) * s], 1)
    nrm = nrm / nrm.norm(dim=1, keepdim=True)
    m = torch.full((n,), L / n, dtype=DT)
    return pos, nrm, m


def facing_pair(kappa1, kappa2, gap, h, L, phase2=0.0, jitter=0.0,
                seed=0, reverse_b=False):
    """The clean L1 window fixture: two anti-aligned parabolic sheets
    facing across gap. Sheet A below (normal +y), sheet B above
    (normal -y)."""
    gen = torch.Generator().manual_seed(seed)
    pa, na, ma = parabola_sheet(kappa1, h, L, y0=0.0, up=True,
                                jitter=jitter, gen=gen)
    pb, nb, mb = parabola_sheet(kappa2, h, L, y0=gap, up=False,
                                phase=phase2, jitter=jitter, gen=gen)
    # certified orientation: facing arcs of two CCW loops traverse
    # ANTI-parallel in space, so sheet B's certified order is
    # right-to-left -> flip. reverse_b=True SKIPS the flip = the
    # order-reversing-pairing negative (arrays not in certified
    # order), which the oriented monotonicity test must reject.
    if not reverse_b:
        pb, nb, mb = pb.flip(0), nb.flip(0), mb.flip(0)
    return (pa, na, ma), (pb, nb, mb)


def third_arc(gap, h, L):
    """Negative: a third flat sheet inside the window at gap/2."""
    return parabola_sheet(0.0, h, L * 0.6, y0=gap / 2, up=True)


# ------------------------------------------------------------------
# certificate measurements
# ------------------------------------------------------------------

def mutual_nn_pairing(pa, pb):
    """Mutual nearest-neighbor pairs (ia, ib)."""
    D = torch.cdist(pa, pb)
    jb = D.argmin(dim=1)
    ja = D.argmin(dim=0)
    ia = torch.arange(pa.shape[0])
    mutual = ja[jb[ia]] == ia
    return ia[mutual], jb[mutual]


def order_preserving(ib):
    """Pairing map monotone NON-INCREASING in the certified
    orientations. Design lesson (v0 smoke): a direction-agnostic
    test accepts a reversed sheet -- the certificate must consume the
    certified cyclic orientation (L0J certify_loop_orders), under
    which two ANTI-aligned facing sheets pair in OPPOSITE arc order
    (A ascending pairs B descending). Allowing the single cyclic
    wrap of a closed loop is L1's job; fixtures are open arcs ->
    strict."""
    d = ib[1:] - ib[:-1]
    return bool((d <= 0).all())


def measure_geom(A, B, h):
    """Design lesson (v2 sweep): MUTUAL-NN collapses to 2-3 pairs at a
    half-spacing sampling phase (each A particle equidistant to two B
    particles), so mutual coverage is NOT a robust h-scale certificate
    quantity (the real L-A3 windows are mirror-symmetric, phase 0, and
    hid this). Certificate pairing = ONE-SIDED NN (every A particle to
    its nearest B): order monotonicity and anti-alignment are then
    phase-robust; mutual coverage stays as telemetry."""
    (pa, na, ma), (pb, nb, mb) = A, B
    D = torch.cdist(pa, pb)
    ib = D.argmin(dim=1)
    ia = torch.arange(pa.shape[0])
    rec = {}
    rec["n_a"], rec["n_b"] = int(pa.shape[0]), int(pb.shape[0])
    ia_m, ib_m = mutual_nn_pairing(pa, pb)
    rec["n_pairs_mutual"] = int(ia_m.shape[0])
    rec["coverage_mutual"] = rec["n_pairs_mutual"] / min(rec["n_a"],
                                                          rec["n_b"])
    core = (ia_m >= rec["n_a"] // 4) & (ia_m < 3 * rec["n_a"] // 4)
    n_core_slots = rec["n_a"] // 2
    rec["coverage_core"] = (int(core.sum()) / n_core_slots
                            if n_core_slots else None)
    rec["order_preserving"] = order_preserving(ib)
    anti = 1.0 + (na * nb[ib]).sum(dim=1)
    rec["eps_anti_max"] = float(anti.max())
    rec["eps_anti_med"] = float(anti.median())
    gaps = D.min(dim=1).values
    rec["gap_min"] = float(gaps.min())
    rec["gap_med"] = float(gaps.median())
    rec["gap_over_h_med"] = rec["gap_med"] / h
    return rec, (ia, ib)


def measure_raw(A, B, pair, h):
    """Window-sum quantities (phase-robust): mass balance of the two
    window sheets and the cos^2-tapered raw current of the whole
    window (A in certified order, B in certified order, each tapered
    along its own arc)."""
    (pa, na, ma), (pb, nb, mb) = A, B
    rec = {}
    Ma, Mb = float(ma.sum()), float(mb.sum())
    rec["mass_balance"] = Ma / Mb - 1.0

    def taper(n):
        t = torch.linspace(0.0, 1.0, n, dtype=DT)
        return torch.sin(math.pi * t) ** 2
    ta, tb = taper(pa.shape[0]), taper(pb.shape[0])
    cur = ((ma * ta)[:, None] * na).sum(dim=0) \
        + ((mb * tb)[:, None] * nb).sum(dim=0)
    rec["raw_current_rel"] = float(cur.norm()
                                   / ((ma * ta).sum() + (mb * tb).sum()))
    cur0 = (ma[:, None] * na).sum(dim=0) + (mb[:, None] * nb).sum(dim=0)
    rec["raw_current_rel_untapered"] = float(cur0.norm() / (Ma + Mb))
    return rec


def l1_certificate(rec, thr):
    """L1-0^geom/raw window certificate: all-or-nothing, fail-closed.
    thr = dict(eps_anti_max, mass_balance_max, raw_current_rel_max);
    coverage is telemetry only (mutual-NN is sampling-phase fragile).
    Returns (certified, failing list)."""
    failing = []
    if not rec.get("order_preserving"):
        failing.append("order_preserving")
    if rec.get("eps_anti_max") is None \
            or rec["eps_anti_max"] > thr["eps_anti_max"]:
        failing.append("eps_anti")
    if rec.get("mass_balance") is None \
            or abs(rec["mass_balance"]) > thr["mass_balance_max"]:
        failing.append("mass_balance")
    if rec.get("raw_current_rel") is None \
            or rec["raw_current_rel"] > thr["raw_current_rel_max"]:
        failing.append("raw_current")
    return (not failing), failing


def l1_certificate_orderfree(rec, thr):
    """Order-free variant (user 2026-08-24): anti-alignment + window
    mass balance + tapered raw-current cancellation ONLY -- no
    order-preserving clause, hence consumable by a pure oriented
    point cloud. The question: which negatives still fail?"""
    failing = []
    if rec.get("eps_anti_max") is None \
            or rec["eps_anti_max"] > thr["eps_anti_max"]:
        failing.append("eps_anti")
    if rec.get("mass_balance") is None \
            or abs(rec["mass_balance"]) > thr["mass_balance_max"]:
        failing.append("mass_balance")
    if rec.get("raw_current_rel") is None \
            or rec["raw_current_rel"] > thr["raw_current_rel_max"]:
        failing.append("raw_current")
    return (not failing), failing


def parabolic_ratio(L, h, kappa1, kappa2):
    ks = abs(kappa1) + abs(kappa2)
    return L / math.sqrt(h / ks) if ks > 0 else None


# ------------------------------------------------------------------
# sweeps
# ------------------------------------------------------------------

def sweep_geom(n_samples, h_ref, jitter_max=0.3, L_factor=3.0,
               n_min=8):
    """Clean-family sweep: curvatures, gap/h, sampling phase, jitter.
    Ranges bracket the two-ellipses contact layer (kappa ~ 1/a..1/b,
    g/h in (0.3, 1.5), L ~ few L_W). jitter_max=0.05 is the
    physically matched family (arc-length sampling is uniform to
    0.1%); n_min rejects under-sampled windows (a window with fewer
    than n_min particles per sheet is not an h-scale window)."""
    gen = torch.Generator().manual_seed(20260822)
    out = []
    tries = 0
    while len(out) < n_samples and tries < 20 * n_samples:
        tries += 1
        i = tries
        k1 = float(torch.empty(1).uniform_(0.4, 3.0,
                                           generator=gen))
        k2 = float(torch.empty(1).uniform_(0.4, 3.0,
                                           generator=gen))
        gph = float(torch.empty(1).uniform_(0.3, 1.5,
                                            generator=gen))
        ph = float(torch.empty(1).uniform_(0.0, 1.0, generator=gen))
        jit = float(torch.empty(1).uniform_(0.0, jitter_max,
                                            generator=gen))
        L = L_factor * math.sqrt(h_ref / (k1 + k2))
        if int(L / h_ref) < n_min:
            continue
        A, B = facing_pair(k1, k2, gph * h_ref, h_ref, L,
                           phase2=ph, jitter=jit, seed=i)
        rec, pair = measure_geom(A, B, h_ref)
        rec.update(measure_raw(A, B, pair, h_ref))
        rec.update(kappa1=k1, kappa2=k2, gap_over_h=gph, phase2=ph,
                   jitter=jit, L_over_LW=parabolic_ratio(
                       L, h_ref, k1, k2))
        out.append(rec)
    return out


def negatives(h_ref):
    """The mandated negative fixtures; each must FAIL its
    certificate (recorded, not thresholded here)."""
    out = {}
    # order reversal
    A, B = facing_pair(1.0, 1.0, 0.8 * h_ref, h_ref,
                       3.0 * math.sqrt(h_ref / 2.0), reverse_b=True)
    rec, _ = measure_geom(A, B, h_ref)
    out["order_reversal"] = dict(rec,
                                 expect="order_preserving == False")
    # three-arc: pair A against the union of B and the third sheet --
    # mutual-NN spans two targets -> coverage/order break
    A, B = facing_pair(1.0, 1.0, 1.2 * h_ref, h_ref,
                       3.0 * math.sqrt(h_ref / 2.0))
    pc, nc, mc = third_arc(1.2 * h_ref, h_ref,
                           3.0 * math.sqrt(h_ref / 2.0))
    Bu = (torch.cat([B[0], pc]), torch.cat([B[1], nc]),
          torch.cat([B[2], mc]))
    rec3, _ = measure_geom(A, Bu, h_ref)
    out["three_arc"] = dict(
        rec3, expect="pairing captured by the interposed sheet: "
        "gap_med ~ gap/2, eps_anti inflated vs the clean pair")
    # self-contact stand-in: anti-parallel same-loop sheets => the
    # L1 certificate must REQUIRE distinct loop labels upstream; here
    # we record that geometry alone cannot distinguish it
    out["self_contact_note"] = (
        "geometrically identical to a clean pair -- the certificate "
        "must consume certified loop labels (L0J certify_loop_orders "
        "fail-closed), not geometry")
    return out


def measure_states(states_path, steps, gamma=1.5):
    """Real-trajectory corpus: extract the h-scale contact window
    (cross-loop NN gap <= gamma h) from checkpoints of a production
    run and measure the SAME certificate quantities. Sheet order is
    taken from the stored particle order (= certified curve order);
    the facing arcs of two CCW loops traverse anti-parallel, so
    sheet B is presented flipped, exactly as in the analytic family."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import two_ellipses_benchmark as teb
    from src.torch.oriented_varifold.mass import (
        compute_recommended_params,
    )
    ck = torch.load(states_path, weights_only=True)
    v0, m1, _, _ = teb.build_cloud(0.0)
    delta, tau = map(float,
                     compute_recommended_params(v0.positions))
    out = []
    for step in steps:
        if step not in ck:
            continue
        st = ck[step]
        pos, ang = st["positions"], st["angles"]
        nor = torch.stack([ang.cos(), ang.sin()], 1)
        m = teb.resolve_m(pos, nor, delta, tau)
        pa_all, pb_all = pos[m1], pos[~m1]
        D = torch.cdist(pa_all, pb_all)
        h_ab = max(float(m[m1].median()), float(m[~m1].median()))
        wa = D.min(dim=1).values <= gamma * h_ab
        wb = D.min(dim=0).values <= gamma * h_ab
        if int(wa.sum()) < 3 or int(wb.sum()) < 3:
            out.append(dict(step=step, window=None))
            continue
        # canonicalize the CYCLIC window: a facing arc may span the
        # array seam (the ellipse sampling starts inside it) -- roll
        # so each sheet's window is contiguous in certified order
        # (this is the single legal wrap; >1 runs = not a window)
        def cyclic_indices(w):
            idx = w.nonzero().flatten()
            n = w.shape[0]
            starts = [int(i) for i in idx
                      if not bool(w[(int(i) - 1) % n])]
            if len(starts) != 1:
                return None            # split window -> reject
            s0 = starts[0]
            return torch.tensor(
                [(s0 + k) % n for k in range(int(w.sum()))],
                dtype=torch.long)
        ia_ = cyclic_indices(wa)
        ib_ = cyclic_indices(wb)
        if ia_ is None or ib_ is None:
            out.append(dict(step=step, window="split"))
            continue
        # stored CCW particle order IS the certified order, and the
        # facing arcs of two CCW loops already traverse anti-parallel
        # -- present both sheets as stored (no flip; flipping B here
        # would double-flip against the analytic convention)
        A = (pa_all[ia_], nor[m1][ia_], m[m1][ia_])
        B = (pb_all[ib_], nor[~m1][ib_], m[~m1][ib_])
        rec, pair = measure_geom(A, B, h_ab)
        rec.update(measure_raw(A, B, pair, h_ab))
        rec.update(step=step, h_ab=h_ab,
                   g_min=float(D.min()),
                   g_over_h=float(D.min()) / h_ab)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["geom", "negatives", "all"],
                    default="all")
    ap.add_argument("--states", default=None,
                    help="ALSO measure the real-trajectory corpus "
                         "from this states .pt (two-ellipses runs)")
    ap.add_argument("--states-steps", type=int, nargs="*",
                    default=None)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--h-ref", type=float,
                    default=2.0 * math.pi / 256.0)
    ap.add_argument("--jitter-max", type=float, default=0.3)
    ap.add_argument("--mass-balance-floor", type=float, default=0.05,
                    help="mass-balance envelope floor: the analytic "
                         "family has exactly equal sheet masses, so "
                         "the floor comes from the L0-0 sampling pin "
                         "h_max/h_min < 1.05 (pre-registered)")
    ap.add_argument("--write-thresholds", action="store_true",
                    help="write l1_thresholds.json = family envelope "
                         "(max/min over the clean family; "
                         "PRE-REGISTRATION source, never refit)")
    ap.add_argument("--tag", default="_l1cal")
    args = ap.parse_args()
    torch.set_default_dtype(DT)
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(meta=dict(
        h_ref=args.h_ref, n_samples=args.n_samples,
        status="DESIGN SKELETON -- distributions only, no "
               "thresholds fixed; calibration source for L1 "
               "pre-registration"))
    if args.family in ("geom", "all"):
        rows = sweep_geom(args.n_samples, args.h_ref,
                          jitter_max=args.jitter_max)
        res["geom"] = rows
        cov = sorted(r["coverage_mutual"] for r in rows)
        covc = sorted(r["coverage_core"] for r in rows)
        anti = sorted(r["eps_anti_max"] for r in rows)
        cur = sorted(r["raw_current_rel"] for r in rows)
        res["geom_summary"] = dict(
            coverage_min=cov[0], coverage_p05=cov[len(cov) // 20],
            coverage_core_min=covc[0],
            coverage_core_p05=covc[len(covc) // 20],
            eps_anti_max_p95=anti[-max(1, len(anti) // 20)],
            eps_anti_max_max=anti[-1],
            raw_current_rel_p95=cur[-max(1, len(cur) // 20)],
            raw_current_rel_max=cur[-1],
            order_preserving_all=all(r["order_preserving"]
                                     for r in rows),
            mass_balance_absmax=max(abs(r["mass_balance"])
                                    for r in rows))
        if args.write_thresholds:
            thr = dict(
                eps_anti_max=res["geom_summary"]["eps_anti_max_max"],
                mass_balance_max=max(
                    res["geom_summary"]["mass_balance_absmax"],
                    args.mass_balance_floor),
                raw_current_rel_max=res["geom_summary"]["raw_current_rel_max"],
                provenance=dict(family="analytic facing-parabola",
                                n_samples=len(rows),
                                jitter_max=args.jitter_max,
                                h_ref=args.h_ref,
                                rule="clean-family envelope (max), "
                                     "fixed before any L1 shadow; "
                                     "L-A3 states are validation only",
                                mass_balance_floor_source=
                                "L0-0 spacing pin h_max/h_min < 1.05"))
            (OUT / "l1_thresholds.json").write_text(
                json.dumps(thr, indent=1))
            res["thresholds"] = thr
    if args.family in ("negatives", "all"):
        res["negatives"] = negatives(args.h_ref)
    if args.states:
        ck = torch.load(args.states, weights_only=True)
        steps = (args.states_steps if args.states_steps
                 else sorted(k for k in ck if isinstance(k, int)))
        traj = measure_states(args.states, steps)
        thr_path = OUT / "l1_thresholds.json"
        if thr_path.exists():
            thr = json.loads(thr_path.read_text())
            for r in traj:
                if r.get("order_preserving") is not None:
                    ok, failing = l1_certificate(r, thr)
                    r["certified"], r["failing"] = ok, failing
        res["trajectory"] = dict(source=args.states, rows=traj)
    out = OUT / f"l1_calibration{args.tag}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    if "geom_summary" in res:
        print(json.dumps(res["geom_summary"], indent=1))
    if "negatives" in res:
        print("negatives:", {k: (v if isinstance(v, str) else
                                 {kk: v[kk] for kk in
                                  ("order_preserving", "coverage",
                                   "eps_anti_max", "gap_med")
                                  if kk in v})
                             for k, v in res["negatives"].items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
