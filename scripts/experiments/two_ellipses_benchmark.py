"""Phase L0: two-ellipses local-contact reconnaissance on the FINAL
production stack (reviewer 2026-08-20 GO; observation only, no
mechanism changes).

Model convention (one-phase interior Mullins-Sekerka / Hele-Shaw):
    Delta u = 0 in each connected component of E(t),
    u = kappa on the boundary,   V = -d_nu u.
Exact structure on smooth intervals: EACH component conserves its area
AND its barycenter. Initial data: two ellipses a=0.4, b=1.0 at centers
(-+0.45, 0) (gap 0.1). |E_j| = pi a b = 0.4 pi, bar(E_j) = (-+0.45, 0);
after a (future) quotient |E| = 0.8 pi, bar(E) = 0. R_eq = sqrt(2ab)
~ 0.894427 is the stationarity CHECK target, not a claimed attractor.

The central L0 question (reviewer): does the loopwise q^WB see LOCAL
contact correctly? Predicted failure mode r_l ~ 1 - L_c/L (uniform
attenuation instead of local annihilation). Mandatory telemetry:
window/complement decomposition of q_full/q_self/q_WB and P
contributions, eta_loc = (1 - qbar_WB^W)/(1 - qbar_full^W),
eta_spill = 1 - qbar_WB^{W^c}.

Purpose wording: "local-contact entry and State-II continuation" --
NOT "contact-layer passage" (that needs L1/L2 machinery).

Pre-registered stop set (first hit):
  (1) any existing fail-closed guard (raises),
  (2) raw winding certificate failure (raises),
  (3) after b_finer onset: g_min <= h_loc (median window mass),
  (4) b_finer onset + K steps (K = 100),
  (5) --steps cap.
Coarse sanity gates (NOT calibrated envelopes; labeled as such):
per-component |dA|/A > 1e-3 or |d bar| > 1e-2, total the same.

Usage:
    uv run python scripts/experiments/two_ellipses_benchmark.py \
        --grid 768 --steps 2000 [--dt 1e-5] [--rotate-deg 22.5] --tag _l0_768
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import build_config  # noqa: E402
from quotient_switch_stability import production_args, wrap  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.shapes.generator import generate_oriented_ellipse  # noqa: E402

DT = torch.float64
A_EL, B_EL, GAP = 0.4, 1.0, 0.1
CX = (2 * A_EL + GAP) / 2.0            # 0.45
H_REF = 2.0 * math.pi / 256.0          # concentric-benchmark spacing
SIGMA = 0.1
K_POST_BFINER = 100
OUT = Path("results/two_ellipses")
R_EQ = math.sqrt(2.0 * A_EL * B_EL)


def ellipse_perimeter(a, b, n=200000):
    t = torch.linspace(0, 2 * math.pi, n + 1, dtype=DT)
    return float(torch.hypot(a * torch.sin(t), b * torch.cos(t))
                 .mul(2 * math.pi / n)[:-1].sum())


def signed_area_order(pos):
    """Shoelace in the GIVEN (curve) order -- exact regardless of
    star-shapedness (particle order is preserved by the flow)."""
    x, y = pos[:, 0], pos[:, 1]
    x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
    return float(0.5 * (x * y2 - x2 * y).sum())


def centroid_order(pos):
    x, y = pos[:, 0], pos[:, 1]
    x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
    cr = x * y2 - x2 * y
    a = 0.5 * cr.sum()
    cx = ((x + x2) * cr).sum() / (6.0 * a)
    cy = ((y + y2) * cr).sum() / (6.0 * a)
    return float(cx), float(cy)


def curvature_at(pos, i):
    """3-point circumcircle curvature at cyclic index i."""
    n = pos.shape[0]
    p0, p1, p2 = pos[(i - 1) % n], pos[i], pos[(i + 1) % n]
    a = (p1 - p0).norm()
    b = (p2 - p1).norm()
    c = (p2 - p0).norm()
    cross = (p1 - p0)[0] * (p2 - p0)[1] - (p1 - p0)[1] * (p2 - p0)[0]
    den = a * b * c
    return float(2.0 * cross / den) if den > 0 else 0.0


def cyclic_runs(mask):
    """Number of connected runs of True in a cyclic boolean mask."""
    m = mask.tolist()
    n = len(m)
    if not any(m):
        return 0
    if all(m):
        return 1
    runs = 0
    for i in range(n):
        if m[i] and not m[i - 1]:
            runs += 1
    return runs


def build_cloud(rotate_deg=0.0):
    L_ell = ellipse_perimeter(A_EL, B_EL)
    n_el = round(L_ell / H_REF)
    e1 = generate_oriented_ellipse(n_el, A_EL, B_EL, (-CX, 0.0), "cpu",
                                   DT, initial_sampling="arc_length")
    e2 = generate_oriented_ellipse(n_el, A_EL, B_EL, (CX, 0.0), "cpu",
                                   DT, initial_sampling="arc_length")
    pos = torch.cat([e1.positions, e2.positions])
    ang = torch.cat([e1.angles, e2.angles])
    if rotate_deg != 0.0:
        al = math.radians(rotate_deg)
        R = torch.tensor([[math.cos(al), -math.sin(al)],
                          [math.sin(al), math.cos(al)]], dtype=DT)
        pos = pos @ R.T
        ang = ang + al
    m1 = torch.zeros(pos.shape[0], dtype=torch.bool)
    m1[:n_el] = True
    return (OrientedPointCloudVarifold(positions=pos, angles=ang),
            m1, n_el, L_ell)


# L-A gate floors (reviewer 2026-08-22 hardening 6): PRE-FIXED from the
# M2 W0 arm's one-step series (two_ellipses_l0j2_768wb.json): max
# per-step |dA| = 5.99e-8, max per-step |dbar| = 2.39e-8; floors = 3x
# (eps_M = 3 (maxdbar A + |xbar| maxdA)). Never fitted on L-A runs.
EPS_A_FLOOR = 1.796e-7
EPS_M_FLOOR = 1.709e-7
EPS_ALG = 1e-12

# L-A amendment (reviewer 2026-08-22 verdict): activation ELIGIBILITY
# eq (1) adds a state-functional geometric margin on top of
# mature_unique -- "looked mature once" and "the hard contact complex
# is persistently definable" are different things. Threshold fixed
# BEFORE L-A2/3 from pre-L-A data (never fitted on L-A runs): the
# L0J-2b three-configuration clean trajectories (768 axis / 1024 axis
# / 768 rot pi/8) are per-step satellite-free for g_min/sigma <=
# 0.555; onset-band flapping was observed for g/sigma in [0.67, 0.91].
# Not claimed universal: a robustness margin for the current Wendland
# support, fixed sigma, two-ellipses class and sampling.
ACT_GAP_OVER_SIGMA = 0.55
ACT_GAP_PROVENANCE = ("pre-LA L0J-2b three-configuration clean "
                      "trajectories")


def activation_gap_ratio(pos, labels, probe_res):
    """g_min / sigma_X of eligibility eq (1): the cross-loop minimum
    gap between the CERTIFIED candidate loop pair (the two loops of
    the unique mature complex), over the source-frozen perimeter
    scale sigma_X actually used to build the complex (probe_res
    carries it; never a live scale or mass bandwidth)."""
    sides = probe_res["cx"]["complexes"][0].sides
    la, lb = sides[0].loop, sides[1].loop
    g = float(torch.cdist(pos[labels == la], pos[labels == lb]).min())
    return g / float(probe_res["sigma"])


def polygon_moments_order(pos):
    x, y = pos[:, 0], pos[:, 1]
    x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
    cr = x * y2 - x2 * y
    return (float(((x + x2) * cr).sum() / 6.0),
            float(((y + y2) * cr).sum() / 6.0))


def fingerprint_of(st, v):
    """L-A hardening 3: everything both arms must share on the SOURCE
    (only the frozen perimeter q^P may differ)."""
    sn = st._last_grid_snapshot
    return dict(
        masses=st.fixed_masses.clone(),
        labels=st._loop_labels_last.clone(),
        q_full=st.fixed_coherence.clone(),
        delta=float(st.mass_delta), tau=float(st.mass_tau),
        sigma=float(st._sigma),
        AB=st.param.AB_solve.clone(),
        rho=(st.grid_wasserstein.phase.rho_metric.clone()
             if st.grid_wasserstein is not None else None),
        compat=st.metric_object.compat_labels.clone()
        if st.metric_object.compat_labels is not None else None,
        masked=(st.metric_object.masked.clone()
                if getattr(st.metric_object, "masked", None) is not None
                else None),
        rows=st._rows_used_last.clone(),
        constraint_sv=list(sn.constraint_row_singular_values),
        r_loop=st.perimeter_r_per_particle.clone(),
        target=float(st._grid_target_volume_initial),
    )


def fingerprints_match(fa, fb):
    bad = []
    for k in fa:
        a, b = fa[k], fb[k]
        if a is None and b is None:
            continue
        if torch.is_tensor(a):
            if not torch.equal(a, b):
                bad.append(k)
        elif isinstance(a, list):
            if any(abs(x - y) > 0 for x, y in zip(a, b)):
                bad.append(k)
        elif a != b:
            bad.append(k)
    # r_loop is DERIVED from q^P branch inputs but must agree because
    # it is the loopwise-global ratio, identical under both modes
    return bad


def run_activation_transaction(pos, ang, cfg_factory, delta, tau,
                               target0, step_n):
    """L-A: same-source WB+rows vs CC+rows committed one-step
    (reviewer hardening 3-6, 9). Returns (passed_or_transient, record,
    committed_cc). passed_or_transient in {'pass','transient','fail'}.
    cfg_factory(q_mode) must return the RUN's config with only the
    perimeter q branch varied (fingerprint parity is asserted, not
    assumed)."""
    from src.torch.perimeter.contact_complex import (
        probe_contact_complex,
    )
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.mm_step import MMStepper
    from src.torch.perimeter.coherence_perimeter import (
        compute_coherence_loopwise,
    )
    from src.torch.transport.bem_wasserstein import compute_coherence

    rec = dict(step=int(step_n), type="wbcc_activation",
               eps_A=EPS_A_FLOOR, eps_M=EPS_M_FLOOR,
               activation_gap_threshold=ACT_GAP_OVER_SIGMA,
               activation_threshold_provenance=ACT_GAP_PROVENANCE)
    gates = {}
    rec["gates"] = gates
    n = pos.shape[0] // 2
    lbl = torch.zeros(pos.shape[0], dtype=torch.long)
    lbl[n:] = 1
    arms = {}
    steppers = {}
    committed = {}
    v_src = OrientedPointCloudVarifold(positions=pos.clone(),
                                       angles=ang.clone())
    try:
        for arm, mode in (("WB", "self_renormalized"),
                          ("CC", "contact_complex_renormalized")):
            cfg = cfg_factory(mode)
            assert cfg.grid_bulk_first_moment_rows, \
                "activation arms require first-moment rows ON"
            solver = MMSolver(cfg)
            st = MMStepper(cfg)
            st._grid_target_volume_initial = target0
            res, com = solver._advance_committed(st, v_src)
            steppers[arm] = st
            committed[arm] = com
            arms[arm] = dict(
                n_iter=int(res.n_iter), converged=bool(res.converged),
                objective=float(res.objective),
                objective_initial=float(res.objective_initial),
                wasserstein=float(res.wasserstein),
                P_sharp_committed=st.sharp_perimeter(com),
                P_sharp_src=st.sharp_perimeter(v_src))
    except Exception as ex:                     # noqa: BLE001
        rec["exception"] = f"{type(ex).__name__}: {str(ex)[:160]}"
        return "fail", rec, None
    # ---- fingerprint parity (3.1) ------------------------------
    fp = {a: fingerprint_of(steppers[a], v_src) for a in ("WB", "CC")}
    bad = fingerprints_match(fp["WB"], fp["CC"])
    gates["fingerprint_parity"] = not bad
    rec["fingerprint_mismatch"] = bad
    # ---- neutrality decomposition (5.1-5.3) --------------------
    m = steppers["WB"].fixed_masses
    q_full = steppers["WB"].fixed_coherence
    q_wb = steppers["WB"].perimeter_coherence
    q_cc = steppers["CC"].perimeter_coherence
    nor = v_src.normals
    q_self, _ = compute_coherence_loopwise(
        pos, nor, m, SIGMA, "wendland_c2", lbl)
    neu = {}
    for li in (0, 1):
        sel = lbl == li
        neu[f"loop{li}_wb"] = float((m[sel] * (q_wb - q_full)[sel])
                                    .sum())
    probe = probe_contact_complex(pos, nor, m, lbl, SIGMA,
                                  "wendland_c2", q_full, q_self)
    rec["source_probe"] = probe["status"]
    sides_src = (probe["res"]["cx"]["complexes"][0].sides
                 if probe["status"] == "mature_unique" else [])
    rec["source_sides"] = [(s_.loop, s_.n) for s_ in sides_src]
    if probe["status"] == "mature_unique":
        rec["activation_gap_over_sigma"] = activation_gap_ratio(
            pos, lbl, probe["res"])
    for side in sides_src:
        sel = side.mask
        neu[f"side{side.loop}_cc"] = float(
            (m[sel] * (q_cc - q_full)[sel]).sum())
    far = torch.ones_like(lbl, dtype=torch.bool)
    for side in sides_src:
        far &= ~side.mask
    neu["far_cc_max"] = float((q_cc - q_full)[far].abs().max())
    neu["total"] = float((m * (q_cc - q_wb)).sum())
    rec["neutrality"] = neu
    gates["neutrality"] = (
        all(abs(v) < EPS_ALG * 10 for k, v in neu.items()
            if k != "far_cc_max") and neu["far_cc_max"] == 0.0)
    # ---- conservation gates (6) --------------------------------
    def one_step_deltas(com):
        d = {}
        for li, sl in ((0, slice(0, n)), (1, slice(n, None))):
            A0 = signed_area_order(pos[sl])
            A1 = signed_area_order(com.positions[sl])
            M0 = polygon_moments_order(pos[sl])
            M1 = polygon_moments_order(com.positions[sl])
            d[f"dA{li}"] = A1 - A0
            d[f"dMx{li}"] = M1[0] - M0[0]
            d[f"dMy{li}"] = M1[1] - M0[1]
            b0, b1 = centroid_order(pos[sl]), \
                centroid_order(com.positions[sl])
            d[f"dbar{li}"] = (b1[0] - b0[0], b1[1] - b0[1])
            # barycenter consistency (6.1)
            pred = ((M1[0] - M0[0]) - b0[0] * (A1 - A0)) / A1
            d[f"bar_consistency{li}"] = abs(pred - (b1[0] - b0[0]))
        return d
    dw = one_step_deltas(committed["WB"])
    dc = one_step_deltas(committed["CC"])
    rec["deltas"] = dict(WB=dw, CC=dc)
    ok = True
    for li in (0, 1):
        ok &= abs(dc[f"dA{li}"]) <= max(abs(dw[f"dA{li}"]),
                                        EPS_A_FLOOR)
        for k in ("dMx", "dMy"):
            ok &= abs(dc[f"{k}{li}"]) <= max(abs(dw[f"{k}{li}"]),
                                             EPS_M_FLOOR)
    gates["conservation"] = bool(ok)
    gates["bar_consistency"] = all(
        d[f"bar_consistency{li}"] < 1e-10
        for d in (dw, dc) for li in (0, 1))
    # ---- split residual (drop-style) ---------------------------
    def split(a):
        return arms[a]["P_sharp_committed"] + arms[a]["wasserstein"] \
            - arms[a]["P_sharp_src"]
    rec["split"] = dict(WB=split("WB"), CC=split("CC"))
    gates["split_residual"] = max(split("CC"), 0.0) <= max(
        max(split("WB"), 0.0), 1e-9)
    # ---- both-arm post-step certificate (4.1) ------------------
    post = {}
    for a in ("WB", "CC"):
        com = committed[a]
        nor1 = com.normals
        m1 = resolve_m(com.positions, nor1, delta, tau)
        qf1 = compute_coherence(_VV(com.positions, nor1), m1, SIGMA,
                                "wendland_c2")
        qs1, _ = compute_coherence_loopwise(
            com.positions, nor1, m1, SIGMA, "wendland_c2", lbl)
        post[a] = probe_contact_complex(com.positions, nor1, m1, lbl,
                                        SIGMA, "wendland_c2", qf1,
                                        qs1)
    rec["post_probe"] = {a: post[a]["status"] for a in post}
    if post["WB"]["status"] != "mature_unique":
        rec["transient"] = True
        return "transient", rec, None
    def side_ns(pr):
        return sorted((s.loop, s.n) for s in
                      pr["res"]["cx"]["complexes"][0].sides)
    src_ns = sorted((s.loop, s.n) for s in sides_src)
    gates["post_both_mature_same_pair"] = (
        post["CC"]["status"] == "mature_unique"
        and [x[0] for x in side_ns(post["CC"])]
        == [x[0] for x in src_ns]
        and all(b[1] >= a[1] for a, b in zip(src_ns,
                                             side_ns(post["CC"])))
        and all(b[1] >= a[1] for a, b in zip(src_ns,
                                             side_ns(post["WB"]))))
    # ---- optimizer gates (9) -----------------------------------
    gates["objective_decrease"] = all(
        arms[a]["objective"] <= arms[a]["objective_initial"] + 1e-12
        for a in arms)
    gates["iterations_under_cap"] = all(
        arms[a]["n_iter"] < 300 for a in arms)
    rec["arms"] = arms
    rec["dX_arms_over_h"] = float(
        (committed["WB"].positions - committed["CC"].positions)
        .abs().max()) / float(m.median())
    rec["dtheta_arms"] = float(
        wrap(committed["WB"].angles - committed["CC"].angles)
        .abs().max())
    passed = all(gates.values())
    rec["passed"] = passed
    if not passed:
        rec["reason"] = [k for k, v in gates.items() if not v]
        return "fail", rec, None
    return "pass", rec, committed["CC"]


class _VV:
    def __init__(self, pos, nor):
        self.positions, self.normals = pos, nor


def resolve_m(pos, nor, delta, tau):
    from src.torch.oriented_varifold.loopwise_mass import (
        resolve_loopwise_oriented_mass_source,
    )
    return resolve_loopwise_oriented_mass_source(
        pos, nor, delta, tau).m_loop


def geometry_pins(v, m1, n_el, L_ell, delta, tau, rotate_deg):
    """L0-0: initial sampling / geometry pins (printed AND asserted at
    O(h^2)-scale tolerances; h ~ 0.0245)."""
    pos = v.positions
    p1, p2 = pos[m1], pos[~m1]
    g_min = float(torch.cdist(p1, p2).min())
    A1, A2 = signed_area_order(p1), signed_area_order(p2)
    b1, b2 = centroid_order(p1), centroid_order(p2)
    seg1 = (torch.roll(p1, -1, 0) - p1).norm(dim=1)
    cv = float(seg1.max() / seg1.min())
    m = resolve_m(pos, v.normals, delta, tau)
    h_med = float(m.median())
    al = math.radians(rotate_deg)
    bar_t = [(-CX * math.cos(al), -CX * math.sin(al)),
             (CX * math.cos(al), CX * math.sin(al))]
    A_t = math.pi * A_EL * B_EL
    rep = dict(n_el=n_el, N=int(pos.shape[0]), L_ell=L_ell,
               spacing=L_ell / n_el, h_ref=H_REF,
               mass_median=h_med, g_min=g_min,
               A=(A1, A2), A_target=A_t, bar=(b1, b2),
               bar_target=bar_t, spacing_max_over_min=cv)
    print("L0-0 pins:", json.dumps(rep, default=str))
    tol = 3e-3          # ~ 5 h^2
    assert abs(g_min - GAP) < 3e-3, f"gap pin: {g_min}"
    for A in (A1, A2):
        assert abs(A - A_t) / A_t < tol, f"area pin: {A} vs {A_t}"
    for b, bt in zip((b1, b2), bar_t):
        assert math.hypot(b[0] - bt[0], b[1] - bt[1]) < 2e-3, \
            f"barycenter pin: {b} vs {bt}"
    assert cv < 1.05, f"arc-length spacing h_max/h_min pin: {cv}"
    assert abs(h_med - H_REF) / H_REF < 0.1, \
        f"0K mass median vs h_ref: {h_med} vs {H_REF}"
    return rep


def window_telemetry(pos, nor, m, m1, q_full, q_self, q_wb, gamma):
    """Contact window W = per-particle cross-loop NN gap <= gamma h_ab
    (per loop, cyclic-connectivity recorded), plus the reviewer's
    q decomposition and parabolic scales."""
    p1, p2 = pos[m1], pos[~m1]
    D = torch.cdist(p1, p2)
    g1, j1 = D.min(dim=1)
    g2, i2 = D.min(dim=0)
    h_ab = max(float(m[m1].median()), float(m[~m1].median()))
    g_min = float(g1.min())
    out = dict(g_min=g_min, h_ab=h_ab, g_min_over_h=g_min / h_ab)
    w1 = g1 <= gamma * h_ab
    w2 = g2 <= gamma * h_ab
    out["n_W"] = (int(w1.sum()), int(w2.sum()))
    out["runs_W"] = (cyclic_runs(w1), cyclic_runs(w2))
    Wm = torch.zeros(pos.shape[0], dtype=torch.bool)
    Wm[m1.nonzero().flatten()[w1]] = True
    Wm[(~m1).nonzero().flatten()[w2]] = True
    out["_W_mask"] = Wm            # stripped before JSON
    i_min = int(g1.argmin())
    j_min = int(j1[i_min])
    k1 = curvature_at(p1, i_min)
    k2 = curvature_at(p2, j_min)
    out["kappa_sum"] = abs(k1) + abs(k2)
    if int(w1.sum()) >= 2 and int(w2.sum()) >= 2:
        idx = torch.zeros(pos.shape[0], dtype=torch.bool)
        idx[m1.nonzero().flatten()[w1]] = True
        idx[(~m1).nonzero().flatten()[w2]] = True
        W, Wc = idx, ~idx
        L_W = float(m[W].sum()) / 2.0        # per-sheet arc length
        out["L_W"] = L_W
        out["L_W_over_h"] = L_W / h_ab
        out["parab"] = out["kappa_sum"] * L_W ** 2 / h_ab

        def mbar(q, sel):
            return float((m[sel] * q[sel]).sum() / m[sel].sum())
        out["qbar_full_W"] = mbar(q_full, W)
        out["qbar_self_W"] = mbar(q_self, W)
        out["qbar_wb_W"] = mbar(q_wb, W)
        out["qbar_full_Wc"] = mbar(q_full, Wc)
        out["qbar_wb_Wc"] = mbar(q_wb, Wc)
        out["P_full_W"] = float((m[W] * q_full[W]).sum())
        out["P_wb_W"] = float((m[W] * q_wb[W]).sum())
        out["P_full_Wc"] = float((m[Wc] * q_full[Wc]).sum())
        out["P_wb_Wc"] = float((m[Wc] * q_wb[Wc]).sum())
        den = 1.0 - out["qbar_full_W"]
        out["eta_loc"] = ((1.0 - out["qbar_wb_W"]) / den
                          if den > 1e-12 else None)
        out["eta_spill"] = 1.0 - out["qbar_wb_Wc"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=768)
    ap.add_argument("--metric", default="grid", choices=("grid", "bie"),
                    help="Wasserstein metric backend (grid Poisson or "
                         "block-diagonal boundary integral)")
    ap.add_argument("--bridge-gap", type=float, default=1.0,
                    help="BIE only: merge gap in units of ell")
    ap.add_argument("--cut-gap", type=float, default=0.08,
                    help="arc-splice cut window (default 2*0.04)")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--rotate-deg", type=float, default=0.0)
    ap.add_argument("--gamma-window", type=float, default=1.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pins-only", action="store_true")
    # L0J-2b (reviewer 2026-08-21)
    ap.add_argument("--q-mode", default="self_renormalized",
                    choices=["full", "self_renormalized",
                             "contact_complex_renormalized",
                             "contact_complex_tb"])
    ap.add_argument("--resume", type=int, default=None,
                    help="resume from this step of --states-from")
    ap.add_argument("--states-from", default=None)
    ap.add_argument("--stop-on-window", action="store_true",
                    help="stop 10 steps after the h-scale window "
                         "(gamma=1.5, both sides n>=2) first forms")
    ap.add_argument("--stop-at-gap", type=float, default=None,
                    help="stop when g_min <= this value (matched-depth"
                         " controls)")
    ap.add_argument("--activation", default="off",
                    choices=["off", "wbcc"],
                    help="L-A: WB->CC certified energy-branch "
                         "activation (requires --first-moment-rows)")
    ap.add_argument("--auto-quotient", action="store_true",
                    help="one-stroke run: at the pre-registered depth "
                         "g <= 0.0336 (L-A3 matched depth) with the "
                         "L1 window certificate PASSING, run the "
                         "keep/drop shadow in-process and commit the "
                         "2 bulk -> 1 bulk quotient on PASS")
    ap.add_argument("--auto-splice", action="store_true",
                    help="one-stroke run: at local-contact entry "
                         "(g <= h_loc) with the quotient active, run "
                         "the certified arc splice in-process and "
                         "continue as a single loop to --steps")
    ap.add_argument("--commit-quotient", default=None,
                    help="L1-1: at --resume, commit the 2 bulk -> 1 "
                         "bulk quotient certified by this shadow JSON "
                         "(irreversible; State III rows [G; C_M1+C_M2])")
    ap.add_argument("--stop-after-activation", type=int, default=None,
                    help="L-A2 smoke: stop this many steps after "
                         "t_act (telemetry-complete short run)")
    ap.add_argument("--moment-rows-form", default="current_centered",
                    choices=["polygon", "current", "current_centered"],
                    help="first-moment row family. PRODUCTION preset "
                         "= current_centered (order-free flux family, "
                         "merged-barycenter centering after quotient "
                         "aggregation; reviewer 2026-08-25). polygon "
                         "= historical bitwise baseline (library "
                         "MMConfig default stays polygon).")
    ap.add_argument("--first-moment-rows", action="store_true",
                    help="L0J-M: exact polygon first-moment rows in "
                         "the admissible basis")
    args = ap.parse_args()
    torch.set_default_dtype(DT)
    OUT.mkdir(parents=True, exist_ok=True)

    from src.torch.perimeter.coherence_perimeter import (
        compute_coherence_loopwise,
    )
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.transport.bem_wasserstein import compute_coherence

    v0, m1, n_el, L_ell = build_cloud(args.rotate_deg)
    delta, tau = map(float, compute_recommended_params(v0.positions))
    step_offset = 0
    if args.resume is not None:
        src = args.states_from if args.states_from else args.tag
        ck = torch.load(OUT / f"two_ellipses{src}_states.pt",
                        weights_only=True)
        st_r = ck[args.resume]
        v0 = OrientedPointCloudVarifold(
            positions=st_r["positions"].clone(),
            angles=st_r["angles"].clone())
        step_offset = args.resume + 1
        pins = {"resumed_from": src, "step": args.resume}
        print(f"resuming from step {args.resume} (states of "
              f"{src!r}, N={v0.n_points})", flush=True)
    else:
        pins = geometry_pins(v0, m1, n_el, L_ell, delta, tau,
                             args.rotate_deg)
    if args.pins_only:
        return
    q_mode_now = args.q_mode
    resumed_activation = None
    if args.activation == "wbcc":
        assert args.first_moment_rows, \
            "--activation wbcc requires --first-moment-rows (the rows " \
            "are a PDE invariant, on in BOTH phases)"
        assert args.q_mode == "self_renormalized", \
            "activation starts from the WB branch"
    if args.resume is not None:
        resumed_activation = st_r.get("activation_state")
        resumed_quotient = st_r.get("quotient_state")
        if args.commit_quotient is not None:
            assert resumed_quotient is None, \
                "checkpoint already carries a quotient (irreversible)"
            assert resumed_activation, \
                "L1 quotient requires the CC (activated) branch"
            sh = json.loads(Path(args.commit_quotient).read_text())
            assert sh.get("passed") and sh.get("step") == args.resume \
                and sh.get("source_tag") == src, \
                "shadow JSON must be a PASS for exactly this source"
            resumed_quotient = {
                "quotient_bulk_groups": [torch.ones(v0.n_points,
                                                    dtype=torch.bool)],
                "quotient_at": int(args.resume),
                "certificate_at_switch": sh["source_certificate"],
                "raw_bulk_pair_at_switch": (0, 1),
                "particle_count": int(v0.n_points),
                "shadow_record": {k: sh[k] for k in
                                  ("gates", "measures", "verdict")},
            }
            print(f"L1-1: committing 2 bulk -> 1 bulk quotient at step "
                  f"{args.resume} (shadow {args.commit_quotient} PASS)",
                  flush=True)
        if resumed_activation:
            # hardening 8: a resumed post-activation run starts in CC
            # and NEVER re-activates
            q_mode_now = "contact_complex_renormalized"
            print(f"imported activation_state (activated_at "
                  f"{resumed_activation.get('activated_at')}): "
                  "resuming in CC, re-activation disabled", flush=True)

    def make_cfg(q_mode):
        c = build_config(production_args(grid=args.grid,
                                         redist_monotone=True,
                                         quotient_mode="off",
                                         q_mode=q_mode,
                                         metric=args.metric,
                                         bridge_gap=args.bridge_gap),
                         delta, tau)
        c.time_step = args.dt
        c.grid_bulk_first_moment_rows = args.first_moment_rows
        c.grid_bulk_first_moment_rows_form = args.moment_rows_form
        return c

    cfg = make_cfg(q_mode_now)
    print(f"L0: one-phase interior MS, N={v0.n_points} "
          f"(2 x {n_el} arc-length), H={args.grid}, dt={args.dt:g}, "
          f"rot={args.rotate_deg} deg, q_mode={args.q_mode}, "
          f"R_eq check {R_EQ:.6f}", flush=True)
    labels = (~m1).long()          # static: order/count never change
    # conservation baseline = the DISCRETE t=0 areas (the analytic
    # pi*a*b differs by the static O(h^2) polygon offset ~2.5e-4,
    # which is not drift); the offset itself is pinned in L0-0
    A0 = (signed_area_order(v0.positions[m1]),
          signed_area_order(v0.positions[~m1]))
    al = math.radians(args.rotate_deg)
    bar0 = [(-CX * math.cos(al), -CX * math.sin(al)),
            (CX * math.cos(al), CX * math.sin(al))]
    series, states = [], {}
    state = {"bfiner_at": None, "stop_reason": None, "t0": time.time()}
    _off = {"v": step_offset}
    _mode = {"q": q_mode_now}
    if args.resume is not None and resumed_quotient:
        state["_quotient_state"] = resumed_quotient
        state["quotient_at"] = resumed_quotient.get("quotient_at")
        # merged-invariant baseline at the quotient (telemetry)
        state["_quot_A0"] = (signed_area_order(v0.positions[m1])
                             + signed_area_order(v0.positions[~m1]))
        _Ma = polygon_moments_order(v0.positions[m1])
        _Mb = polygon_moments_order(v0.positions[~m1])
        state["_quot_M0"] = (_Ma[0] + _Mb[0], _Ma[1] + _Mb[1])
        print(f"quotient state active (quotient_at "
              f"{state['quotient_at']}): State III rows", flush=True)
    if resumed_activation:
        state["activated_at"] = resumed_activation.get("activated_at")
        state["_activation_state"] = resumed_activation
        # target continuity across resume (hardening 8): the stored
        # t=0-derived target, not one re-derived from the resumed cloud
        if resumed_activation.get("target_volume_initial") is not None:
            state["_target0"] = float(
                resumed_activation["target_volume_initial"])
    out_json = OUT / f"two_ellipses{args.tag}.json"
    states_path = OUT / f"two_ellipses{args.tag}_states.pt"

    def dump(err=None):
        tmp = out_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(
            meta=dict(a=A_EL, b=B_EL, gap=GAP, n_el=n_el,
                      N=v0.n_points, grid=args.grid, dt=args.dt,
                      metric=args.metric, bridge_gap=args.bridge_gap,
                      cut_gap=args.cut_gap,
                      rotate_deg=args.rotate_deg,
                      gamma_window=args.gamma_window, pins=pins,
                      model="one-phase interior MS: Delta u=0 per "
                            "component, u=kappa, V=-d_nu u",
                      r_eq_check=R_EQ, error=err,
                      stop_reason=state.get("stop_reason"),
                      bfiner_at=state.get("bfiner_at"),
                      activated_at=state.get("activated_at"),
                      quotient_at=state.get("quotient_at"),
                      spliced_at=state.get("spliced_at"),
                      l1_first_fail=state.get("l1_first_fail"),
                      act_transients=state.get("act_transients"),
                      act_events=state.get("act_events")),
            series=series), indent=1, default=str))
        tmp.replace(out_json)
        torch.save(states, states_path)

    # 0J-E production refresh envelope (max+ over the audited window,
    # results/reports/phase3c0j_report.md) -- the EXISTING bound the
    # L0J-2b positive-refresh hard stop reads (no new threshold)
    REFRESH_ENVELOPE = 0.0323

    def cb(step, res):
        step = step + _off["v"]
        vv = (res.committed_varifold
              if res.committed_varifold is not None else res.varifold)
        pos, nor = vv.positions, vv.normals
        def _ckpt():
            # every checkpoint (periodic, event, stop) carries the
            # activation/quotient state so ANY saved step can be
            # resumed with the correct branch (the 1826 stop state of
            # L-A3 lacked activation_state -- fixed here)
            c = {"positions": pos.clone(), "angles": vv.angles.clone()}
            if state.get("_activation_state") is not None:
                c["activation_state"] = state["_activation_state"]
            if state.get("_quotient_state") is not None:
                c["quotient_state"] = state["_quotient_state"]
            return c
        if step % 25 == 0:
            states[step] = _ckpt()
        m = resolve_m(pos, nor, delta, tau)
        q_full = compute_coherence(vv, m, SIGMA, cfg.perimeter_kernel)
        q_self, cross_max = compute_coherence_loopwise(
            pos, nor, m, SIGMA, cfg.perimeter_kernel, labels)
        r = []
        q_wb = torch.empty_like(q_full)
        for sel in (m1, ~m1):
            r_l = float((m[sel] * q_full[sel]).sum()
                        / (m[sel] * q_self[sel]).sum())
            r.append(r_l)
            q_wb[sel] = r_l * q_self[sel]
        p1, p2 = pos[m1], pos[~m1]
        A = (signed_area_order(p1), signed_area_order(p2))
        bar = (centroid_order(p1), centroid_order(p2))
        win = window_telemetry(pos, nor, m, m1, q_full, q_self, q_wb,
                               args.gamma_window)
        W_mask = win.pop("_W_mask", None)
        # ---- L-A activation probe (WB phase; pure observation) ----
        act_stop = None
        act_probe = None
        act_gap_ratio = None
        if (args.activation == "wbcc"
                and state.get("activated_at") is None):
            from src.torch.perimeter.contact_complex import (
                probe_contact_complex,
            )
            act_probe = probe_contact_complex(
                pos, nor, m, labels, SIGMA, cfg.perimeter_kernel,
                q_full, q_self)
            rec_probe = act_probe["status"]
            if rec_probe in ("ambiguous", "geometry_error",
                             "out_of_scope"):
                act_stop = (f"activation probe fatal ({rec_probe}): "
                            f"{act_probe['reason']}")
            elif rec_probe == "mature_unique":
                # eligibility eq (1) (reviewer 2026-08-22 verdict):
                # mature_unique AND both |I| >= 3 (probe-enforced)
                # AND g_min/sigma_X <= 0.55 -- state-functional,
                # history-free; outside the margin the state is
                # recorded but NOT a candidate (onset-band flapping)
                act_gap_ratio = activation_gap_ratio(
                    pos, labels, act_probe["res"])
                if act_gap_ratio <= ACT_GAP_OVER_SIGMA:
                    # segment break: the controller runs the
                    # transaction on THIS committed state (zero-time
                    # event at step n)
                    state["_act_candidate"] = step
                    state["_act_source"] = (pos.clone(),
                                            vv.angles.clone())
        else:
            rec_probe = None
        # ---- L0J-2b: q^CC telemetry + hard stops -----------------
        cc_stop = None
        rec_cc = {}
        if _mode["q"] in ("contact_complex_renormalized",
                          "contact_complex_tb"):
            # telemetry q is the PLAIN q^CC (under TB the production q
            # differs on the sides; the moment/eta telemetry stays
            # comparable across arms)
            from src.torch.perimeter.contact_complex import (
                SigmaContactComplexError,
                certify_loop_orders,
                contact_complex_q,
            )
            try:
                res_cc = contact_complex_q(pos, nor, m, labels, SIGMA,
                                           cfg.perimeter_kernel,
                                           q_full, q_self)
            except SigmaContactComplexError as e:
                res_cc = None
                cc_stop = (f"complex certificate failure: "
                           f"{str(e)[:140]}")
            if res_cc is not None:
                cxx, q_cc = res_cc["cx"], res_cc["q_cc"]
                orders = certify_loop_orders(pos, labels)
                sides_tel = []
                for comp in cxx["complexes"]:
                    for side in comp.sides:
                        ro = orders[side.loop]["rank_of"][
                            side.particles]
                        sides_tel.append(dict(
                            loop=side.loop, n=side.n,
                            rank_lo=int(ro.min()),
                            rank_hi=int(ro.max())))
                rec_cc = dict(n_complexes=cxx["n_complexes"],
                              cc_sides=res_cc["sides"],
                              cc_side_ranks=sides_tel)
                prev_q = state.get("prev_qcc")
                if prev_q is not None:
                    rec_cc["refresh_pos"] = max(
                        float((m * (q_cc - prev_q)).sum()), 0.0)
                state["prev_qcc"] = q_cc
                # birth-energy certificate (reviewer 2026-08-21 sec 2:
                # the source-template energy change of NEWLY admitted
                # side particles; the general refresh non-positivity is
                # NOT a theorem -- this quantity is what the future
                # activation certificate will gate)
                cur_members = set()
                for comp in cxx["complexes"]:
                    for side in comp.sides:
                        cur_members |= set(side.particles.tolist())
                prev_members = state.get("prev_cc_members")
                if prev_members is not None:
                    born = sorted(cur_members - prev_members)
                    if born:
                        bi = torch.tensor(born, dtype=torch.long)
                        rec_cc["delta_birth"] = float(
                            (m[bi] * (q_full[bi] - q_self[bi])).sum())
                        rec_cc["n_born"] = len(born)
                state["prev_cc_members"] = cur_members
                d_full = q_self - q_full
                d_cc = q_self - q_cc
                D12 = torch.cdist(pos[m1], pos[~m1])
                gpp = torch.empty(pos.shape[0], dtype=pos.dtype)
                gpp[m1.nonzero().flatten()] = D12.min(dim=1).values
                gpp[(~m1).nonzero().flatten()] = D12.min(dim=0).values
                mom = {}
                for comp in cxx["complexes"]:
                    for side in comp.sides:
                        sel = side.particles
                        od = orders[side.loop]
                        i_c = sel[int(gpp[sel].argmin())]
                        s_all, L = od["arclen"], od["length"]
                        ds = s_all[od["rank_of"][sel]] \
                            - s_all[od["rank_of"][i_c]]
                        ds = torch.where(ds > L / 2, ds - L, ds)
                        ds = torch.where(ds < -L / 2, ds + L, ds)
                        row = {}
                        for nm, d in (("full", d_full), ("cc", d_cc)):
                            for k in (0, 1, 2):
                                row[f"M{k}_{nm}"] = float(
                                    (m[sel] * ds ** k * d[sel]).sum())
                        mom[f"loop{side.loop}"] = row
                rec_cc["moments"] = mom
                if W_mask is not None:
                    Isig = cxx["complex_id"] >= 0
                    def _eta(selm):
                        den = float((m[selm] * d_full[selm]).sum())
                        num = float((m[selm] * d_cc[selm]).sum())
                        return dict(num=num, den=den, n=int(selm.sum()),
                                    eta=(num / den if den > 1e-12
                                         else None))
                    rec_cc["eta_h"] = dict(
                        core=_eta(W_mask & Isig),
                        collar=_eta(Isig & ~W_mask),
                        far=_eta(~Isig))
                # ---- L-A standing certificate (hardening 7) -------
                if state.get("activated_at") is not None:
                    cur_sets = {s.loop: set(s.particles.tolist())
                                for comp in cxx["complexes"]
                                for s in comp.sides}
                    prev_sets = state.get("_act_side_sets")
                    if prev_sets is not None and cc_stop is None:
                        if set(cur_sets) != set(prev_sets):
                            cc_stop = ("standing cert: loop pair "
                                       f"changed {sorted(prev_sets)} "
                                       f"-> {sorted(cur_sets)}")
                        else:
                            deaths = {lb: len(prev_sets[lb]
                                              - cur_sets[lb])
                                      for lb in prev_sets}
                            births = {lb: len(cur_sets[lb]
                                              - prev_sets[lb])
                                      for lb in prev_sets}
                            rec_cc["side_deaths"] = deaths
                            rec_cc["side_births"] = births
                            if any(deaths.values()):
                                cc_stop = ("standing cert: side "
                                           f"death {deaths} "
                                           "(unexplained, fail)")
                            db_ = rec_cc.get("delta_birth")
                            if cc_stop is None and db_ is not None \
                                    and db_ > EPS_ALG * 10:
                                cc_stop = ("standing cert: "
                                           f"Delta_birth {db_:.2e} "
                                           "> eps_alg")
                    state["_act_side_sets"] = cur_sets
                # hard stops (L0J-2b)
                if cxx["n_complexes"] != 1:
                    cc_stop = (f"complex split/merge: n = "
                               f"{cxx['n_complexes']}")
                else:
                    ns = tuple(s["n"] for s in res_cc["sides"])
                    prev_ns = state.get("prev_side_ns")
                    if prev_ns is not None and (
                            ns[0] < prev_ns[0] or ns[1] < prev_ns[1]):
                        cc_stop = (f"side interval shrink: {prev_ns} "
                                   f"-> {ns}")
                    state["prev_side_ns"] = ns
                    rp = rec_cc.get("refresh_pos")
                    if cc_stop is None and rp is not None \
                            and rp > REFRESH_ENVELOPE:
                        cc_stop = (f"positive refresh {rp:.3e} > 0J "
                                   f"envelope {REFRESH_ENVELOPE}")
        # ---- L1: merged invariants + standing window certificate ---
        l1_tel = {}
        if state.get("_quotient_state") is not None:
            _Ma = polygon_moments_order(p1)
            _Mb = polygon_moments_order(p2)
            l1_tel["A_merged"] = A[0] + A[1]
            l1_tel["M_merged"] = (_Ma[0] + _Mb[0], _Ma[1] + _Mb[1])
            l1_tel["dA_merged_rel"] = (l1_tel["A_merged"]
                                       - state["_quot_A0"]) \
                / state["_quot_A0"]
            l1_tel["dM_merged"] = (
                l1_tel["M_merged"][0] - state["_quot_M0"][0],
                l1_tel["M_merged"][1] - state["_quot_M0"][1])
            from l1_quotient_shadow import window_certificate
            _thr = state.get("_l1_thr")
            if _thr is None:
                _thr = json.loads(Path(
                    "results/two_ellipses/l1_calibration/"
                    "l1_thresholds.json").read_text())
                state["_l1_thr"] = _thr
            _rec, _ok, _fail = window_certificate(
                pos, vv.angles, m1, delta, tau, _thr)
            l1_tel["l1_cert"] = bool(_ok)
            l1_tel["l1_failing"] = _fail
            if _rec is not None:
                l1_tel["l1_metrics"] = {k: _rec[k] for k in (
                    "eps_anti_max", "raw_current_rel", "mass_balance",
                    "order_preserving", "coverage_core", "n_W",
                    "g_over_h") if k in _rec}
            if not _ok and state.get("l1_first_fail") is None:
                state["l1_first_fail"] = step
                print(f"== EVENT L1 window certificate first failure "
                      f"at step {step}: {_fail} (telemetry; run "
                      "continues under the existing guards) ==",
                      flush=True)
        sn = res.grid_setup_snapshot
        rel = sn.partition_relation_grid_bulk if sn else None
        rec = dict(step=step, t=(step + 1) * args.dt,
                   P_frozen=float(res.perimeter),
                   W=float(res.wasserstein), n_iter=int(res.n_iter),
                   A=A, bar=bar, r_l=r, cross_max=cross_max,
                   dA_rel=[(A[j] - A0[j]) / A0[j] for j in (0, 1)],
                   dbar=[math.hypot(bar[j][0] - bar0[j][0],
                                    bar[j][1] - bar0[j][1])
                         for j in (0, 1)],
                   act_probe=rec_probe,
                   act_gap_over_sigma=act_gap_ratio,
                   **l1_tel,
                   partition_relation=rel,
                   n_components=(sn.n_components if sn else None),
                   r_stack=(getattr(sn, "r_stack", None) if sn
                            else None),
                   max_s=float(res.displacements.abs().max()),
                   **{k: v for k, v in win.items()
                      if not k.startswith("_")},
                   **rec_cc)
        series.append(rec)
        if step % 25 == 0:
            el = time.time() - state["t0"]
            print(f"step {step:4d}: g_min {win['g_min']:.4f} "
                  f"(/h {win['g_min_over_h']:.2f}) dA "
                  f"{max(abs(x) for x in rec['dA_rel']):.2e} dbar "
                  f"{max(rec['dbar']):.2e} r_l "
                  f"[{r[0]:.4f} {r[1]:.4f}] rel {rel} "
                  f"({el / (step + 1):.1f}s/step)", flush=True)
            if win.get("eta_loc") is not None:
                print(f"    W: n {win['n_W']} runs {win['runs_W']} "
                      f"L_W/h {win.get('L_W_over_h'):.2f} eta_loc "
                      f"{win['eta_loc']:.4f} eta_spill "
                      f"{win['eta_spill']:.4f}", flush=True)
            dump()
        # ---- pre-registered stop set --------------------------------
        if state["bfiner_at"] is None and rel == "b_finer":
            state["bfiner_at"] = step
            states[step] = _ckpt()
            print(f"== EVENT b_finer onset at step {step} ==",
                  flush=True)
        stop = None
        if cc_stop is not None:
            stop = cc_stop
        elif act_stop is not None:
            stop = act_stop
        elif state.get("_splice_candidate") == step:
            dump()
            return True
        elif state.get("_act_candidate") == step:
            # segment break for the activation transaction -- NOT a
            # failure stop (stop_reason stays unset)
            dump()
            return True
        elif (args.stop_after_activation is not None
              and state.get("activated_at") is not None
              and step >= state["activated_at"]
              + args.stop_after_activation):
            stop = (f"t_act {state['activated_at']} + "
                    f"{args.stop_after_activation} steps "
                    "(L-A2 smoke window complete)")
        elif (args.auto_quotient
              and state.get("activated_at") is not None
              and state.get("_quotient_state") is None
              and state.get("_quot_candidate") is None
              and ((win["g_min"] <= 0.0336) if args.metric == "grid"
                   else bool(sn is not None and sn.bie_merged_pairs))):
            state["_quot_candidate"] = step
            state["_quot_source"] = (pos.clone(), vv.angles.clone())
            dump()
            return True
        elif (args.stop_at_gap is not None
              and win["g_min"] <= args.stop_at_gap):
            stop = (f"matched-depth stop: g_min {win['g_min']:.5f} <= "
                    f"{args.stop_at_gap}")
        elif args.stop_on_window:
            formed = (win.get("n_W", (0, 0))[0] >= 2
                      and win["n_W"][1] >= 2)
            if formed and state.get("window_at") is None:
                state["window_at"] = step
                states[step] = _ckpt()
                print(f"== EVENT h-window formed at step {step} ==",
                      flush=True)
            if state.get("window_at") is not None \
                    and step >= state["window_at"] + 10:
                stop = (f"h-window formed at {state['window_at']} + 10"
                        " steps (L0J-2b measurement window complete)")
        if stop:
            pass
        elif max(abs(x) for x in rec["dA_rel"]) > 1e-3:
            stop = (f"component area drift {rec['dA_rel']} > 1e-3 "
                    "(coarse sanity bound, not a calibrated envelope)")
        elif max(rec["dbar"]) > 1e-2:
            stop = (f"component barycenter drift {rec['dbar']} > 1e-2 "
                    "(coarse sanity bound)")
        elif state["bfiner_at"] is not None:
            h_loc = win["h_ab"]
            if win.get("n_W", (0, 0))[0] >= 2:
                sel = torch.cdist(p1, p2).min(dim=1).values \
                    <= args.gamma_window * win["h_ab"]
                h_loc = float(m[m1][sel].median())
            if args.metric == "bie":
                # BIE: the cancelling pair is masked and held fixed at
                # the merge, so local contact is entered when the gap
                # has stopped decreasing for 3 consecutive steps with
                # the quotient committed (same rule as the concentric
                # removal event)
                gp = state.get("_gap_prev")
                flat = gp is not None and win["g_min"] >= gp * (1 - 1e-9)
                state["_n_gap_flat"] = (state.get("_n_gap_flat", 0) + 1
                                        if flat else 0)
                state["_gap_prev"] = win["g_min"]
                entry = (state.get("_quotient_state") is not None
                         and bool(sn is not None and sn.bie_n_masked)
                         and state["_n_gap_flat"] >= 3)
            else:
                entry = win["g_min"] <= h_loc
            if entry:
                if (args.auto_splice
                        and state.get("_quotient_state") is not None
                        and state.get("_splice_candidate") is None):
                    # segment break NOW (the quotient branch pattern):
                    # setting the candidate without returning loses
                    # the step-matched break in the earlier stop
                    # chain (one-step skew -> the plain stop fired;
                    # canonical attempt 2)
                    state["_splice_candidate"] = step
                    state["_splice_source"] = (pos.clone(),
                                               vv.angles.clone())
                    dump()
                    return True
                else:
                    stop = (f"local-contact entry: g_min "
                            f"{win['g_min']:.5f} <= h_loc "
                            f"{h_loc:.5f}")
            elif (not args.stop_on_window
                  and args.stop_at_gap is None
                  and not args.auto_quotient and not args.auto_splice
                  and step >= state["bfiner_at"] + K_POST_BFINER):
                # L0 reconnaissance stop; DISABLED for the L0J-2b
                # window runs, whose purpose is to pass this point and
                # reach h-window formation (+10 measurement steps)
                stop = (f"b_finer + {K_POST_BFINER} steps "
                        "(State-II continuation window complete)")
        if stop:
            state["stop_reason"] = stop
            states[step] = _ckpt()
            print(f"== L0 stop at step {step}: {stop} ==", flush=True)
            dump()
            return True
        return False

    err = None

    def _do_auto_quotient(step_n):
        """One-stroke controller: L1 window certificate (hard) +
        in-process keep/drop committed one-step at the candidate
        state; on PASS commit the irreversible 2 bulk -> 1 bulk
        quotient (state passed to every later segment). Returns
        (err, ok)."""
        from l1_quotient_shadow import window_certificate
        from src.torch.solver.mm_step import MMStepper
        if args.metric == "bie":
            from src.torch.solver.quotient_gates_bie import (
                QUOTIENT_GATES_BIE as QUOTIENT_GATES,
            )
        else:
            from src.torch.solver.quotient_gates import QUOTIENT_GATES
        posQ, angQ = state["_quot_source"]
        thr = json.loads(Path(
            "results/two_ellipses/l1_calibration/l1_thresholds.json"
        ).read_text())
        w_rec, w_ok, w_fail = window_certificate(posQ, angQ, m1,
                                                 delta, tau, thr)
        if not w_ok:
            return (f"auto-quotient: L1 certificate failed at "
                    f"{step_n}: {w_fail}"), False
        cert = dict(kind="L1-0 window certificate", metrics=w_rec,
                    thresholds=thr, certified=True, failing=[],
                    source_step=int(step_n), auto=True)
        v_srcQ = OrientedPointCloudVarifold(
            positions=posQ.clone(), angles=angQ.clone())
        arms = {}
        for arm in ("keep", "drop"):
            cfgQ = make_cfg("contact_complex_renormalized")
            svQ = MMSolver(cfgQ)
            stQ = MMStepper(cfgQ)
            stQ._grid_target_volume_initial = state["_target0"]
            if arm == "drop":
                stQ.commit_quotient(
                    torch.ones(posQ.shape[0], dtype=torch.bool),
                    int(step_n), cert, (0, 1))
            try:
                resQ, comQ = svQ._advance_committed(stQ, v_srcQ)
            except Exception as ex:              # noqa: BLE001
                return (f"auto-quotient {arm} arm raised at "
                        f"{step_n}: {type(ex).__name__}: "
                        f"{str(ex)[:160]}"), False
            pq = comQ.positions
            arms[arm] = dict(
                pos=pq, ang=comQ.angles,
                obj0=float(resQ.objective_initial),
                obj=float(resQ.objective),
                n_iter=int(resQ.n_iter),
                W=float(resQ.wasserstein),
                Pc=stQ.sharp_perimeter(comQ),
                Ps=stQ.sharp_perimeter(v_srcQ),
                A=sum(signed_area_order(pq[labels == lb])
                      for lb in (0, 1)),
                M=[sum(t) for t in zip(
                    *[polygon_moments_order(pq[labels == lb])
                      for lb in (0, 1)])])
        k, d_ = arms["keep"], arms["drop"]
        h_w = float(resolve_m(posQ, torch.stack(
            [angQ.cos(), angQ.sin()], 1), delta, tau).median())
        A0q = sum(signed_area_order(posQ[labels == lb])
                  for lb in (0, 1))
        M0q = [sum(t) for t in zip(
            *[polygon_moments_order(posQ[labels == lb])
              for lb in (0, 1)])]
        g = dict(
            split=max(d_["Pc"] + d_["W"] - d_["Ps"], 0.0)
            <= QUOTIENT_GATES["eps_split"],
            switch_x=float((d_["pos"] - k["pos"]).abs().max()) / h_w
            <= QUOTIENT_GATES["eps_switch_x"],
            switch_theta=float(wrap(d_["ang"] - k["ang"]).abs().max())
            <= QUOTIENT_GATES["eps_switch_theta"],
            merged_area=abs(d_["A"] - k["A"]) / A0q
            <= QUOTIENT_GATES["eps_volume"],
            merged_moment=all(
                abs(d_["M"][i] - M0q[i])
                <= max(abs(k["M"][i] - M0q[i]), EPS_M_FLOOR)
                for i in (0, 1)),
            objective=all(a["obj"] <= a["obj0"] + 1e-12
                          for a in arms.values()),
            iters=all(a["n_iter"] < 300 for a in arms.values()))
        rec_q = dict(step=int(step_n), type="auto_quotient", gates=g,
                     passed=all(g.values()),
                     gate_values=dict(
                         split=max(d_["Pc"] + d_["W"] - d_["Ps"], 0.0),
                         switch_x=float((d_["pos"] - k["pos"]).abs().max())
                         / h_w,
                         switch_theta=float(
                             wrap(d_["ang"] - k["ang"]).abs().max()),
                         merged_area=abs(d_["A"] - k["A"]) / A0q),
                     gate_thresholds=dict(QUOTIENT_GATES),
                     certificate_metrics={kk: w_rec[kk] for kk in
                                          ("g_over_h", "n_W",
                                           "eps_anti_max",
                                           "raw_current_rel")})
        state.setdefault("act_events", []).append(rec_q)
        if not all(g.values()):
            print(f"== auto-quotient shadow FAIL at {step_n}: "
                  f"{[kk for kk, vv in g.items() if not vv]} ==",
                  flush=True)
            return None, False
        state["_quotient_state"] = {
            "quotient_bulk_groups": [torch.ones(posQ.shape[0],
                                                dtype=torch.bool)],
            "quotient_at": int(step_n),
            "certificate_at_switch": cert,
            "raw_bulk_pair_at_switch": (0, 1),
            "particle_count": int(posQ.shape[0]),
            "shadow_record": rec_q,
        }
        state["quotient_at"] = int(step_n)
        state["_quot_A0"] = A0q
        state["_quot_M0"] = tuple(M0q)
        c_ = {"positions": posQ.clone(), "angles": angQ.clone(),
              "quotient_state": state["_quotient_state"]}
        if state.get("_activation_state") is not None:
            c_["activation_state"] = state["_activation_state"]
        states[int(step_n)] = c_
        print(f"== EVENT t_quot: bulks 2 -> 1 COMMITTED at step "
              f"{step_n} (auto; all gates) ==", flush=True)
        dump()
        return None, True

    def _do_auto_splice(step_n):
        """One-stroke controller: certified arc splice at the
        committed state step_n (zero-time; S0 event gates hard,
        fail-closed) followed by the single-loop continuation up to
        --steps, all in THIS process/series."""
        from l1_quotient_shadow import window_certificate
        from src.torch.solver.arc_splice import (
            ArcSpliceError,
            _has_self_intersection,
            _polygon_A_M,
            arc_splice,
        )
        posS, angS = state.pop("_splice_source")
        thr = json.loads(Path(
            "results/two_ellipses/l1_calibration/l1_thresholds.json"
        ).read_text())
        norS = torch.stack([angS.cos(), angS.sin()], 1)
        mS = resolve_m(posS, norS, delta, tau)
        # L1 certificate on the calibrated 1.5h window (hard)
        w_rec, w_ok, w_fail = window_certificate(posS, angS, m1,
                                                 delta, tau, thr)
        # cut window: 2 eps_fill = 0.08 on the grid backend (the
        # grid-bridging-safe scale); the same width (3.2 ell) is kept
        # on the BIE backend so the reconnection geometry is unchanged
        CUT_GAP = float(args.cut_gap)
        D = torch.cdist(posS[m1], posS[~m1])

        def cyc(w, base):
            idx = w.nonzero().flatten()
            nn = w.shape[0]
            st_ = [int(i) for i in idx
                   if not bool(w[(int(i) - 1) % nn])]
            if len(st_) != 1:
                return None
            return base + torch.tensor(
                [(st_[0] + k) % nn for k in range(int(w.sum()))],
                dtype=torch.long)
        w1 = cyc(D.min(dim=1).values <= CUT_GAP, 0)
        w2 = cyc(D.min(dim=0).values <= CUT_GAP, int(m1.sum()))
        if not w_ok or w1 is None or w2 is None:
            return (f"auto-splice: window not certified at {step_n} "
                    f"({w_fail})")
        h_ab = max(float(mS[m1].median()), float(mS[~m1].median()))
        try:
            sp = arc_splice(posS, angS, labels, w1, w2, h_ab)
        except ArcSpliceError as e:
            return f"auto-splice ArcSpliceError at {step_n}: {e}"
        if not sp["certified"]:
            return (f"auto-splice S0 certificates failed at {step_n}:"
                    f" {sp['certificates']}")
        Xs, ANs = sp.pop("positions"), sp.pop("angles")
        sp.pop("provenance")
        state["spliced_at"] = int(step_n)
        state["_splice_record"] = {k: v for k, v in sp.items()}
        states[step_n] = _last = dict(
            positions=posS.clone(), angles=angS.clone())
        states[f"spliced_{step_n}"] = dict(
            positions=Xs.clone(), angles=ANs.clone(),
            splice_record=state["_splice_record"])
        print(f"== EVENT t_splice: loops 2 -> 1 at step {step_n} "
              f"(N {posS.shape[0]} -> {Xs.shape[0]}, corr residual "
              f"{sp['corr_residual']}) ==", flush=True)
        # ---- single-loop continuation ---------------------------
        cfg1 = make_cfg("self_renormalized")
        sv = MMSolver(cfg1)
        if state.get("_target0") is not None:
            sv._pending_target_volume = state["_target0"]
        A0s = float(_polygon_A_M(Xs)[0])
        off = step_n + 1

        def cb1(k, res):
            vv = (res.committed_varifold
                  if res.committed_varifold is not None
                  else res.varifold)
            pp = vv.positions
            gstep = off + k
            Aq, Mxq, Myq = (float(x) for x in _polygon_A_M(pp))
            c = (Mxq / Aq, Myq / Aq)
            r = (pp - torch.tensor(c)).norm(dim=1)
            L = float((pp.roll(-1, 0) - pp).norm(dim=1).sum())
            rec = dict(step=gstep, t=(gstep + 1) * args.dt,
                       phase="single_loop", A=Aq,
                       dA_rel=(Aq - A0s) / A0s, bar=c, L=L,
                       isoperimetric=4 * math.pi * Aq / L ** 2,
                       r_mean=float(r.mean()), r_std=float(r.std()),
                       r_mean_over_req=float(r.mean()) / R_EQ,
                       P_frozen=float(res.perimeter),
                       W=float(res.wasserstein),
                       n_iter=int(res.n_iter),
                       max_s=float(res.displacements.abs().max()))
            series.append(rec)
            if gstep % 25 == 0:
                states[gstep] = dict(positions=pp.clone(),
                                     angles=vv.angles.clone())
                print(f"step {gstep:4d} [1-loop]: dA "
                      f"{rec['dA_rel']:+.2e} isoper "
                      f"{rec['isoperimetric']:.5f} r/R_eq "
                      f"{rec['r_mean_over_req']:.5f}", flush=True)
                dump()
            stop1 = None
            if abs(rec["dA_rel"]) > 5e-3:
                stop1 = f"1-loop area sanity {rec['dA_rel']:.2e}"
            elif math.hypot(*c) > 1e-2:
                stop1 = f"1-loop barycenter sanity {c}"
            elif k % 5 == 0 and _has_self_intersection(pp):
                stop1 = "1-loop self-intersection"
            if stop1:
                state["stop_reason"] = stop1
                states[gstep] = dict(positions=pp.clone(),
                                     angles=vv.angles.clone())
                print(f"== stop at {gstep}: {stop1} ==", flush=True)
                dump()
                return True
            return False

        n1 = args.steps - off
        if n1 <= 0:
            return None
        try:
            hist = sv.solve(OrientedPointCloudVarifold(
                positions=Xs.clone(), angles=ANs.clone()), n1,
                callback=cb1)
            if getattr(hist, "stop_message", None):
                return (f"{getattr(hist, 'stop_exception_type', '?')}"
                        f": {hist.stop_message}")
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return f"{type(exc).__name__}: {exc}"
        return None

    def _segment(v_start):
        seg_err = None
        sv = MMSolver(cfg)
        if state.get("_target0") is not None:
            sv._pending_target_volume = state["_target0"]
        if state.get("_quotient_state") is not None:
            sv._pending_quotient_state = state["_quotient_state"]
        try:
            hist = sv.solve(v_start, args.steps - _off["v"],
                            callback=cb)
            if getattr(hist, "stop_message", None):
                seg_err = (f"{getattr(hist, 'stop_exception_type', '?')}"
                           f": {hist.stop_message}")
        except BaseException as exc:  # noqa: BLE001 -- recorded
            seg_err = f"{type(exc).__name__}: {exc}"
            print(f"    [FAIL-CLOSED] {seg_err}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                dump(seg_err)
                raise
        return seg_err

    if args.activation != "wbcc":
        err = _segment(v0)
    else:
        # L-A segmented controller: WB probe -> transaction ->
        # zero-time CC commit (t_act = n, X^n unchanged, first CC
        # physical step = n+1)
        v_start = v0
        n_transient = 0
        # the run's frozen volume target, derived ONCE from the t=0
        # cloud (deterministic: identical to what segment 1's own
        # stepper would derive); every segment and both transaction
        # arms are seeded with it so target continuity is exact
        if state.get("_target0") is None:
            from src.torch.solver.mm_step import MMStepper
            st_probe = MMStepper(make_cfg(q_mode_now))
            st_probe._setup_step(OrientedPointCloudVarifold(
                positions=v_start.positions.clone(),
                angles=v_start.angles.clone()))
            state["_target0"] = float(
                st_probe._grid_target_volume_initial)
            del st_probe
        while True:
            err = _segment(v_start)
            qc = state.pop("_quot_candidate", None)
            if (err is None and qc is not None
                    and state.get("_quotient_state") is None
                    and state.get("stop_reason") is None):
                err, ok = _do_auto_quotient(qc)
                if err is None and ok:
                    posQ, angQ = state.pop("_quot_source")
                    v_start = OrientedPointCloudVarifold(
                        positions=posQ.clone(), angles=angQ.clone())
                    _off["v"] = qc + 1
                    continue
                if err is None:
                    err = f"auto-quotient shadow FAILED at {qc}"
                break
            spl = state.get("_splice_candidate")
            if (err is None and spl is not None
                    and state.get("spliced_at") is None
                    and state.get("stop_reason") is None):
                err = _do_auto_splice(spl)
                break
            cand = state.pop("_act_candidate", None)
            if (err is not None or cand is None
                    or state.get("stop_reason") is not None
                    or state.get("activated_at") is not None):
                break
            posA, angA = state.pop("_act_source")
            print(f"== L-A activation transaction at source step "
                  f"{cand} ==", flush=True)
            status, a_rec, com_cc = run_activation_transaction(
                posA, angA, make_cfg, delta, tau, state["_target0"],
                cand)
            state.setdefault("act_events", []).append(a_rec)
            if series and series[-1]["step"] == cand:
                series[-1]["activation_event"] = a_rec
            if status == "pass":
                state["activated_at"] = cand
                state["_activation_state"] = dict(
                    mode="contact_complex_renormalized",
                    activated_at=int(cand),
                    source_loop_pair=[x[0] for x in
                                      a_rec.get("source_sides", [])],
                    source_side_intervals=a_rec.get("source_sides"),
                    sigma_at_activation=SIGMA,
                    first_moment_rows=True,
                    target_volume_initial=state["_target0"],
                    activation_gap_over_sigma=a_rec.get(
                        "activation_gap_over_sigma"),
                    activation_gap_threshold=ACT_GAP_OVER_SIGMA,
                    activation_threshold_provenance=
                    ACT_GAP_PROVENANCE)
                states[cand] = {"positions": posA.clone(),
                                "angles": angA.clone(),
                                "activation_state":
                                    state["_activation_state"]}
                _mode["q"] = "contact_complex_renormalized"
                cfg = make_cfg("contact_complex_renormalized")
                print(f"== EVENT t_act: WB -> CC ACTIVATED at step "
                      f"{cand} (dX_arms/h "
                      f"{a_rec.get('dX_arms_over_h', 0):.2e}) ==",
                      flush=True)
                v_start = OrientedPointCloudVarifold(
                    positions=posA.clone(), angles=angA.clone())
                _off["v"] = cand + 1
                dump()
                continue
            if status == "transient":
                n_transient += 1
                state.setdefault("act_transients", []).append(cand)
                print(f"== activation transient at {cand} (keep arm "
                      "lost the complex) -- continuing WB ==",
                      flush=True)
                if n_transient > 20:
                    err = "activation transient more than 20 times"
                    break
                v_start = OrientedPointCloudVarifold(
                    positions=posA.clone(), angles=angA.clone())
                _off["v"] = cand + 1
                continue
            # fail
            (OUT / f"two_ellipses{args.tag}_activation_failure.json"
             ).write_text(json.dumps(a_rec, indent=1, default=str))
            err = (f"ActivationRejected at {cand}: "
                   f"{a_rec.get('reason', a_rec.get('exception'))}")
            print(f"    [FAIL-CLOSED] {err}", flush=True)
            break
    dump(err)
    print(f"completed {len(series)}/{args.steps} steps in "
          f"{(time.time() - state['t0']) / 3600.0:.2f} h; stop = "
          f"{state.get('stop_reason')}; err = {err}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
