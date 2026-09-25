"""0L-B2-0.5: pre-registration of the quotient-switch gates.

Sources (NEVER the switch run itself):
  (i)   clean State-II window: B0 cell B (768, 1569-1583; the
        `_b2cal_Bsave` re-run stores every committed state) --
        split residual R_n = objective_initial[n+1] + W_n -
        objective_initial[n] (objective_initial = P#(X^n) since W = 0
        at y = 0), merged geometric volume per-step relative change,
        and the natural step-to-step motion |X^{n+1}-X^n|_inf/h_wall,
        |dtheta|_inf (positive envelope for the switch difference)
  (ii)  B1 positive fixture (concentric disk + hole at residual gap
        h + outer, on the 512 grid, State II): keep/drop MM one-step
        difference (positive)
  (iii) B0 cell B vs D one-step at 768/1568 (uncertified switch,
        NEGATIVE reference for the switch difference)
Rule: upper-bound gates tau = sqrt(p_max * n_min) when the positive
max and the negative min separate, else the positive max (+ eps_num).
Writes src/torch/solver/quotient_gates.py (constants with provenance)
and results/reports/phase3c0l_b2_calibration.json.

Usage:
    uv run python scripts/experiments/quotient_switch_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import OUT  # noqa: E402

DT = torch.float64
EPS_NUM = 1e-9


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def window_stats(tag, lo, hi):
    d = json.load(open(OUT / f"exact_merger{tag}.json"))
    s = [r for r in d["series"] if lo <= r["step"] <= hi]
    R, dV, mmax = [], [], []
    for k in range(len(s) - 1):
        a, b = s[k], s[k + 1]
        if a.get("objective_initial") is None:
            continue
        R.append(b["objective_initial"] + a["wasserstein"]
                 - a["objective_initial"])
        va = a["vol_geom_disk"] + a["vol_geom_ann"]
        vb = b["vol_geom_disk"] + b["vol_geom_ann"]
        dV.append(abs(vb - va) / va)
    return dict(n=len(s), split_pos_max=max((max(x, 0.0) for x in R),
                                             default=None),
                split_list=R, dV_rel_max=max(dV, default=None),
                h_wall=s[-1]["h_wall"])


def state_motion(tag, lo, hi):
    ck = torch.load(OUT / f"exact_merger{tag}_states.pt",
                    weights_only=True)
    ks = sorted(k for k in ck if lo <= k <= hi)
    dx, dth = [], []
    for a, b in zip(ks[:-1], ks[1:]):
        pa, pb = ck[a]["positions"], ck[b]["positions"]
        dx.append(float((pb - pa).abs().max()))
        dth.append(float(wrap(ck[b]["angles"] - ck[a]["angles"])
                         .abs().max()))
    return ks, dx, dth


def one_step_diff(tag_a, tag_b, step):
    ca = torch.load(OUT / f"exact_merger{tag_a}_states.pt",
                    weights_only=True)[step]
    cb = torch.load(OUT / f"exact_merger{tag_b}_states.pt",
                    weights_only=True)[step]
    return (float((ca["positions"] - cb["positions"]).abs().max()),
            float(wrap(ca["angles"] - cb["angles"]).abs().max()))


def fixture_shadow_switch():
    """B1 positive fixture on the grid: keep vs drop MM one-step."""
    import tilt_row_ablation as tra
    from exact_merger_benchmark import build_cloud
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.oriented_varifold.loopwise_mass import (
        resolve_loopwise_oriented_mass_source,
    )
    from src.torch.oriented_varifold.mass import compute_recommended_params
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    tra.DELTA, tra.TAU = delta, tau
    m0 = resolve_loopwise_oriented_mass_source(
        v0.positions, v0.normals, delta, tau).m_loop
    target0 = float(0.5 * (m0 * (v0.positions * v0.normals).sum(-1)).sum())
    R = 0.35
    h = 2 * math.pi * R / 90

    def circ(Rr, n, inward=False):
        t = torch.arange(n, dtype=DT) * 2 * math.pi / n
        pos = torch.stack([Rr * t.cos(), Rr * t.sin()], 1)
        return pos, t + (math.pi if inward else 0.0)
    pd, ad = circ(R, 90)
    ph, ah = circ(R + h, 141, inward=True)
    po, ao = circ(0.93, 256)
    v = OrientedPointCloudVarifold(positions=torch.cat([pd, ph, po]),
                                   angles=torch.cat([ad, ah, ao]))
    st = tra.production_stepper("current", "current", "augment", target0)
    st.config.grid_contact_rows_mode = "aligned_prequotient"
    st.config.angle_constraint_scope = "loopwise"
    st.config.angle_constraint_measure = "raw_loopwise"
    st.config.redistribution_density_scope = "loopwise"
    st.config.redistribution_curvature_scope = "loopwise"
    st.config.redistribution_q_policy = "r_loop"
    out = {}
    try:
        r_keep = st.step(v)
        certs = st._contact_certificates_source
        cert = certs[0].as_dict() if isinstance(certs, list) and certs \
            else certs
        ctx = st._contact_pair_context
        rel = st._last_grid_snapshot.partition_relation_grid_bulk
        out.update(relation=rel, certificate=cert)
        if rel != "b_finer" or not certs or not certs[0].certified:
            out["note"] = "fixture not in certified State II"
            return out
        ba, bb = certs[0].candidate.bulk_pair
        labels = st._loop_labels_last
        bulk_pp = torch.tensor([ctx["loop_to_raw_bulk"][int(l)]
                                for l in labels.tolist()])
        mask = (bulk_pp == ba) | (bulk_pp == bb)
        st2 = tra.production_stepper("current", "current", "augment",
                                     target0)
        st2.config.grid_contact_rows_mode = "aligned_prequotient"
        st2.config.angle_constraint_scope = "loopwise"
        st2.config.angle_constraint_measure = "raw_loopwise"
        st2.config.redistribution_density_scope = "loopwise"
        st2.config.redistribution_curvature_scope = "loopwise"
        st2.config.redistribution_q_policy = "r_loop"
        st2.commit_quotient(mask, 0, cert, (ba, bb))
        r_drop = st2.step(v)
        h_wall = float(st.fixed_masses.median())
        out.update(
            dx_over_h=float((r_drop.varifold.positions
                             - r_keep.varifold.positions).abs().max())
            / h_wall,
            dtheta=float(wrap(r_drop.varifold.angles
                              - r_keep.varifold.angles).abs().max()),
            r_stack_keep=st._last_grid_snapshot.r_stack,
            r_stack_drop=st2._last_grid_snapshot.r_stack,
            quotient_classes_drop=st2._last_grid_snapshot.quotient_classes,
        )
    except Exception as ex:      # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out


def main():
    torch.set_default_dtype(DT)
    rep = {}
    # (i) clean State-II window
    w768 = window_stats("_0lb0_768_B", 1569, 1583)
    w1024 = window_stats("_0lb0_1024_B", 1575, 1589)
    ks, dx, dth = state_motion("_b2cal_Bsave", 1568, 1583)
    hw = w768["h_wall"]
    rep["window_768"] = w768
    rep["window_1024"] = w1024
    rep["motion_768"] = dict(steps=ks, dx_over_h=[x / hw for x in dx],
                             dtheta=dth)
    print(f"(i) split (R)+ max 768 {w768['split_pos_max']:.3e} / 1024 "
          f"{w1024['split_pos_max']:.3e}; dV_merged rel max 768 "
          f"{w768['dV_rel_max']:.2e} / 1024 {w1024['dV_rel_max']:.2e}")
    print(f"    natural motion 768: |dX|/h max {max(dx) / hw:.3e}, "
          f"|dtheta| max {max(dth):.3e}")
    # (ii) fixture
    fx = fixture_shadow_switch()
    rep["fixture"] = fx
    print(f"(ii) fixture shadow: {fx}")
    # (iii) B vs D one-step at 1568 (negative)
    dxBD, dthBD = one_step_diff("_b2cal_Bsave", "_b2cal_D1", 1569)
    rep["BD_1569"] = dict(dx_over_h=dxBD / hw, dtheta=dthBD)
    print(f"(iii) B vs D one-step at 1569: |dX|/h {dxBD / hw:.3e}, "
          f"|dtheta| {dthBD:.3e}")
    # thresholds
    pos_x = [max(dx) / hw] + ([fx["dx_over_h"]] if "dx_over_h" in fx else [])
    pos_t = [max(dth)] + ([fx["dtheta"]] if "dtheta" in fx else [])
    neg_x, neg_t = dxBD / hw, dthBD

    def tau(pmax, nmin):
        if nmin > pmax > 0:
            return math.sqrt(pmax * nmin), True
        return pmax + EPS_NUM, False
    tx, sx = tau(max(pos_x), neg_x)
    tt, st_ = tau(max(pos_t), neg_t)
    gates = dict(
        eps_split=max(w768["split_pos_max"], w1024["split_pos_max"])
        + EPS_NUM,
        eps_switch_x=tx, eps_switch_theta=tt,
        eps_volume=max(w768["dV_rel_max"], w1024["dV_rel_max"]) + EPS_NUM,
        _separated=dict(switch_x=sx, switch_theta=st_),
        _sources=dict(pos_x=pos_x, pos_t=pos_t, neg_x=neg_x, neg_t=neg_t),
    )
    rep["gates"] = gates
    print("gates:", {k: v for k, v in gates.items() if not k.startswith("_")})
    Path("results/reports/phase3c0l_b2_calibration.json").write_text(
        json.dumps(rep, indent=1, default=str))
    body = (
        '"""0L-B2 pre-registered quotient-switch gates -- GENERATED by\n'
        'scripts/experiments/quotient_switch_calibration.py from the clean\n'
        'State-II window (B0 cell B, 768/1024), the B1 positive-fixture\n'
        'shadow switch and the B0 B/D one-step (negative reference).\n'
        'Never re-fit on a switch run.\n\n'
        f'    eps_split        (R_drop)+ floor      : '
        f'{gates["eps_split"]:.6e}\n'
        f'    eps_switch_x     |dX|_inf / h_wall    : '
        f'{gates["eps_switch_x"]:.6e}  (separated: {sx})\n'
        f'    eps_switch_theta |dtheta|_inf         : '
        f'{gates["eps_switch_theta"]:.6e}  (separated: {st_})\n'
        f'    eps_volume       |dV_merged|/V        : '
        f'{gates["eps_volume"]:.6e}\n"""\n\n'
        'QUOTIENT_GATES = {\n'
        f'    "eps_split": {gates["eps_split"]!r},\n'
        f'    "eps_switch_x": {gates["eps_switch_x"]!r},\n'
        f'    "eps_switch_theta": {gates["eps_switch_theta"]!r},\n'
        f'    "eps_volume": {gates["eps_volume"]!r},\n'
        '}\n')
    Path("src/torch/solver/quotient_gates.py").write_text(body)
    print("wrote src/torch/solver/quotient_gates.py")


if __name__ == "__main__":
    main()
