"""L0J-1a: static locality audit + refresh defects of the four
coherence arms on the three L0 stop states (reviewer 2026-08-21).

Arms:
    F  q^full   locality positive control (NOT a production candidate)
    W  q^WB     loopwise scalar -- spill negative control
    C  q^CC     contact-complex candidate
    S  q^self   interaction-off control

Locality metric (reviewer sec. 4): interaction deficit
    d_i^arm = q_i^self - q_i^arm       (NEVER 1 - q),
    eta_R^arm = sum_R m d^arm / sum_R m d^full,
undefined when the denominator is below the far-field floor. Regions:
complex window I_sigma, tight core (g <= sigma/2), fringe collar
(I_sigma minus core), far field.

Refresh defects (reviewer sec. 7), both kinds:
    common-candidate  D^common = [P_Y(Y) - P_X(Y)]_+  at shared Y
        (Y1 = tapered gap-closing displacement, Y2 = window shape
        perturbation, Y3 = the production one-step of the CURRENT
        production arm W),
    arm-specific      D^dyn = [P_{X1}(X1) - P_X(X1)]_+ with X1 the
        arm's own production one-step.
For arm C both include the hard-membership refresh jumps by
construction (the complex is re-derived on the new state).

Usage:
    uv run python scripts/experiments/l0j_locality_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import build_config  # noqa: E402
from quotient_switch_stability import production_args  # noqa: E402
from two_ellipses_benchmark import SIGMA, resolve_m  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.perimeter.coherence_perimeter import (  # noqa: E402
    compute_coherence_loopwise,
)
from src.torch.perimeter.contact_complex import (  # noqa: E402
    SigmaContactComplexError,
    contact_complex_q,
)
from src.torch.transport.bem_wasserstein import compute_coherence  # noqa: E402

DT = torch.float64
STATES = (("_l0_768", 768), ("_l0_1024", 1024), ("_l0_768rot", 768))
KERNEL = "wendland_c2"


def arms_q(pos, nor, m, lbl):
    q_full = compute_coherence(_V(pos, nor), m, SIGMA, KERNEL)
    q_self, _ = compute_coherence_loopwise(pos, nor, m, SIGMA, KERNEL,
                                           lbl)
    q_wb = torch.empty_like(q_full)
    for lb in lbl.unique():
        sel = lbl == lb
        r = float((m[sel] * q_full[sel]).sum()
                  / (m[sel] * q_self[sel]).sum())
        q_wb[sel] = r * q_self[sel]
    res = contact_complex_q(pos, nor, m, lbl, SIGMA, KERNEL,
                            q_full, q_self)
    return dict(F=q_full, W=q_wb, C=res["q_cc"], S=q_self), res


class _V:
    def __init__(self, pos, nor):
        self.positions, self.normals = pos, nor


def regions(pos, lbl, cx):
    n = pos.shape[0]
    m1 = lbl == 0
    g = torch.full((n,), float("inf"), dtype=DT)
    D = torch.cdist(pos[m1], pos[~m1])
    g[m1.nonzero().flatten()] = D.min(dim=1).values
    g[(~m1).nonzero().flatten()] = D.min(dim=0).values
    I = cx["complex_id"] >= 0
    core = I & (g <= SIGMA / 2)
    collar = I & ~core
    far = ~I
    return dict(window=I, core=core, collar=collar, far=far)


def eta_table(qs, m, regs, floor):
    d = {a: qs["S"] - qs[a] for a in ("F", "W", "C")}
    out = {}
    for rname, sel in regs.items():
        den = float((m[sel] * d["F"][sel]).sum()) if bool(sel.any()) \
            else 0.0
        row = dict(deficit_full=den, n=int(sel.sum()))
        for a in ("W", "C"):
            num = float((m[sel] * d[a][sel]).sum())
            row[f"deficit_{a}"] = num
            row[f"eta_{a}"] = (num / den if den > floor else None)
        out[rname] = row
    return out


def sharp_p(m, q):
    return float((m * q).sum())


def template_p(pos_y, nor_y, lbl, delta, tau, arm, m_y=None):
    """q^arm evaluated ON state Y (template refreshed at Y)."""
    m_y = resolve_m(pos_y, nor_y, delta, tau) if m_y is None else m_y
    qs, res = arms_q(pos_y, nor_y, m_y, lbl)
    return m_y, qs[arm], res


def one_step(pos, ang, grid, delta, tau, q_mode):
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.mm_step import MMStepper
    over = dict(q_mode=q_mode)
    if q_mode == "full":
        # the r_loop redistribution policy needs a WB mode; the F
        # positive-control arm runs with the historical stale_full
        over["redist_q_policy"] = "stale_full"
    cfg = build_config(production_args(grid=grid, redist_monotone=True,
                                       **over), delta, tau)
    solver = MMSolver(cfg)
    st = MMStepper(cfg)
    v = OrientedPointCloudVarifold(positions=pos.clone(),
                                   angles=ang.clone())
    res, committed = solver._advance_committed(st, v)
    return committed


def candidates(pos, ang, nor, m, lbl, cx, grid, delta, tau):
    """Common candidates Y1 (tapered gap closing), Y2 (window shape),
    Y3 (production one-step of the current production arm W)."""
    h = float(m.median())
    I = cx["complex_id"] >= 0
    m1 = lbl == 0
    ys = {}
    # facing normals point INTO the gap, so +normal displacement on
    # the window CLOSES the gap and -normal OPENS it
    for name, amp, mode in (("Y1_gapclose", 0.2 * h, "close"),
                            ("Y1b_gapopen", 0.2 * h, "open"),
                            ("Y2_shape", 0.2 * h, "shape")):
        d = torch.zeros_like(pos)
        for sel_loop in (m1, ~m1):
            sel = I & sel_loop
            if not bool(sel.any()):
                continue
            idx = sel.nonzero().flatten()
            k = idx.numel()
            s = torch.linspace(0, 1, k, dtype=DT)
            taper = torch.sin(torch.pi * s) ** 2
            if mode == "close":
                disp = amp * taper
            elif mode == "open":
                disp = -amp * taper
            else:
                disp = amp * taper * torch.sin(2 * torch.pi * s)
            d[idx] = nor[idx] * disp.unsqueeze(1)
        ys[name] = (pos + d, ang.clone())
    y3 = one_step(pos, ang, grid, delta, tau, "self_renormalized")
    ys["Y3_prod_step_W"] = (y3.positions, y3.angles)
    return ys


def main():
    torch.set_default_dtype(DT)
    rep = {}
    for tag, grid in STATES:
        p = Path(f"results/two_ellipses/two_ellipses{tag}_states.pt")
        if not p.exists():
            print(f"-- {tag} missing")
            continue
        ck = torch.load(p, weights_only=True)
        step = max(ck)
        pos, ang = ck[step]["positions"], ck[step]["angles"]
        n = pos.shape[0] // 2
        lbl = torch.zeros(pos.shape[0], dtype=torch.long)
        lbl[n:] = 1
        nor = torch.stack([ang.cos(), ang.sin()], 1)
        delta, tau = map(float, compute_recommended_params(pos))
        m = resolve_m(pos, nor, delta, tau)
        qs, res = arms_q(pos, nor, m, lbl)
        regs = regions(pos, lbl, res["cx"])
        far_floor = float((m[regs["far"]]
                           * (qs["S"] - qs["F"])[regs["far"]]).abs()
                          .sum()) * 10 + 1e-14
        eta = eta_table(qs, m, regs, far_floor)
        r_rep = dict(step=int(step),
                     sides=res["sides"],
                     eta=eta,
                     P={a: sharp_p(m, qs[a]) for a in qs})
        # ---- refresh defects ---------------------------------------
        ys = candidates(pos, ang, nor, m, lbl, res["cx"], grid,
                        delta, tau)
        common = {}
        for yname, (py, ay) in ys.items():
            ny = torch.stack([ay.cos(), ay.sin()], 1)
            row = {}
            m_y = resolve_m(py, ny, delta, tau)
            for arm in ("F", "W", "C", "S"):
                try:
                    _, q_y, _ = template_p(py, ny, lbl, delta, tau,
                                           arm, m_y)
                    q_x = qs[arm]
                    row[arm] = max(sharp_p(m_y, q_y)
                                   - sharp_p(m_y, q_x), 0.0)
                except SigmaContactComplexError as e:
                    row[arm] = f"FAIL-CLOSED: {str(e)[:80]}"
            common[yname] = row
        r_rep["refresh_common"] = common
        dyn = {}
        for arm, mode in (("F", "full"), ("W", "self_renormalized"),
                          ("C", "contact_complex_renormalized")):
            try:
                x1 = one_step(pos, ang, grid, delta, tau, mode)
                n1 = torch.stack([x1.angles.cos(), x1.angles.sin()], 1)
                m1_ = resolve_m(x1.positions, n1, delta, tau)
                _, q1, _ = template_p(x1.positions, n1, lbl, delta,
                                      tau, arm, m1_)
                q_x = qs[arm]
                dyn[arm] = max(sharp_p(m1_, q1) - sharp_p(m1_, q_x),
                               0.0)
            except Exception as e:            # noqa: BLE001 recorded
                dyn[arm] = f"FAIL: {type(e).__name__}: {str(e)[:80]}"
        r_rep["refresh_dynamic"] = dyn
        rep[tag] = r_rep
        print(f"== {tag} @ {step}: sides "
              f"{[(s['loop'], s['n'], round(s['r'], 4)) for s in res['sides']]}")
        for rname in ("window", "core", "collar", "far"):
            e = eta[rname]
            print(f"  {rname:7s} n {e['n']:4d} dF {e['deficit_full']:.3e}"
                  f" eta_W {e['eta_W']} eta_C {e['eta_C']}")
        print(f"  refresh common: "
              + " | ".join(f"{k}: C={v['C'] if isinstance(v['C'], str) else format(v['C'], '.2e')}"
                           f" W={v['W'] if isinstance(v['W'], str) else format(v['W'], '.2e')}"
                           for k, v in common.items()))
        print(f"  refresh dynamic: "
              + " ".join(f"{a}={v if isinstance(v, str) else format(v, '.2e')}"
                         for a, v in dyn.items()))
    Path("results/reports/l0j_locality_audit.json").write_text(
        json.dumps(rep, indent=1, default=str))
    print("wrote results/reports/l0j_locality_audit.json")


if __name__ == "__main__":
    main()
