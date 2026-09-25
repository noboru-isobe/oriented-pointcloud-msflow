"""P1.1: staged production-integration comparison (F14 protocol),
DECOMPOSED and fail-closed.

Review correction: the earlier C2 -> C3 jump changed FOUR things at
once (compatibility, operator measure, epsilon scale, velocity
weighting), so its effect is the PRODUCTION METRIC BUNDLE effect, not
the componentwise-compatibility effect. The chain now isolates each
axis with cheap one-step ablations:

    C1    legacy exterior + legacy global projection    (published)
    C2    + interior trace                    (trace effect, isolated)
    C2S   + spectral_bordered compatibility ONLY
          (visible operator, sigma epsilon, q-velocity kept).
          INVARIANT: for phase rank C = 1 shapes C2 ~ C2S -- a large
          difference here is a spectral IMPLEMENTATION effect, not
          componentwise theory. For the two-ellipse pair (C = 2) this
          step IS the componentwise-compatibility effect.
    C2O   + carrier operator measure
    C2E   + carrier-segment epsilon
    C3    + M velocity variant               (= production candidate)

All configs share the SAME frozen delta/tau and sigma = 0.1, real
coherence. FAIL-CLOSED: every one-step run stores optimizer/gradient
diagnostics and finiteness checks; any invalid row makes the script
exit nonzero. Effects are reported RELATIVE and ABSOLUTE
(max|s|, max|s|/h_local, max|dtheta|) -- a 67% relative difference
means nothing without the displacement scale. The initial FROZEN
perimeter is cross-checked across configs (same sigma/q => it must
agree; the earlier rel_dP compared perimeters at different minimizers
and checked nothing).

Usage
-----
    uv run python scripts/experiments/p1_production_comparison.py \
        --out results/p1_production
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.shapes.generator import (
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_flower,
    generate_oriented_star,
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_step import MMConfig, MMStepper

sys.path.insert(0, str(Path(__file__).parent))

DT = 1e-4
SIGMA = 0.1
DTYPE = torch.float64
RELGRAD_TOL = 1e-5
CHAIN = ("C1", "C2", "C2S", "C2O", "C2E", "C3")
PAIRS = (("trace", "C1", "C2"),
         ("spectral_compat_only", "C2", "C2S"),
         ("operator_measure", "C2S", "C2O"),
         ("epsilon_scale", "C2O", "C2E"),
         ("velocity_variant", "C2E", "C3"),
         ("production_metric_bundle", "C2", "C3"))


def shapes():
    return dict(
        circle=(generate_oriented_circle(64, 1.0, (0.0, 0.0), "cpu",
                                         DTYPE), 1),
        ellipse2to1=(generate_oriented_ellipse(96, 1.0, 0.5, (0.0, 0.0),
                                               "cpu", DTYPE), 1),
        flower=(generate_oriented_flower(128, 5, 0.7, 1.3, (0.0, 0.0),
                                         "cpu", DTYPE), 1),
        star=(generate_oriented_star(128, device="cpu", dtype=DTYPE), 1),
        # PAPER configuration (manuscript 6.2.2): vertical ellipses
        # a = 0.4, b = 1.0 at centers +-0.45, minor axes facing,
        # initial gap 0.1 -- NOT the generator default (+-0.6, gap 0.4)
        two_ellipses_paper=(generate_oriented_two_ellipses(
            128, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
            a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu",
            dtype=DTYPE), 2),
    )


def make_cfg(kind: str, delta, tau, rank: int) -> MMConfig:
    common = dict(time_step=DT, perimeter_sigma=SIGMA, angle_sigma=SIGMA,
                  use_unit_coherence=False,
                  mass_bandwidth_type="global",
                  mass_delta=delta, mass_tau=tau,
                  mass_delta_adaptive=False,
                  redistribute=False, remove_dead_points=False,
                  optimizer_method="trust-ncg", optimizer_tol=1e-8,
                  optimizer_max_iter=300)
    spectral = dict(bem_solver_mode="spectral_bordered",
                    bem_rank_mode="oracle", bem_component_rank=rank)
    if kind == "C1":
        return MMConfig(**common)
    if kind == "C2":
        return MMConfig(bem_trace_side="interior", **common)
    if kind == "C2S":
        return MMConfig(bem_trace_side="interior", **spectral, **common)
    if kind == "C2O":
        return MMConfig(bem_trace_side="interior", **spectral,
                        bem_operator_measure="carrier", **common)
    if kind == "C2E":
        return MMConfig(bem_trace_side="interior", **spectral,
                        bem_operator_measure="carrier",
                        bem_epsilon_mode="carrier_segment_length",
                        **common)
    if kind == "C3":
        return MMConfig(bem_trace_side="interior", **spectral,
                        bem_operator_measure="carrier",
                        bem_epsilon_mode="carrier_segment_length",
                        wlin_use_coherence_velocity=False, **common)
    raise ValueError(kind)


def _circ(a, b):
    return ((a - b + math.pi) % (2 * math.pi) - math.pi)


def one_step(kind, v, delta, tau, rank):
    stepper = MMStepper(make_cfg(kind, delta, tau, rank))
    r = stepper.step(v)
    s = r.displacements.detach().flatten().cpu()
    # permutation-invariant kNN scale (review: the ordered-roll median
    # picked up the two cross-component chords on the pair)
    dmat = torch.cdist(v.positions, v.positions)
    dmat.fill_diagonal_(float("inf"))
    h_local = dmat.min(dim=1).values.median().item()
    dtheta = _circ(r.varifold.angles.cpu(), v.angles.cpu())
    relgrad = r.relative_gradient_norm
    valid = bool(
        r.objective_decreased
        and relgrad is not None and math.isfinite(relgrad)
        and relgrad < RELGRAD_TOL
        and torch.isfinite(s).all()
        and torch.isfinite(dtheta).all())
    return dict(
        s=s, valid=valid,
        perimeter=r.perimeter, wasserstein=r.wasserstein,
        frozen_perimeter_initial=r.frozen_perimeter_initial,
        converged=bool(r.converged), n_iter=r.n_iter,
        objective_decreased=bool(r.objective_decreased),
        relative_gradient_norm=relgrad,
        gradient_norm_final=r.gradient_norm_final,
        step_norm=r.step_norm,
        max_abs_s=s.abs().max().item(),
        max_abs_s_over_h=s.abs().max().item() / h_local,
        max_abs_dtheta=dtheta.abs().max().item(),
        h_local=h_local,
        # spectral diagnostics (None on legacy configs)
        detected_rank=getattr(r, "detected_rank", None),
        rank_status=getattr(r, "rank_status", None),
        rank_gap_ratio=getattr(r, "rank_gap_ratio", None),
        rank_abs_level=getattr(r, "rank_abs_level", None),
        r_comp=getattr(r, "r_comp", None),
        compat_row_angle_deg=_compat_row_angle(stepper, rank))


def _compat_row_angle(stepper, rank):
    """Angle between the spectral particle-space compatibility row and
    the legacy global constraint row (q^2 m weights) -- the finite-N
    nullspace tilt the C=1 C2 ~ C2S invariant rides on. None for
    legacy configs and for C > 1 (row spaces, not single rows)."""
    C_full = getattr(stepper, "_spectral_C_full", None)
    if C_full is None or rank != 1:
        return None
    row = C_full[0].detach().cpu()
    w = (stepper.fixed_coherence ** 2 * stepper.fixed_masses)
    w = w.detach().cpu()
    cosang = torch.dot(row, w).abs() / (row.norm() * w.norm())
    return math.degrees(math.acos(min(max(cosang.item(), -1.0), 1.0)))


def compare(a, b):
    sa, sb = a["s"], b["s"]
    denom = sb.abs().max().item()
    return dict(
        rel_max_ds=(sa - sb).abs().max().item() / denom
        if denom > 0 else None,
        abs_max_ds=(sa - sb).abs().max().item(),
        direction_cos=torch.dot(sa, sb).item()
        / max(sa.norm().item() * sb.norm().item(), 1e-30),
        rel_dW=(abs(a["wasserstein"] - b["wasserstein"])
                / abs(b["wasserstein"])
                if abs(b["wasserstein"]) > 1e-30 else None),
        # SAME initial geometry, same sigma/q: the frozen initial
        # perimeter must agree across configs (energy sanity)
        frozen_P0_rel_diff=(
            abs(a["frozen_perimeter_initial"]
                - b["frozen_perimeter_initial"])
            / abs(b["frozen_perimeter_initial"])))




def condition_numbers(v, rank, delta, tau):
    """Conditioning of the three realizations of the singular BEM
    solve (P1.2 diagnostics): the RAW weighted operator (near-null
    kappa ~ 1/s_min, grows with N), the BORDERED matrix (near-null
    directions absorbed into the border: kappa set by the separated
    spectrum, uniform in N), and the legacy GLOBAL projection (ONE
    weighted-constant row: fine at C = 1 up to the finite-N tilt, but
    at C = 2 it cannot remove the second near-null mode -- the hidden
    ill-conditioning that is the numerical face of the exchange
    mode)."""
    st = MMStepper(make_cfg("C3", delta, tau, rank))
    st.step(v)
    A = st.bem_wasserstein._spec_A_weighted.detach().cpu()
    U, S, Vh = torch.linalg.svd(A)
    NK = A.shape[0]
    L = U[:, -rank:]
    R = Vh[-rank:, :].T
    Bord = torch.zeros(NK + rank, NK + rank, dtype=A.dtype)
    Bord[:NK, :NK] = A
    Bord[:NK, NK:] = L
    Bord[NK:, :NK] = R.T
    sb = torch.linalg.svdvals(Bord)
    w = st.bem_wasserstein._endpoint_weights_cache.detach().cpu().sqrt()
    w = w / w.norm()
    Qc = torch.linalg.qr(torch.eye(NK, dtype=A.dtype)
                         - torch.outer(w, w))[0][:, :NK - 1]
    sp = torch.linalg.svdvals(Qc.T @ A @ Qc)
    return dict(
        NK=NK, C=rank,
        raw=dict(s_min=S[-1].item(), s_max=S[0].item(),
                 kappa=(S[0] / S[-1]).item()),
        bordered=dict(s_min=sb[-1].item(),
                      kappa=(sb[0] / sb[-1]).item()),
        global_projected=dict(s_min=sp[-1].item(),
                              kappa=(sp[0] / sp[-1]).item()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("results/p1_production"))
    ap.add_argument("--conds-only", action="store_true",
                    help="append the condition-number section to the "
                         "existing JSON without redoing the chain")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "p1_one_step_comparison.json"

    if args.conds_only:
        out = json.loads(path.read_text())
        out["condition_numbers"] = {}
        for name, (v, rank) in shapes().items():
            delta, tau = compute_recommended_params(v.positions)
            c = condition_numbers(v, rank, delta, tau)
            out["condition_numbers"][name] = c
            print(f"  {name:20s} raw kappa={c['raw']['kappa']:.1e}  "
                  f"bordered={c['bordered']['kappa']:.1f}  "
                  f"global={c['global_projected']['kappa']:.1e}",
                  flush=True)
        path.write_text(json.dumps(out, indent=1, default=str))
        print(f"raw results -> {path}")
        return

    out = dict(meta=dict(dt=DT, sigma=SIGMA, relgrad_tol=RELGRAD_TOL,
                         chain=list(CHAIN)))
    any_invalid = False
    for name, (v, rank) in shapes().items():
        delta, tau = compute_recommended_params(v.positions)
        row = dict(n_points=v.n_points, rank=rank, delta=delta)
        res = {}
        for kind in CHAIN:
            try:
                res[kind] = one_step(kind, v, delta, tau, rank)
            except Exception as e:
                res[kind] = None
                row[f"{kind}_error"] = f"{type(e).__name__}: {e}"
                any_invalid = True
        for kind in CHAIN:
            if res.get(kind):
                d = {k: val for k, val in res[kind].items() if k != "s"}
                row[f"diag_{kind}"] = d
                if not res[kind]["valid"]:
                    any_invalid = True
        for label, a, b in PAIRS:
            if res.get(a) and res.get(b):
                row[label] = compare(res[a], res[b])
        out[name] = row
        parts = []
        for label, a, b in PAIRS[:-1]:
            c = row.get(label, {})
            parts.append(f"{label}={c.get('rel_max_ds')}")
        print(f"{name:20s} N={v.n_points:4d} C={rank} "
              f"max|s|={row.get('diag_C2', {}).get('max_abs_s')} "
              f"(/h={row.get('diag_C2', {}).get('max_abs_s_over_h')})",
              flush=True)
        for p in parts:
            print(f"    {p}", flush=True)

    # --- pair C2/C2S at tight optimizer tolerance (review item 3:
    # 0.23% is 'numerically small at the present one-step resolution',
    # not a resolved physical magnitude until it is tolerance-stable)
    v, rank = shapes()["two_ellipses_paper"]
    delta, tau = compute_recommended_params(v.positions)
    tight = {}
    tight_diag = {}
    for kind in ("C2", "C2S"):
        cfg = make_cfg(kind, delta, tau, rank)
        cfg.optimizer_tol = 1e-10
        st = MMStepper(cfg)
        r = st.step(v)
        tight[kind] = r.displacements.detach().flatten().cpu()
        # review: without these the 'tolerance-stable' claim cannot
        # distinguish a genuinely tighter solve from an optimizer that
        # stopped at the same point for a different reason
        tight_diag[kind] = dict(
            converged=bool(r.converged), n_iter=r.n_iter,
            optimizer_message=str(r.optimizer_message),
            relative_gradient_norm=r.relative_gradient_norm,
            gradient_norm_final=r.gradient_norm_final,
            objective_decreased=bool(r.objective_decreased))
    denom = tight["C2S"].abs().max().item()
    out["pair_tight_tol"] = dict(
        optimizer_tol=1e-10,
        rel_max_ds=(tight["C2"] - tight["C2S"]).abs().max().item()
        / denom,
        diagnostics=tight_diag)
    print(f"pair C2/C2S @tol=1e-10: rel_max_ds="
          f"{out['pair_tight_tol']['rel_max_ds']} "
          f"diag={tight_diag}", flush=True)

    # OPTIMIZER-INDEPENDENT closure (review): the projected initial
    # gradients on the two admissible spaces at y = 0. No optimizer in
    # the loop at all -- if these agree to the same order, the
    # symmetric initial perimeter gradient only weakly excites the
    # component-exchange correction, full stop.
    pg = {}
    for kind in ("C2", "C2S"):
        st = MMStepper(make_cfg(kind, delta, tau, rank))
        st._setup_step(v)
        y0 = st.param.init_params().requires_grad_(True)
        st.objective(y0).backward()
        pg[kind] = y0.grad.detach().cpu()
    na, nb = pg["C2"].norm().item(), pg["C2S"].norm().item()
    # the parameter bases differ (dim N-1 vs N-C): compare the
    # PHYSICAL displacement directions of one gradient step
    def _phys(kind, g):
        st = MMStepper(make_cfg(kind, delta, tau, rank))
        st._setup_step(v)
        s_disp, _ = st.param.unpack_params(-g.to(
            st.param.prev_positions.dtype))
        return s_disp.detach().cpu().flatten()
    da, db = _phys("C2", pg["C2"]), _phys("C2S", pg["C2S"])
    out["pair_projected_gradient"] = dict(
        grad_norm_C2=na, grad_norm_C2S=nb,
        rel_norm_diff=abs(na - nb) / max(nb, 1e-30),
        phys_direction_cos=(torch.dot(da, db)
                            / (da.norm() * db.norm())).item())
    print(f"projected initial gradients: |g|_C2={na:.6e} "
          f"|g|_C2S={nb:.6e} "
          f"dir_cos={out['pair_projected_gradient']['phys_direction_cos']:.8f}",
          flush=True)

    # --- N-refinement of the C=1 C2~C2S invariant (review item 4): is
    # the star's 18.2% a decaying finite-N nullspace error or a
    # persistent bordered/global implementation difference?
    out["c2s_refinement"] = {}
    for name, gen in (
            ("flower", lambda n: generate_oriented_flower(
                n, 5, 0.7, 1.3, (0.0, 0.0), "cpu", DTYPE)),
            ("star", lambda n: generate_oriented_star(
                n, device="cpu", dtype=DTYPE))):
        for n in (128, 256, 512):
            vv = gen(n)
            dd, tt = compute_recommended_params(vv.positions)
            a = one_step("C2", vv, dd, tt, 1)
            b = one_step("C2S", vv, dd, tt, 1)
            c = compare(a, b)
            out["c2s_refinement"][f"{name}_n{n}"] = dict(
                rel_max_ds=c["rel_max_ds"],
                abs_max_ds=c["abs_max_ds"],
                compat_row_angle_deg=b["compat_row_angle_deg"],
                rank_abs_level=b["rank_abs_level"],
                valid=a["valid"] and b["valid"])
            print(f"  c2s_refinement {name} n={n}: "
                  f"rel={c['rel_max_ds']:.4f} "
                  f"row_angle={b['compat_row_angle_deg']:.3f} deg "
                  f"valid={a['valid'] and b['valid']}", flush=True)

    # --- state-machine preflight on the paper pair (review item 5):
    # the full-SVD gap ratio starts ~15 < rank_gap_min = 30, so the
    # PRODUCTION rank mode may refuse to initialize -- measure, do not
    # silently fall back to oracle
    cfg = make_cfg("C3", delta, tau, rank)
    cfg.bem_rank_mode = "state_machine"
    st = MMStepper(cfg)
    try:
        r = st.step(v)
        out["pair_state_machine_preflight"] = dict(
            initialized=True, detected_rank=r.detected_rank,
            rank_status=r.rank_status,
            rank_gap_ratio=r.rank_gap_ratio)
    except Exception as e:
        out["pair_state_machine_preflight"] = dict(
            initialized=False,
            error=f"{type(e).__name__}: {str(e)[:200]}")
    print(f"state-machine preflight: "
          f"{out['pair_state_machine_preflight']}", flush=True)

    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"raw results -> {path}")
    if any_invalid:
        print("INVALID ROWS PRESENT -- failing closed")
        sys.exit(1)


if __name__ == "__main__":
    main()
