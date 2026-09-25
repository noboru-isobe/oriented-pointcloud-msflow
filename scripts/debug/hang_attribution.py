"""Hang/crawl attribution v2 on the SAME snapshot (closure patch 2,
reviewer-approved redesign).

Each (snapshot, kind) case runs ONE MM step in a `spawn` child process
with a HARD parent-side timeout (the optimizer callback can only act at
iteration boundaries, so it is soft telemetry, never the timeout
mechanism). Telemetry survives timeouts: the child streams one JSONL
record per trust-region iteration (line-buffered flush) and overwrites
the latest iterate to disk, so a kill inside a single HVP still leaves
the last completed iterate and its callback record.

Diagnostics per case (all computed from the ACTUAL code-level objective
decomposition -- the implemented W_lin term already contains the 1/dt
factor, so nothing is rescaled by hand):

    H_P    = hessian of the frozen perimeter term
    H_W_a  = hessian of the Wasserstein term AS USED in the objective
    H_obj  = hessian of the full objective, with the consistency check
             H_obj ~= H_P + H_W_a  (dense case)
computed at TWO points: y = 0 and the last completed iterate y_k. The
dt dependence is probed by REBUILDING the objective at each dt in
--dt-sweep (relative scale H_P vs H_W/dt), never by dividing H_W again.
Dimension gate: dim_y <= 400 -> dense eigvalsh (exact
negative_eigenvalue_count); larger -> matrix-free Lanczos reporting
Ritz values only (n_negative_ritz_values, negative_curvature_detected).

Counters: torchmin does not expose HVP counts; the child wraps
torch.autograd.grad (child-process scope only) and reports the honest
name `autograd_grad_calls` alongside n_fev. It is NOT an HVP count.

Spectral cases additionally record rho_rank = s_(C)/s_(C+1) and
kappa_A_range = s_max/s_(C+1) (BIE range conditioning -- NOT a full
KKT condition number), plus min_collocation_over_eps for ALL modes.

Usage
-----
    uv run python scripts/debug/hang_attribution.py \
        results/p12_long/snapshots/pair_C2_step37.pt \
        --kinds C1 C2 C3 --timeout-s 600 \
        --out results/p12_long/hang_attribution
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
import sys

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

DT_DEFAULT = 1e-5


def _sym_eig_report(H, label):
    import torch
    H = 0.5 * (H + H.T)
    ev = torch.linalg.eigvalsh(H)
    mn, mx = float(ev.min()), float(ev.max())
    pos = ev[ev > 0]
    rep = dict(label=label, dim=H.shape[0], method="dense_eigvalsh",
               smallest_eigenvalue=mn, largest_eigenvalue=mx,
               negative_eigenvalue_count=int((ev < 0).sum()))
    if mn > 0:
        rep["condition_number"] = mx / mn
    elif len(pos):
        rep["positive_spectrum_condition"] = mx / float(pos.min())
    return rep


def _ritz_report(hvp, dim, label, k=6):
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh
    import torch

    def mv(x):
        v = torch.from_numpy(np.asarray(x, dtype=np.float64))
        return hvp(v).numpy()

    op = LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
    small = eigsh(op, k=k, which="SA", return_eigenvectors=False)
    large = eigsh(op, k=1, which="LA", return_eigenvectors=False)
    small = sorted(float(x) for x in small)
    return dict(label=label, dim=dim, method="lanczos_ritz",
                smallest_ritz_values=small,
                largest_ritz_value=float(large[0]),
                n_negative_ritz_values=sum(1 for x in small if x < 0),
                negative_curvature_detected=small[0] < 0,
                positive_ritz_condition_estimate=(
                    float(large[0]) / small[0] if small[0] > 0 else None))


def _hessian_block(stepper, y_point, label, dense_gate=400):
    """H_P / H_W_actual / H_obj at a given point, from the code-level
    decomposition, with the additivity check in the dense case."""
    import torch
    from src.torch.oriented_varifold.mass import compute_masses

    param = stepper.param
    cfg = stepper.config
    q = stepper.fixed_coherence

    def wlin(p):
        s, dth = param.unpack_params(p)
        return stepper.bem_wasserstein(
            displacements=s, delta_angles=dth, time_step=cfg.time_step)

    def perim(p):
        vf = param.reconstruct_varifold(p)
        m = compute_masses(vf.positions, stepper._mass_delta_for_kde,
                           stepper._mass_tau_for_kde, cfg.mass_kernel)
        return (m * q).sum()

    def obj(p):
        return stepper.objective(p)

    dim = y_point.shape[0]
    out = dict(point=label, dim=dim)
    if dim <= dense_gate:
        from torch.autograd.functional import hessian
        H_P = hessian(perim, y_point)
        H_W = hessian(wlin, y_point)
        H_O = hessian(obj, y_point)
        add = float((H_O - (H_P + H_W)).abs().max()
                    / H_O.abs().max().clamp_min(1e-300))
        out["decomposition_residual"] = add
        out["H_P"] = _sym_eig_report(H_P, "H_perimeter")
        out["H_W_actual"] = _sym_eig_report(H_W, "H_wasserstein_actual")
        out["H_obj"] = _sym_eig_report(H_O, "H_objective")
    else:
        def hvp_of(f):
            def hvp(vv):
                p = y_point.detach().clone().requires_grad_(True)
                g, = torch.autograd.grad(f(p), p, create_graph=True)
                Hv, = torch.autograd.grad(g, p, grad_outputs=vv)
                return Hv.detach()
            return hvp
        out["H_W_actual"] = _ritz_report(hvp_of(wlin), dim,
                                         "H_wasserstein_actual")
        out["H_obj"] = _ritz_report(hvp_of(obj), dim, "H_objective")
    return out


def child_case(snapshot_path, kind, dt_sweep, timeout_soft_s,
               tmp_json, callback_log, last_x_path):
    """Runs in a SPAWN child. Writes tmp_json atomically at the end;
    streams per-iteration callback records to callback_log (JSONL)."""
    # test-only hook: emulate a child stuck inside a single long
    # operation (e.g. one HVP) AFTER one callback record was streamed,
    # so the hard-timeout path can be pinned without a real hang
    if os.environ.get("HANG_ATTR_TEST_BLOCK"):
        with open(callback_log, "a", buffering=1) as f:
            f.write(json.dumps(dict(iteration=1, wall_s=0.0,
                                    fval=1.0, trust_radius=0.1,
                                    rho=0.5,
                                    trust_region_boundary_hit=False,
                                    step_norm=0.01)) + "\n")
            f.flush()
        time.sleep(3600)

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.oriented_varifold.mass import (
        compute_recommended_params,
    )
    from src.torch.diagnostics import classify_support
    from src.torch.solver.mm_step import MMStepper
    from p1_production_comparison import make_cfg

    # honest counter: child-scope wrap; this counts EVERY autograd.grad
    # call (gradients, HVPs, diagnostics) -- NOT an HVP count
    grad_calls = [0]
    _orig_grad = torch.autograd.grad

    def _counting_grad(*a, **k):
        grad_calls[0] += 1
        return _orig_grad(*a, **k)

    torch.autograd.grad = _counting_grad

    snap = torch.load(snapshot_path, weights_only=False)
    v = OrientedPointCloudVarifold(positions=snap["positions"],
                                   angles=snap["angles"])
    delta, tau = compute_recommended_params(v.positions)
    out = dict(case=f"{Path(snapshot_path).stem}::{kind}", kind=kind,
               snapshot=str(snapshot_path),
               recorded_wall=snap.get("wall"),
               recorded_median=snap.get("median"))

    cfg = make_cfg(kind, float(delta), float(tau), 2)
    cfg.time_step = DT_DEFAULT
    cfg.bem_setup_telemetry = True
    st = MMStepper(cfg)

    # --- pre-step diagnostics on the setup -------------------------
    st._setup_step(v)
    sn = st._last_bem_snapshot
    if sn is not None and sn.endpoint_positions is not None \
            and sn.epsilon_bem_used:
        E = sn.endpoint_positions
        K = sn.endpoints_per_particle
        pid = torch.arange(E.shape[0]) // K
        d = torch.cdist(E, E)
        d = d.masked_fill(pid[:, None] == pid[None, :], float("inf"))
        out["min_collocation_over_eps_raw"] = float(
            d.min() / sn.epsilon_bem_used)
        # DANGER metric: exclude by-design adjacency (each particle's
        # 2 nearest neighbours) -- nonlocal near-duplication only
        P = sn.positions
        dp = torch.cdist(P, P)
        dp.fill_diagonal_(float("inf"))
        nn2 = dp.argsort(dim=1)[:, :2]
        adj = torch.zeros_like(dp, dtype=torch.bool)
        adj.scatter_(1, nn2, True)
        adj = adj | adj.T
        adj_e = adj.repeat_interleave(K, 0).repeat_interleave(K, 1)
        out["min_collocation_over_eps_nonlocal"] = float(
            d.masked_fill(adj_e, float("inf")).min()
            / sn.epsilon_bem_used)
    if sn is not None and sn.singular_values_ascending is not None:
        s = sn.singular_values_ascending
        C = 2
        assert 1 <= C < len(s)
        out["rho_rank"] = float(s[C - 1] / s[C])
        out["kappa_A_range"] = float(s[-1] / s[C])
        out["s_null_edge"] = float(s[C - 1])
        out["s_range_edge"] = float(s[C])
        out["s_max"] = float(s[-1])

    loops = snap.get("loops")
    if loops is not None and all(int(c.max()) < v.n_points
                                 for c in loops):
        try:
            rep = classify_support(
                v, loops, delta=float(delta),
                sigma=cfg.perimeter_sigma or 0.1,
                h=float(torch.cat([
                    (v.positions[c].roll(-1, 0) - v.positions[c])
                    .norm(dim=1) for c in loops]).median()))
            out["support_state"] = rep.state
            out["support_regime"] = rep.interaction_regime
        except Exception as e:
            out["support_state"] = f"classifier_error: {e}"

    # --- Hessian attribution at y = 0, dt sweep --------------------
    n_params = st.param.n_params
    y0 = torch.zeros(n_params, dtype=torch.float64)
    hess = {"y0": {}}
    for dt in dt_sweep:
        cfg_dt = make_cfg(kind, float(delta), float(tau), 2)
        cfg_dt.time_step = dt
        st_dt = MMStepper(cfg_dt)
        st_dt._setup_step(v)
        hess["y0"][f"dt={dt:g}"] = _hessian_block(
            st_dt, torch.zeros(st_dt.param.n_params,
                               dtype=torch.float64), f"y0@dt={dt:g}")
    out["hessians"] = hess

    # --- the actual step with streaming callback -------------------
    t_start = time.perf_counter()
    it_count = [0]
    cb_f = open(callback_log, "a", buffering=1)

    def cb(x, **kw):
        it_count[0] += 1
        rec = dict(iteration=it_count[0],
                   wall_s=round(time.perf_counter() - t_start, 3),
                   fval=float(kw.get("fval", float("nan")))
                   if kw.get("fval") is not None else None,
                   trust_radius=(float(kw["trust_radius"])
                                 if kw.get("trust_radius") is not None
                                 else None),
                   rho=(float(kw["rho"])
                        if kw.get("rho") is not None else None),
                   trust_region_boundary_hit=bool(
                       kw.get("hits_boundary", False)),
                   step_norm=(float(kw["step_norm"])
                              if kw.get("step_norm") is not None
                              else None))
        cb_f.write(json.dumps(rec) + "\n")
        cb_f.flush()
        torch.save(x.detach().clone(), last_x_path)
        # soft stop only -- acts at iteration boundaries
        return (time.perf_counter() - t_start) > timeout_soft_s

    st.optimizer_callback = cb
    try:
        r = st.step(v)
        out["step"] = dict(
            wall=round(time.perf_counter() - t_start, 2),
            n_iter=r.n_iter, n_fev=r.n_fev,
            converged=bool(r.converged),
            optimizer_message=r.optimizer_message,
            objective=float(r.objective),
            relative_gradient_norm=r.relative_gradient_norm,
            max_disp=float(r.displacements.abs().max()))
    except Exception as e:
        out["step"] = dict(error=f"{type(e).__name__}: {e}",
                           wall=round(time.perf_counter() - t_start, 2))
    out["autograd_grad_calls"] = grad_calls[0]
    out["iterations_completed"] = it_count[0]

    # --- Hessian at the last completed iterate ---------------------
    if Path(last_x_path).exists():
        yk = torch.load(last_x_path, weights_only=False)
        try:
            st2 = MMStepper(cfg)
            st2._setup_step(v)
            out["hessians"]["y_last"] = _hessian_block(st2, yk, "y_last")
        except Exception as e:
            out["hessians"]["y_last"] = dict(error=str(e))

    tmp = Path(tmp_json)
    tmp.write_text(json.dumps(out, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--kinds", nargs="+", default=["C1", "C2", "C3"])
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--dt-sweep", type=float, nargs="+",
                    default=[1e-5, 1e-4, 1e-3])
    ap.add_argument("--out", type=Path,
                    default=Path("results/p12_long/hang_attribution"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    for kind in args.kinds:
        stem = f"{args.snapshot.stem}__{kind}"
        final = args.out / f"{stem}.json"
        tmp = args.out / f"{stem}.tmp.json"
        cb_log = args.out / f"{stem}.callback.jsonl"
        last_x = args.out / f"{stem}.last_x.pt"
        for p in (tmp, cb_log, last_x):
            if p.exists():
                p.unlink()

        proc = ctx.Process(
            target=child_case,
            args=(str(args.snapshot), kind, args.dt_sweep,
                  args.timeout_s * 0.9, str(tmp), str(cb_log),
                  str(last_x)))
        proc.start()
        proc.join(args.timeout_s)
        if proc.is_alive():
            proc.terminate()
            proc.join(10)
            if proc.is_alive():
                proc.kill()
                proc.join()
            last_cb = None
            n_it = 0
            if cb_log.exists():
                lines = cb_log.read_text().strip().splitlines()
                n_it = len(lines)
                if lines:
                    last_cb = json.loads(lines[-1])
            final.write_text(json.dumps(dict(
                case=stem, timed_out=True,
                timeout_s=args.timeout_s,
                child_exitcode=proc.exitcode,
                last_completed_iteration=n_it,
                last_callback=last_cb,
                callback_records_path=str(cb_log)), indent=1))
            print(f"  {stem}: TIMED OUT after {args.timeout_s}s "
                  f"({n_it} iterations completed)", flush=True)
        else:
            if tmp.exists():
                tmp.replace(final)          # atomic rename
                d = json.loads(final.read_text())
                stp = d.get("step", {})
                print(f"  {stem}: wall={stp.get('wall')}s "
                      f"n_iter={stp.get('n_iter')} "
                      f"grad_calls={d.get('autograd_grad_calls')} "
                      f"colloc_nl/eps="
                      f"{d.get('min_collocation_over_eps_nonlocal')}",
                      flush=True)
            else:
                final.write_text(json.dumps(dict(
                    case=stem, timed_out=False,
                    child_exitcode=proc.exitcode,
                    error="child exited without writing result"),
                    indent=1))
                print(f"  {stem}: CHILD FAILED "
                      f"(exit {proc.exitcode})", flush=True)
    print(f"results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
