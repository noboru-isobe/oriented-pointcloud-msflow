"""P3 feasibility: warm-started LOBPCG vs full SVD for the spectral
near-null basis.

MEASUREMENT ONLY -- nothing is integrated into the solver. Per the
review, truncated extraction is a performance optimization to be
introduced against the full-SVD baseline with parity gates; this script
supplies the feasibility numbers first:

  - parity: principal angle between the LOBPCG smallest-C left subspace
    and the full-SVD U0; singular-value agreement; gap-ratio agreement
    (the rank machinery consumes sigma_(C), sigma_(C+1));
  - warm start: on an evolving trajectory the operator moves O(dt) per
    step, so LOBPCG seeded with the previous step's basis should
    converge in a few matvecs;
  - cost: full torch.linalg.svd wall time vs LOBPCG wall time. The
    Gram matrix is NEVER formed (that would be O(N^3) again): scipy's
    LOBPCG gets a LinearOperator whose matvec is two dense mat-vecs,
    O(NK^2) per application. Left singular vectors <-> eigenvectors of
    A A^T.

Usage
-----
    uv run python scripts/experiments/p3_lobpcg_feasibility.py \
        --out results/p3_lobpcg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, lobpcg

from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.solver.mm_step import MMStepper

sys.path.insert(0, str(Path(__file__).parent))
from d1a_mm_shadow import make_config  # noqa: E402
from d2b0_timelevel_shadow import snapshot  # noqa: E402
from p1_production_comparison import make_cfg, shapes  # noqa: E402

K_EXTRA = 3          # smallest C + K_EXTRA pairs (gap diagnostics)


def lobpcg_left_smallest(A_w: torch.Tensor, k: int,
                         X0: np.ndarray | None, tol: float = 1e-10,
                         maxiter: int = 200):
    """Smallest k left singular pairs of A_w via LOBPCG on A A^T,
    Gram-free (matvec = two dense mat-vecs)."""
    A = A_w.numpy()
    n = A.shape[0]

    def mv(x):
        return A @ (A.T @ x)

    op = LinearOperator((n, n), matvec=mv, matmat=lambda X: A @ (A.T @ X),
                        dtype=A.dtype)
    rng = np.random.default_rng(0)
    if X0 is None:
        X0 = rng.standard_normal((n, k))
    t0 = time.perf_counter()
    vals, vecs = lobpcg(op, X0, largest=False, tol=tol, maxiter=maxiter)
    wall = time.perf_counter() - t0
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    svals = np.sqrt(np.clip(vals, 0.0, None))
    return svals, vecs, wall


def principal_angle_deg(U1: np.ndarray, U2: np.ndarray) -> float:
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


def run_trajectory(label, v, cfg, C, n_steps):
    stepper = MMStepper(cfg)
    rows = []
    X_warm = None
    for step in range(n_steps):
        r = stepper.step(v)
        bw = stepper.bem_wasserstein
        A_w = bw._spec_A_weighted.detach().cpu()
        U_full, S_full, _ = bw._spec_svd
        t_full = bw.last_timings["svd"]
        # full-SVD reference: torch svd is DESCENDING -> smallest at end
        U0_full = U_full[:, -C:].cpu().numpy()
        s_small_full = S_full[-(C + K_EXTRA):].flip(0).cpu().numpy()

        k = C + K_EXTRA
        cold = X_warm is None
        svals, vecs, t_lob = lobpcg_left_smallest(A_w, k, X_warm)
        X_warm = vecs.copy()

        ang = principal_angle_deg(U0_full, vecs[:, :C])
        sv_rel = float(np.max(np.abs(svals - s_small_full)
                              / np.maximum(s_small_full, 1e-300)))
        gap_full = s_small_full[C] / max(s_small_full[C - 1], 1e-300)
        gap_lob = svals[C] / max(svals[C - 1], 1e-300)
        rows.append(dict(step=step, cold=cold, NK=A_w.shape[0],
                         t_full_svd=t_full, t_lobpcg=t_lob,
                         speedup=t_full / t_lob if t_lob > 0 else None,
                         principal_angle_deg=ang,
                         sval_rel_err=sv_rel,
                         gap_ratio_full=float(gap_full),
                         gap_ratio_lobpcg=float(gap_lob)))
        v = r.varifold
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/p3_lobpcg"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = {}

    # C = 1: annulus trajectory (corrected config)
    v0, _, _ = snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    v, n_out, n_in = snapshot(0.5, warp=0.0)
    rows = run_trajectory("annulus", v, make_config(2e-5, delta, tau),
                          1, 12)
    out["annulus_C1"] = rows

    # C = 2: paper two-ellipses (production-candidate config C3)
    v2, rank = shapes()["two_ellipses_paper"]
    d2, t2 = compute_recommended_params(v2.positions)
    rows2 = run_trajectory("two_ellipses", v2,
                           make_cfg("C3", d2, t2, rank), 2, 12)
    out["two_ellipses_C2"] = rows2

    for name, rows in out.items():
        warm = [r for r in rows if not r["cold"]]
        print(f"=== {name} (NK={rows[0]['NK']}) ===")
        print(f"  cold step: angle={rows[0]['principal_angle_deg']:.2e} "
              f"deg, sval_rel={rows[0]['sval_rel_err']:.2e}, "
              f"t={rows[0]['t_lobpcg']:.3f}s vs full "
              f"{rows[0]['t_full_svd']:.3f}s")
        if warm:
            print(f"  warm steps (n={len(warm)}): "
                  f"max angle={max(r['principal_angle_deg'] for r in warm):.2e} deg, "
                  f"max sval_rel={max(r['sval_rel_err'] for r in warm):.2e}, "
                  f"median speedup="
                  f"{sorted(r['speedup'] for r in warm)[len(warm)//2]:.1f}x, "
                  f"max |gap_full-gap_lob|="
                  f"{max(abs(r['gap_ratio_full'] - r['gap_ratio_lobpcg']) for r in warm):.2e}")

    path = args.out / "p3_lobpcg_feasibility.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
