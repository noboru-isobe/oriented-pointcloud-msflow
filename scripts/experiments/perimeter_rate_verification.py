#!/usr/bin/env python3
"""Numerical verification of Corollary cor:explicit-rate-with-BB-bandwidth.

Corollary (docs/draft (5).tex):
    E |hat P_{σ_N, δ_N, N} − P(V; Ω)| ≤ C · N^{−β² r / (d−1+2β)},
    r = c/(d−1+2c), c = min(α, γ),
    δ_N ≍ N^{−1/(d−1+2c)},
    σ_N = N^{−a*},  a* = β r / (d−1+2β).

This script picks the unit circle Γ = S^1 ⊂ R^2 (so d=2, α=1) with uniform
sampling density (γ=1), and sweeps β ∈ {1.0, 0.5} via two mark-distribution
scenarios:

  - β=1   : N_i ≡ X_i (deterministic outward normal), m̄(x) = x.
  - β=0.5 : N_i = ±X_i Bernoulli with success prob p(φ) = (1 + c(φ))/2,
            c(φ) = sqrt(d_S¹(φ, φ*)/π) (Hölder-1/2 cusp at φ* = 0).
            Then m̄(x) = c(φ) x, |m̄| ≤ 1, Hölder regularity exactly 1/2.

For each (β, N), we draw M i.i.d. trials. Each trial computes the BB-corrected
weights W_i^{δ,N} = χ_τ(θ̂_δ(X_i)) / θ̂_δ(X_i), masses m_i = W_i / N, then
P̂_σ = Σ m_i q_i via the existing src/torch utilities.

The MC mean error E_N = mean_m |P̂_σ_N,δ_N,N − 2π| is fit on a log-log plot
against N; the fitted slope is compared to the theoretical −β² r / (d−1+2β).

Outputs:
  - PNG: log-log with both scenarios and dashed theoretical slope lines
  - JSON: fit slope ± CI per scenario, per-N stats
  - CSV: per-trial raw errors (beta, N, trial_idx, error)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from src.torch.oriented_varifold.state import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_kde_density, chi_tau
from src.torch.perimeter.coherence_perimeter import compute_perimeter_coherence


D = 2                  # ambient dim
ALPHA = 1.0            # surface smoothness (unit circle is C^∞)
GAMMA = 1.0            # sampling density smoothness (uniform)
THETA_MINUS = 1.0 / (2.0 * math.pi)   # uniform-on-S^1 density lower bound

# Mark-model parameters
MARKED_A0 = 0.5                         # β=1 "marked-constant": |\bar n|=a_0
WEIERSTRASS_A0 = 0.5                    # β=0.5 "weierstrass" baseline
WEIERSTRASS_ETA = 0.1                   # amplitude of multi-scale roughness
WEIERSTRASS_M = 15                      # # levels (smallest scale ≈ 2π/2^15)
WEIERSTRASS_BASE = 2                    # dyadic series

LEGACY_MARK_MODEL = {1.0: "deterministic", 0.5: "localized-cusp"}
ALL_MARK_MODELS = ["deterministic", "marked-constant",
                   "localized-cusp", "weierstrass"]


def weierstrass_a(phi: np.ndarray, *, beta: float = 0.5,
                  a0: float = WEIERSTRASS_A0, eta: float = WEIERSTRASS_ETA,
                  M: int = WEIERSTRASS_M, base: int = WEIERSTRASS_BASE,
                  ) -> np.ndarray:
    """Truncated Weierstrass coherence amplitude on S¹:

        a(φ) = a_0 + η Σ_{m=0}^{M} b^{-mβ} cos(b^m φ)

    Construction is C^{0,β} but nowhere differentiable for finite β<1 in
    the M→∞ limit. ``cos(b^m φ)`` has period 2π for integer b, so each
    level integrates to zero ⇒ ∫a dφ = 2π a_0.

    Bound: |a(φ) - a_0| ≤ η/(1 - b^{-β}).
    For β=0.5, b=2: |a - a_0| ≤ 3.41·η ≈ 0.341 with η=0.1
    ⇒ a(φ) ∈ [0.16, 0.84], strictly positive (so |a|=a).
    """
    out = np.full_like(phi, a0, dtype=np.float64)
    for m in range(M + 1):
        out = out + eta * (base ** (-m * beta)) * np.cos((base ** m) * phi)
    return out


def true_perimeter(beta: float, mark_model: str = None) -> float:
    """The corollary's limit P(V; Ω) = ∫_Γ |m̄(x)| dH¹(x), depends on the
    mean coherence field m̄.

    - ``"deterministic"`` (β=1, no noise): m̄ = x, |m̄|≡1  ⇒ P = 2π
    - ``"marked-constant"`` (β=1, Bernoulli sign with const ā=a_0):
            m̄ = a_0·x, |m̄|≡a_0  ⇒ P = 2π·a_0
    - ``"localized-cusp"`` (β=0.5, single Hölder cusp at φ=0):
            m̄(φ) = c(φ)·x, c=sqrt(d_S¹/π)  ⇒ P = 4π/3
    - ``"weierstrass"`` (β=0.5, multi-scale Hölder roughness):
            m̄(φ) = a(φ)·x, a∈[a_min,a_max]⊂(0,1)  ⇒ P = 2π·a_0

    Earlier versions used 2π for both legacy cases — that was wrong for
    β=0.5 (the Bernoulli sign-flip variance shrinks |m̄|).
    """
    if mark_model is None:                     # backward compat
        mark_model = LEGACY_MARK_MODEL.get(beta)
        if mark_model is None:
            raise ValueError(f"no legacy mark_model for beta={beta!r}")
    if mark_model == "deterministic":
        return 2.0 * math.pi
    if mark_model == "marked-constant":
        return 2.0 * math.pi * MARKED_A0
    if mark_model == "localized-cusp":
        return 4.0 * math.pi / 3.0
    if mark_model == "weierstrass":
        return 2.0 * math.pi * WEIERSTRASS_A0
    raise ValueError(f"unknown mark_model={mark_model!r}")


# =============================================================================
# Sampling
# =============================================================================

def sample_scenario(
    N: int, beta: float, rng: np.random.Generator,
    *, mark_model: str = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample (X_i, N_i) ∈ S^1 × S^1 from the marked i.i.d. model.

    The ``mark_model`` selects how the conditional mean ``ā(φ) =
    E[N_i·u(X_i) | X_i = X(φ)]`` is built; unit normals N_i are then
    drawn by Bernoulli sign-flip with probability (1+a(φ))/2 (a_marks
    with |\bar n| < 1 + non-trivial mark variance), or set deterministically
    when ``mark_model="deterministic"``.

    - ``"deterministic"`` (legacy β=1): N_i = u(X_i), |m̄|≡1.
    - ``"marked-constant"`` (new β=1): a(φ)=a_0 const, Bernoulli sign-flip.
            Non-trivial mark noise but field is C^∞.
    - ``"localized-cusp"`` (legacy β=0.5): a(φ) = sqrt(d_S¹(φ,0)/π),
            Hölder-1/2 cusp localized at φ=0 (a(0)=0 ⇒ pure coin flip there).
    - ``"weierstrass"`` (new β=0.5): a(φ) = a_0 + η Σ b^{-mβ}cos(b^m φ),
            genuine multi-scale C^{0,β} roughness, a∈(0,1) a.e.
    """
    if mark_model is None:
        mark_model = LEGACY_MARK_MODEL.get(beta)
        if mark_model is None:
            raise ValueError(f"no legacy mark_model for beta={beta!r}")
    phi = rng.uniform(0.0, 2.0 * math.pi, size=N)
    X = np.stack([np.cos(phi), np.sin(phi)], axis=1)            # (N, 2)
    n_hat = X.copy()                                            # outward unit
    if mark_model == "deterministic":
        N_marks = n_hat
    else:
        if mark_model == "marked-constant":
            a = np.full_like(phi, MARKED_A0)
        elif mark_model == "localized-cusp":
            d_arc = np.minimum(np.abs(phi), 2.0 * math.pi - np.abs(phi))
            a = np.sqrt(d_arc / math.pi)                        # ∈ [0, 1]
        elif mark_model == "weierstrass":
            a = weierstrass_a(phi, beta=beta)
        else:
            raise ValueError(f"unknown mark_model={mark_model!r}")
        p = 0.5 * (1.0 + a)                                     # ∈ (0, 1)
        sign = np.where(rng.uniform(0.0, 1.0, size=N) < p, 1.0, -1.0)
        N_marks = sign[:, None] * n_hat
    return torch.from_numpy(X).to(dtype), torch.from_numpy(N_marks).to(dtype)


# =============================================================================
# BB bandwidth schedule
# =============================================================================

def bandwidths(
    N: int, *, beta: float,
    c_delta: float, c_sigma: float, c_tau: float,
) -> Tuple[float, float, float]:
    """Return (δ_N, σ_N, τ) per the corollary's prescription.

    δ_N = c_δ · N^{−1/(d−1+2c)},   c = min(α, γ)
    σ_N = c_σ · N^{−a*},            a* = β c / ((d−1+2c)(d−1+2β))
    τ   = c_τ · θ_−
    """
    c = min(ALPHA, GAMMA)
    delta_N = c_delta * N ** (-1.0 / (D - 1 + 2 * c))
    a_star = beta * c / ((D - 1 + 2 * c) * (D - 1 + 2 * beta))
    sigma_N = c_sigma * N ** (-a_star)
    tau = c_tau * THETA_MINUS
    return delta_N, sigma_N, tau


def theoretical_slope(beta: float) -> float:
    """Predicted slope of log E_N vs log N from the corollary."""
    c = min(ALPHA, GAMMA)
    r = c / (D - 1 + 2 * c)
    return -(beta ** 2) * r / (D - 1 + 2 * beta)


# =============================================================================
# Single trial
# =============================================================================

def single_trial(
    N: int, beta: float, rng: np.random.Generator,
    *, c_delta: float, c_sigma: float, c_tau: float,
    mark_model: str = None,
    kernel: str = "wendland_c2", dtype: torch.dtype = torch.float64,
    backend: str = "keops",
) -> float:
    """One Monte-Carlo realisation: sample N points, build BB weights, return
    |P̂_{σ_N, δ_N, N} − true_perimeter(β, mark_model)|. The true target
    depends on β AND the mark model because the coherent perimeter
    integrates |m̄|, which is the constant 1 only in the deterministic
    legacy case.

    Wrapped in ``torch.no_grad()`` — the rate-verification estimator
    doesn't need autograd, and the wrapper drops the autograd graph
    that would otherwise keep the (N, N) intermediates alive in the
    naive backend. Independent of the KeOps switch but stacks on top.
    """
    with torch.no_grad():
        X, normals = sample_scenario(N, beta, rng, mark_model=mark_model, dtype=dtype)
        delta_N, sigma_N, tau = bandwidths(
            N, beta=beta, c_delta=c_delta, c_sigma=c_sigma, c_tau=c_tau,
        )
        theta_hat = compute_kde_density(X, delta_N, kernel=kernel, backend=backend)
        chi = chi_tau(theta_hat, tau)
        W = torch.where(theta_hat > 0, chi / theta_hat,
                        torch.zeros_like(theta_hat))
        masses = W / N                                                  # m_i = W_i / N

        angles = torch.atan2(normals[:, 1], normals[:, 0])
        varifold = OrientedPointCloudVarifold(positions=X, angles=angles)
        P_hat = compute_perimeter_coherence(
            varifold, masses, sigma=sigma_N, kernel=kernel, backend=backend,
        )
        return float(abs(P_hat.item() - true_perimeter(beta, mark_model)))


# =============================================================================
# Monte-Carlo loop + fit
# =============================================================================

def _trial_core(packed):
    """Plain trial executor. **Does NOT mutate** ``torch.set_num_threads``
    — the caller controls torch's intra-op pool. In the sequential path
    this preserves the main-process budget (e.g. 4 threads); in the
    parallel path ``_trial_in_subprocess`` caps the subprocess to 1
    thread before delegating here.
    """
    (N, beta, mark_model, child_seed,
     c_delta, c_sigma, c_tau, kernel, backend, dtype) = packed
    rng = np.random.default_rng(child_seed)
    return single_trial(
        N, beta, rng,
        c_delta=c_delta, c_sigma=c_sigma, c_tau=c_tau,
        mark_model=mark_model,
        kernel=kernel, backend=backend, dtype=dtype,
    )


def _trial_in_subprocess(packed):
    """ProcessPoolExecutor entry point: cap torch to 1 intra-op thread so
    N workers × N threads ≤ n_cores (no over-subscription / cache
    thrash), then delegate to ``_trial_core`` for the actual work."""
    import torch as _torch
    _torch.set_num_threads(1)
    return _trial_core(packed)


def _build_scenario_dict(
    beta: float, mark_model: str, N_grid_done: list[int],
    E_N: list[float], std_N: list[float], se_N: list[float],
    rows: list, *, completed: bool,
) -> dict:
    """Snapshot the per-scenario state into a JSON-friendly dict. Used
    both for partial checkpoints (after each N is processed) and for the
    final return value (after the whole N_grid)."""
    d = {
        "beta": beta,
        "mark_model": mark_model,
        "true_perimeter": true_perimeter(beta, mark_model),
        "N": list(N_grid_done),
        "E_N": list(E_N),
        "std_N": list(std_N),
        "se_N": list(se_N),
        "theory_slope": theoretical_slope(beta),
        "_completed": completed,
        "_rows": list(rows),
    }
    n_pts = len(E_N)
    if n_pts >= 2:
        logN = np.log(np.asarray(N_grid_done))
        logE = np.log(np.asarray(E_N))
        slope, intercept = np.polyfit(logN, logE, 1)
        d["fit_slope"] = float(slope)
        d["fit_intercept"] = float(intercept)
        if n_pts > 2:
            pred = slope * logN + intercept
            resid = logE - pred
            s_err = math.sqrt((resid ** 2).sum() / (n_pts - 2))
            sx = math.sqrt(((logN - logN.mean()) ** 2).sum())
            slope_se = float(s_err / sx)
            t_crit = stats.t.ppf(0.975, df=n_pts - 2)
            d["fit_slope_se"] = slope_se
            d["fit_slope_ci"] = [slope - t_crit * slope_se,
                                 slope + t_crit * slope_se]
        else:
            d["fit_slope_se"] = float("nan")
            d["fit_slope_ci"] = [float("nan"), float("nan")]
    return d


def run_one_scenario(
    beta: float, N_grid: list[int], M: int, *,
    c_delta: float, c_sigma: float, c_tau: float,
    seed: int, kernel: str, backend: str, dtype: torch.dtype,
    mark_model: str,
    pool: ProcessPoolExecutor | None = None,
    checkpoint_callback=None,
) -> dict:
    """Run M trials for each N in N_grid and return per-N stats + fit.

    When ``pool`` is given, trials are dispatched in parallel; otherwise
    the loop runs sequentially in the calling process. The deterministic
    per-(beta, N, trial) seed makes both paths bit-reproducible.
    """
    rows = []        # list[(N, trial_idx, error)] for CSV
    E_N = []         # mean over M trials
    std_N = []       # sample std over M trials (per-trial variability)
    se_N = []        # std / √M  (uncertainty of the mean estimate)
    # Mix mark_model into the seed so different scenarios with the same β
    # don't share trial sequences.
    mark_seed = abs(hash(mark_model)) & ((1 << 32) - 1)
    for N in N_grid:
        base = (np.uint64(seed * 1_000_003)
                ^ np.uint64(int(N))
                ^ np.uint64(int(beta * 1e6))
                ^ np.uint64(mark_seed))
        # Per-trial seeds via spawn (independent streams).
        child_seeds = np.random.default_rng(base).integers(
            0, 2**63 - 1, size=M, dtype=np.int64,
        )
        packed = [(N, beta, mark_model, int(child_seeds[k]),
                   c_delta, c_sigma, c_tau, kernel, backend, dtype)
                  for k in range(M)]

        errs = np.empty(M, dtype=np.float64)
        desc = f"β={beta:.2f}/{mark_model:<16s} N={N:>5d}"
        if pool is None:
            # Sequential path — direct ``_trial_core`` call so the main
            # process's torch thread budget (set in main, typically =
            # n_cores) actually applies to the trial's torch ops.
            pbar = tqdm(range(M), desc=desc, leave=False, dynamic_ncols=True)
            for k in pbar:
                errs[k] = _trial_core(packed[k])
                rows.append((N, k, float(errs[k])))
                if (k + 1) % max(1, M // 10) == 0:
                    running = errs[: k + 1]
                    pbar.set_postfix(E=f"{running.mean():.3e}",
                                     std=f"{running.std(ddof=1):.1e}")
        else:
            # Parallel path. ``executor.map`` preserves order so errs[k]
            # corresponds to packed[k] / child_seeds[k] deterministically.
            pbar = tqdm(total=M, desc=desc, leave=False, dynamic_ncols=True)
            for k, err in enumerate(pool.map(_trial_in_subprocess, packed)):
                errs[k] = err
                rows.append((N, k, float(err)))
                pbar.update(1)
                if (k + 1) % max(1, M // 10) == 0:
                    running = errs[: k + 1]
                    pbar.set_postfix(E=f"{running.mean():.3e}",
                                     std=f"{running.std(ddof=1):.1e}")
            pbar.close()

        E_N.append(float(errs.mean()))
        std_N.append(float(errs.std(ddof=1)))
        se_N.append(float(errs.std(ddof=1) / math.sqrt(M)))
        print(f"  β={beta:.2f}/{mark_model:<16s} N={N:>5d}: E_N={E_N[-1]:.4e}  "
              f"std={std_N[-1]:.2e}  se={se_N[-1]:.2e}", flush=True)

        # Partial checkpoint after each N — lets the caller persist the
        # in-progress scenario so a kill mid-N_grid doesn't lose data.
        if checkpoint_callback is not None:
            partial = _build_scenario_dict(
                beta, mark_model, N_grid[: len(E_N)],
                E_N, std_N, se_N, rows, completed=False,
            )
            checkpoint_callback(partial)

    return _build_scenario_dict(
        beta, mark_model, list(N_grid), E_N, std_N, se_N, rows,
        completed=True,
    )


# =============================================================================
# Plot + dump
# =============================================================================

def plot_loglog(scenarios: list[dict], png_path: Path):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    # Color by β, linestyle/marker by mark_model so multiple scenarios per
    # β are visually separated.
    beta_color = {1.0: "C0", 0.5: "C1"}
    mark_style = {
        "deterministic":   {"linestyle": "-",  "marker": "o"},
        "marked-constant": {"linestyle": "--", "marker": "s"},
        "localized-cusp":  {"linestyle": "-",  "marker": "o"},
        "weierstrass":     {"linestyle": "--", "marker": "s"},
    }
    seen_theory = set()
    for s in scenarios:
        Ns = np.asarray(s["N"])
        P_true = float(s["true_perimeter"])
        E_rel = np.asarray(s["E_N"]) / P_true
        std_rel = np.asarray(s["std_N"]) / P_true
        beta = s["beta"]
        mm = s.get("mark_model", LEGACY_MARK_MODEL.get(beta, "?"))
        col = beta_color.get(beta, "C2")
        sty = mark_style.get(mm, {"linestyle": ":", "marker": "x"})

        rows = s.get("_rows", [])
        if rows:
            N_dots = np.array([r[0] for r in rows], dtype=float)
            err_dots = np.array([r[2] for r in rows], dtype=float) / P_true
            ax.scatter(N_dots, err_dots, color=col, alpha=0.10, s=10,
                       edgecolors="none", zorder=1)

        fit_slope = s.get("fit_slope")
        fit_slope_se = s.get("fit_slope_se", float("nan"))
        if fit_slope is None:
            slope_txt = "(N=1, no fit)"
        else:
            slope_txt = rf"slope = {fit_slope:.3f}$\pm${fit_slope_se:.3f}"
        ax.errorbar(
            Ns, E_rel, yerr=std_rel,
            fmt=sty["marker"] + sty["linestyle"], color=col, capsize=4,
            elinewidth=1.5, markeredgewidth=1.5, zorder=5,
            label=(rf"$\beta={beta}$ / {mm}  "
                   rf"($P_{{\rm true}}={P_true:.4f}$, {slope_txt})"),
        )
        # Theoretical reference: one per β (avoid duplicate legend entries).
        N0, E0_rel = Ns[0], E_rel[0]
        th = s["theory_slope"]
        if beta not in seen_theory and len(Ns) >= 2:
            ax.plot(Ns, E0_rel * (Ns / N0) ** th, ":", color=col, alpha=0.55,
                    zorder=4, label=rf"theory slope (β={beta}) = {th:.3f}")
            seen_theory.add(beta)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\mathbb{E}\,|\,\widehat P_{\sigma_N,\delta_N,N} - P_{\rm true}\,|"
                  r"\,/\,P_{\rm true}$")
    ax.set_title("Empirical relative-error convergence vs Corollary "
                 "cor:explicit-rate-with-BB-bandwidth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def write_csv(scenarios: list[dict], csv_path: Path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "mark_model", "N", "trial_idx",
                    "error", "error_relative"])
        for s in scenarios:
            P_true = float(s["true_perimeter"])
            mm = s.get("mark_model", LEGACY_MARK_MODEL.get(s["beta"], "?"))
            for N, k, e in s["_rows"]:
                w.writerow([s["beta"], mm, N, k, e, e / P_true])


def write_json(scenarios: list[dict], json_path: Path, *,
               args: argparse.Namespace, walltime: float):
    out = {
        "args": vars(args),
        "walltime_s": walltime,
        "scenarios": scenarios,
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Numerical verification of Corollary "
                    "cor:explicit-rate-with-BB-bandwidth.",
    )
    parser.add_argument("--n-grid", type=int, nargs="+",
                        default=[100, 1000, 10000, 30000, 100000])
    parser.add_argument("--m-trials", type=int, default=20)
    parser.add_argument("--betas", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--mark-models", type=str, nargs="+", default=None,
                        choices=ALL_MARK_MODELS,
                        help="One per β. Default: legacy pairing "
                             "1.0→deterministic, 0.5→localized-cusp.")
    parser.add_argument("--backend", type=str, default="keops",
                        choices=["naive", "keops"],
                        help="pairwise backend; default 'keops' (memory-light + faster on CPU).")
    parser.add_argument("--threads", type=int, default=0,
                        help="torch intra-op threads in workers; 0 = all available cores")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes; default 1 (sequential). KeOps backend "
                             "is memory-cheap so workers>1 is also fine, but the default favours "
                             "predictable single-process behaviour for reproducibility.")
    parser.add_argument("--c-delta", type=float, default=1.0,
                        help="δ_N = c_δ · N^{−1/(d−1+2c)} (default 1.0)")
    parser.add_argument("--c-sigma", type=float, default=1.0,
                        help="σ_N = c_σ · N^{−a*} (default 1.0)")
    parser.add_argument("--c-tau", type=float, default=0.5,
                        help="τ = c_τ · θ_− (default 0.5; must be ≤ 1)")
    parser.add_argument("--kernel", type=str, default="wendland_c2",
                        choices=["wendland_c2", "biweight", "epanechnikov"])
    parser.add_argument("--dtype", type=str, default="float64",
                        choices=["float32", "float64"],
                        help="numerical precision. FP32 is ~30× faster on Tesla T4 "
                             "(weak FP64), enough precision for E ~ 1e-2.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str,
                        default="scripts/outputs/experiments/rate_verification")
    parser.add_argument("--replot-only", type=Path, default=None,
                        help="既存 JSON を読んでプロットのみ再生成。実験は走らない。")
    args = parser.parse_args()

    if args.replot_only is not None:
        data = json.loads(args.replot_only.read_text())
        scenarios = data["scenarios"]
        png_path = args.replot_only.with_name("rate_verification.png")
        plot_loglog(scenarios, png_path)
        print(f"Saved: {png_path.resolve()}")

        # Augment existing per-trial CSV with `error_relative` column if it
        # already has only the legacy 4 columns. Reads true_perimeter from
        # the JSON's scenarios.
        csv_path = args.replot_only.with_name("rate_verification.csv")
        if csv_path.exists():
            P_by_beta = {float(s["beta"]): float(s["true_perimeter"])
                         for s in scenarios}
            with open(csv_path) as f:
                rdr = csv.reader(f)
                rows = list(rdr)
            header = rows[0]
            if header == ["beta", "N", "trial_idx", "error"]:
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["beta", "N", "trial_idx", "error",
                                "error_relative"])
                    for beta_s, N_s, k_s, e_s in rows[1:]:
                        beta = float(beta_s)
                        e = float(e_s)
                        w.writerow([beta_s, N_s, k_s, e_s, e / P_by_beta[beta]])
                print(f"Augmented: {csv_path.resolve()} (+error_relative)")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cores = os.cpu_count() or 4
    n_workers = args.workers if args.workers > 0 else n_cores
    # Per-worker thread count: when running n_workers processes, give each
    # 1 thread to avoid over-subscribing (n_workers * threads > n_cores
    # would thrash). When sequential (--workers 1), let torch use the
    # full thread budget.
    if n_workers == 1:
        torch.set_num_threads(args.threads if args.threads > 0 else n_cores)
    else:
        torch.set_num_threads(1)

    # Pair each β with a mark_model; legacy default if not given.
    if args.mark_models is None:
        try:
            mark_models = [LEGACY_MARK_MODEL[b] for b in args.betas]
        except KeyError as e:
            raise SystemExit(
                f"no legacy mark_model for β={e.args[0]!r}; "
                "pass --mark-models explicitly."
            )
    else:
        if len(args.mark_models) != len(args.betas):
            raise SystemExit(
                f"--mark-models has {len(args.mark_models)} entries; "
                f"expected {len(args.betas)} (one per β)."
            )
        mark_models = args.mark_models

    print("=== Perimeter rate verification (torch, CPU) ===")
    print(f"  N grid : {args.n_grid}")
    print(f"  M      : {args.m_trials}")
    print(f"  scenarios:")
    for b, mm in zip(args.betas, mark_models):
        print(f"    β={b}  mark_model={mm}")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    print(f"  c_δ,c_σ,c_τ : {args.c_delta}, {args.c_sigma}, {args.c_tau}")
    print(f"  kernel : {args.kernel}")
    print(f"  backend: {args.backend}")
    print(f"  dtype  : {args.dtype}")
    print(f"  workers: {n_workers}  (torch threads/worker: {torch.get_num_threads()})")
    print(f"  output : {out_dir}")

    t0 = time.perf_counter()
    pool_ctx = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    json_path = out_dir / "rate_verification.json"
    csv_path = out_dir / "rate_verification.csv"
    png_path = out_dir / "rate_verification.png"

    def _checkpoint(partial_scenario: dict) -> None:
        """Write a JSON snapshot of completed scenarios + the in-progress
        one. Called after each N is processed inside run_one_scenario."""
        snapshot = scenarios + [partial_scenario]
        write_json(snapshot, json_path, args=args,
                   walltime=time.perf_counter() - t0)

    try:
        scenarios = []
        for beta, mark_model in zip(args.betas, mark_models):
            print(f"\n--- scenario β = {beta}  mark_model = {mark_model} ---")
            done = run_one_scenario(
                beta, args.n_grid, args.m_trials,
                c_delta=args.c_delta, c_sigma=args.c_sigma, c_tau=args.c_tau,
                seed=args.seed, kernel=args.kernel, backend=args.backend,
                dtype=dtype, mark_model=mark_model, pool=pool_ctx,
                checkpoint_callback=_checkpoint,
            )
            scenarios.append(done)
            # Also overwrite the checkpoint without any in-progress dict
            # so the saved file matches the final state of completed work.
            write_json(scenarios, json_path, args=args,
                       walltime=time.perf_counter() - t0)
    finally:
        if pool_ctx is not None:
            pool_ctx.shutdown(wait=True)
    walltime = time.perf_counter() - t0
    print(f"\nTotal wall: {walltime:.1f}s")

    plot_loglog(scenarios, png_path)
    write_csv(scenarios, csv_path)
    write_json(scenarios, json_path, args=args, walltime=walltime)
    print(f"\nSaved:\n  {png_path}\n  {json_path}\n  {csv_path}")

    print("\n--- fitted vs theoretical slopes ---")
    for s in scenarios:
        mm = s.get("mark_model", "?")
        print(f"  β={s['beta']:.2f} / {mm:<16s}: fit = {s['fit_slope']:+.4f} "
              f"± {s['fit_slope_se']:.4f}  "
              f"(theory {s['theory_slope']:+.4f})")


if __name__ == "__main__":
    main()
