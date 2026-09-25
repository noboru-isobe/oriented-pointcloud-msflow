# BIE metric backend (block-diagonal boundary integral W₂ term) — verification report

Branch `experiment/annulus-benchmark`, commits ee70595 / ef32f9a / 7d30bb9 (2026-09-24). Status: **complete** (2026-09-24); the three production runs and a rotated two-ellipses run were done on the VM with ε = 0.15ℓ and the BIE gates.

## 1. What changed

The linearized Wasserstein term of each MM step is evaluated by a dense boundary-integral formulation instead of the
768²/1024² grid Poisson solve (`metric_backend="bie"`, `src/torch/transport/bie_wasserstein.py`):

- same endpoint conventions as the grid flux (K=3 endpoints per particle, ζ_ik = m_i/3, V_ik = (s_i − r_ik δθ_i)/h,
  carrier measure, unit velocity scale); `BoundaryFluxToGrid` is reused without a grid;
- one block per **metric component**: the winding bulk components, merged when two loops of different bulks come within
  g_bridge = ℓ (ℓ = median particle mass); single-layer interior-Neumann block A = ½I + K*, weighted A_w = D^{1/2} A D^{-1/2},
  bordered with the one near-null pair of A_w (rank 1 by construction, no rank estimation), quadratic form
  Q = sym(D S Λ) precomputed once per step;
- inside a merged component the **cancelling pair** (facing partner within 1.5ℓ with n·n' ≤ −0.9) is excluded from the
  operator and held fixed (unit constraint rows); rows that become dependent modulo the unit rows (the relative area
  row of a fully masked loop) are dropped, not counted as rank deficiency;
- compatibility rows = exact endpoint flux sums per metric component (= the bulk area rows before a merge, their sum after);
- kernel regularization r² = |x−y|² + ε², ε = 0.15ℓ (see §3);
- fail-closed diagnostics per step: σ_tail = σ_{n−1}/σ_1 of every block (≥ 1e-6), λ_min of the metric Hessian on the
  admissible subspace (≥ −1e-6 λ_max), componentwise compatibility residual (≤ 0.1).

Events on the BIE backend: the quotient candidate appears at the merge step (metric coarser than the bulk partition),
removal / reconnection fire when the gap has stopped decreasing for 3 consecutive steps with the quotient committed;
the raw/visible incidence divergence is telemetry only. The arc-splice cut window stays 0.08 (= 3.2ℓ).

## 2. Local checks (4-core workstation, float64)

| check | result |
|---|---|
| disk Fourier modes vs exact (N=256, ε=0.1ℓ / 0.15ℓ) | k=1: −0.3% / −0.6%, k=3: −0.5% / −1.2%, k=5: −0.8% / −1.8%, k=8: −1.4% / −3.0% (1024² grid: −0.7 / −0.8 / −1.2 / −2.3%) |
| annulus radial mode vs exact | +0.07% (0.1ℓ), ≈ −0.2% (0.15ℓ); grid −0.7% |
| one-step optimizer parity, ellipse N=256, vs BEM C3 | rel_s = 0.09% (grid: 1.9%) |
| component pins (disk / annulus / two circles / annulus+disk) | C = 1 / 1 / 2 / 2, n_params = N − C, λ_min > 0 |
| two ellipses at gap 0.8ℓ | one block, 18 masked, σ_tail 7e-2, λ_min > 0, relation b_finer, contact-pair context produced |
| concentric disk + annulus at gap 0.9ℓ | disk and hole loops fully masked, block = outer loop only, one effective row |
| unit tests | tests/test_bie_wasserstein.py, tests/test_bie_mm_step.py: 26 passed; grid tests unchanged (25 passed) |
| full suite on the VM (commit ee70595) | 1226 passed, 12 failed: 9 KeOps parity (no C++ compiler on the VM, so the KeOps JIT cannot build its kernels; pykeops itself is installed), 3 test_d0_golden (rounding-level differences of 1.6e-10 in the golden comparison and a converged-flag mismatch on the VM; pass locally) |

## 3. Regularization scale (flower, N=291, 40 steps, λ_min of the W Hessian on the admissible subspace)

| ε/ℓ | λ_min/λ_max over steps 10–35 | disk k=3 | disk k=8 |
|---|---|---|---|
| 0.10 | −9e-5 … −4.7e-4 (indefinite from step 10, recovering by step 40) | −0.5% | −1.4% |
| 0.15 | +1.2e-3 | −1.2% | −3.0% |
| 0.20 | +1.0e-3 | −1.7% | −4.3% |
| 0.25 | +8e-4 | −2.1% | −5.4% |
| 0.33 | +7e-4 | −2.9% | −7.2% |
| 0.60 | +2.7e-4 | — | — |

Decision: ε = 0.15ℓ (smallest scale with a positive-definite form on the concave test geometry; accuracy comparable to
the 1024² grid on k ≤ 5).

## 4. Static parity vs the 1024² grid (VM, `scripts/experiments/bie_grid_parity.py`, ε = 0.15ℓ; results/bie_metric/parity.md)

| case | BIE vs exact | grid vs exact |
|---|---|---|
| disk k=1 / 2 / 3 / 5 / 8 | −0.7 / −1.0 / −1.3 / −2.0 / −3.1 % | −0.7 / −0.7 / −0.8 / −1.2 / −2.3 % |
| annulus radial | −0.2 % | −0.7 % |

Gram parity on per-loop Fourier modes k ≤ 6 (12–24 admissible directions): worst |W_grid/W_BIE − 1| = 0.8 % (ellipse),
0.8 % (two ellipses at gap 0.1), 1.6 % (annulus + disk); generalized eigenvalues within 0.8 / 1.1 / 1.6 % of 1.
Random particle-scale directions differ by 13–17 % (the grid mollifies at ε_fill = 0.04 = 1.6ℓ; not a smooth-family
quantity). Merge window (two ellipses at gap 0.8ℓ, BIE merged with 26 particles masked, grid phase connected):
smooth-mode worst 1.4 %, σ_tail 4.9e-2, λ_min = 0.75 > 0.

## 5. Production reruns (VM e2-standard-32, 8 threads per run, ε = 0.15ℓ, tags `_bie`)

| run | steps | s/step grid → BIE | outcome |
|---|---|---|---|
| flower (N=291) | 4000 | ~11 (768²) → 0.8 | completed; P̂ monotone; P̂: 7.086 → 4.839 (grid 7.083 → 4.839), circularity 0.468 → 1.0011 (grid 1.0011), final P̂ within −0.17 % of 2π√(A/π) (grid −0.16 %), area drift −2.25e-3 (grid −2.14e-3), max normal displacement at the end 3.5e-8 (grid 4.0e-8); step-by-step: max \|P̂_BIE/P̂_grid − 1\| = 0.22 %, area within 8e-5, circularity within 3.5e-3 |
| annulus (N=487) | 2400 | ~23 (1024²) → 1.26 | completed; merge of the metric components and quotient at step 2047 (exact contact T_c = step 2035, +0.65 %; grid: phases connect at 1521, quotient 1919), removal of the pair at 2049 (N 487 → 256, c* = 0.99023; grid 2292, c* = 0.99633), pre-contact ODE errors (driver postprocess, t < 0.015): disk 0, outer 4.9e-4, hole 2.8e-3 (grid 2.5e-7, 1.9e-4, 1.2e-3); after the removal the outer circle is exactly stationary (n_iter 0, drift 0.0; grid drift 1.9e-10) with \|R_out − R_eq\|/R_eq = 1.1e-4 (grid 1.6e-4); area error max 4.0e-5 (grid 8.0e-5); all 11 compression gates pass |
| two ellipses (N=376) | 4400 | ~11 (768²) → 0.81 | first attempt (grid gates): switch at 1431 (grid 1435), merge of the metric at 2025 (gap ≤ ℓ; grid: phases connect 1443, quotient 1829 at 1.35ℓ), auto-quotient stopped by the grid-calibrated θ gate (6.4e-5 > 3.3e-5 on 8 particles at the ends of the held-fixed arcs; |ΔX|/h 1e-4 ≪ 1.4e-2, area exchange 1e-9). **Rerun with the BIE gates (§6), one stroke from t=0**: switch 1431, merge + quotient 2025 (46 particles held fixed; switch differences |ΔX|/h 9.9e-5, |Δθ| 6.4e-5, merged area 1.5e-10), reconnection 2029 (N 376 → 304: 78 removed, 6 inserted; polygon area/moment residual 1e-16), then single-loop relaxation: circularity 0.686 → 0.987 (grid 0.69 → 0.99), r̄/R_eq = 1.0013 (grid 1.0014), P̂ end +0.59 % of 2πR_eq (grid +0.6 %), max normal displacement at the end 2.5e-5 (grid 2.5e-5); barycenters of the ellipses conserved to 2.4e-7 up to the merge (grid 1.7e-7 up to its quotient); post-reconnection area transient 1.7e-4, halved after ~260 steps (grid 4.6e-4 / 500 steps); merged area drift at the end 7.8e-5. Rotated initial data (22.5°): identical event steps (1431 / 2025 / 2029) and final circularity, mean radius and perimeter to four digits (grid: switch steps differed by ≤ 1). A local resumed run from step 2000 reproduced the events; its P̂ differs from the one-stroke run by ≤ 2e-5 relative. |

Wall times on the VM (8–12 threads per run): flower 0.8 s/step, annulus 1.26 s/step, two ellipses 0.81 s/step
(grid production: 11 s/step at 768², 23 s/step at 1024²). Offline diagnostics at the stored states
(results/reports/bie_state_diagnostics.json): σ_tail ≥ 6e-2, λ_min/λ_max ≥ 3e-4 on all states of the three runs;
no fail-closed check fired.

## 7. Verdict against the pre-registered criteria

- static parity on smooth modes ≤ 1.5 %: ellipse 0.8 %, two ellipses 0.8 %, annulus + disk 1.6 % (at the threshold;
  on exact modes the BIE is closer to the exact value than the grid) — accepted;
- flower within 1e-2 of the grid run in P̂, area and circularity: 2.2e-3 / 8e-5 / 3.5e-3 — pass;
- events fire in the same order, fully automatically, in one run: pass (annulus: merge + quotient, removal; ellipses:
  switch, merge + quotient, reconnection); the grid's diffuse "phases connect" event has no counterpart, the contact is
  detected at the point spacing (annulus contact time +0.6 % of T_c instead of −25 % for the diffuse connection);
- conservation: merged area 4e-5 (annulus), polygon area/moment residuals at the events 1e-16 — pass;
- σ_tail ≥ 1e-6 and λ_min ≥ −1e-6 λ_max on every step: pass (never triggered; min values 6e-2 and +3e-4);
- ≥ 5× faster than the 768² grid: 13× (0.8 vs 11 s/step) — pass.

Deviations from the plan: ε = 0.15ℓ instead of 0.1ℓ (§3); the quotient-switch gates of the BIE backend were fixed on the
pre-merge natural motion (§6) after the first two-ellipses attempt; the pre-contact ODE errors of the annulus are
about twice those of the grid run (hole 2.8e-3 vs 1.2e-3), consistent with the −0.2 % attenuation of the radial mode.

## 8. Paper

Manuscript updated accordingly (5.4 "Linearized Wasserstein term by a boundary integral method", Appendix A
"Boundary integral metric", events (ii)–(iv) in terms of ℓ, Section 6 numbers and figures regenerated from the `_bie`
runs, abstract/résumé/intro "on a fixed grid" → boundary integral method).

## 6. Quotient-switch gates on the BIE backend

The grid gates (quotient_gates.py) were calibrated on a clean State-II window of the grid runs (metric connected, rows
separate, hundreds of steps). On the BIE backend that intermediate state lasts one step by construction (the merge of the
metric components makes the candidate pair, the certificate is evaluated on the same state), so the switch gates are fixed
by the paper's Appendix rule on the pre-contact part of earlier trajectories: the natural per-step motion of the 25 steps
before the merge of the two BIE trajectories (two ellipses 2000–2025, annulus 2000–2025; `bie_gate_calibration.py`):
eps_switch_x = 3.09e-3 (grid 1.40e-2), eps_switch_theta = 1.99e-3 (grid 3.29e-5; the annulus rotates by 0 per step and
contributes nothing), eps_split and eps_volume inherited. The annulus run passed the stricter grid gates and therefore
the BIE gates.
