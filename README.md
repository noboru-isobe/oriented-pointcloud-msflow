# Mullins–Sekerka Flow via Oriented Point Cloud Varifolds

A mesh-free solver for **Mullins–Sekerka flow** — the Wasserstein gradient
flow of perimeter — discretised on an oriented point cloud varifold and
advanced by a minimizing-movements scheme with a BEM-based linearised
Wasserstein cost.

## Highlights

- **BEM-based linearised Wasserstein**: the transport cost is computed with a
  Boundary Element Method (with endpoint collocation) instead of Sinkhorn.
- **Volume-preserving parametrisation**: the constraint $\sum_i s_i w_i = 0$ is
  satisfied automatically via the orthogonal complement $s = Qy$.
- **Grid-free**: computed from boundary points only — no phase field, no
  winding number.
- **Coherence-based perimeter**: hidden-boundary–aware perimeter estimator.
- **Buet–Leonardi–Masnou mass**: per-point mass from a KDE density and a
  cutoff function.
- **Optional KeOps backend**: matrix-free pairwise kernels, O(N) memory,
  beneficial at N ≳ 1000.

## Installation

```bash
uv sync
```

## Quickstart

Command line (`run.py`, select the shape by argument; mp4 is written with an
in-process PyAV encoder):

```bash
# Two ellipses merging (showcase video)
python run.py --shape two-ellipses --n-points 64 --gap 0.1 --time-step 1e-5 \
    --optimizer-method trust-ncg --gtol 1e-8 --redistribute --mass-knn-k 9 \
    --remove-dead --dead-threshold 0.15 --n-steps 5000 --output out.mp4

# Flower rounding (no video, just final perimeter/volume)
python run.py --shape flower --n-points 128 --n-steps 2000 --no-video

# Quick circle / ellipse sanity check
python run.py --shape circle --n-points 64 --n-steps 10 --no-video
```

Python API:

```python
from src.torch.shapes.generator import generate_oriented_flower
from src.torch.solver.mm_step import MMConfig
from src.torch.solver.mm_solver import MMSolver

varifold = generate_oriented_flower(n_points=128)
config = MMConfig(
    time_step=0.001,
    bem_method="point",      # collocation BEM
    bem_epsilon_scale=0.1,   # epsilon for the point method
)
history = MMSolver(config).solve(varifold, n_steps=100)
```

## Repository layout

```
run.py                         # generic entry point (--shape selects the shape)
results/                       # published artifacts (regenerable)
├── rate_verification.{png,csv,json}  # Corollary verification (N=10²..10⁶)
├── two_ellipses.mp4           # showcase: two ellipses merging
└── flower.mp4                 # showcase: flower rounding

src/torch/
├── oriented_varifold/
│   ├── state.py               # OrientedPointCloudVarifold dataclass
│   └── mass.py                # Buet–Leonardi–Masnou mass (KDE + χ_τ)
├── perimeter/
│   ├── coherence_perimeter.py # coherence-based perimeter
│   └── angle_constraint.py    # weak-form angle constraint (A Δα = B s)
├── transport/
│   └── bem_wasserstein.py     # BEM-based linearised Wasserstein
├── solver/
│   ├── mm_step.py             # single MM step + MMConfig
│   ├── mm_solver.py           # full evolution loop
│   ├── redistribute.py        # tangential point redistribution
│   └── remove_dead_points.py  # effective-mass dead-point removal
├── shapes/generator.py        # shape factory (circle/ellipse/flower/two-ellipses)
├── math_utils/
│   ├── linalg.py              # TruncatedSVD
│   ├── pairwise.py            # pairwise kernels (naive)
│   └── keops_pairwise.py      # KeOps backend (matrix-free, N≳1000)
└── visualization/animation.py # in-process PyAV mp4 encoder

scripts/
├── experiments/
│   └── perimeter_rate_verification.py  # Corollary rate verification
└── examples/
    ├── run_two_ellipses_batch.py       # two-ellipses showcase (with diagnostics)
    └── animate_flower.py               # flower showcase (with diagnostics)

tests/                         # pytest suite (uv run pytest tests/)
```

## Mathematical background

### MM scheme

At each time step, minimise

$$\mu^n = \underset{y,\,\delta\theta}{\mathrm{argmin}}
\left[ \hat{P}(\mu) + W_{\mathrm{lin}}(s, \delta\theta) \right]$$

where:
- $\hat{P}(\mu)$: coherence-based perimeter
- $W_{\mathrm{lin}}(s, \delta\theta)$: BEM-based linearised Wasserstein
  (includes both positional displacement and angle change)
- $s = Qy$: normal displacement satisfying volume conservation
- $\delta\theta$: angle change from the previous step

### Buet–Leonardi–Masnou mass

Estimate each point's "mass" (its boundary contribution) from the cloud:

$$\theta_{\delta,N}(x_i) = \frac{1}{N C_\eta \delta}
\sum_j \eta\!\left(\frac{|x_i - x_j|}{\delta}\right)$$

$$m_i = \frac{1}{N}\,\frac{\chi_\tau(\theta)}{\theta}$$

where:
- $\eta$: compactly-supported kernel (Wendland C² recommended)
- $\delta$: kernel bandwidth
- $\chi_\tau(t) = \mathrm{ReLU}(2t/\tau - 1) - \mathrm{ReLU}(2t/\tau - 2)$: cutoff
- $\tau$: cutoff threshold

Kernel choices:

| Name | $\eta(u)$ | $C_\eta$ | Notes |
|------|-----------|----------|-------|
| Wendland C² (recommended) | $(1-u)_+^4(4u+1)$ | $2/3$ | smooth + compact, AD-friendly |
| Biweight | $(1-u^2)_+^2$ | $16/15$ | simple |
| Epanechnikov | $(1-u^2)_+$ | $4/3$ | classic |

### BEM-based linearised Wasserstein

Linearise the Wasserstein distance in the small normal displacement $s_i$ and
angle change $\delta\theta_i$.

**Endpoint collocation (K = 3 points/segment).** Place K collocation points on
each segment $i$ to capture the effect of the angle change:

$$p_{ik} = x_i + r_k \cdot t_i, \qquad r_k \in \{-m_i/2,\ 0,\ +m_i/2\}$$

velocity at each collocation point:

$$V_{ik} = \frac{q_i}{h}\,(s_i - r_k \cdot \delta\theta_i)$$

discretised $W_{\mathrm{lin}}$:

$$W_{\mathrm{lin}} = \frac{h}{2} \sum_{i,k} w_{ik}\,\phi_{ik}\,V_{ik}$$

where $w_{ik} = m_i/K$ are quadrature weights, $\phi_{ik}$ is the BEM potential
at the collocation point, and $q_i$ is the coherence.

**Why endpoint collocation is needed**: with the centre point only (K = 1) the
$\delta\theta$ term vanishes ($r = 0 \Rightarrow V_i = q_i s_i / h$); K ≥ 2 lets
the angle change enter the transport cost.

Neumann problem:

$$\begin{cases}
\Delta\phi = 0 & \text{in } \Omega \\
-\partial_\nu \phi = V & \text{on } \Gamma
\end{cases}$$

BEM discretisation:
- Single layer: $S_{ij} = \int G(x_i, y)\, ds_j$
- Adjoint double layer: $K^*_{ij} = \int \partial_{n_x} G(x_i, y)\, ds_j$
- Linear system: $A\lambda = -V$ with $A = -\tfrac{1}{2}I + K^*$

### Coherence-based perimeter

Estimate the perimeter from boundary points only (hidden-boundary aware):

$$\hat{P}_\sigma(\mu) = \sum_i m_i q_i$$

where the coherence $q_i$ measures local normal alignment:

$$U_\sigma(x_i) = \sum_j m_j \rho_\sigma(x_i - x_j), \qquad
V_\sigma(x_i) = \sum_j m_j \rho_\sigma(x_i - x_j)\, n_j$$

$$q_i = \frac{|V_\sigma(x_i)|}{U_\sigma(x_i)} \in [0,1]$$

- Ordinary boundary: aligned normals → $q \approx 1$ → counted
- Hidden boundary: opposing normals cancel → $q \approx 0$ → not counted

### Volume conservation

The constraint $\sum_i s_i w_i = 0$ (with $w_i = q_i^2 m_i$) is expressed in the
orthogonal complement: $Q \in \mathbb{R}^{N \times (N-1)}$ spans the complement
of the constraint vector, so $s = Qy$ satisfies it for any
$y \in \mathbb{R}^{N-1}$.

## BEM method (point collocation)

The solver uses a regularised point-collocation BEM, which is stable for all
tested shapes (circle / ellipse / star / two-ellipses, 32–1024 points).

Parameters: `bem_epsilon_scale=0.1` (regularisation, stable for all shapes),
`bem_n_endpoints=3` (K = 3 collocation points per segment makes the Hessian
positive semi-definite and lets the angle change enter the transport cost).

## Tests

```bash
uv run pytest tests/
uv run pytest tests/test_bem_wasserstein.py -v          # BEM Wasserstein only
uv run pytest tests/test_bem_wasserstein.py -v -k point # point method only
```

## Reproducing the results (`results/`)

Each artifact in `results/` is regenerated by:

```bash
# results/rate_verification.{png,csv,json} — numerical verification of the Corollary
#   β=1 (deterministic) and β=0.5 (localized cusp), N=10²..10⁶, M=10 trials
uv run python scripts/experiments/perimeter_rate_verification.py \
    --betas 1.0 0.5 --mark-models deterministic localized-cusp \
    --n-grid 100 1000 10000 30000 100000 1000000 --m-trials 10 --dtype float32

# results/{two_ellipses.mp4,two_ellipses_data.json} — two-ellipses merging
# showcase. Uses scripts/examples/run_two_ellipses_batch.py for the
# diagnostic-rich 2×2 layout in the mp4 plus the data.json with the full
# per-step (perimeter / volume) history. run.py with the same flags
# reproduces the same trajectory but with a simpler default layout.
uv run python scripts/examples/run_two_ellipses_batch.py \
    --n-steps 5000 --n-per-ellipse 64 --mass-knn-k 9 --gap 0.1 \
    --optimizer-method trust-ncg --gtol 1e-8 --redistribute \
    --remove-dead --dead-threshold 0.15 --device auto --skip-frames 10

# results/{flower.mp4,flower_data.json} — flower rounding showcase
uv run python scripts/examples/animate_flower.py \
    --petals 5 --n-points 128 --time-step 1e-5 \
    --bem-n-endpoints 3 --n-steps 5000 --save-history flower_history.npz

# results/ablation/ — cumulative ablation (coherence q_i, redistribute, dead-point removal)
bash scripts/experiments/run_ablation.sh             # 4 cells × 5000 steps on CUDA
uv run python scripts/experiments/ablation_summary.py  # aggregate + plots
```

`scripts/examples/{run_two_ellipses_batch,animate_flower}.py` are shape-specific
drivers that produce the same showcases with extra diagnostic plots.

### Ablation: which mechanisms matter?

`results/ablation/` contains a 5-cell ablation on the two-ellipses merging
scenario: cells 1–4 switch the mechanisms on cumulatively, cell 5 is the
counterpart of cell 4 with coherence taken back out.

| cell | coherence $q_i$ | redistribute | dead-point removal | outcome |
|------|-----------------|--------------|--------------------|---------|
| 1. none      | ✗ (q≡1) | ✗ | ✗ | **early-LU break** at step 9 |
| 2. +coh      | ✓        | ✗ | ✗ | **early-LU break** at step 12 |
| 3. +coh +redist | ✓     | ✓ | ✗ | **silent divergence** (positions explode to ‖x‖∞ ≈ 484 over 5000 steps; the BB-corrected perimeter stays bounded — even $P_\text{err} \approx 0.023$ vs. the single-disk asymptote — because diverged points get near-zero mass, masking the failure. Only the trajectory geometry reveals it.) |
| 4. full      | ✓        | ✓ | ✓ | converges ($P_\text{err} \approx 0.027$ vs. the single-disk asymptote $P_\star = 2\sqrt{\pi V_0} \approx 5.572$, volume err ≈ 1.1%) |
| 5. full − coh | ✗ (q≡1) | ✓ | ✓ | **wrong asymptote**: the trajectory completes 5000 steps but the perimeter stalls at $P_f \approx 6.19$ ($P_\text{err} \approx 0.62$, about 23× cell 4) and circularity stays around 0.8. The snapshot grid shows a residual seam down the merger axis — without $q \to 0$ at opposing normals the hidden boundary can't cancel, so the perimeter functional treats the merged shape as if it still had an interior interface. Coherence is what closes the gap. |

`ablation_comparison.png` (trajectories + summary table) and
`ablation_snapshots.png` (4 cells × 5 time snapshots, points coloured by
the per-point perimeter contribution $m_i q_i$) make the ablation
visually obvious.

## References

- Chambolle, A., & Laux, T. (2021). *Mullins–Sekerka as the Wasserstein
  gradient flow of the perimeter.*
- Buet, B., Leonardi, G. P., & Masnou, S. (2017). *A varifold approach to
  surface approximation.*

## Citation

```bibtex
@software{isobe_msflow_2026,
  author = {Isobe, Noboru},
  title  = {Mullins--Sekerka Flow via Oriented Point Cloud Varifolds},
  year   = {2026},
  url    = {https://github.com/noboru-isobe/MSflow_via_oriented_point_cloud_varifolds}
}
```

## License

MIT License — see [LICENSE](LICENSE).
