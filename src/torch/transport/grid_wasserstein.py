"""Grid metric G2: the linearized Wasserstein quadratic form on the
fixed Eulerian grid.

    W_grid(s, dtheta) = (1 / 2 dt) < delta_rho, L_rho0^dagger delta_rho >

with delta_rho = B(s, dtheta) the boundary-flux scatter (G1) and
L_rho0 the frozen weighted-Poisson operator (G0) built from the
current-to-phase filling of the PRE-STEP cloud. Same call signature as
BEMWasserstein; torchmin's outer optimizer never changes.

Autograd contract:
- forward solves are FAIL-CLOSED (componentwise compatibility +
  inactive-mass guards): the caller must evaluate on admissible
  directions (a constraint basis whose component fluxes vanish).
- backward cotangents are generically incompatible: the adjoint of the
  projected inverse is L^dagger P_comp, so backward projects first and
  never rejects (solve_projected). Both faces are custom Functions
  whose backward re-applies the projected solve -- double backward is
  therefore another solve, exactly like the analytic-W_lin precedent.
- all FFT stays inside no_grad solver internals.

Setup gates (G1.1, enforced here): valid phase reconstruction (no
dropped components, projection and volume within tolerance), identical
phase/Poisson grid configs, phase component count == operator
nullspace count, PCG convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch

from .boundary_flux_grid import BoundaryFluxPolicies, BoundaryFluxToGrid
from .phase_grid import (
    CurrentToPhase,
    PhaseGrid,
    PhaseGridConfig,
    assert_grid_configs_consistent,
    validate_phase_reconstruction,
)
from .weighted_poisson import (
    WeightedPoissonConfig,
    WeightedPoissonOperator,
    WeightedPoissonSolveError,
)

_T = torch.Tensor


class _ProjectedPoissonSolve(torch.autograd.Function):
    """L^dagger P_comp with implicit self-adjoint backward."""

    @staticmethod
    def forward(ctx, rhs: _T, op: WeightedPoissonOperator) -> _T:
        phi, info = op.solve_projected(rhs.detach())
        if not info.converged:
            raise WeightedPoissonSolveError(info)
        ctx.op = op
        return phi

    @staticmethod
    def backward(ctx, grad_phi: _T):
        return _ProjectedPoissonSolve.apply(grad_phi, ctx.op), None


class SymmetricPoissonSolve(torch.autograd.Function):
    """Fail-closed forward (public solve), projected backward."""

    @staticmethod
    def forward(ctx, rhs: _T, op: WeightedPoissonOperator) -> _T:
        phi, info = op.solve(rhs.detach())
        if not info.converged:
            raise WeightedPoissonSolveError(info)
        ctx.op = op
        return phi

    @staticmethod
    def backward(ctx, grad_phi: _T):
        return _ProjectedPoissonSolve.apply(grad_phi, ctx.op), None


class _GradReplay(torch.autograd.Function):
    """Value = the ALREADY-COMPUTED phi (zero solves); Jacobian =
    L^dagger P_comp (one projected solve per cotangent). The node that
    makes the analytic quadratic's first backward free while keeping
    double backward exact."""

    @staticmethod
    def forward(ctx, rhs: _T, phi: _T, op: WeightedPoissonOperator) -> _T:
        ctx.op = op
        return phi

    @staticmethod
    def backward(ctx, grad_phi: _T):
        return _ProjectedPoissonSolve.apply(grad_phi, ctx.op), None, None


class GridQuadraticForm(torch.autograd.Function):
    """Analytic Q(drho) = <drho, L^dagger drho> (G3a reviewer item 2).

    The generic composition (SymmetricPoissonSolve + inner) makes
    autograd re-solve for L^dagger drho in the first backward even
    though the forward already computed it. Here the gradient is
    closed-form by symmetry, dQ/ddrho = 2 phi, and reuses the forward
    solution: objective + gradient = ONE fail-closed solve; each HVP =
    ONE projected solve (through _GradReplay). Same pattern as the
    audited BEM analytic-W_lin speedup.
    """

    @staticmethod
    def forward(ctx, drho: _T, op: WeightedPoissonOperator) -> _T:
        phi, info = op.solve(drho.detach())
        if not info.converged:
            raise WeightedPoissonSolveError(info)
        ctx.op = op
        ctx.save_for_backward(drho, phi)
        return op.inner(drho, phi)

    @staticmethod
    def backward(ctx, grad_out: _T):
        drho, phi = ctx.saved_tensors
        # Euclidean (autograd-convention) gradient of the MEASURE-
        # weighted form <d, A d> = cell_area * d^T A d is
        # 2 * cell_area * A d = 2 * cell_area * phi. (The projected
        # solve map A = P L^+ P is Euclidean-symmetric, so the
        # _GradReplay Jacobian needs no extra measure factor.)
        grad = (2.0 * ctx.op.cell_area) * grad_out \
            * _GradReplay.apply(drho, phi, ctx.op)
        return grad, None


@dataclass
class GridMetricConfig:
    phase: PhaseGridConfig = None            # type: ignore
    poisson: WeightedPoissonConfig = None    # type: ignore
    # G3a: "analytic" reuses the forward phi for the first gradient
    # (objective+gradient = ONE solve, HVP = ONE solve); "generic" is
    # the G2a composition kept as the parity/audit reference (its first
    # backward re-solves for what the forward already computed).
    quadratic_path: str = "analytic"
    # Soft-bridge merger window (2026-08-07): which partition of the
    # active set defines the compatibility/projection subspace.
    # "operating" (default, all pre-existing behavior bitwise): the
    # operating-threshold connected components. "conservative_sweep":
    # the most conservative reading of the confidence sweep -- while
    # the 3*thr superlevel set still separates blobs that the
    # operating support has just bridged, the blobs keep separate
    # volume constraints and gauge projections, deflating the soft
    # exchange mode (see conservative_component_labels). Label-free
    # and self-removing; used by the Otto pass-through pair run.
    compatibility_components: str = "operating"

    def __post_init__(self):
        if self.phase is None:
            self.phase = PhaseGridConfig()
        if self.poisson is None:
            # G3a adoption (user decision, measured 59x): the metric's
            # frozen-operator many-solves-per-step structure makes the
            # sparse LU the default here (3.3 s/step vs 94.8 s PCG at
            # 512^2, residual 1e-13). native_pcg remains the
            # correctness reference (per-backend parity tests) and the
            # GPU path -- sparse_direct raises on CUDA, so GPU metric
            # configs must pass an explicit poisson config with
            # solver_backend="native_pcg" (no silent fallback).
            self.poisson = WeightedPoissonConfig(
                grid_shape=self.phase.grid_shape,
                box_min=self.phase.box_min,
                box_max=self.phase.box_max,
                support_threshold=self.phase.support_threshold,
                solver_backend="sparse_direct")
        assert_grid_configs_consistent(self.phase, self.poisson)


class ComposedConstraintBasis(NamedTuple):
    basis: _T                 # (N, N - C) orthonormal kernel basis
    singular_values: _T       # (C,) of the composed rows (confidence)
    n_rows_effective: int = 0  # rows kept after freeze-dependent drop


def reduce_constraint_rows(stacked_rows: _T, rank_rtol: float = 1e-10
                           ) -> tuple:
    """Rank-revealing reduction of stacked constraint rows on the
    concatenated (s, dtheta) space. Returns (orthonormal rows spanning
    the same row space, numerical rank, singular values). Kernel-exact:
    the admissible space downstream depends only on the row SPAN, so
    replacing possibly-dependent stacked rows [C_grid; C_bulk] by an
    orthonormal row basis changes nothing except making the composed
    rank check well-posed. Post-bridge rank 3 (1 grid + 2 bulk with the
    grid row generally independent, w_act != 1 near the boundary) is a
    NORMAL value, not an anomaly."""
    U, S, Vh = torch.linalg.svd(stacked_rows, full_matrices=False)
    rank = int((S > rank_rtol * S[0]).sum()) if S.numel() else 0
    return Vh[:rank].contiguous(), rank, S


def constraint_row_independence(rows_grid: _T, rows_bulk: _T,
                                rank_rtol: float = 1e-10) -> float:
    """delta_compat = ||C_grid (I - P_row(C_bulk))||_2 / ||C_grid||_2:
    how much numerically-independent constraint the grid compatibility
    rows add on top of the physical bulk rows. Large values mean the
    stacked cell over-constrains relative to bulk-only -- required
    telemetry before attributing residual velocity errors to operator
    bridge coupling (reviewer item 3/4)."""
    if rows_grid.numel() == 0 or rows_bulk.numel() == 0:
        return float("nan")
    _, Sb, Vb = torch.linalg.svd(rows_bulk, full_matrices=False)
    rb = int((Sb > rank_rtol * Sb[0]).sum())
    Vb = Vb[:rb]
    resid = rows_grid - (rows_grid @ Vb.T) @ Vb
    num = torch.linalg.svdvals(resid)[0]
    den = torch.linalg.svdvals(rows_grid)[0]
    return float(num / den)


def bulk_grid_incidence(grid_pp: _T, bulk_pp: _T, n_grid: int,
                        n_bulk: int) -> _T:
    """0L-B0: aggregation matrix A_gb in {0,1}^{G x B}, A_gb[g, b] = 1
    iff bulk b lies inside grid component g. Only defined when every
    bulk class sits in ONE grid class (partition_relation 'equal' or
    'b_finer'); a bulk class straddling grid classes is refused."""
    A = torch.zeros(n_grid, n_bulk, dtype=torch.float64,
                    device=grid_pp.device)
    for b in range(n_bulk):
        gs = grid_pp[bulk_pp == b].unique()
        if gs.numel() != 1:
            raise ValueError(
                f"bulk class {b} spans {gs.numel()} grid classes -- "
                "aggregation is only defined for equal/b_finer")
        A[int(gs[0]), b] = 1.0
    return A


def relative_bulk_rows(bulk_rows: _T, A_gb: _T) -> _T:
    """0L-B0 aligned prequotient rows: the bulk volume rows projected
    onto the RELATIVE (intra-grid-component) coefficient space,
        P_rel = I_B - A_gb^T (A_gb A_gb^T)^dagger A_gb,   C_rel = P_rel C_bulk.
    ker A_gb = coefficient vectors summing to zero inside every grid
    component, so span(C_rel) constrains only volume EXCHANGE between
    bulks of the same grid component and never the per-grid-component
    totals (those are the grid rows' job). Permutation-equivariant in
    the bulk labels (P_rel is a projector, no basis choice); rank
    B - rank(A_gb) generically. For (G, B) = (1, 2): span{C_d - C_a}."""
    A = A_gb.to(bulk_rows.dtype)
    AAt = A @ A.T
    P = torch.eye(A.shape[1], dtype=A.dtype, device=A.device) \
        - A.T @ torch.linalg.pinv(AAt) @ A
    return P @ bulk_rows


def row_space_discrepancy(rows_ref: _T, rows_test: _T,
                          rank_rtol: float = 1e-10) -> float:
    """Scale-invariant row-space discrepancy (reviewer 0L-B0 item 2):
        eps = ||rows_test (I - P_row(rows_ref))||_2 / ||rows_test||_2,
    i.e. the sine of the largest principal angle from span(rows_test)
    into span(rows_ref) -- for one row each this is
    sqrt(1 - <G,C>^2/(|G|^2|C|^2)), zero whenever the two rows define
    the same kernel (any scalar multiple). This is the third,
    non-physical direction delta_comp that makes the post-bridge stack
    [G; C_d; C_a] rank 3 instead of 2."""
    if rows_ref.numel() == 0 or rows_test.numel() == 0:
        return float("nan")
    _, S, V = torch.linalg.svd(rows_ref, full_matrices=False)
    r = int((S > rank_rtol * S[0]).sum())
    V = V[:r]
    resid = rows_test - (rows_test @ V.T) @ V
    num = torch.linalg.svdvals(resid)[0]
    den = torch.linalg.svdvals(rows_test)[0]
    return float(num / den) if float(den) > 0 else float("nan")


def compose_constraint_basis(component_rows: _T, AB_solve: _T,
                             coherence: _T, q_suppress: bool,
                             rank_rtol: float = 1e-10,
                             frozen_mask: "_T | None" = None,
                             drop_dependent_under_freeze: bool = False
                             ) -> ComposedConstraintBasis:
    """Orthonormal basis of the admissible pre-scale displacement space.

    The grid compatibility constraints live on (s, dtheta):
    C_alpha[(s, dtheta)] = rows_s . s + rows_th . dtheta = 0 per phase
    component alpha. The parametrization generates
        s = D_q (Q y),  dtheta = D_q AB D_q (Q y)   (D_q = diag(q) when
    q_suppress, else identity), so in the pre-scale variable u = Q y the
    composed row is
        r_alpha = q o rows_s + q o AB^T (q o rows_th)
    (drop the q's when q_suppress is False). Returns the orthonormal
    kernel basis (N, N - C) -- to be passed as constraint_basis --
    plus the composed rows' singular values (confidence telemetry,
    reviewer item 6); with the basis in place every
    parameter direction produces an EXACTLY component-flux-free delta_rho
    and the fail-closed forward solve accepts it by construction.

    Fail-closed: raises if the numerical rank of the composed rows is
    not C (degenerate constraints, e.g. a component whose boundary is
    entirely coherence-suppressed)."""
    C, twoN = component_rows.shape
    N = twoN // 2
    rows_s, rows_th = component_rows[:, :N], component_rows[:, N:]
    if q_suppress:
        composed = coherence[None, :] * (
            rows_s + (coherence[None, :] * rows_th) @ AB_solve)
    else:
        composed = rows_s + rows_th @ AB_solve
    if C == 0:
        raise RuntimeError("no phase components -- empty constraint set")
    # arm 7 (horizontal-space optimization): a dynamically dead mark's
    # displacement direction is a lift-FIBER direction -- moving a
    # cancelling pair together changes neither the phase nor (as its
    # amplitude fades) the current, so its objective cost vanishes and
    # trust-ncg wanders there (measured: 310x displacement scatter,
    # folds, 4223 s/step). Freezing those DOF restricts the MM step to
    # the HORIZONTAL directions; the vertical handling belongs to
    # advection (transport) and fade/GC (reduction). Implemented as
    # extra unit constraint rows e_i for frozen marks.
    n_frozen = 0
    if frozen_mask is not None and bool(frozen_mask.any()):
        idx = torch.nonzero(frozen_mask).flatten()
        n_frozen = idx.numel()
        if drop_dependent_under_freeze:
            # BIE backend: a row supported entirely on frozen particles
            # (e.g. the relative area row of a fully masked loop) is
            # implied by the unit rows e_i; the kernel is unchanged
            # when the frozen columns are zeroed first and the rows
            # are reduced to a basis of their row space.
            composed = composed.clone()
            composed[:, idx] = 0.0
            Ur, Sr, Vhr = torch.linalg.svd(composed, full_matrices=False)
            r_eff = int((Sr > rank_rtol * Sr[0]).sum()) if Sr.numel() \
                else 0
            composed = Sr[:r_eff, None] * Vhr[:r_eff]
            C = r_eff
        e_rows = torch.zeros(n_frozen, N, dtype=composed.dtype)
        e_rows[torch.arange(n_frozen), idx] = 1.0
        composed = torch.cat([composed, e_rows], dim=0)
    expected = C + n_frozen
    U, S, Vh = torch.linalg.svd(composed, full_matrices=True)
    rank = int((S > rank_rtol * S[0]).sum())
    if rank != expected:
        raise RuntimeError(
            f"grid constraint rows are rank-deficient: numerical rank "
            f"{rank} != n_components {C}"
            + (f" + n_frozen {n_frozen}" if n_frozen else "")
            + f" (spectrum head "
            f"{[float(x) for x in S[:min(4, len(S))]]}) -- a component's "
            f"flux constraint degenerated; refusing to build the basis")
    return ComposedConstraintBasis(Vh[rank:].T.contiguous(), S, C)


class GridWassersteinMetric:
    """Frozen-coefficient grid metric; BEMWasserstein-compatible call."""

    def __init__(self, config: Optional[GridMetricConfig] = None):
        self.config = config or GridMetricConfig()
        self.phase: Optional[PhaseGrid] = None
        self.poisson: Optional[WeightedPoissonOperator] = None
        self.flux: Optional[BoundaryFluxToGrid] = None
        # "projected" is reserved for the 0F bulk_only_diagnostic cell
        self.forward_solve: str = "fail_closed"
        # 0L-B0 telemetry (reset every setup_for_step)
        self.gauge_hold_fired = False
        self.gamma_cross: "float | None" = None
        self.n_compat_base: "int | None" = None
        self.compat_labels_base = None
        self.max_projection_rel = 0.0

    def setup_for_step(self, varifold, carrier_masses: _T,
                       coherence: _T, target_volume: float,
                       policies: Optional[BoundaryFluxPolicies] = None,
                       gauge_seed: "tuple | None" = None) -> None:
        """gauge_seed = (positions, bulk_pp, n_bulk) -- 0L-B0 cell E
        DIAGNOSTIC gauge hold: when the conservative sweep reads fewer
        compatibility components than the raw bulk incidence, the
        active set is re-partitioned by nearest-particle bulk label
        (deflation override; matrix untouched). base/used telemetry
        is kept separately so the b_finer onset never disappears."""
        cfg = self.config
        assert_grid_configs_consistent(cfg.phase, cfg.poisson)
        self.gauge_hold_fired = False
        self.gamma_cross = None
        self.n_compat_base = None
        self.compat_labels_base = None
        self.max_projection_rel = 0.0
        pg = CurrentToPhase(cfg.phase).reconstruct(
            varifold, carrier_masses, target_volume)
        ambiguous = False
        if cfg.compatibility_components == "conservative_sweep":
            from .weighted_poisson import connected_components
            thr = cfg.phase.support_threshold
            # the window is read on rho_RAW: the projected/dropped
            # field erases the very crumbs whose flicker is the
            # instability evidence (v5 measured: two dropped
            # single-cell crumbs at step 793 while the POST-drop
            # metric field read stable). Raw-field counts at
            # thr/3, thr, 3*thr keep the window open through the
            # healing fuzz and close it only when all three agree.
            raw = pg.rho_raw
            _, n_lo = connected_components(raw > thr / 3.0)
            _, n_op = connected_components(raw > thr)
            _, n_hi = connected_components(raw > 3.0 * thr)
            ambiguous = not (n_lo == n_op == n_hi)
        validate_phase_reconstruction(pg, cfg.phase,
                                      ambiguous_window=ambiguous)
        if cfg.compatibility_components == "conservative_sweep":
            from .phase_grid import conservative_component_labels
            import math as _math
            min_area = (cfg.phase.min_component_area_factor
                        * _math.pi * cfg.phase.fill_epsilon ** 2 / 4.0)
            labels, n_compat = conservative_component_labels(
                pg.rho_metric, cfg.phase.support_threshold,
                min_core_cells=int(_math.ceil(
                    min_area / pg.cell_area)))
            self.n_compat_base = n_compat
            self.compat_labels_base = labels
            if gauge_seed is not None:
                seed_pos, bulk_pp, n_bulk = gauge_seed
                if n_compat < n_bulk:
                    from .phase_grid import bulk_seeded_component_labels
                    labels, _cert = bulk_seeded_component_labels(
                        labels >= 0, cfg.phase, seed_pos, bulk_pp,
                        n_bulk)
                    n_compat = n_bulk
                    self.gauge_hold_fired = True
            op = WeightedPoissonOperator(pg.rho_metric, cfg.poisson,
                                         component_labels=labels)
            if self.gauge_hold_fired:
                from .phase_grid import cross_label_conductivity
                self.gamma_cross = cross_label_conductivity(op)
            # v8: the sweep may legitimately read fewer components
            # than the raw operating support when the difference is
            # coreless threshold debris (absorbed by the v7/v8
            # invariant). What it must never lose is a component
            # with a substantial 3*thr core.
            from .phase_grid import substantial_component_count
            n_substantial = substantial_component_count(
                pg.rho_metric, cfg.phase.support_threshold,
                min_core_cells=int(_math.ceil(
                    min_area / pg.cell_area)))
            if n_compat < n_substantial:
                raise RuntimeError(
                    "conservative sweep produced FEWER components "
                    f"({n_compat}) than the substantial operating "
                    f"support ({n_substantial})")
        elif cfg.compatibility_components == "operating":
            op = WeightedPoissonOperator(pg.rho_metric, cfg.poisson)
            if op.n_components != pg.n_components:
                raise RuntimeError(
                    f"phase components ({pg.n_components}) != operator "
                    f"nullspace components ({op.n_components}) -- "
                    f"config drift between phase and Poisson layers")
        else:
            raise ValueError(
                "compatibility_components must be 'operating' or "
                f"'conservative_sweep', got "
                f"{cfg.compatibility_components!r}")
        # single source for every compatibility consumer (constraint
        # rows in the stepper, projections in the operator)
        self.compat_labels = op.labels
        self.n_compat_components = op.n_components
        if self.n_compat_base is None:
            self.n_compat_base = op.n_components
        self.ambiguous_window = ambiguous
        self.phase, self.poisson = pg, op
        self.flux = BoundaryFluxToGrid(
            varifold.positions, varifold.normals, carrier_masses,
            coherence, cfg.phase, policies)

    def __call__(self, displacements: _T, delta_angles: _T,
                 time_step: float,
                 drho_offset: "_T | None" = None) -> _T:
        """drho_offset (arm V2, coupled minimizing movement): a fixed
        grid field added to the flux BEFORE the quadratic form, so the
        metric prices flux(s, dtheta) + offset jointly. The cross term
        <flux, Ldag offset> is exactly the transport-compensates-
        removal coupling the split scheme discards; None is bitwise
        the pre-V2 path."""
        assert self.poisson is not None, "setup_for_step first"
        drho = self.flux(displacements, delta_angles)
        if drho_offset is not None:
            drho = drho + drho_offset
        if self.forward_solve == "projected":
            # 0F bulk_only_diagnostic ONLY (never production): the
            # admissible space then enforces just the physical bulk
            # rows, so the flux is generically grid-incompatible and
            # the fail-closed forward would reject it. K = <P drho,
            # Ldag P drho> isolates the effect of the extra
            # C_grid-independent constraint directions.
            b = self.poisson.project_componentwise_zero_mean(drho)
            # 0L-B0 cell C gate telemetry: how much grid-incompatible
            # flux the projection discards (per-step max; the
            # pre-registered gate eps_proj,* bounds it)
            with torch.no_grad():
                nd = float(drho.detach().norm())
                if nd > 0:
                    self.max_projection_rel = max(
                        self.max_projection_rel,
                        float((drho.detach() - b.detach()).norm()) / nd)
            phi = _ProjectedPoissonSolve.apply(b, self.poisson)
            return (0.5 / time_step) * self.poisson.inner(b, phi)
        if self.forward_solve != "fail_closed":
            raise ValueError(
                f"forward_solve must be 'fail_closed' or 'projected', "
                f"got {self.forward_solve!r}")
        if self.config.quadratic_path == "analytic":
            return (0.5 / time_step) * GridQuadraticForm.apply(
                drho, self.poisson)
        if self.config.quadratic_path != "generic":
            raise ValueError(
                f"quadratic_path must be 'analytic' or 'generic', got "
                f"{self.config.quadratic_path!r}")
        phi = SymmetricPoissonSolve.apply(drho, self.poisson)
        return (0.5 / time_step) * self.poisson.inner(drho, phi)

    def quadratic(self, displacements: _T, delta_angles: _T,
                  time_step: float,
                  drho_offset: "_T | None" = None) -> _T:
        return self(displacements, delta_angles, time_step,
                    drho_offset=drho_offset)

    def hessp(self, dir_s: _T, dir_th: _T, time_step: float
              ) -> tuple[_T, _T]:
        """Metric HVP in (s, dtheta) coordinates: B^* L^dagger B / dt.
        Audit use; the outer optimizer gets HVPs through autograd."""
        assert self.poisson is not None
        drho = self.flux(dir_s, dir_th)
        phi, info = self.poisson.solve(drho)
        if not info.converged:
            raise WeightedPoissonSolveError(info)
        # adjoint of the scatter: v_ik = zeta q^pol kernel^T phi
        st = self.flux.stencil
        phi_flat = phi.reshape(-1)
        w_phi = (phi_flat[st.flat_index]
                 * st.kernel_weights).sum(dim=1) * st.cell_area
        w_phi = w_phi.reshape(self.flux.n_particles,
                              self.flux.n_endpoints)
        base = self.flux.zeta * w_phi \
            * self.flux.flux_scale[:, None]
        g_s = base.sum(dim=1) / time_step
        g_th = -(base * self.flux.r_physical).sum(dim=1) / time_step
        return g_s, g_th
