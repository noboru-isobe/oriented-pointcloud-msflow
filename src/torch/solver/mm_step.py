"""Single step of the MM (Minimizing Movements) scheme.

Implements one step of the MM scheme for Mullins-Sekerka flow using
BEM-based linearized Wasserstein distance:

    μ^n = argmin_{y} [ P̂(μ) + W_lin(s, Δθ(s)) ]

where:
    - P̂ is the coherence-based perimeter
    - W_lin is the linearized Wasserstein via BEM (with endpoint collocation)
    - s = Q @ y (volume-conserving displacements)
    - Δθ is derived from s via weak form angle constraint with coherence suppression:
      A @ Δθ = B @ s  =>  Δθ_i = q_i × [A^† (B @ s)]_i
"""

import torch
from dataclasses import dataclass
from typing import Literal

from torchmin import minimize

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
    compute_per_point_bandwidths_knn,
    compute_per_point_bandwidths_abramson,
)
from src.torch.perimeter.angle_constraint import compute_angle_constraint_matrices
from src.torch.transport import BEMWasserstein, compute_coherence
from src.torch.math_utils.linalg import TruncatedSVD
from src.torch.math_utils.angles import wrap_angles


@dataclass
class MMConfig:
    """Configuration for MM scheme.

    Attributes:
        time_step: Time step h for MM scheme.
        perimeter_sigma: Bandwidth for coherence perimeter (None for auto).
        perimeter_kernel: Kernel for perimeter computation.
        perimeter_c_sigma: Multiplier for auto sigma computation.
        mass_delta: Bandwidth for KDE mass (None for auto).
        mass_tau: Cutoff threshold for mass (None for auto).
        mass_kernel: Kernel for mass computation.
        optimizer_method: Optimization method (bfgs recommended).
        optimizer_max_iter: Maximum optimizer iterations.
        optimizer_tol: Optimizer tolerance.
        optimizer_lr: Step size for BFGS/L-BFGS.
        optimizer_disp: Verbosity level (0 = silent).
        bem_method: BEM method ("point" or "panel").
        bem_epsilon_scale: Epsilon scale for point method.
        bem_n_endpoints: Number of collocation points per segment (1=center only, 3=endpoints).
    """
    # Time step. Default 1e-4 matches the production-tested scale: animate_flower.py
    # and run_two_ellipses_batch.py both pass 1e-5 explicitly for long runs, and 1e-4
    # gives near-production volume conservation in short integration tests.
    time_step: float = 1e-4

    # Perimeter parameters
    perimeter_sigma: float | None = None
    perimeter_kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2"
    perimeter_c_sigma: float = 3.0

    # Mass parameters
    mass_delta: float | None = None
    mass_tau: float | None = None
    mass_k_min: float = 1.0
    mass_kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2"

    # Optimizer parameters
    optimizer_method: str = "bfgs"
    optimizer_max_iter: int = 100
    optimizer_tol: float = 1e-6
    optimizer_lr: float = 0.5
    optimizer_disp: int = 0

    # BEM Wasserstein parameters
    bem_method: str = "point"       # "point" (recommended) or "panel"
    bem_epsilon_scale: float = 0.1  # only for "point"; must be > 0
    bem_n_endpoints: int = 3        # 1 = center only (no Δθ), 3 = endpoint collocation

    # Angle constraint (weak form: A @ Δθ = B @ s)
    angle_constraint_rcond: float | None = None  # SVD truncation threshold (None = auto)

    # Coherence option for debugging
    use_unit_coherence: bool = False  # If True, use coherence=1 instead of computed value

    # W_lin velocity coherence scaling
    wlin_use_coherence_velocity: bool = True  # If False, V_{ik} = (1/h)(...) instead of (q_i/h)(...)

    # Displacement q suppression: s_i *= q_i, dθ_i *= q_i in MM step
    displacement_q_suppress: bool = False

    # torch.compile
    compile: bool = False

    # Dead point removal
    remove_dead_points: bool = False
    dead_point_threshold: float = 1e-4  # remove if m_i*q_i < threshold * max(m*q)
    dead_point_interval: int = 1        # check every N steps

    # Mass bandwidth type
    mass_bandwidth_type: Literal["global", "per_point_knn", "per_point_abramson"] = "global"
    mass_delta_adaptive: bool = False   # True: recompute δ/τ every step, False: fixed at initial
    mass_knn_k: int = 10               # k for kNN distance in δ computation

    # Redistribution parameters
    redistribute: bool = False
    redistribute_n_iters: int = 10
    redistribute_step_size: float = 0.01
    redistribute_tol: float = 1e-4
    redistribute_interval: int = 1
    redistribute_delta_ratio: float = 0.5  # δ_redist = mass_delta * ratio
    redistribute_max_disp_ratio: float = 0.05  # max per-iter displacement as fraction of δ_redist
    redistribute_n_iters_after_removal: int = 0  # extra redistribute after dead point removal (0 = off)

    # Pairwise kernel backend for coherence (scalar_density / vector_field).
    # "naive" is faster at small N; "keops" lazy is faster at N≳1000.
    backend: Literal["naive", "keops"] = "naive"


@dataclass
class MMStepResult:
    """Result of a single MM step.

    Attributes:
        varifold: Updated varifold after optimization.
        perimeter: Final perimeter value.
        wasserstein: Final linearized Wasserstein value.
        objective: Final objective value (P + W_lin).
        converged: Whether optimizer converged.
        n_iter: Number of optimizer iterations.
        displacements: Normal direction displacements (N,).
        delta_angles: Angle changes (N,).
        masses: (N,) KDE masses from previous step (fixed during optimization).
        effective_masses: (N,) masses * coherence from previous step.
    """
    varifold: OrientedPointCloudVarifold
    perimeter: float
    wasserstein: float
    objective: float
    converged: bool
    n_iter: int
    displacements: torch.Tensor
    delta_angles: torch.Tensor
    masses: torch.Tensor
    effective_masses: torch.Tensor


class NormalGraphParametrization:
    """Normal graph parametrization with volume conservation and angle constraint.

    Position update: x_k = x_k^{n-1} + s_k * n_k^{n-1}
    Angle update: Δθ_i = q_i × [A^†B @ s]_i  (coherence-suppressed)
    Displacement: s = Q @ y where y ∈ R^{N-1}

    Volume conservation: Σ s_i w_i = 0 (via Q).
    Angle constraint: A @ Δθ = B @ s, pre-solved as AB_solve = A^† B,
    then Δθ = (AB_solve @ s) * coherence for hidden boundary suppression.
    """

    def __init__(
        self,
        prev_positions: torch.Tensor,
        prev_normals: torch.Tensor,
        prev_angles: torch.Tensor,
        constraint_weights: torch.Tensor,
        A_angle: torch.Tensor,
        B_angle: torch.Tensor,
        coherence: torch.Tensor,
        rcond: float | None = None,
        q_suppress: bool = False,
    ):
        """Initialize parametrization.

        Args:
            prev_positions: Positions from previous step (N, 2).
            prev_normals: Unit normals from previous step (N, 2).
            prev_angles: Angles from previous step (N,).
            constraint_weights: Weights w_i for constraint Σ s_i w_i = 0.
            A_angle: (N, N) angle constraint matrix for Δθ.
            B_angle: (N, N) angle constraint matrix for s.
            coherence: (N,) coherence values from previous step for displacement scaling.
            rcond: SVD truncation threshold. None = eps * N (torch.linalg.lstsq style).
            q_suppress: If True, multiply displacements and delta_angles by coherence q_i.
        """
        from src.torch.transport.bem_wasserstein import _orthogonal_complement

        self.prev_positions = prev_positions.detach()
        self.prev_normals = prev_normals.detach()
        self.prev_angles = prev_angles.detach()
        self.coherence = coherence.detach()
        self.q_suppress = q_suppress
        self.N = prev_positions.shape[0]
        self.Q = _orthogonal_complement(constraint_weights)

        # Pre-compute AB_solve = A^† B for efficient angle constraint application
        A_det = A_angle.detach()
        B_det = B_angle.detach()

        if A_det.device.type == "cpu":
            # Use lstsq (LAPACK gelsd) for better numerical stability on CPU.
            # Explicit driver='gelsd' ensures SVD-based solve and returns singular_values.
            result = torch.linalg.lstsq(A_det, B_det, rcond=rcond, driver="gelsd")
            self.AB_solve = result.solution
            S = result.singular_values
            if S.numel() > 0 and (S > 0).any():
                self.angle_cond = (S.max() / S[S > 0].min()).item()
            else:
                self.angle_cond = float("nan")
            self.angle_n_truncated = (
                max(0, A_det.shape[0] - result.rank.item())
                if result.rank.numel() > 0 else 0
            )
        else:
            # CUDA: SVD-based pseudo-inverse (gelsd not available on CUDA)
            svd = TruncatedSVD.from_matrix(A_det, rcond=rcond)
            self.AB_solve = svd.Vh.T @ (svd.S_inv.unsqueeze(-1) * (svd.U.T @ B_det))
            S_inv = svd.S_inv
            active = S_inv > 0
            if active.any():
                S_active = 1.0 / S_inv[active]
                self.angle_cond = (S_active.max() / S_active.min()).item()
            else:
                self.angle_cond = float("inf")
            self.angle_n_truncated = svd.n_truncated

    def init_params(self) -> torch.Tensor:
        """Create initial parameter vector y=0.

        Returns:
            params: (N-1,) parameter vector y.
        """
        device, dtype = self.prev_positions.device, self.prev_positions.dtype
        return torch.zeros(self.N - 1, device=device, dtype=dtype)

    def unpack_params(
        self,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack parameter vector into displacements and delta_angles.

        Displacements are scaled by coherence so that hidden boundary points
        (q ≈ 0) have near-zero displacement: s_i = q_i × (Q @ y)_i.

        delta_angles is derived from displacements via the weak form angle constraint
        and then suppressed by coherence:
            Δθ_i = q_i × [A^†B @ s]_i

        This ensures hidden boundary points (q ≈ 0) have near-zero Δθ,
        preventing the normal rotation instability caused by ill-conditioned A.

        Returns:
            (displacements (N,), delta_angles (N,))
        """
        y = params
        s_raw = self.Q @ y
        displacements = self.coherence * s_raw if self.q_suppress else s_raw

        # Derive Δθ from s via pre-computed AB_solve = A^† B
        delta_angles = self.AB_solve @ displacements
        if self.q_suppress:
            delta_angles = delta_angles * self.coherence

        return displacements, delta_angles

    def reconstruct_varifold(
        self,
        params: torch.Tensor,
    ) -> OrientedPointCloudVarifold:
        """Reconstruct varifold from parameters."""
        displacements, delta_theta = self.unpack_params(params)
        positions = self.prev_positions + displacements.unsqueeze(-1) * self.prev_normals
        angles = self.prev_angles + delta_theta
        # Wrap angles to [-π, π) for numerical stability
        angles = wrap_angles(angles)
        return OrientedPointCloudVarifold(positions=positions, angles=angles)


class MMStepper:
    """Stateful MM stepper that reuses BEM objects across steps.

    Designed for torch.compile compatibility: self.objective is a stable method
    (fixed address) so Dynamo can cache the trace across MM steps.

    Usage (standalone):
        stepper = MMStepper(config)
        result = stepper.step(varifold)

    Usage (in solver loop — reuse across steps):
        stepper = MMStepper(config)
        for step in range(n_steps):
            result = stepper.step(current_varifold)
            current_varifold = result.varifold
        # After step(), stepper.mass_delta and stepper.mass_tau hold current values
    """

    def __init__(self, config: MMConfig):
        self.config = config
        self.bem_wasserstein = BEMWasserstein(
            method=config.bem_method,
            epsilon_scale=config.bem_epsilon_scale,
            n_endpoints=config.bem_n_endpoints,
        )
        self.param: NormalGraphParametrization | None = None
        self.fixed_coherence: torch.Tensor | None = None
        self.mass_delta: float | None = config.mass_delta
        self.mass_tau: float | None = config.mass_tau
        # Per-point bandwidth (set by _setup_step when per_point mode)
        self._mass_delta_for_kde: float | torch.Tensor | None = config.mass_delta
        self._mass_tau_for_kde: float | torch.Tensor | None = config.mass_tau

        # Tensor diagnostics updated by objective(), converted in step()
        self._last_perimeter: torch.Tensor | None = None
        self._last_wasserstein: torch.Tensor | None = None

        # Lazy-compiled objective
        self._compiled_objective = None

        # Optional optimizer callback (e.g. for trust-region diagnostics)
        self.optimizer_callback = None

    def set_mass_params(self, delta: float | None, tau: float | None):
        """Set mass parameters (called by MMSolver after auto-computation)."""
        self.mass_delta = delta
        self.mass_tau = tau

    def _setup_step(self, varifold: OrientedPointCloudVarifold):
        """Prepare per-step state: parametrization, coherence, BEM cache.

        Corresponds to the setup portion of the old mm_step().
        """
        from src.torch.perimeter.coherence_perimeter import compute_recommended_sigma

        config = self.config

        # Compute mass parameters
        # adaptive=True: recompute every step; adaptive=False: compute once, cache
        need_compute = config.mass_delta_adaptive or self._mass_delta_for_kde is None

        if config.mass_bandwidth_type == "per_point_knn":
            if need_compute or isinstance(self._mass_delta_for_kde, float):
                delta_pp, tau_pp = compute_per_point_bandwidths_knn(
                    varifold.positions.detach(),
                    k=config.mass_knn_k,
                    k_min=config.mass_k_min,
                    kernel=config.mass_kernel,
                )
                self.mass_delta = delta_pp.median().item()
                self.mass_tau = tau_pp.median().item()
                self._mass_delta_for_kde = delta_pp
                self._mass_tau_for_kde = tau_pp
        elif config.mass_bandwidth_type == "per_point_abramson":
            if need_compute or isinstance(self._mass_delta_for_kde, float):
                delta_pp, tau_pp = compute_per_point_bandwidths_abramson(
                    varifold.positions.detach(),
                    k0=config.mass_knn_k,
                    k_min=config.mass_k_min,
                    kernel=config.mass_kernel,
                )
                self.mass_delta = delta_pp.median().item()
                self.mass_tau = tau_pp.median().item()
                self._mass_delta_for_kde = delta_pp
                self._mass_tau_for_kde = tau_pp
        else:  # "global"
            if need_compute:
                if config.mass_delta is None or config.mass_tau is None:
                    delta, tau = compute_recommended_params(
                        varifold.positions.detach(),
                        k0=config.mass_knn_k,
                        kernel=config.mass_kernel,
                        k_min=config.mass_k_min,
                    )
                    mass_delta = config.mass_delta if config.mass_delta is not None else delta
                    mass_tau = config.mass_tau if config.mass_tau is not None else tau
                else:
                    mass_delta = config.mass_delta
                    mass_tau = config.mass_tau
                self.mass_delta = mass_delta
                self.mass_tau = mass_tau
                self._mass_delta_for_kde = mass_delta
                self._mass_tau_for_kde = mass_tau

        # Compute sigma for perimeter and coherence
        if config.perimeter_sigma is None:
            sigma = compute_recommended_sigma(varifold.positions, config.perimeter_c_sigma)
        else:
            sigma = config.perimeter_sigma
        self._sigma = sigma

        # Compute masses and coherence from PREVIOUS step (FIXED during optimization)
        prev_positions = varifold.positions.detach()
        prev_normals = varifold.normals.detach()
        prev_masses = compute_masses(
            prev_positions,
            self._mass_delta_for_kde,
            self._mass_tau_for_kde,
            config.mass_kernel,
        )
        self.fixed_masses = prev_masses

        if config.use_unit_coherence:
            self.fixed_coherence = torch.ones_like(prev_masses)
        else:
            self.fixed_coherence = compute_coherence(
                varifold, prev_masses, sigma, config.perimeter_kernel,
                backend=config.backend,
            )

        # effective_masses = masses * coherence
        effective_masses = prev_masses * self.fixed_coherence
        self.fixed_effective_masses = effective_masses

        # Constraint weights for volume conservation: w_i = q_i² * m_i
        constraint_weights = self.fixed_coherence * self.fixed_coherence * prev_masses

        # Compute weak form angle constraint matrices
        prev_tangents = torch.stack(
            [-prev_normals[:, 1], prev_normals[:, 0]], dim=1
        )
        A_angle, B_angle = compute_angle_constraint_matrices(
            positions=prev_positions,
            tangents=prev_tangents,
            masses=prev_masses,
            coherence=self.fixed_coherence,
            sigma=sigma,
            kernel=config.perimeter_kernel,
        )

        # Create parametrization with volume conservation and angle constraint
        self.param = NormalGraphParametrization(
            prev_positions=varifold.positions,
            prev_normals=varifold.normals,
            prev_angles=varifold.angles,
            constraint_weights=constraint_weights,
            A_angle=A_angle,
            B_angle=B_angle,
            coherence=self.fixed_coherence,
            rcond=config.angle_constraint_rcond,
            q_suppress=config.displacement_q_suppress,
        )

        # Update BEM Wasserstein cache (same object, no reallocation)
        self.bem_wasserstein.setup_for_step(
            positions=prev_positions,
            normals=prev_normals,
            effective_masses=effective_masses,
            sigma=sigma,
            c_sigma=config.perimeter_c_sigma,
        )
        self.bem_wasserstein.setup_coherence(
            self.fixed_coherence,
            use_coherence_velocity=config.wlin_use_coherence_velocity,
        )

        # Reset diagnostics
        self._last_perimeter = None
        self._last_wasserstein = None

    def objective(self, params: torch.Tensor) -> torch.Tensor:
        """Compute objective: P + W_lin.

        This is the stable method that torch.compile can trace and cache.
        """
        displacements, delta_angles = self.param.unpack_params(params)
        current_varifold = self.param.reconstruct_varifold(params)

        # Compute masses for CURRENT positions
        masses = compute_masses(
            current_varifold.positions,
            self._mass_delta_for_kde,
            self._mass_tau_for_kde,
            self.config.mass_kernel,
        )

        # Perimeter with FIXED coherence from previous step:
        # P = Σ m_i * q_i^{n-1}, where m_i depends on current positions
        P = (masses * self.fixed_coherence).sum()

        # Compute BEM Wasserstein (with endpoint collocation and delta_angles)
        W_lin = self.bem_wasserstein(
            displacements=displacements,
            delta_angles=delta_angles,
            time_step=self.config.time_step,
        )

        total = P + W_lin

        # Store tensors (not .item()) to avoid graph break under torch.compile.
        # Converted to float in step() after optimization finishes.
        self._last_perimeter = P
        self._last_wasserstein = W_lin

        return total

    def _build_optimizer_options(self) -> dict:
        """Build optimizer options dict from config."""
        config = self.config
        if config.optimizer_method in ("bfgs", "l-bfgs"):
            return {
                "max_iter": config.optimizer_max_iter,
                "gtol": config.optimizer_tol,
                "lr": config.optimizer_lr,
                "disp": config.optimizer_disp,
            }
        elif config.optimizer_method in ("trust-ncg", "dogleg", "trust-exact", "trust-krylov"):
            return {
                "max_iter": config.optimizer_max_iter,
                "gtol": config.optimizer_tol,
            }
        else:
            return {
                "max_iter": config.optimizer_max_iter,
            }

    def _make_debug_wrapper(self, obj_fn):
        """Wrap objective with debug output (for optimizer_disp > 0)."""
        _call_count = [0]
        _prev_params = [None]

        def objective_with_debug(params: torch.Tensor) -> torch.Tensor:
            result = obj_fn(params)
            if self.config.optimizer_disp > 0:
                _call_count[0] += 1
                if _prev_params[0] is not None:
                    step_size = (params - _prev_params[0]).norm().item()
                else:
                    step_size = 0.0
                _prev_params[0] = params.detach().clone()
                if params.requires_grad:
                    grad = torch.autograd.grad(result, params, create_graph=False, retain_graph=True)[0]
                    print(f"  [call {_call_count[0]}] obj={result.item():.6f}, |grad|={grad.norm().item():.4e}, |step|={step_size:.4e}")
            return result

        return objective_with_debug

    def step(self, varifold: OrientedPointCloudVarifold) -> MMStepResult:
        """Perform a single MM step.

        Args:
            varifold: Current varifold μ^{n-1}.

        Returns:
            MMStepResult containing updated varifold and diagnostics.
        """
        # Per-step setup (parametrization, coherence, BEM cache)
        self._setup_step(varifold)

        # Select objective function (compiled or plain)
        if self.config.compile:
            if self._compiled_objective is None:
                self._compiled_objective = torch.compile(self.objective)
            obj_fn = self._compiled_objective
        else:
            obj_fn = self.objective

        # Wrap with debug output if needed
        if self.config.optimizer_disp > 0:
            obj_fn = self._make_debug_wrapper(obj_fn)

        # Initial parameters: y=0
        params_init = self.param.init_params()
        params_init.requires_grad_(True)

        # Run optimization
        result = minimize(
            obj_fn,
            params_init,
            method=self.config.optimizer_method,
            options=self._build_optimizer_options(),
            disp=self.config.optimizer_disp,
            callback=self.optimizer_callback,
        )

        # Extract optimized varifold and parameters
        new_varifold = self.param.reconstruct_varifold(result.x)
        new_varifold = OrientedPointCloudVarifold(
            positions=new_varifold.positions.detach(),
            angles=new_varifold.angles.detach(),
        )
        displacements, delta_angles = self.param.unpack_params(result.x)

        return MMStepResult(
            varifold=new_varifold,
            perimeter=self._last_perimeter.item(),
            wasserstein=self._last_wasserstein.item(),
            objective=result.fun.item() if hasattr(result.fun, 'item') else result.fun,
            converged=result.success,
            n_iter=result.nit if hasattr(result, 'nit') else -1,
            displacements=displacements.detach(),
            delta_angles=delta_angles.detach(),
            masses=self.fixed_masses,
            effective_masses=self.fixed_effective_masses,
        )


def mm_step(
    varifold: OrientedPointCloudVarifold,
    config: MMConfig,
) -> MMStepResult:
    """Perform a single MM step using BEM-based linearized Wasserstein.

    Thin wrapper around MMStepper for backward compatibility.
    For repeated calls (e.g., in a solver loop), prefer using
    MMStepper directly to reuse BEM objects across steps.

    Args:
        varifold: Current varifold μ^{n-1}.
        config: MM scheme configuration.

    Returns:
        MMStepResult containing updated varifold and diagnostics.
    """
    stepper = MMStepper(config)
    return stepper.step(varifold)
