"""MM (Minimizing Movements) solver for Mullins-Sekerka flow.

Implements the full evolution loop using BEM-based linearized Wasserstein:

    μ^n = argmin_{y, θ} [ P̂(μ) + W_lin(s) ]

This iterates mm_step for multiple time steps, tracking the evolution
of the boundary.
"""

import torch
from dataclasses import dataclass, field
from typing import Callable

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params

from .mm_step import MMConfig, MMStepResult, MMStepper, mm_step
from .redistribute import redistribute_points
from .remove_dead_points import remove_dead_points


@dataclass
class EvolutionHistory:
    """History of MM evolution.

    Stores varifolds as a list to support variable N (point removal).
    Scalar quantities (perimeter, volume, etc.) use pre-allocated tensors.

    Attributes:
        _varifolds: List of varifolds at each step (length = n_completed + 1).
        perimeters: (n_steps,) tensor of perimeter values.
        wassersteins: (n_steps,) tensor of Wasserstein values.
        objectives: (n_steps,) tensor of objective values.
        volumes: (n_steps+1,) tensor of computed volumes.
        converged: (n_steps,) tensor of convergence flags.
        n_completed: Number of completed steps.
    """
    _varifolds: list[OrientedPointCloudVarifold] = field(repr=False)
    perimeters: torch.Tensor     # (n_steps,)
    wassersteins: torch.Tensor   # (n_steps,)
    objectives: torch.Tensor     # (n_steps,)
    volumes: torch.Tensor        # (n_steps+1,)
    converged: torch.Tensor      # (n_steps,)
    n_completed: int

    def __len__(self) -> int:
        return self.n_completed + 1

    def get_varifold(self, step: int) -> OrientedPointCloudVarifold:
        """Get varifold at given step."""
        return self._varifolds[step]

    def get_final_varifold(self) -> OrientedPointCloudVarifold:
        """Get varifold at final completed step."""
        return self.get_varifold(self.n_completed)

    @property
    def positions(self) -> torch.Tensor:
        """(n_completed+1, N, 2) positions tensor (backward compat).

        Raises RuntimeError if N varies across steps.
        """
        vfs = self._varifolds[:self.n_completed + 1]
        return torch.stack([v.positions for v in vfs])

    @property
    def angles(self) -> torch.Tensor:
        """(n_completed+1, N) angles tensor (backward compat).

        Raises RuntimeError if N varies across steps.
        """
        vfs = self._varifolds[:self.n_completed + 1]
        return torch.stack([v.angles for v in vfs])


def create_history(
    n_steps: int,
    device: str,
    dtype: torch.dtype,
) -> EvolutionHistory:
    """Create history with pre-allocated scalar tensors."""
    return EvolutionHistory(
        _varifolds=[],
        perimeters=torch.zeros(n_steps, device=device, dtype=dtype),
        wassersteins=torch.zeros(n_steps, device=device, dtype=dtype),
        objectives=torch.zeros(n_steps, device=device, dtype=dtype),
        volumes=torch.zeros(n_steps + 1, device=device, dtype=dtype),
        converged=torch.zeros(n_steps, device=device, dtype=torch.bool),
        n_completed=0,
    )


def compute_volume_divergence(
    varifold: OrientedPointCloudVarifold,
    masses: torch.Tensor,
) -> float:
    """Compute volume (area in 2D) using the divergence theorem.

    Area = (1/2) Σ_i (x_i cos θ_i + y_i sin θ_i) · m_i

    This does NOT require point ordering and works for multi-component
    boundaries and after topology changes (e.g., fusion).

    Args:
        varifold: Oriented point cloud with positions and angles.
        masses: (N,) KDE masses (arc length weights).

    Returns:
        Area enclosed by the boundary.
    """
    pos = varifold.positions
    normals = torch.stack(
        [torch.cos(varifold.angles), torch.sin(varifold.angles)], dim=1
    )
    area = 0.5 * (pos * normals).sum(dim=1).dot(masses)
    return area.item()


def compute_volume_shoelace(varifold: OrientedPointCloudVarifold) -> float:
    """Compute volume (area in 2D) using the shoelace formula.

    Warning:
        This function assumes points are ordered consecutively around the
        boundary (i.e., point i is adjacent to points i-1 and i+1).
        If points are unordered or represent multiple disconnected components,
        the result will be incorrect. For multi-component boundaries, each
        component must be processed separately.

    Args:
        varifold: Oriented point cloud with positions ordered along boundary.

    Returns:
        Signed area enclosed by the boundary (positive for counter-clockwise).
    """
    positions = varifold.positions
    x = positions[:, 0]
    y = positions[:, 1]

    x_next = torch.roll(x, -1)
    y_next = torch.roll(y, -1)

    area = 0.5 * torch.abs((x * y_next - x_next * y).sum())
    return area.item()


# Keep old name for backward compatibility
compute_volume = compute_volume_shoelace


class MMSolver:
    """MM scheme solver for Mullins-Sekerka flow using BEM.

    Example:
        config = MMConfig(time_step=0.01)
        solver = MMSolver(config)
        history = solver.solve(initial_varifold, n_steps=100)
    """

    def __init__(self, config: MMConfig):
        """Initialize solver."""
        self.config = config

    def solve(
        self,
        initial_varifold: OrientedPointCloudVarifold,
        n_steps: int,
        callback: Callable[[int, MMStepResult], bool] | None = None,
    ) -> EvolutionHistory:
        """Run MM scheme for multiple steps.

        Args:
            initial_varifold: Initial boundary configuration.
            n_steps: Number of time steps.
            callback: Optional callback(step, result). Return True to stop.

        Returns:
            EvolutionHistory containing all states and diagnostics.
        """
        device = initial_varifold.positions.device
        dtype = initial_varifold.positions.dtype

        # Create history
        history = create_history(n_steps, device, dtype)

        # Store initial state
        history._varifolds.append(OrientedPointCloudVarifold(
            positions=initial_varifold.positions.detach(),
            angles=initial_varifold.angles.detach(),
        ))

        # Create stepper once, reuse across steps (enables torch.compile caching)
        # δ, τ are recomputed adaptively in stepper._setup_step() when config is None
        stepper = MMStepper(self.config)

        # Compute initial volume (divergence theorem: order-independent)
        cfg = self.config
        delta, tau = compute_recommended_params(
            initial_varifold.positions, kernel=cfg.mass_kernel, k_min=cfg.mass_k_min,
        )
        mass_delta = cfg.mass_delta if cfg.mass_delta is not None else delta
        mass_tau = cfg.mass_tau if cfg.mass_tau is not None else tau
        initial_masses = compute_masses(
            initial_varifold.positions, mass_delta, mass_tau, cfg.mass_kernel,
        )
        initial_volume = compute_volume_divergence(initial_varifold, initial_masses)
        history.volumes[0] = initial_volume

        current_varifold = initial_varifold
        self._stepper = stepper  # expose for diagnostics

        # Evolution loop
        for step in range(n_steps):
            try:
                result = stepper.step(current_varifold)
            except Exception as e:
                print(f"    [ERROR] Exception at step {step + 1}: {e}")
                break

            current_varifold = result.varifold

            # Redistribute points if enabled
            if (self.config.redistribute and
                (step + 1) % self.config.redistribute_interval == 0):
                new_pos, new_angles, _ = redistribute_points(
                    positions=current_varifold.positions,
                    angles=current_varifold.angles,
                    delta=stepper.mass_delta,
                    kernel=self.config.mass_kernel,
                    n_iters=self.config.redistribute_n_iters,
                    step_size=self.config.redistribute_step_size,
                    tol=self.config.redistribute_tol,
                    max_disp_ratio=self.config.redistribute_max_disp_ratio,
                    mass_tau=stepper.mass_tau,
                    delta_redist=stepper.mass_delta * self.config.redistribute_delta_ratio,
                    coherence=stepper.fixed_coherence,
                )
                current_varifold = OrientedPointCloudVarifold(
                    positions=new_pos, angles=new_angles,
                )

            # Remove dead points if enabled
            if (self.config.remove_dead_points and
                (step + 1) % self.config.dead_point_interval == 0):
                new_varifold, keep_mask = remove_dead_points(
                    current_varifold, result.masses, stepper.fixed_coherence,
                    threshold=self.config.dead_point_threshold,
                )
                if new_varifold.n_points < current_varifold.n_points:
                    n_removed = current_varifold.n_points - new_varifold.n_points
                    print(f"    [REMOVE] step {step+1}: removed {n_removed} dead points, "
                          f"N={current_varifold.n_points}→{new_varifold.n_points}")
                    current_varifold = new_varifold

                    # Extra redistribute after removal to fix density holes
                    if (self.config.redistribute and
                        self.config.redistribute_n_iters_after_removal > 0):
                        # Slice coherence to match surviving points
                        post_coherence = (stepper.fixed_coherence[keep_mask]
                                          if stepper.fixed_coherence is not None
                                          else None)
                        new_pos, new_angles, rd_info = redistribute_points(
                            positions=current_varifold.positions,
                            angles=current_varifold.angles,
                            delta=stepper.mass_delta,
                            kernel=self.config.mass_kernel,
                            n_iters=self.config.redistribute_n_iters_after_removal,
                            step_size=self.config.redistribute_step_size,
                            tol=self.config.redistribute_tol,
                            max_disp_ratio=self.config.redistribute_max_disp_ratio,
                            mass_tau=stepper.mass_tau,
                            delta_redist=stepper.mass_delta * self.config.redistribute_delta_ratio,
                            coherence=post_coherence,
                        )
                        current_varifold = OrientedPointCloudVarifold(
                            positions=new_pos, angles=new_angles,
                        )
                        cv_hist = rd_info["cv_history"]
                        print(f"    [REDIST] post-removal: CV {cv_hist[0]:.4f}→{cv_hist[-1]:.4f} "
                              f"({rd_info['n_iters']} iters, conv={rd_info['converged']})")

            # Check for NaN
            if (result.perimeter is None or
                torch.isnan(torch.tensor(result.perimeter)) or
                torch.isnan(current_varifold.positions).any()):
                print(f"    [WARNING] NaN at step {step + 1}, stopping")
                break

            # Store in history
            history._varifolds.append(OrientedPointCloudVarifold(
                positions=current_varifold.positions.detach(),
                angles=current_varifold.angles.detach(),
            ))
            history.perimeters[step] = result.perimeter
            history.wassersteins[step] = result.wasserstein
            history.objectives[step] = result.objective
            history.converged[step] = result.converged
            history.n_completed = step + 1

            # Compute current volume (divergence theorem)
            # After dead-point removal, current_varifold may have fewer points
            # than result.masses, so recompute masses from current positions.
            if current_varifold.n_points != len(result.masses):
                vol_masses = compute_masses(
                    current_varifold.positions,
                    stepper.mass_delta, stepper.mass_tau,
                    self.config.mass_kernel,
                )
            else:
                vol_masses = result.masses
            current_volume = compute_volume_divergence(current_varifold, vol_masses)
            history.volumes[step + 1] = current_volume

            # Callback
            if callback is not None:
                if callback(step, result):
                    break

        return history
