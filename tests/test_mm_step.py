"""Tests for MM step integration.

Phase 3: Tests that mm_step correctly optimizes the objective
P + W_lin with volume conservation.
"""

import math
import pytest
import torch

from src.torch.solver.mm_step import mm_step, MMConfig
from src.torch.solver.mm_solver import compute_volume_divergence
from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params
from src.torch.shapes.generator import (
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_star,
    generate_oriented_two_ellipses,
)
from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params
from src.torch.transport import compute_coherence
from src.torch.transport.bem_wasserstein import _orthogonal_complement
from src.torch.perimeter.coherence_perimeter import compute_recommended_sigma


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

# Shapes for testing
SHAPES = [
    ("circle", lambda n, d: generate_oriented_circle(n, device=d)),
    ("ellipse", lambda n, d: generate_oriented_ellipse(n, a=1.0, b=0.5, device=d)),
    ("star", lambda n, d: generate_oriented_star(n, n_star_points=5, device=d)),
    ("two_ellipses", lambda n, d: generate_oriented_two_ellipses(n // 2, device=d)),
]

# Time steps to test
TIME_STEPS = [1e-4, 1e-3]

# Max iterations
MAX_ITER = 200


class TestMMStepFromMinimum:
    """Test 1: Start from minimum (initial point). Ignore success flag."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    @pytest.mark.parametrize("time_step", TIME_STEPS)
    def test_no_nan_inf(self, device, shape_name, shape_fn, time_step):
        """Test that mm_step produces valid results (no NaN/inf)."""
        varifold = shape_fn(64, device)
        config = MMConfig(
            time_step=time_step,
            optimizer_max_iter=MAX_ITER,
            optimizer_tol=1e-6,
        )

        result = mm_step(varifold, config)

        # Compute displacement
        displacement = (result.varifold.positions - varifold.positions).norm(dim=1).max().item()

        # Print debug info
        print(f"\n{shape_name}(h={time_step}, {device}): "
              f"n_iter={result.n_iter}, converged={result.converged}, "
              f"P={result.perimeter:.6f}, W={result.wasserstein:.6f}, "
              f"obj={result.objective:.6f}, max_disp={displacement:.6f}")

        # Check no NaN/inf (ignore converged flag)
        assert not torch.isnan(torch.tensor(result.perimeter)), "Perimeter is NaN"
        assert not torch.isnan(torch.tensor(result.objective)), "Objective is NaN"
        assert not torch.isinf(torch.tensor(result.wasserstein)), "Wasserstein is inf"
        assert result.perimeter > 0, "Perimeter should be positive"
        assert result.objective > 0, "Objective should be positive"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    @pytest.mark.parametrize("time_step", TIME_STEPS)
    def test_objective_not_increased(self, device, shape_name, shape_fn, time_step):
        """Test that objective doesn't increase from initial point."""
        varifold = shape_fn(64, device)
        config = MMConfig(
            time_step=time_step,
            optimizer_max_iter=MAX_ITER,
            optimizer_tol=1e-6,
        )

        # Compute initial objective (at y=0, W_lin=0)
        from src.torch.perimeter import compute_perimeter_coherence
        from src.torch.perimeter.coherence_perimeter import compute_recommended_sigma

        delta, tau = compute_recommended_params(varifold.positions)
        masses = compute_masses(varifold.positions, delta, tau)
        sigma = compute_recommended_sigma(varifold.positions)

        initial_perimeter = compute_perimeter_coherence(varifold, masses, sigma=sigma)
        initial_objective = initial_perimeter.item()

        result = mm_step(varifold, config)

        print(f"\n{shape_name}(h={time_step}, {device}): "
              f"obj {initial_objective:.6f} -> {result.objective:.6f}")

        # Allow small numerical tolerance
        assert result.objective <= initial_objective + 1e-5, \
            f"Objective increased: {initial_objective:.6f} -> {result.objective:.6f}"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    @pytest.mark.parametrize("time_step", TIME_STEPS)
    def test_volume_conservation(self, device, shape_name, shape_fn, time_step):
        """Test that volume is approximately conserved."""
        varifold = shape_fn(64, device)
        config = MMConfig(
            time_step=time_step,
            optimizer_max_iter=MAX_ITER,
            optimizer_tol=1e-6,
        )

        def compute_vol(v):
            delta, tau = compute_recommended_params(v.positions)
            masses = compute_masses(v.positions, delta, tau)
            return compute_volume_divergence(v, masses)

        initial_volume = compute_vol(varifold)
        result = mm_step(varifold, config)
        final_volume = compute_vol(result.varifold)

        relative_change = abs(final_volume - initial_volume) / initial_volume

        print(f"\n{shape_name}(h={time_step}, {device}): "
              f"volume {initial_volume:.6f} -> {final_volume:.6f} ({relative_change*100:.2f}%)")

        assert relative_change < 0.05, \
            f"Volume changed by {relative_change*100:.1f}%"


def create_perturbed_varifold(varifold, perturbation_scale=0.01, seed=42):
    """Create a perturbed varifold with volume-conserving displacement."""
    torch.manual_seed(seed)

    delta, tau = compute_recommended_params(varifold.positions)
    masses = compute_masses(varifold.positions, delta, tau)
    sigma = compute_recommended_sigma(varifold.positions)
    coherence = compute_coherence(varifold, masses, sigma)
    constraint_weights = coherence * coherence * masses

    Q = _orthogonal_complement(constraint_weights)
    n = varifold.n_points
    y = torch.randn(n - 1, device=varifold.positions.device, dtype=varifold.positions.dtype)
    y = y * perturbation_scale
    displacements = Q @ y

    new_positions = varifold.positions + displacements.unsqueeze(-1) * varifold.normals

    return OrientedPointCloudVarifold(positions=new_positions, angles=varifold.angles.clone())


class TestMMStepStability:
    """Test that positions and angles recover after optimization from perturbed state."""

    # Use only convex shapes for stability test
    CONVEX_SHAPES = [
        ("circle", lambda n, d: generate_oriented_circle(n, device=d)),
        ("ellipse", lambda n, d: generate_oriented_ellipse(n, a=1.0, b=0.5, device=d)),
    ]

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("shape_name,shape_fn", CONVEX_SHAPES)
    @pytest.mark.parametrize("time_step", [1e-4, 1e-3])
    def test_positions_and_angles_stable(self, device, shape_name, shape_fn, time_step):
        """Test that positions and angles don't drift significantly after MM step optimization."""
        original = shape_fn(64, device)
        perturbed = create_perturbed_varifold(original, perturbation_scale=0.01)

        config = MMConfig(
            time_step=time_step,
            optimizer_max_iter=MAX_ITER,
            optimizer_tol=1e-6,
        )

        result = mm_step(perturbed, config)

        # Compute position difference
        pos_diff = (result.varifold.positions - original.positions).norm(dim=1)
        max_pos_diff = pos_diff.max().item()
        mean_pos_diff = pos_diff.mean().item()

        # Compute angle difference (wrap to [-π, π])
        angle_diff = result.varifold.angles - original.angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        max_angle_diff = angle_diff.abs().max().item()
        mean_angle_diff = angle_diff.abs().mean().item()

        print(f"\n{shape_name}(h={time_step}, {device}): "
              f"max_pos_diff={max_pos_diff:.4f}, mean_pos_diff={mean_pos_diff:.4f}, "
              f"max_angle_diff={max_angle_diff:.4f}, mean_angle_diff={mean_angle_diff:.4f}")

        # Positions should not drift too much (< 0.1 for scale ~1 shapes)
        assert max_pos_diff < 0.1, \
            f"Positions drifted too much: max_diff={max_pos_diff:.4f}"

        # Angles should not drift too much (< 0.5 rad ≈ 29 degrees)
        assert max_angle_diff < 0.5, \
            f"Angles drifted too much: max_diff={max_angle_diff:.4f} rad"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("shape_name,shape_fn", CONVEX_SHAPES)
    def test_optimization_produces_valid_output(self, device, shape_name, shape_fn):
        """Test that optimization produces valid (non-NaN) output."""
        varifold = shape_fn(64, device)

        config = MMConfig(
            time_step=1e-4,
            optimizer_max_iter=MAX_ITER,
            optimizer_tol=1e-6,
        )

        result = mm_step(varifold, config)

        # The optimization should converge without NaN
        assert not torch.isnan(torch.tensor(result.objective)), "Objective is NaN"
        assert result.objective > 0, "Objective should be positive"

        print(f"\n{shape_name}({device}): "
              f"n_iter={result.n_iter}, converged={result.converged}, "
              f"P={result.perimeter:.6f}, W={result.wasserstein:.6f}")
