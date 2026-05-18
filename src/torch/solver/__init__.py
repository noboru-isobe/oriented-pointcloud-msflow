"""Solver module for MM (Minimizing Movements) scheme.

Implements the Chambolle-Laux MM scheme for Mullins-Sekerka flow.
"""

from .mm_step import (
    MMConfig,
    MMStepResult,
    MMStepper,
    NormalGraphParametrization,
    mm_step,
)
from .mm_solver import (
    EvolutionHistory,
    MMSolver,
    create_history,
    compute_volume,
    compute_volume_shoelace,
    compute_volume_divergence,
)
from .redistribute import redistribute_points
from .remove_dead_points import remove_dead_points

__all__ = [
    # Configuration and types
    "MMConfig",
    "MMStepResult",
    "EvolutionHistory",
    "NormalGraphParametrization",
    # Single step
    "MMStepper",
    "mm_step",
    # Full solver
    "MMSolver",
    "create_history",
    "compute_volume",
    "compute_volume_shoelace",
    "compute_volume_divergence",
    # Redistribution
    "redistribute_points",
    # Dead point removal
    "remove_dead_points",
]
