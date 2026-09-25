"""Transport module: linearized Wasserstein backends.

BEM (boundary integral) backend -- production default -- and the grid
weighted-Poisson backend (G-series): the same continuum MS tangent
metric, discretized on a fixed Eulerian grid via the current-to-phase
filling map.
"""

from .bem_wasserstein import (
    BEMWasserstein,
    compute_tangents_from_normals,
    compute_coherence,
    build_bem_matrices_point,
    build_bem_matrices_panel,
    solve_neumann_interior,
)
from .boundary_flux_grid import (
    BoundaryFluxPolicies,
    BoundaryFluxToGrid,
)
from .grid_wasserstein import (
    GridMetricConfig,
    GridWassersteinMetric,
    compose_constraint_basis,
)
from .phase_grid import (
    CurrentToPhase,
    PhaseGrid,
    PhaseGridConfig,
)
from .weighted_poisson import (
    IncompatibleGridVelocityError,
    WeightedPoissonConfig,
    WeightedPoissonOperator,
    WeightedPoissonSolveError,
)

__all__ = [
    "BEMWasserstein",
    "compute_tangents_from_normals",
    "compute_coherence",
    "build_bem_matrices_point",
    "build_bem_matrices_panel",
    "solve_neumann_interior",
    "BoundaryFluxPolicies",
    "BoundaryFluxToGrid",
    "GridMetricConfig",
    "GridWassersteinMetric",
    "compose_constraint_basis",
    "CurrentToPhase",
    "PhaseGrid",
    "PhaseGridConfig",
    "IncompatibleGridVelocityError",
    "WeightedPoissonConfig",
    "WeightedPoissonOperator",
    "WeightedPoissonSolveError",
]
