"""Transport module for linearized Wasserstein computation via BEM."""

from .bem_wasserstein import (
    BEMWasserstein,
    compute_tangents_from_normals,
    compute_coherence,
    build_bem_matrices_point,
    build_bem_matrices_panel,
    solve_neumann_interior,
)

__all__ = [
    "BEMWasserstein",
    "compute_tangents_from_normals",
    "compute_coherence",
    "build_bem_matrices_point",
    "build_bem_matrices_panel",
    "solve_neumann_interior",
]
