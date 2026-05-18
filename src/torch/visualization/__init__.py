"""Visualization module for oriented point cloud varifolds.

This module provides matplotlib-based visualization capabilities:
- Static plots for publication-quality figures
- Animation support for MP4/GIF generation
- Core utilities for data preparation
"""

from .core import *
from .static import *
from .animation import *

__all__ = [
    # Core utilities
    "compute_mass_colors",
    "compute_coherence_colors",
    "prepare_boundary_data",

    # Static (Matplotlib)
    "plot_boundary_evolution",
    "plot_diagnostics",
    "plot_coherence_field",
    "create_publication_figure",
    "plot_shape_grid",
    "plot_perimeter_comparison",

    # Animation (Matplotlib)
    "VarifoldAnimator",
    "CircleAnimator",
    "create_circle_animation",
    "setup_animation_backend",
]