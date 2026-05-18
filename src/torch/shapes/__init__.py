"""Shape generation modules."""

from .curves import (
    ParametricCurve,
    circle,
    ellipse,
    flower,
    star,
)
from .generator import (
    sample_curve,
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_flower,
    generate_oriented_star,
    generate_oriented_two_ellipses,
    generate_oriented_two_circles,
    generate_oriented_rectangle,
    generate_oriented_two_rectangles,
)
from .arclength import (
    sample_curve_arc_length,
    compute_arc_length_cumulative,
    verify_arc_length_with_torchquad,
    find_parameters_for_arc_lengths,
)

__all__ = [
    "ParametricCurve",
    "circle",
    "ellipse",
    "flower",
    "star",
    "sample_curve",
    "generate_oriented_circle",
    "generate_oriented_ellipse",
    "generate_oriented_flower",
    "generate_oriented_star",
    "generate_oriented_two_ellipses",
    "generate_oriented_two_circles",
    "generate_oriented_rectangle",
    "generate_oriented_two_rectangles",
    # Arc length parametrization
    "sample_curve_arc_length",
    "compute_arc_length_cumulative",
    "verify_arc_length_with_torchquad",
    "find_parameters_for_arc_lengths",
]
