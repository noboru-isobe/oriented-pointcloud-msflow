"""Explicit-device policy (review, P1.2 pre-patch): every public
tensor-allocating generator/helper must REJECT a missing device at
runtime. AST call-site scans only prove current callers pass it; the
public API itself must fail closed (the silent cuda default put audit
geometries on the GPU unnoticed; a silent cpu default would hide the
same class of mistake)."""

import pytest
import torch

from src.torch.shapes import generator as G


CASES = [
    (G.sample_curve, lambda: (__import__(
        "src.torch.shapes.curves", fromlist=["circle"]).circle(1.0,
                                                               (0.0, 0.0)),
        16)),
    (G.generate_oriented_circle, lambda: (16,)),
    (G.generate_oriented_ellipse, lambda: (16,)),
    (G.generate_oriented_flower, lambda: (16,)),
    (G.generate_oriented_star, lambda: (16,)),
    (G.generate_oriented_two_ellipses, lambda: (16,)),
    (G.generate_oriented_annulus, lambda: (16, 1.0, 0.5)),
    (G.annulus_exact_masses, lambda: (16, 8, 1.0, 0.5)),
    (G.generate_oriented_two_circles, lambda: (16,)),
    (G.generate_oriented_rectangle, lambda: (16,)),
    (G.generate_oriented_two_rectangles, lambda: (8,)),
]


@pytest.mark.parametrize("fn,mkargs", CASES,
                         ids=[c[0].__name__ for c in CASES])
def test_public_generators_reject_missing_device(fn, mkargs):
    with pytest.raises((ValueError, TypeError)):
        fn(*mkargs())


@pytest.mark.parametrize("fn,mkargs", CASES,
                         ids=[c[0].__name__ for c in CASES])
def test_public_generators_accept_explicit_cpu(fn, mkargs):
    out = fn(*mkargs(), device="cpu", dtype=torch.float64)
    t = out.positions if hasattr(out, "positions") else out
    assert t.device.type == "cpu"
