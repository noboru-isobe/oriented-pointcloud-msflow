"""D0 golden regression: the legacy pipeline is pinned.

Each fixture reruns the fully explicit legacy configuration and compares
against the stored golden artifact (results/d0_golden/G*.pt):

  - bitwise (torch.equal) within the locked environment;
  - discrete decisions (keep masks, IDs, deletion steps, level flags,
    support states) ALWAYS exact;
  - if the bitwise layer ever fails after an environment change, the
    semantic layer (atol 1e-10..1e-12) is the documented fallback --
    a semantic failure means the legacy baseline moved and every 0D
    comparison is void until this is resolved.

The D1/D2 variants will be compared against these artifacts; the
production default (legacy_hybrid + legacy_lagged) must reproduce them
until the audits conclude and any change is made deliberately.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from d0_golden_regression import (  # noqa: E402
    FIXTURES,
    compare,
    run_fixture,
    validate_fixture,
)

GOLDEN_DIR = Path(__file__).parent.parent / "results" / "d0_golden"


@pytest.mark.parametrize("name", list(FIXTURES))
def test_legacy_pipeline_reproduces_golden(name):
    path = GOLDEN_DIR / f"{name}.pt"
    assert path.exists(), (
        f"golden {path} missing -- generate with "
        "scripts/experiments/d0_golden_regression.py")
    golden = torch.load(path, weights_only=False)
    fresh = run_fixture(name, FIXTURES[name])
    validate_fixture(name, fresh)          # events must actually fire
    bad = compare(golden, fresh, exact=True)
    if bad:
        sem = compare(golden, fresh, exact=False)
        assert not sem, f"legacy baseline MOVED: {sem}"
        pytest.fail(f"bitwise drift (semantic PASS -- environment "
                    f"change?): {bad}")


def test_deletion_fixture_actually_deletes():
    """The G2 golden is only meaningful if deletion genuinely fired."""
    golden = torch.load(GOLDEN_DIR / "G2.pt", weights_only=False)
    assert golden["n_removed"][0] > 0
    assert any(not all(m) for m in golden["keep_masks"] if m is not None)


def test_variant_enums_guarded():
    """Both variant enums are live (D1b: dead_point_rule, D2b-1:
    redistribution_rule -- routed through the shared helpers; the
    goldens above prove legacy stayed bitwise). Unknown values must
    fail LOUDLY inside the helpers -- a silent fallthrough to legacy
    behavior would invalidate the audits."""
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.mm_step import MMConfig
    from d0_golden_regression import make_initial

    cfg = MMConfig(redistribute=True, use_unit_coherence=True,
                   time_step=1e-6, optimizer_method="trust-ncg",
                   optimizer_max_iter=30)
    cfg.redistribution_rule = "bogus"
    with pytest.raises(ValueError):
        MMSolver(cfg).solve(make_initial(0), 1)
    # unknown dead-point rule fails loudly inside the helper
    cfg2 = MMConfig(remove_dead_points=True, dead_point_threshold=0.3,
                    use_unit_coherence=True, time_step=1e-6,
                    optimizer_method="trust-ncg", optimizer_max_iter=30)
    cfg2.dead_point_rule = "bogus"
    with pytest.raises(ValueError):
        MMSolver(cfg2).solve(make_initial(12), 1)
