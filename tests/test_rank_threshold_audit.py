"""0C-5d threshold-audit pins (see rank_threshold_audit_0c5d.py; raw
tables in results/rank_threshold_0c5d).

The two facts that redesign the transition rule:

  1. on the overlap frames the fixed tau vetoes, the max-gap C = 1
     candidate aligns with the phase constant at 2.5-4.1 deg -- the
     disagreement is a threshold artifact, not a physical-mode failure;
  2. the true null floor is consistent with O(h) (1.8e-3 at h = 0.03 ->
     8.6e-4 at h = 0.015) while the thin-neck quasi-null level is nearly
     h-independent (3.4e-2 -> 2.8e-2 at g = -0.02), so a tau placed
     between the floors recovers the overlap side ON THE SAMPLED GAP
     GRID: at h = 0.015, tau = 0.025 moves the first clean C = 1 from
     g = -0.05 to g = -0.02. (Not a production formula: an absolute-h
     threshold is not scale invariant and only one circular family was
     tested; the 0C-5d-2 design is a previous-clean-regime null floor.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from rank_threshold_audit_0c5d import frame  # noqa: E402


def test_vetoed_candidate_is_a_physical_mode():
    row = frame(-0.02, 0.03)
    assert row["tau0.05"]["status"] == "detector_disagreement"
    assert row["align_c1_deg"] < 5.0               # measured 3.98
    # true floor vs quasi-null neck mode, h = 0.03
    assert row["s1_rel"] < 5e-3                    # measured 1.76e-3
    assert row["s2_rel"] > 1e-2                    # measured 3.41e-2


def test_resolution_dependent_tau_recovers_overlap_side():
    row = frame(-0.02, 0.015)
    # fixed legacy tau still vetoes ...
    assert row["tau0.05"]["status"] == "detector_disagreement"
    # ... while a tau between the O(h) floor and the neck level is clean
    assert row["tau0.025"]["status"] == "clean"
    assert row["tau0.025"]["c_gap"] == 1
    assert row["align_c1_deg"] < 5.0               # measured 2.56
    # floor separation grows under refinement (vs 1.76e-3 at h = 0.03)
    assert row["s1_rel"] < 1.2e-3                  # measured 8.46e-4
