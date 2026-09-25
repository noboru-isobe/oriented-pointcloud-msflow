"""0C-5d-0 regressions: the hysteretic rank state machine.

Unit tests drive the machine with synthetic evidence and bases; the
integration tests pin the static audit of
scripts/experiments/rank_state_machine_0c5d0.py (raw tables in
results/rank_state_0c5d0). Design being pinned:

- weak_gap holds, ambiguity holds, switches need k_persist consecutive
  clean frames PLUS physical-mode validation;
- the raw-crossing artificial C = 1 modes carry enormous gap ratios (up to
  2.5e5 -- a single-criterion detector would confidently switch) but sit
  62-72 deg away from the phase-constant direction and are rejected;
  hysteresis must never manufacture a valid merged boundary from an
  invalid support.
"""

import sys
from pathlib import Path

import torch

from src.torch.transport.bem_wasserstein import RankEvidence
from src.torch.transport.rank_state import (
    RankStateConfig,
    RankStateMachine,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))


def ev(status, c=1, gap=100.0):
    return RankEvidence(c_abs=c, c_gap=c, gap_ratio=gap,
                        abs_levels=(1e-3, 0.3, 0.5, 1.0), status=status)


def ev_disagree():
    return RankEvidence(c_abs=2, c_gap=1, gap_ratio=10.0,
                        abs_levels=(1e-3, 0.03, 0.5, 1.0),
                        status="detector_disagreement")


def test_state_machine_unit_logic():
    n = 8
    sqrt_w = torch.ones(n, dtype=torch.float64)
    const = (sqrt_w / sqrt_w.norm()).unsqueeze(1)      # aligned candidate
    bad = torch.zeros(n, 1, dtype=torch.float64)       # orthogonal to const
    bad[0, 0] = 2 ** -0.5
    bad[1, 0] = -(2 ** -0.5)
    two = torch.linalg.qr(torch.randn(n, 2, dtype=torch.float64))[0]

    m = RankStateMachine(rank=2, config=RankStateConfig(k_persist=3))

    # same-rank weak evidence holds; ambiguity holds
    assert m.update(ev("weak_gap", c=2), two, sqrt_w).action == "hold_weak"
    assert m.update(ev_disagree(), const, sqrt_w).action == "hold_ambiguous"
    assert m.rank == 2

    # misaligned clean C=1 candidate: rejected, never pending
    d = m.update(ev("clean", c=1), bad, sqrt_w)
    assert d.action == "reject_artificial" and m.rank == 2
    assert d.align_const_deg > 45.0

    # aligned clean C=1 evidence must persist k_persist CONSECUTIVE times:
    # an interrupting weak frame resets the counter entirely
    assert m.update(ev("clean", c=1), const, sqrt_w).pending == 1
    assert m.update(ev("weak_gap", c=1), const, sqrt_w).action == "hold_weak"
    assert m.update(ev("clean", c=1), const, sqrt_w).pending == 1
    assert m.update(ev("clean", c=1), const, sqrt_w).pending == 2
    d = m.update(ev("clean", c=1), const, sqrt_w)
    assert d.action == "switch" and m.rank == 1

    # an ambiguity resets any new pending run
    m2 = RankStateMachine(rank=2, config=RankStateConfig(k_persist=2))
    m2.update(ev("clean", c=1), const, sqrt_w)
    m2.update(ev_disagree(), const, sqrt_w)
    assert m2.update(ev("clean", c=1), const, sqrt_w).pending == 1
    assert m2.rank == 2


def test_clean_union_sequence_switches_once_after_hysteresis():
    """Static audit pin (spacing 0.03, measured): clean C = 2 down to
    g = +0.02, weak/ambiguous band [+0.01, -0.04] HELD at rank 2, clean
    C = 1 from g = -0.05 with phase-constant alignment 3.8 deg, switch
    completed at g = -0.10 after k_persist = 3 frames. No rejections on
    the valid union family."""
    from rank_state_machine_0c5d0 import UNION_GAPS, run_sequence

    rows = run_sequence("union", 0.03, UNION_GAPS)
    switches = [r for r in rows if r["action"] == "switch"]
    assert len(switches) == 1
    assert switches[0]["g"] == -0.10
    assert switches[0]["align_const_deg"] < 5.0     # measured 3.5
    # ambiguous band held at rank 2
    for r in rows:
        if r["status"] != "clean":
            assert r["rank"] == 2 and r["action"] in ("hold_weak",
                                                      "hold_ambiguous")
    assert sum(r["action"] == "reject_artificial" for r in rows) == 0
    assert rows[-1]["rank"] == 1


def test_raw_crossing_never_switches():
    """Static audit pin (spacing 0.03, measured): the raw two-loop overlap
    produces clean-LOOKING C = 1 evidence with gap ratios 575..2e4, but the
    candidate mode sits 62-71 deg from W^{1/2}1 (threshold 15) and 62-71
    deg from the previous subspace -- all rejected; the machine never
    leaves rank 2 and the caller sees explicit rejections (Gate-B
    conditional territory, deferred to 0D/0E support diagnostics)."""
    from rank_state_machine_0c5d0 import RAW_GAPS, run_sequence

    rows = run_sequence("raw", 0.03, RAW_GAPS)
    assert all(r["rank"] == 2 for r in rows)
    assert not any(r["action"] == "switch" for r in rows)
    rejected = [r for r in rows if r["action"] == "reject_artificial"]
    assert len(rejected) >= 4
    assert all(r["align_const_deg"] > 45.0 for r in rejected)
