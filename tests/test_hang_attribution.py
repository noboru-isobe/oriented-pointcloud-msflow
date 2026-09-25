"""Closure patch 2: hang_attribution v2 hard-timeout harness.

The parent must survive a child that blocks INSIDE a single long
operation (where the optimizer callback can never fire again): the
test hook streams one callback record and then sleeps, and the parent
must produce a timed_out JSON that preserves the streamed telemetry.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent


def test_hard_timeout_preserves_callback_telemetry(tmp_path):
    snap = tmp_path / "snap.pt"
    torch.save(dict(positions=torch.zeros(8, 2, dtype=torch.float64),
                    angles=torch.zeros(8, dtype=torch.float64),
                    shape="pair", kind="C2", step=1, wall=1.0,
                    median=0.1), snap)
    out = tmp_path / "out"
    env = dict(os.environ, HANG_ATTR_TEST_BLOCK="1")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "scripts/debug/hang_attribution.py",
         str(snap), "--kinds", "C2", "--timeout-s", "6",
         "--out", str(out)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    wall = time.time() - t0
    assert r.returncode == 0, r.stderr[-500:]
    assert wall < 60, "parent did not enforce the hard timeout"

    res = json.loads((out / "snap__C2.json").read_text())
    assert res["timed_out"] is True
    assert res["last_completed_iteration"] == 1
    assert res["last_callback"]["trust_radius"] == 0.1
    assert res["last_callback"]["trust_region_boundary_hit"] is False
    assert Path(res["callback_records_path"]).exists()
