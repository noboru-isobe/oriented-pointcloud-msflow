"""0M-1 analysis: rotation / subcell-shift / jitter discrimination.

Reads exact_merger_0m1_*.json (38-step windows from endgame3 state
1500 under the L_m production baseline) and reports, per arm and
loop: the final complex z4, its GROWTH over the window (final minus
initial, complex -- the inherited deformation subtracts out), the
growth phase in the world frame and in the cloud frame (world minus
the applied rotation), the growth amplitude, and the stop step.

Discrimination table (reviewer 0M-1):
  grid channel      -> growth phase fixed in WORLD frame across
                       rotations; jitter seeds converge to the same
                       phase
  particle channel  -> growth phase follows the CLOUD frame
  scatter alias     -> growth amplitude varies with subcell shift
  isotropic layer   -> no deterministic phase
"""

import json
import math
from pathlib import Path

OUT = Path("results/exact_merger")
ARMS = [
    ("rot0", 0.0), ("rot11", 11.25), ("rot22", 22.5),
    ("rot33", 33.75),
    ("sx", 0.0), ("sy", 0.0), ("sxy", 0.0),
    ("jit1", 0.0), ("jit2", 0.0), ("jit3", 0.0),
]


def phs(re, im):
    return math.degrees((-math.atan2(im, re) / 4.0)
                        % (math.pi / 2.0))


def main():
    rows = []
    for tag, alpha in ARMS:
        p = OUT / f"exact_merger_0m1_{tag}.json"
        if not p.exists():
            print(f"-- missing {tag}")
            continue
        d = json.load(open(p))
        s = [r for r in d["series"]
             if r.get("m4") and "error" not in r["m4"]]
        if not s:
            print(f"-- no m4 records {tag}")
            continue
        first, last = s[0], s[-1]
        row = dict(tag=tag, alpha=alpha,
                   first=first["step"], last=last["step"])
        for role in ("disk", "hole"):
            f4 = first["m4"].get(role)
            l4 = last["m4"].get(role)
            if not f4 or not l4:
                continue
            gre, gim = l4["re"] - f4["re"], l4["im"] - f4["im"]
            ga = math.hypot(gre, gim)
            pw = phs(gre, gim)
            pc = (pw - alpha) % 90.0
            row[role] = dict(
                A4_final=l4["A4"], growth_amp=ga,
                phase_world=pw, phase_cloud=pc,
                phase_final_world=l4["phi4_deg"])
        rows.append(row)
        dk, hl = row.get("disk", {}), row.get("hole", {})
        print(f"[{tag:5s} a={alpha:5.2f}] steps {row['first']}-"
              f"{row['last']}  disk grow {dk.get('growth_amp', 0):.2e}"
              f" phW {dk.get('phase_world', -1):5.1f} "
              f"phC {dk.get('phase_cloud', -1):5.1f} | hole grow "
              f"{hl.get('growth_amp', 0):.2e} phW "
              f"{hl.get('phase_world', -1):5.1f} phC "
              f"{hl.get('phase_cloud', -1):5.1f}", flush=True)
    outp = Path("results/reports/phase3c0m1_discrimination.json")
    outp.write_text(json.dumps(rows, indent=1))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
