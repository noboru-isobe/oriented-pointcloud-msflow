"""P1-1D (part 2): lift generated capsule profiles into the 1D
lubrication solver (reviewer program section 5) -- healing /
finite-time / infinite-time exemplars for pilot initial-data
selection, plus the P1-2 rate fixture for 1D/2D background
comparison.

Lift (CENV frame): ell = the |x| where H = 1.5 (w/2) (the SAME
fixed convention as p_eff_of_profile), xi = x/ell,
hhat(xi) = H(xi ell)/H(ell). Then hhat(+-1) = 1 and
hhat''(+-1) = ell^2 H''(ell)/H(ell) = P_eff, so the CENV boundary
data (h = 1, h_xx = P_eff) is exactly the lifted profile's own.
Time is reported in 1D units tau; the dimensional map back to the
2D solver clock is tau -> t = tau * ell^4 / (c_mob * H_ell)
(recorded, applied only once c_mob is fixed by P1-2 -- never fitted
to 2D trajectories).

Classification per the pre-registered table via classify_min_h
(finite-time C(T*-t)^alpha vs infinite-time exponential tail).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lubrication1d import classify_min_h, solve_neck  # noqa: E402
from src.torch.shapes.dumbbell import (  # noqa: E402
    CapsuleProfile,
    p_eff_of_profile,
)

FIXTURES = [
    # (name, R, L)  -- w = 0.29 throughout (pilot pre-registration)
    ("rate_R03_L36", 0.3, 3.6),      # P1-2 rate fixture (2D bg comp)
    ("pilot_low_R03_L03", 0.3, 0.3),         # P_eff < 1 class
    ("pilot_mod_R055_L10", 0.55, 1.0),       # moderate P_eff > 4
    ("pilot_high_R055_L20", 0.55, 2.0),      # higher P_eff
]
W = 0.29
MULT = 1.5


def lift(prof: CapsuleProfile):
    target = MULT * W / 2
    # bisection bracket ends at the lobe apex (H monotone up to
    # there; beyond the far cap H = 0 and the bracket collapses)
    lo, hi = 0.0, abs(prof.r["c"])
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if prof.H(mid) < target else (lo, mid)
    ell = 0.5 * (lo + hi)
    H_ell = prof.H(ell)
    xi = np.linspace(-1.0, 1.0, 401)
    h0 = np.array([prof.H(x * ell) / H_ell for x in xi])
    p_eff = ell * ell * prof.ddH(ell) / H_ell
    return ell, H_ell, h0, p_eff


def main():
    out = Path("results/pinchoff")
    res = {}
    for name, R, L in FIXTURES:
        prof = CapsuleProfile(R, R, L, W)
        ell, H_ell, h0, p_eff = lift(prof)
        p_chk = p_eff_of_profile(prof.H, prof.ddH, W, +1, MULT, x_max=abs(prof.r["c"]))
        assert abs(p_eff - p_chk) < 1e-9 * max(1, abs(p_chk))
        r = solve_neck(P=p_eff, t_end=3.0, dt=1e-5, h0=h0)
        fits = classify_min_h(r["t"], r["min_h"], 1e-3)
        res[name] = dict(
            R=R, L=L, w=W, ell=ell, H_ell=H_ell, P_eff=p_eff,
            status=r["status"],
            t_last=r["t"][-1] if r["t"] else None,
            min_h_last=float(r["h"].min()),
            min_h_series_head=[round(v, 5) for v in r["min_h"][:5]],
            fits=fits,
            time_map="t_2D = tau * ell^4 / (c_mob * H_ell)")
        print(name, f"P_eff={p_eff:.3f} ell={ell:.3f}",
              r["status"], f"min_h={res[name]['min_h_last']:.4f}",
              fits if isinstance(fits, dict) else "", flush=True)
    (out / "lifted_profiles_1d.json").write_text(
        json.dumps(res, indent=1, default=str))
    print("wrote", out / "lifted_profiles_1d.json")


if __name__ == "__main__":
    main()
