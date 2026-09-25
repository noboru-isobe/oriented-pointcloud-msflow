"""P1b-c0 static pins (no solver run) for the corrected seed:

1. branch parity: sinuous dtheta_+ = dtheta_-, varicose
   dtheta_+ = -dtheta_- (the P1b-1 implementation flipped the
   upper-wall sign: an O(eps) defect).
2. taper contact: d = d' = d'' = 0 at the support edge x1 (C^2
   quintic taper; the old exp(-(2x/L)^8) x Boolean-mask seed had
   chi(x1) = 0.944, an O(eps) hard edge).
3. stored vs graph normal on the discrete perturbed wall (central
   FD): new seed must be consistent to O(h^2 d'''); the old seed's
   mask-edge kink is reported for reference.
4. FFT strip Fourier-multiplier diagnostic (reviewer P1b-c0): the
   seeds are NOT eigenfunctions of M_v = |D|^3 tanh(w|D|/2) /
   M_s = |D|^3 coth(w|D|/2); compute the exact strip prediction
   A(t) = <g_k, exp(-c_mob t M) f_k> for the windowed observable
   g_k = 1_{|x|<x1} cos(kx), fit the SAME windows as the 2D runs,
   and compare to the pure-mode rate c_mob k^3 {tanh,coth}(kw/2).
   This quantifies how much of the measured varicose excess the
   fixed window edge (spectral leakage) alone can produce, for the
   old (hard-edge) and corrected (C^2) seeds.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dumbbell_benchmark import H_REF, seed_mode  # noqa: E402

W0 = 0.29
L_STR = 3.6
EPS = 0.004
WINDOWS = [(0.001, 0.004), (0.001, 0.006), (0.002, 0.008),
           (0.002, 0.010)]
C_MOB = 1.3022  # s18 calibration (P1b-3 judge)


def flat_walls(n_half):
    """Synthetic ordered flat-wall pair at spacing H_REF."""
    xs = torch.arange(-n_half, n_half + 1, dtype=torch.float64) \
        * H_REF
    top = torch.stack([xs, torch.full_like(xs, W0 / 2)], 1)
    bot = torch.stack([xs, torch.full_like(xs, -W0 / 2)], 1)
    pos = torch.cat([top, bot])
    ang = torch.cat([torch.full_like(xs, math.pi / 2),
                     torch.full_like(xs, -math.pi / 2)])
    return xs, pos, ang


def pin_parity_and_normals(k, mode):
    xs, pos, ang = flat_walls(int(0.45 * L_STR / H_REF))
    pos2, ang2 = seed_mode(pos, ang, W0, L_STR, k, EPS, mode)
    n = len(xs)
    dth_t = (ang2[:n] - ang[:n]).numpy()
    dth_b = (ang2[n:] - ang[n:]).numpy()
    par = dth_t + dth_b if mode == "varicose" else dth_t - dth_b
    parity_err = float(np.abs(par).max())
    # stored vs graph normal (central FD on each wall, matched to
    # atan2 of the exact outward-normal convention)
    errs = []
    for s, sl in ((+1, slice(0, n)), (-1, slice(n, 2 * n))):
        p = pos2[sl].numpy()
        slope = (p[2:, 1] - p[:-2, 1]) / (p[2:, 0] - p[:-2, 0])
        th_graph = np.arctan2(s * np.ones_like(slope), -s * slope)
        d = ang2[sl].numpy()[1:-1] - th_graph
        errs.append(np.abs(np.arctan2(np.sin(d), np.cos(d))).max())
    return parity_err, float(max(errs)), (pos2, ang2, xs, n)


def pin_taper_contact():
    x1 = 0.7 * L_STR / 2
    x0 = 0.5 * x1

    def chi(x):
        u = min(max((x1 - abs(x)) / (x1 - x0), 0.0), 1.0)
        return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)
    # contact order at the edge: chi(x1 - h) = O(h^p) with p >= 3
    # certifies chi = chi' = chi'' = 0 at x1 (one-sided; chi == 0
    # outside by construction)
    d0 = chi(x1)
    v1, v2 = chi(x1 - 1e-2), chi(x1 - 1e-3)
    p = math.log(v1 / v2) / math.log(10.0)
    return d0, v2, p


def old_seed_window(x):
    x1 = 0.7 * L_STR / 2
    return np.where(np.abs(x) < x1,
                    np.exp(-(2.0 * x / L_STR) ** 8), 0.0)


def new_seed_window(x):
    x1 = 0.7 * L_STR / 2
    x0 = 0.5 * x1
    u = np.clip((x1 - np.abs(x)) / (x1 - x0), 0.0, 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def old_normal_kink(k, mode):
    """FD graph-normal error of the OLD seed (hard edge + flipped
    upper sign), for reference against the new pin."""
    xs, pos, ang = flat_walls(int(0.45 * L_STR / H_REF))
    x = pos[:, 0]
    W = torch.from_numpy(old_seed_window(x.numpy()))
    sgn = -torch.sign(pos[:, 1]) if mode == "varicose" \
        else torch.ones_like(pos[:, 1])
    dy = EPS * sgn * torch.cos(k * x) * W
    dW = -(16.0 / L_STR) * (2.0 * x / L_STR) ** 7 * \
        torch.exp(-(2.0 * x / L_STR) ** 8) * (x.abs() < 0.7 * L_STR / 2)
    ddy = EPS * sgn * (-k * torch.sin(k * x) * torch.cos(x * 0) * W
                       + torch.cos(k * x) * dW)
    nor_y = ang.sin()
    pos2 = pos.clone()
    pos2[:, 1] += dy
    ang2 = ang - nor_y * ddy  # old (flipped) update
    n = len(xs)
    errs = []
    for s, sl in ((+1, slice(0, n)), (-1, slice(n, 2 * n))):
        p = pos2[sl].numpy()
        slope = (p[2:, 1] - p[:-2, 1]) / (p[2:, 0] - p[:-2, 0])
        th_graph = np.arctan2(s * np.ones_like(slope), -s * slope)
        d = ang2[sl].numpy()[1:-1] - th_graph
        errs.append(np.abs(np.arctan2(np.sin(d), np.cos(d))).max())
    return float(max(errs))


def strip_response(k, mode, window_fn, c_mob):
    """A(t) = <g, exp(-c_mob t M) f>: exact evaluation of the
    frozen-width infinite-strip linear semigroup for the actual
    windowed seed f and observable g (NOT a finite-capsule solution).
    Returns (t, A)."""
    N, X = 1 << 16, 32.0
    x = (np.arange(N) - N // 2) * (2 * X / N)
    f = window_fn(x) * np.cos(k * x)
    g = (np.abs(x) < 0.7 * L_STR / 2) * np.cos(k * x)
    xi = 2 * np.pi * np.fft.fftfreq(N, d=2 * X / N)
    axi = np.abs(xi)
    if mode == "varicose":
        m = axi ** 3 * np.tanh(axi * W0 / 2)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            m = axi ** 3 / np.tanh(axi * W0 / 2)
        m[axi == 0] = 0.0
    fh, gh = np.fft.fft(f), np.fft.fft(g)
    t = np.arange(0.0, 0.0125, 2.5e-5)
    A = np.array([float(np.real(np.vdot(gh, fh * np.exp(
        -c_mob * tt * m)))) for tt in t])
    return t, A


def fit_windows(t, A):
    lams = []
    for lo, hi in WINDOWS:
        msk = (t >= lo) & (t <= hi) & (A > 0)
        c = np.polyfit(t[msk], np.log(A[msk]), 1)
        lams.append(-float(c[0]))
    return lams


def strip_prediction(k, mode, window_fn, c_mob=C_MOB):
    """Windowed fitted rates of the strip semigroup at mobility c_mob.
    NOTE (reviewer P1b-c corrigendum): the fitted rate on a fixed
    window is NOT linear in c_mob (the windowed seed is a mode
    mixture), so the semigroup must be re-evaluated for every c_mob;
    never rescale fitted rates."""
    return fit_windows(*strip_response(k, mode, window_fn, c_mob))


def main():
    out = {"pins": {}, "strip_multiplier": {}}
    print("== P1b-c0 pins (corrected seed) ==")
    d0, v_near, p = pin_taper_contact()
    ok = d0 == 0.0 and p >= 2.9
    print(f"taper contact at x1: chi(x1)={d0:.1e}, "
          f"chi(x1-1e-3)={v_near:.2e}, contact order p={p:.2f} "
          f"(need >= 3)  {'PASS' if ok else 'FAIL'}")
    out["pins"]["taper"] = dict(chi=d0, near=v_near,
                                order=round(p, 3), ok=bool(ok))
    for mode in ("sinuous", "varicose"):
        for k in (1.8, 2.7):
            pe, ne, _ = pin_parity_and_normals(k, mode)
            ok = pe < 1e-13 and ne < 5e-4
            print(f"{mode:8s} k={k}: parity_err={pe:.2e} "
                  f"graph_normal_err={ne:.2e} rad  "
                  f"{'PASS' if ok else 'FAIL'}")
            out["pins"][f"{mode}_k{k}"] = dict(
                parity=pe, graph_normal=ne, ok=bool(ok))
    for mode in ("sinuous", "varicose"):
        kink = old_normal_kink(1.8, mode)
        print(f"[reference] OLD seed {mode} k=1.8 graph-normal "
              f"error = {kink:.3f} rad ({math.degrees(kink):.1f} deg)")
        out["pins"][f"old_kink_{mode}"] = kink

    print("\n== strip Fourier-multiplier prediction (windowed "
          "observable, same fit windows; c_mob = %.4f) ==" % C_MOB)
    for mode, ks in (("varicose", (1.8, 2.2, 2.7)),
                     ("sinuous", (1.8, 2.7))):
        for k in ks:
            z = k * W0 / 2
            sym = C_MOB * k ** 3 * (
                math.tanh(z) if mode == "varicose"
                else 1 / math.tanh(z))
            row = {}
            for name, wf in (("old", old_seed_window),
                             ("new", new_seed_window)):
                lams = strip_prediction(k, mode, wf)
                row[name] = [round(v, 3) for v in lams]
                print(f"{mode:8s} k={k} {name}: lam_windows="
                      + " ".join(f"{v:7.2f}" for v in lams)
                      + f"  pure-mode={sym:6.2f}"
                      f"  ratio(last)={lams[-1] / sym:.2f}")
            row["pure_mode"] = round(sym, 3)
            out["strip_multiplier"][f"{mode}_k{k}"] = row
    Path("results/pinchoff/p1bc_seed_pins.json").write_text(
        json.dumps(out, indent=1))
    print("\nwrote results/pinchoff/p1bc_seed_pins.json")


if __name__ == "__main__":
    main()
