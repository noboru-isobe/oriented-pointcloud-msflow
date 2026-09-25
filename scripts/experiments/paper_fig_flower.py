"""Paper figure: single-component sanity run (five-petal flower) under the
production stack (results/flower/flower_production.*).

Top: snapshots (t = 0, 0.002, 0.006, 0.015, 0.04) colored by w_i q_i.
Bottom: (a) visible perimeter, (b) circularity, (c) relative area error [%].
Output: results/reports/figs/flower_production.pdf (+ numbers json).
"""
from __future__ import annotations

import json
from pathlib import Path

from paper_fig_common import (  # noqa: E402
    draw_cloud, nearest_state, np, plt, run_args, torch,
)

R = Path("results/flower")
FIG = Path("results/reports/figs")


def main():
    torch.set_default_dtype(torch.float64)
    FIG.mkdir(parents=True, exist_ok=True)
    args = run_args("flower_production", "flower_production")
    ck = torch.load(R / f"{args.run}_states.pt", weights_only=True)
    d = json.load(open(R / f"{args.run}.json"))
    ser = d["series"]
    t = np.array([r["t"] for r in ser])
    P = np.array([r["P"] for r in ser])
    A = np.array([r["A"] for r in ser])
    c = np.array([r["circ"] for r in ser])
    R_eq = float(np.sqrt(A[0] / np.pi))
    snaps = [nearest_state(ck, k, prefer_ge=False)
             for k in (0, 200, 600, 1500, 3975)]
    fig = plt.figure(figsize=(7.0, 4.4))
    gs_top = fig.add_gridspec(1, 5, top=0.93, bottom=0.60, left=0.03,
                              right=0.90, wspace=0.08)
    gs = fig.add_gridspec(1, 3, top=0.46, bottom=0.11, left=0.08,
                          right=0.98, wspace=0.55)
    for i, k in enumerate(snaps):
        ax = fig.add_subplot(gs_top[0, i])
        st = ck[k]
        sc = draw_cloud(ax, st["positions"], st["angles"], 1.1,
                        f"step {k}\n$t={(k + 1) * d['meta']['dt']:.3f}$",
                        normals_every=8, vmax=0.03)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R_eq * np.cos(th), R_eq * np.sin(th), "--", color="0.7",
                lw=0.6, zorder=1)
    cax = fig.add_axes([0.915, 0.62, 0.012, 0.29])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$w_i q_i$")
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, P, color="tab:blue", lw=1.0)
    ax.axhline(2 * np.pi * R_eq, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\widehat P(t)$")
    ax.set_title("(a) visible perimeter")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, c, color="tab:green", lw=1.0)
    ax.axhline(1.0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$4\pi A/\widehat P^{\,2}$")
    ax.set_title("(b) circularity")
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, (A - A[0]) / A[0] * 100, color="tab:red", lw=1.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"relative area error [\%]")
    ax.set_title("(c) area conservation")
    fig.savefig(FIG / f"{args.out}.pdf")
    summ = dict(N=d["meta"]["N"], steps=len(ser), P0=float(P[0]),
                P_end=float(P[-1]), circ0=float(c[0]), circ_end=float(c[-1]),
                dA_rel_end=float((A[-1] - A[0]) / A[0]),
                dA_rel_step50=float((A[50] - A[0]) / A[0]),
                dA_rel_min=float(((A - A[0]) / A[0]).min()),
                R_eq=R_eq, P_end_rel_2piReq=float(P[-1] / (2 * np.pi * R_eq) - 1),
                max_s_end=float(ser[-1]["max_s"]),
                P_monotone=bool(np.all(np.diff(P) <= 1e-9)))
    (FIG / f"{args.out}_numbers.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
