"""Paper figure: local-contact merger of two ellipses, canonical
one-stroke H=768 run (results/two_ellipses/two_ellipses_canonical_768.*).

Top row: snapshots at t=0, t_1 (activation), t_3 (quotient), t_4 (splice; zero-time spliced
state), +200 steps, end. Bottom row: (a) perimeter and minimal gap;
(b) isoperimetric ratio and r_mean/R_eq after the splice; (c) merged
area error and barycenter drift.

Usage: uv run python scripts/experiments/paper_fig_two_ellipses.py
Output: results/reports/figs/two_ellipses_merger.pdf (+ numbers json).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from paper_fig_common import (  # noqa: E402
    draw_cloud, nearest_state, np, plt, run_args, torch,
)

R = Path("results/two_ellipses")
FIG = Path("results/reports/figs")
DT = 1e-5
R_EQ = math.sqrt(2.0 * 0.4 * 1.0)
EVENTS = [(1435, r"$t_1$"), (1443, r"$t_2$"), (1829, r"$t_3$"),
          (2017, r"$t_4$")]


def main():
    torch.set_default_dtype(torch.float64)
    FIG.mkdir(parents=True, exist_ok=True)
    args = run_args("two_ellipses_canonical_768", "two_ellipses_merger")
    ck = torch.load(R / f"{args.run}_states.pt", weights_only=True)
    d = json.load(open(R / f"{args.run}.json"))
    ser = d["series"]
    meta = d["meta"]
    global EVENTS
    cand = [meta.get("activated_at"), meta.get("bfiner_at"),
            meta.get("quotient_at"), meta.get("spliced_at")]
    steps_ev = []
    for x in cand:
        if x is not None and int(x) not in steps_ev:
            steps_ev.append(int(x))
    if steps_ev:
        EVENTS = [(k, rf"$t_{{{j + 1}}}$") for j, k in enumerate(steps_ev)]
    t1 = EVENTS[0][0]
    t_sp = int(meta.get("spliced_at") or EVENTS[-1][0])
    lab_sp = [lab for k, lab in EVENTS if k == t_sp][0]
    t_q = int(meta.get("quotient_at") or EVENTS[-2][0])
    lab_q = [lab for k, lab in EVENTS if k == t_q][0]
    last = max(k for k in ck if isinstance(k, int))
    t = np.array([r["t"] for r in ser])
    step = np.array([r["step"] for r in ser])
    P = np.array([r["P_frozen"] for r in ser])
    A = np.array([r["A"] if r.get("phase") == "single_loop"
                  else r["A"][0] + r["A"][1] for r in ser])
    gmin = np.array([r.get("g_min", np.nan) if r.get("g_min") is not None
                     else np.nan for r in ser])
    iso = np.array([r.get("isoperimetric", np.nan) for r in ser])
    rr = np.array([r.get("r_mean_over_req", np.nan) for r in ser])
    bar = np.array([r["bar"] if isinstance(r["bar"][0], (int, float))
                    else [np.nan, np.nan] for r in ser])
    single = np.array([r.get("phase") == "single_loop" for r in ser])
    i_sp = int(np.argmax(single))

    snaps = [(0, r"$t=0$", None), (nearest_state(ck, t1), r"$t_1$", None),
             (nearest_state(ck, t_q), lab_q, None),
             (t_sp, lab_sp, f"spliced_{t_sp}"),
             (nearest_state(ck, t_sp + 200), lab_sp + r"$+200h$", None),
             (nearest_state(ck, last - 25, prefer_ge=False), "end", None)]
    fig = plt.figure(figsize=(7.0, 4.8))
    gs_top = fig.add_gridspec(1, 6, top=0.93, bottom=0.60, left=0.03,
                              right=0.90, wspace=0.08)
    gs = fig.add_gridspec(1, 3, top=0.46, bottom=0.10, left=0.08,
                          right=0.98, wspace=0.6)
    for i, (k, lab, key) in enumerate(snaps):
        ax = fig.add_subplot(gs_top[0, i])
        st = ck[key] if key else ck[k]
        sc = draw_cloud(ax, st["positions"], st["angles"], 1.2,
                        f"{lab}\nstep {k}, $N$={len(st['positions'])}",
                        normals_every=8, vmax=0.03)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R_EQ * np.cos(th), R_EQ * np.sin(th), "--", color="0.7",
                lw=0.6, zorder=1)
    cax = fig.add_axes([0.915, 0.62, 0.012, 0.29])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$w_i q_i$")

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, P, color="tab:blue", lw=1.0)
    ax.axhline(2 * np.pi * R_EQ, color="0.4", ls="--", lw=0.8)
    ax.text(t[-1], 2 * np.pi * R_EQ, r"$2\pi R_{\textup{eq}}$", ha="right",
            va="bottom", fontsize=6)
    ax.set_ylabel(r"$\widehat P(t)$")
    ax.set_xlabel("$t$")
    ylim = ax.get_ylim()
    for j, (k, lab) in enumerate(EVENTS):
        ax.axvline(k * DT, color="0.5", ls=":", lw=0.8)
        top = (j % 2 == 0)
        ax.text(k * DT, ylim[1] if top else ylim[0], lab, rotation=90,
                va="top" if top else "bottom", ha="right", fontsize=6)
    ax.set_title("(a) perimeter and event times")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t[single], iso[single], color="tab:green", lw=1.0)
    ax.set_xlabel("$t$")
    ax.text(0.97, 0.52, r"$4\pi A/\widehat P^{2}$ (left axis)",
            color="tab:green", transform=ax.transAxes, ha="right",
            fontsize=7)
    ax.tick_params(axis="y", colors="tab:green")
    ax2 = ax.twinx()
    ax2.plot(t[single], rr[single], color="tab:purple", lw=1.0)
    ax2.axhline(1.0, color="0.4", ls="--", lw=0.8)
    ax2.text(0.97, 0.42, r"$\overline r/R_{\textup{eq}}$ (right axis)",
             color="tab:purple", transform=ax.transAxes, ha="right",
             fontsize=7)
    ax2.tick_params(axis="y", colors="tab:purple")
    ax.set_title("(b) single-loop relaxation")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, (A - A[0]) / A[0] * 100, color="tab:red", lw=1.0)
    ax.set_ylabel(r"relative area error [\%]")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xlabel("$t$")
    for k, lab in EVENTS:
        ax.axvline(k * DT, color="0.5", ls=":", lw=0.8)
    ax.set_title("(c) merged-area conservation")
    fig.savefig(FIG / f"{args.out}.pdf")

    summ = dict(
        R_eq=R_EQ, N_before=int(len(ck[0]["positions"])),
        N_after=int(len(ck[snaps[-1][0]]["positions"])),
        events=dict(EVENTS), t_splice_index=i_sp,
        iso_first=float(iso[single][0]), iso_end=float(iso[-1]),
        r_over_req_end=float(rr[-1]),
        P_end=float(P[-1]), P_end_rel_2piReq=float(P[-1] / (2 * np.pi * R_EQ) - 1),
        dA_rel_end=float((A[-1] - A[0]) / A[0]),
        dA_rel_end_post_splice=float((A[-1] - A[i_sp]) / A[i_sp]),
        dA_rel_max=float(np.max(np.abs(A - A[0]) / A[0])),
        bar_end=[float(bar[-1, 0]), float(bar[-1, 1])],
        dbar_x_post_splice=float(bar[-1, 0] - bar[i_sp, 0]),
        steps=len(ser))
    (FIG / f"{args.out}_numbers.json").write_text(
        json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
