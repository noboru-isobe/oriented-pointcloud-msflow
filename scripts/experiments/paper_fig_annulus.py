"""Paper figure: exact concentric merger (disk + annulus), final H=1024
t=0 run (results/exact_merger/exact_merger_final_1024.*).

Top row: point-cloud snapshots at t=0, t_1 (phases connect), t_2 (quotient), T_c, t_4 (compression),
end (colored by the effective mass m_i q_i, normals every few points).
Bottom row: (a) radii R_outer / R_hole / R_disk vs the exact radial ODE
up to T_c and R_eq after; (b) relative area error and perimeter;
(c) N(t) and the four event times.

Usage: uv run python scripts/experiments/paper_fig_annulus.py
Output: results/reports/figs/annulus_merger.pdf (+ .json with the
numbers quoted in the paper).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from paper_fig_common import (  # noqa: E402
    draw_cloud, nearest_state, np, plt, run_args, torch,
)
from scipy.integrate import solve_ivp  # noqa: E402

OUT = Path("results/exact_merger")
FIG = Path("results/reports/figs")
DT = 1e-5
T_C = 0.020347043818204673
R_EQ = 0.9055385138149781
R1, R2, R3 = 0.35, 0.55, 1.0
EVENTS = [(1521, r"$t_1$"), (1919, r"$t_2$"),
          (2167, r"$t_3$"), (2292, r"$t_4$")]


def exact_radii(t):
    a = lambda rp, rm: (1.0 / rp + 1.0 / rm) / math.log(rp / rm)
    rhs = lambda _t, y: [-a(y[0], y[1]) / y[0], -a(y[0], y[1]) / y[1]]
    sol = solve_ivp(rhs, [0.0, T_C], [R3, R2], rtol=1e-12, atol=1e-14,
                    dense_output=True)
    tt = np.minimum(t, T_C)
    y = sol.sol(tt)
    return y[0], y[1]


def main():
    torch.set_default_dtype(torch.float64)
    FIG.mkdir(parents=True, exist_ok=True)
    args = run_args("exact_merger_final_1024", "annulus_merger")
    ck = torch.load(OUT / f"{args.run}_states.pt", weights_only=True)
    d = json.load(open(OUT / f"{args.run}.json"))
    ser = d["series"]
    meta = d["meta"]
    # event times from the run metadata (fallback: the paper's grid run);
    # steps that coincide (BIE: the merge of the metric and the quotient
    # happen at the same step) are merged into one event
    global EVENTS
    cand = [meta.get("merged_at_step"), meta.get("quotient_at"),
            meta.get("raw_vis_diverged_at"), meta.get("comp_committed_at")]
    steps_ev = []
    for x in cand:
        if x is not None and int(x) not in steps_ev:
            steps_ev.append(int(x))
    if steps_ev:
        EVENTS = [(k, rf"$t_{{{j + 1}}}$") for j, k in enumerate(steps_ev)]
    t_first, t_last = EVENTS[0][0], EVENTS[-1][0]
    last = max(k for k in ck if isinstance(k, int))
    t = np.array([r["t"] for r in ser])
    step = np.array([r["step"] for r in ser])
    Ro = np.array([r["R_outer"] for r in ser])
    Rh = np.array([r["R_hole"] if r["R_hole"] is not None else np.nan
                   for r in ser])
    Rd = np.array([r["R_disk"] if r["R_disk"] is not None else np.nan
                   for r in ser])
    V = np.array([r["vol_current"] for r in ser])
    P = np.array([r["perimeter"] for r in ser])
    Np = np.array([r["n_points"] for r in ser])
    snaps = [(0, r"$t=0$"), (nearest_state(ck, int(round(T_C / DT)), prefer_ge=False),
                             r"$t\approx T_c$")]
    snaps += [(nearest_state(ck, k), lab) for k, lab in EVENTS]
    snaps += [(nearest_state(ck, t_last + 200), EVENTS[-1][1] + r"$+200h$"),
              (nearest_state(ck, last - 25, prefer_ge=False), "end")]
    while len(snaps) > 6:          # keep six panels (drop T_c, then +200h)
        snaps.pop(1 if len(snaps) > 7 else -2)
    fig = plt.figure(figsize=(7.0, 4.8))
    gs_top = fig.add_gridspec(1, 6, top=0.93, bottom=0.60, left=0.03,
                              right=0.90, wspace=0.08)
    gs = fig.add_gridspec(2, 6, top=0.46, bottom=0.10, left=0.08,
                          right=0.98, wspace=1.2)
    for i, (k, lab) in enumerate(snaps):
        ax = fig.add_subplot(gs_top[0, i])
        st = ck[k]
        sc = draw_cloud(ax, st["positions"], st["angles"], 1.15,
                        f"{lab}\nstep {k}, $N$={len(st['positions'])}",
                        normals_every=8, vmax=0.03)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R_EQ * np.cos(th), R_EQ * np.sin(th), "--", color="0.7",
                lw=0.6, zorder=1)
    cax = fig.add_axes([0.915, 0.62, 0.012, 0.29])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$w_i q_i$")

    # (a) radii vs exact ODE
    ax = fig.add_subplot(gs[:, 0:2])
    Ro_ex, Rh_ex = exact_radii(t)
    ax.plot(t, Ro, color="tab:blue", lw=1.2, label=r"$R_{\textup{outer}}$")
    ax.plot(t, Rh, color="tab:orange", lw=1.2, label=r"$R_{\textup{hole}}$")
    ax.plot(t, Rd, color="tab:green", lw=1.2, label=r"$R_{\textup{disk}}$")
    m = t <= T_C
    ax.plot(t[m], Ro_ex[m], "k:", lw=0.9, label="exact ODE")
    ax.plot(t[m], Rh_ex[m], "k:", lw=0.9)
    ax.axhline(R1, color="k", ls=":", lw=0.9)
    ax.axhline(R_EQ, color="0.4", ls="--", lw=0.8, label=r"$R_{\textup{eq}}$")
    ax.axvline(T_C, color="0.3", lw=0.7)
    ax.set_xlabel("$t$")
    ax.set_ylabel("radius")
    ax.legend(loc="center right", ncol=1)
    ax.set_title("(a) radii vs. exact solution")

    # (b) perimeter with the event times
    ax = fig.add_subplot(gs[:, 2:4])
    ax.plot(t, P, color="tab:blue", lw=1.0)
    ax.set_ylabel(r"$\widehat P(t)$")
    ax.set_xlabel("$t$")
    ylim = ax.get_ylim()
    for j, (k, lab) in enumerate(EVENTS):
        ax.axvline(k * DT, color="0.5", ls=":", lw=0.8)
        top = (j % 2 == 0)
        ax.text(k * DT, ylim[1] if top else ylim[0], lab, rotation=90,
                va="top" if top else "bottom", ha="right", fontsize=6)
    ax.axvline(T_C, color="0.2", lw=0.8)
    ax.text(T_C, 0.5 * (ylim[0] + ylim[1]), r"$T_c$", va="center", ha="left",
            fontsize=6)
    ax.set_title("(b) perimeter and event times")

    # (c) area conservation
    ax = fig.add_subplot(gs[:, 4:6])
    ax.plot(t, (V - V[0]) / V[0] * 100, color="tab:red", lw=1.0)
    ax.set_ylabel(r"relative area error [\%]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set_xlabel("$t$")
    for k, lab in EVENTS:
        ax.axvline(k * DT, color="0.5", ls=":", lw=0.8)
    ax.set_title("(c) area conservation")
    fig.savefig(FIG / f"{args.out}.pdf")

    # numbers for the text
    pre = t <= 0.75 * T_C          # before the hole accelerates
    err = dict(
        disk=float(np.nanmax(np.abs(Rd[pre] - R1) / R1)),
        outer=float(np.nanmax(np.abs(Ro[pre] - Ro_ex[pre]) / Ro_ex[pre])),
        hole=float(np.nanmax(np.abs(Rh[pre] - Rh_ex[pre]) / Rh_ex[pre])))
    post = step >= t_last
    summ = dict(T_c=T_C, T_c_step=int(round(T_C / DT)), R_eq=R_EQ,
                ode_max_rel_err_state_I=err,
                R_out_end=float(Ro[-1]),
                R_out_rel_R_eq=float(abs(Ro[-1] - R_EQ) / R_EQ),
                R_out_post_comp_drift=float(Ro[post].max() - Ro[post].min()),
                area_rel_err_max=float(np.max(np.abs(V - V[0]) / V[0])),
                N_before=int(Np[0]), N_after=int(Np[-1]),
                events=dict(EVENTS), steps=len(ser),
                V_sharp_drift_state_III=None)
    (FIG / f"{args.out}_numbers.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
