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
    # area, barycenter, circularity and mean radius: the order-free
    # quantities of the paper (A = 1/2 sum w x.u, 4 pi A / P^2, mean
    # radius about the barycenter 1/(2A) sum w |x|^2 u). Runs made
    # before these were recorded fall back to the polygon values.
    order_free = "A_cur" in ser[0]
    if not order_free:
        print("WARNING: run has no order-free series (A_cur); polygon "
              "area / length are plotted instead")
    kA = "A_cur" if order_free else "A"
    kbar = "bar_cur" if order_free else "bar"
    A = np.array([r[kA] if r.get("phase") == "single_loop"
                  else r[kA][0] + r[kA][1] for r in ser])
    gmin = np.array([r.get("g_min", np.nan) if r.get("g_min") is not None
                     else np.nan for r in ser])
    iso = np.array([r.get("circ" if order_free else "isoperimetric",
                          np.nan) for r in ser])
    rr = np.array([r.get("r_mean_cur_over_req" if order_free
                         else "r_mean_over_req", np.nan) for r in ser])
    bar = np.array([r[kbar] if isinstance(r[kbar][0], (int, float))
                    else [np.nan, np.nan] for r in ser])
    # barycenter of each loop before the merging of the constraints
    # (order-free and invariant under translations; evaluated on the
    # stored states when the series does not carry it)
    import two_ellipses_benchmark as teb
    from src.torch.oriented_varifold.mass import (
        compute_recommended_params,
    )
    n_el = int(len(ck[0]["positions"])) // 2
    m1 = torch.zeros(2 * n_el, dtype=torch.bool)
    m1[:n_el] = True
    delta, tau = map(float, compute_recommended_params(
        teb.build_cloud(float(meta.get("rotate_deg", 0.0)))[0]
        .positions))

    def bar_cen_of(st):
        pos, ang = st["positions"], st["angles"]
        nor = torch.stack([ang.cos(), ang.sin()], 1)
        m = teb.resolve_m(pos, nor, delta, tau)
        return [teb.centered_barycenter(pos[s_], nor[s_], m[s_])
                for s_ in (m1, ~m1)]
    pre = sorted(k for k in ck if isinstance(k, int) and k <= t_q
                 and len(ck[k]["positions"]) == 2 * n_el)
    b0 = bar_cen_of(ck[pre[0]])
    dbar_pre, n_pre = 0.0, len(pre)
    for k in pre:
        bk = bar_cen_of(ck[k])
        dbar_pre = max(dbar_pre, max(
            math.hypot(bk[j][0] - b0[j][0], bk[j][1] - b0[j][1])
            for j in (0, 1)))
    two = [r for r in ser if r.get("phase") != "single_loop"
           and r["step"] < t_q]
    dbar_poly = max(max(r["dbar"]) for r in two)
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
    # 1 x 4 with a spacer column between (b) and (c): the y label of (c)
    # must not stand next to the right axis ticks of (b)
    gs = fig.add_gridspec(1, 4, top=0.46, bottom=0.10, left=0.08,
                          right=0.98, wspace=0.5,
                          width_ratios=[1.0, 1.0, 0.28, 1.0])
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
    ax.plot(t, P, color="#0072B2", lw=1.0)
    ax.axhline(2 * np.pi * R_EQ, color="0.4", ls="--", lw=0.8)
    ax.set_ylabel(r"$\widehat P^{\,n}$")
    ax.set_xlabel("$t$")
    ylim = ax.get_ylim()
    for j, (k, lab) in enumerate(EVENTS):
        ax.axvline(k * DT, color="0.5", ls=":", lw=0.8)
        if j == 1:
            # t_2 and t_3 coincide at this scale: only t_1 and t_3 are
            # labelled (the caption states the coincidence)
            continue
        ax.text(k * DT, ylim[1], lab, rotation=90, va="top", ha="right",
                fontsize=6)
    ax.set_title("(a) perimeter term and event times")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t[single], iso[single], color="#009E73", lw=1.0)
    ax.set_xlabel("$t$")
    ax.text(0.97, 0.52, r"$4\pi A/(\widehat P^{\,n})^{2}$ (left axis)",
            color="#009E73", transform=ax.transAxes, ha="right",
            fontsize=7)
    ax.tick_params(axis="y", colors="#009E73")
    ax2 = ax.twinx()
    ax2.plot(t[single], rr[single], color="#CC79A7", lw=1.0)
    ax2.axhline(1.0, color="0.4", ls="--", lw=0.8)
    ax2.text(0.97, 0.42, r"$\overline r/R_{\textup{eq}}$ (right axis)",
             color="#CC79A7", transform=ax.transAxes, ha="right",
             fontsize=7)
    ax2.tick_params(axis="y", colors="#CC79A7")
    ax.set_title("(b) single-loop relaxation")

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(t, (A - A[0]) / A[0] * 100, color="#D55E00", lw=1.0)
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
        dA_rel_max_step=int(step[np.argmax(np.abs(A - A[0]))]),
        dA_rel_at_splice=float((A[i_sp] - A[0]) / A[0]),
        dA_rel_min_post_splice=float(np.min(A[i_sp:] - A[0]) / A[0]),
        dA_rel_min_post_splice_step=int(
            step[i_sp + np.argmin(A[i_sp:])]),
        order_free=bool(order_free),
        dbar_pre_merge_max=float(dbar_pre),
        dbar_pre_merge_n_states=int(n_pre),
        dbar_pre_merge_max_polygon=float(dbar_poly),
        circ_first=float(iso[single][0]), circ_end=float(iso[-1]),
        dA_rel_pre_splice=float((A[i_sp - 1] - A[0]) / A[0]),
        bar_end=[float(bar[-1, 0]), float(bar[-1, 1])],
        dbar_x_post_splice=float(bar[-1, 0] - bar[i_sp, 0]),
        steps=len(ser))
    (FIG / f"{args.out}_numbers.json").write_text(
        json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
