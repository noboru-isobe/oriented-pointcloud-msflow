"""Paper figure: schematic of the four topological events (Section 5.7),
drawn from the stored states of the production runs.

(i)  contact region at the switch of the visibility      (two ellipses, step 1450)
(ii) facing pairs at the merging of the constraints           (two ellipses, step 1850)
(iii) removal of the cancelling pair   (annulus, steps 2275 -> 2300)
(iv) reconnection of the arcs                             (two ellipses, step 2000 -> spliced_2017)

Usage: uv run python scripts/experiments/paper_fig_events.py
Output: results/reports/figs/events_schematic.pdf
"""
from __future__ import annotations

from pathlib import Path

from paper_fig_common import np, plt, torch  # noqa: E402
from paper_fig_common import SIGMA  # noqa: E402

FIG = Path("results/reports/figs")
ELL = Path("results/two_ellipses/two_ellipses_canonical_768_states.pt")
ANN = Path("results/exact_merger/exact_merger_final_1024_states.pt")
# steps of the four panels for the paper's grid runs; overridden from
# the run metadata when --ell-run / --ann-run are given
STEPS = dict(switch=1450, pairs=1850, ann_before=2275, ann_after=2300,
             splice_before=2000, splice_key="spliced_2017")


def _args():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--ell-run", default=None)
    ap.add_argument("--ann-run", default=None)
    ap.add_argument("--out", default="events_schematic")
    a = ap.parse_args()
    global ELL, ANN
    if a.ell_run:
        ELL = Path(f"results/two_ellipses/{a.ell_run}_states.pt")
        m = json.load(open(f"results/two_ellipses/{a.ell_run}.json"))["meta"]
        t_act, t_q, t_sp = (int(m["activated_at"]), int(m["quotient_at"]),
                            int(m["spliced_at"]))
        STEPS.update(switch=t_act, pairs=t_q, splice_before=t_q,
                     splice_key=f"spliced_{t_sp}")
    if a.ann_run:
        ANN = Path(f"results/exact_merger/{a.ann_run}_states.pt")
        m = json.load(open(f"results/exact_merger/{a.ann_run}.json"))["meta"]
        STEPS.update(ann_before=int(m["quotient_at"]),
                     ann_after=int(m["comp_committed_at"]) + 1)
    return a


def normals(ang):
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def pairwise_min_dist(a, b):
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return d.min(axis=1), d.argmin(axis=1)


def style(ax, xlim, ylim, title):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=3)


def _nearest(ck, k):
    keys = sorted(kk for kk in ck if isinstance(kk, int))
    if k in ck:
        return k
    ge = [kk for kk in keys if kk >= k]
    return ge[0] if ge else keys[-1]


def main():
    torch.set_default_dtype(torch.float64)
    args = _args()
    FIG.mkdir(parents=True, exist_ok=True)
    ell = torch.load(ELL, weights_only=True)
    ann = torch.load(ANN, weights_only=True)
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.05))
    c_left, c_right, c_new = "tab:blue", "tab:green", "tab:red"

    # (i) contact region at the switch of the visibility
    st = ell[_nearest(ell, STEPS['switch'])]
    p = st["positions"].numpy()
    left = p[:, 0] < 0
    dl, _ = pairwise_min_dist(p[left], p[~left])
    dr, _ = pairwise_min_dist(p[~left], p[left])
    ax = axs[0]
    ax.scatter(p[left, 0], p[left, 1], s=2.5, c=c_left, linewidths=0)
    ax.scatter(p[~left, 0], p[~left, 1], s=2.5, c=c_right, linewidths=0)
    for pts, d in ((p[left], dl), (p[~left], dr)):
        m = d < SIGMA
        ax.scatter(pts[m, 0], pts[m, 1], s=7, c=c_new, linewidths=0,
                   zorder=3)
    g = min(dl.min(), dr.min())
    style(ax, (-1.05, 1.05), (-1.25, 1.25), "(i) contact region")

    # (ii) facing pairs at the merging of the constraints
    st = ell[_nearest(ell, STEPS['pairs'])]
    p = st["positions"].numpy()
    n = normals(st["angles"].numpy())
    left = p[:, 0] < 0
    pl, pr, nl, nr = p[left], p[~left], n[left], n[~left]
    d, j = pairwise_min_dist(pl, pr)
    ax = axs[1]
    ax.scatter(pl[:, 0], pl[:, 1], s=6, c=c_left, linewidths=0, zorder=3)
    ax.scatter(pr[:, 0], pr[:, 1], s=6, c=c_right, linewidths=0, zorder=3)
    npairs = 0
    for i in range(len(pl)):
        if (nl[i] @ nr[j[i]]) <= -0.9 and d[i] < 3 * 2 * np.pi / 256:
            ax.plot([pl[i, 0], pr[j[i], 0]], [pl[i, 1], pr[j[i], 1]],
                    color="0.3", lw=0.6, zorder=2)
            npairs += 1
    q = 6
    for pts, nn in ((pl, nl), (pr, nr)):
        ax.quiver(pts[::q, 0], pts[::q, 1], nn[::q, 0], nn[::q, 1],
                  angles="xy", scale_units="xy", scale=12, width=0.006,
                  color="0.45", zorder=1)
    style(ax, (-0.32, 0.32), (-0.42, 0.42), "(ii) facing pairs")

    # (iii) compression: annulus before/after
    b, a = ann[_nearest(ann, STEPS['ann_before'])], ann[_nearest(ann, STEPS['ann_after'])]
    pb = b["positions"].numpy()
    pa = a["positions"].numpy()
    rb = np.linalg.norm(pb, axis=1)
    outer = rb > 0.7
    ax = axs[2]
    ax.scatter(pb[outer, 0], pb[outer, 1], s=2.5, c="0.75", linewidths=0)
    ax.scatter(pb[~outer, 0], pb[~outer, 1], s=4, c="0.35", linewidths=0,
               zorder=2)
    ax.scatter(pa[:, 0], pa[:, 1], s=2.5, c=c_left, linewidths=0, zorder=3)
    ax.annotate("", xy=(0.0, 0.0), xytext=(0.0, 0.37),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=c_new))
    ax.annotate("", xy=(0.905, 0.0), xytext=(0.99, 0.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=c_new))
    style(ax, (-1.1, 1.1), (-1.1, 1.1), "(iii) removal of the pair")

    # (iv) splice: before/after near the neck
    b, a = ell[_nearest(ell, STEPS['splice_before'])], ell[STEPS['splice_key']]
    pb = b["positions"].numpy()
    pa = a["positions"].numpy()
    da, _ = pairwise_min_dist(pa, pb)
    new = da > 0.75 * (2 * np.pi / 256)   # bridge particles: far from every old particle
    ax = axs[3]
    ax.scatter(pb[:, 0], pb[:, 1], s=4, c="0.75", linewidths=0)
    ax.scatter(pa[~new, 0], pa[~new, 1], s=6, c=c_left, linewidths=0,
               zorder=3)
    ax.scatter(pa[new, 0], pa[new, 1], s=9, c=c_new, linewidths=0, zorder=4)
    style(ax, (-0.6, 0.6), (-0.8, 0.8), "(iv) reconnection")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02,
                        wspace=0.06)
    fig.savefig(FIG / f"{args.out}.pdf")
    print("pairs drawn:", npairs, "bridge particles:", int(new.sum()),
          "N before/after splice:", len(pb), len(pa))


if __name__ == "__main__":
    main()
