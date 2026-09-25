"""Static parity of the block-diagonal BIE metric against the
production grid Poisson metric (1024^2, fill 0.04, box (-2,2)^2) and
against exact interior-Neumann values.

Sections
  A  exact modes: disk Fourier (k = 1..8), annulus radial -- grid, BIE
     and exact; relative errors.
  B  y-subspace Gram parity on separated configurations (ellipse, two
     ellipses at gap 0.1, annulus + disk): random admissible directions
     of the BIE stepper's own parametrization, projected onto the grid
     compatibility kernel; worst |W_grid/W_bie - 1| and the generalized
     eigenvalues of the two Gram matrices (max |lambda - 1|).
  C  merge window (report only): two ellipses at gap 0.8 ell -- the BIE
     merges the components and masks the cancelling pair, the grid
     phase is connected at this gap; same Gram statistics, plus the BIE
     block conditioning (sigma_tail) and admissible-subspace lambda_min.

Output: results/bie_metric/parity.json (+ parity.md summary).
Runs on CPU in float64; ~5-10 min (six 1024^2 sparse factorizations).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import torch  # noqa: E402

from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.shapes.generator import (  # noqa: E402
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402
from src.torch.transport.bie_wasserstein import BIEMetricConfig  # noqa: E402
from src.torch.transport.boundary_flux_grid import (  # noqa: E402
    BoundaryFluxPolicies,
)
from src.torch.transport.grid_wasserstein import (  # noqa: E402
    GridMetricConfig,
    GridWassersteinMetric,
)
from src.torch.transport.phase_grid import PhaseGridConfig  # noqa: E402
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64
DT_STEP = 1e-5
OUT = ROOT / "results" / "bie_metric"


def bie_cfg(delta, tau, **bie):
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.time_step = DT_STEP
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "bie"
    cfg.grid_bulk_rows_mode = "augment"
    cfg.grid_contact_rows_mode = "aligned_prequotient"
    cfg.angle_constraint_scope = "loopwise"
    cfg.mass_estimator = "loopwise_oriented_kde"
    cfg.bie_metric = BIEMetricConfig(**bie)
    return cfg


def bie_setup(v, **bie):
    delta, tau = compute_recommended_params(v.positions)
    st = MMStepper(bie_cfg(float(delta), float(tau), **bie))
    st._setup_step(v)
    return st


def grid_setup(v, st, n=1024, eps=0.04, box=2.0):
    cfg = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(n, n), box_min=(-box, -box), box_max=(box, box),
        fill_epsilon=eps, support_threshold=3e-3,
        projection_rel_tol=5e-2),
        compatibility_components="conservative_sweep")
    gm = GridWassersteinMetric(cfg)
    m = st.fixed_masses
    target = float(0.5 * (m * (v.positions * v.normals).sum(dim=-1)).sum())
    pol = BoundaryFluxPolicies(operator_measure="carrier",
                               use_coherence_velocity=False, n_endpoints=3)
    t0 = time.time()
    gm.setup_for_step(v, m, st.fixed_coherence, target, pol)
    return gm, time.time() - t0


def grid_admissible(gm, s, th, N):
    """project a direction onto the kernel of the grid compatibility
    rows (per-component constant normal speed correction, as in
    grid_metric_bem_parity.admissible_mode but label-free: uses the
    particle attribution of the flux kernel)."""
    rows = gm.flux.component_rows(gm.compat_labels, gm.phase.cell_area)
    frac = gm.flux.particle_component_fractions(gm.compat_labels,
                                                gm.phase.cell_area)
    plab = frac.argmax(dim=1)
    y = torch.cat([s, th])
    s_adj = s.clone()
    for c in range(rows.shape[0]):
        flux_c = float(rows[c] @ y)
        onec = torch.zeros_like(s)
        onec[plab == c] = 1.0
        denom = float(rows[c, :N] @ onec)
        if abs(denom) > 1e-300:
            s_adj = s_adj - (flux_c / denom) * onec
    return s_adj


def section_a():
    out = {}
    # disk Fourier: uniform masses at the generator spacing
    N, R = 256, 1.0
    v = generate_oriented_circle(N, R, (0.0, 0.0), "cpu", DT)
    st = bie_setup(v)
    gm, tg = grid_setup(v, st)
    th = torch.atan2(v.positions[:, 1], v.positions[:, 0])
    z = torch.zeros(N, dtype=DT)
    rows = []
    for k in (1, 2, 3, 5, 8):
        s = torch.cos(k * th)
        s_g = grid_admissible(gm, s, z, N)
        w_b = float(st.bie_wasserstein(s, z, DT_STEP))
        w_g = float(gm(s_g, z, DT_STEP))
        # the BIE masses are the KDE masses (not exactly 2 pi R / N);
        # the exact value uses the circle geometry, so compare ratios
        exact = 0.5 * DT_STEP * (R / k) * math.pi * R / DT_STEP ** 2
        rows.append(dict(k=k, W_bie=w_b, W_grid=w_g, W_exact=exact,
                         bie_vs_exact=w_b / exact - 1,
                         grid_vs_exact=w_g / exact - 1,
                         grid_vs_bie=w_g / w_b - 1))
    out["disk_fourier"] = dict(rows=rows, grid_setup_s=tg,
                               mass_median=float(st.fixed_masses.median()),
                               spacing=2 * math.pi * R / N)
    # annulus radial
    Rp, Rm, c = 1.2, 0.5, 0.05
    v = generate_oriented_annulus(384, Rp, Rm, (0.0, 0.0), "cpu", DT)
    st = bie_setup(v)
    gm, tg = grid_setup(v, st)
    m = st.fixed_masses
    r = v.positions.norm(dim=1)
    inner = r < 0.5 * (Rp + Rm)
    c_in = c * float(m[~inner].sum()) / float(m[inner].sum())
    s = torch.where(inner, torch.full_like(r, -c_in), torch.full_like(r, c))
    z = torch.zeros_like(s)
    w_b = float(st.bie_wasserstein(s, z, DT_STEP))
    s_g = grid_admissible(gm, s, z, v.n_points)
    w_g = float(gm(s_g, z, DT_STEP))
    exact = 0.5 * DT_STEP * c * c * Rp * Rp * 2 * math.pi \
        * math.log(Rp / Rm) / DT_STEP ** 2
    out["annulus_radial"] = dict(W_bie=w_b, W_grid=w_g, W_exact=exact,
                                 bie_vs_exact=w_b / exact - 1,
                                 grid_vs_exact=w_g / exact - 1,
                                 grid_vs_bie=w_g / w_b - 1,
                                 grid_setup_s=tg,
                                 n_components_grid=gm.phase.n_components)
    return out


def smooth_directions(v, st, gm, kmax=6):
    """Per-loop boundary Fourier modes cos/sin(k phi), k = 1..kmax
    (phi = polar angle about the loop centroid), pushed through the
    BIE parametrization (least-squares y so that s = D_q Q y matches
    the mode, dtheta from the angle map) and made grid-admissible.
    This is the smooth family the parity gate applies to; random y
    directions (unresolved particle-scale stress) are reported
    separately."""
    param = st.param
    N = v.n_points
    loops = st._loop_labels_last
    q = st.fixed_coherence
    Q = param.Q                      # (N, n_y): s = q * (Q y)
    out = []
    for lb in loops.unique():
        idx = loops == lb
        c = v.positions[idx].mean(dim=0)
        phi = torch.atan2(v.positions[:, 1] - c[1], v.positions[:, 0] - c[0])
        for k in range(1, kmax + 1):
            for fn in (torch.cos, torch.sin):
                target = torch.where(idx, fn(k * phi), torch.zeros(N, dtype=DT))
                y = torch.linalg.lstsq(q[:, None] * Q, target[:, None]).solution[:, 0]
                s, th = param.unpack_params(y)
                s = grid_admissible(gm, s.detach(), th.detach(), N)
                out.append((s, th.detach()))
    return out


def random_directions(st, gm, v, n_dirs=12, seed=0):
    param = st.param
    N = v.n_points
    torch.manual_seed(seed)
    out = []
    for _ in range(n_dirs):
        y = torch.randn(param.n_params, dtype=DT)
        y = y / y.norm()
        s, th = param.unpack_params(y)
        out.append((grid_admissible(gm, s.detach(), th.detach(), N),
                    th.detach()))
    return out


def gram_parity(v, st, gm, directions=None, n_dirs=12, seed=0):
    """Gram matrices of both metrics on a span of admissible
    directions (default: smooth per-loop Fourier modes)."""
    if directions is None:
        directions = smooth_directions(v, st, gm)
    S_list = [d[0] for d in directions]
    TH_list = [d[1] for d in directions]
    n_dirs = len(directions)
    Mg = torch.zeros(n_dirs, n_dirs, dtype=DT)
    Mb = torch.zeros(n_dirs, n_dirs, dtype=DT)
    per = []
    for i in range(n_dirs):
        gs_b, gth_b = st.bie_wasserstein.hessp(S_list[i], TH_list[i],
                                               DT_STEP)
        gs_g, gth_g = gm.hessp(S_list[i], TH_list[i], DT_STEP)
        for j in range(n_dirs):
            Mb[i, j] = gs_b @ S_list[j] + gth_b @ TH_list[j]
            Mg[i, j] = gs_g @ S_list[j] + gth_g @ TH_list[j]
        w_b = float(st.bie_wasserstein(S_list[i], TH_list[i], DT_STEP))
        w_g = float(gm(S_list[i], TH_list[i], DT_STEP))
        per.append(dict(W_bie=w_b, W_grid=w_g,
                        rel=(w_g - w_b) / abs(w_b)))
    Mb = 0.5 * (Mb + Mb.T)
    Mg = 0.5 * (Mg + Mg.T)
    L = torch.linalg.cholesky(Mb)
    Li = torch.linalg.inv(L)
    gen = torch.linalg.eigvalsh(Li @ Mg @ Li.T)
    sn = st._last_grid_snapshot
    return dict(n_dirs=n_dirs, worst_rel=max(abs(p["rel"]) for p in per),
                gen_eigs=[float(x) for x in gen],
                max_abs_lambda_minus_1=float((gen - 1).abs().max()),
                gram_rel_fro=float((Mg - Mb).norm() / Mb.norm()),
                bie=dict(n_components=sn.n_components,
                         sigma_tail=sn.bie_sigma_tail,
                         null_level=sn.bie_null_level,
                         lambda_min=sn.bie_lambda_min,
                         lambda_max=sn.bie_lambda_max,
                         n_masked=sn.bie_n_masked,
                         merged_pairs=sn.bie_merged_pairs,
                         block_sizes=sn.bie_block_sizes,
                         setup_s=sn.bie_setup_wall_seconds),
                grid=dict(n_components=gm.phase.n_components,
                          n_compat=gm.n_compat_components),
                directions=per)


def fixtures_b():
    ell_v = generate_oriented_ellipse(256, a=1.2, b=0.8, device="cpu",
                                      dtype=DT)
    pair = generate_oriented_two_ellipses(
        188, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu", dtype=DT)
    va = generate_oriented_annulus(256, R_outer=1.0, R_inner=0.55,
                                   center=(0.0, 0.0), device="cpu",
                                   dtype=DT)
    vd = generate_oriented_circle(90, 0.35, (0.0, 0.0), "cpu", DT)
    ann_disk = OrientedPointCloudVarifold(
        positions=torch.cat([va.positions, vd.positions]),
        angles=torch.cat([va.angles, vd.angles]))
    return dict(ellipse=ell_v, two_ellipses_gap010=pair,
                annulus_disk=ann_disk)


def two_ellipses_at(gap_over_ell, n_el=188):
    v0 = generate_oriented_two_ellipses(
        n_el, a1=0.4, b1=1.0, center1=(-1.0, 0.0),
        a2=0.4, b2=1.0, center2=(1.0, 0.0), device="cpu", dtype=DT)
    seg = (v0.positions[:n_el].roll(-1, 0) - v0.positions[:n_el]) \
        .norm(dim=1)
    ell = float(seg.median())
    gap = gap_over_ell * ell
    return generate_oriented_two_ellipses(
        n_el, a1=0.4, b1=1.0, center1=(-(0.4 + gap / 2), 0.0),
        a2=0.4, b2=1.0, center2=((0.4 + gap / 2), 0.0),
        device="cpu", dtype=DT), ell, gap


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    quick = "--quick" in sys.argv
    n = 512 if quick else 1024
    res = {"grid": n}
    t0 = time.time()
    print("A: exact modes", flush=True)
    globals()["grid_setup"].__defaults__ = (n, 0.04, 2.0)
    res["A"] = section_a()
    for k, r in res["A"].items():
        print("  ", k, {kk: vv for kk, vv in r.items()
                        if kk != "rows"}, flush=True)
        for row in r.get("rows", []):
            print("     ", row, flush=True)
    print("B: separated Gram parity", flush=True)
    res["B"] = {}
    for name, v in fixtures_b().items():
        st = bie_setup(v)
        gm, tg = grid_setup(v, st)
        r = gram_parity(v, st, gm)
        r["grid_setup_s"] = tg
        r["random_unresolved"] = {
            k: vv for k, vv in gram_parity(
                v, st, gm, random_directions(st, gm, v)).items()
            if k in ("worst_rel", "max_abs_lambda_minus_1",
                     "gram_rel_fro")}
        res["B"][name] = r
        print(f"   {name}: smooth worst_rel {r['worst_rel']:.4f} "
              f"max|lam-1| {r['max_abs_lambda_minus_1']:.4f} "
              f"fro {r['gram_rel_fro']:.4f}; random "
              f"{r['random_unresolved']}; bie {r['bie']}",
              flush=True)
    print("C: merge window (report only)", flush=True)
    v, ell, gap = two_ellipses_at(0.8)
    st = bie_setup(v)
    gm, tg = grid_setup(v, st)
    r = gram_parity(v, st, gm)
    r.update(grid_setup_s=tg, ell=ell, gap=gap)
    res["C"] = {"two_ellipses_gap0.8ell": r}
    print(f"   merge window: worst_rel {r['worst_rel']:.4f} "
          f"max|lam-1| {r['max_abs_lambda_minus_1']:.4f} bie {r['bie']} "
          f"grid {r['grid']}", flush=True)
    res["wall_s"] = time.time() - t0
    (OUT / "parity.json").write_text(json.dumps(res, indent=1,
                                                default=str))
    md = ["# BIE vs grid static parity", "",
          f"grid {n}^2, fill 0.04, box (-2,2)^2; BIE eps = "
          f"{BIEMetricConfig().epsilon_scale:g} ell, K=3; wall "
          f"{res['wall_s']:.0f} s", "",
          "## A exact modes", "",
          "| case | BIE vs exact | grid vs exact | grid vs BIE |",
          "|---|---|---|---|"]
    for row in res["A"]["disk_fourier"]["rows"]:
        md.append(f"| disk k={row['k']} | {row['bie_vs_exact']:+.4f} | "
                  f"{row['grid_vs_exact']:+.4f} | {row['grid_vs_bie']:+.4f} |")
    a = res["A"]["annulus_radial"]
    md.append(f"| annulus radial | {a['bie_vs_exact']:+.4f} | "
              f"{a['grid_vs_exact']:+.4f} | {a['grid_vs_bie']:+.4f} |")
    md += ["", "## B separated Gram parity (per-loop Fourier modes "
           "k <= 6; random particle-scale directions in the last column)",
           "",
           "| fixture | worst |W_g/W_b - 1| | max|lambda-1| | rel Fro | "
           "sigma_tail | lambda_min | random: worst / max|lambda-1| |",
           "|---|---|---|---|---|---|---|"]
    for name, r in res["B"].items():
        ru = r["random_unresolved"]
        md.append(f"| {name} | {r['worst_rel']:.4f} | "
                  f"{r['max_abs_lambda_minus_1']:.4f} | "
                  f"{r['gram_rel_fro']:.4f} | "
                  f"{min(r['bie']['sigma_tail']):.2e} | "
                  f"{r['bie']['lambda_min']:.3e} | "
                  f"{ru['worst_rel']:.3f} / "
                  f"{ru['max_abs_lambda_minus_1']:.3f} |")
    r = res["C"]["two_ellipses_gap0.8ell"]
    md += ["", "## C merge window (report only)", "",
           f"two ellipses at gap 0.8 ell = {r['gap']:.4f}: BIE merged "
           f"{r['bie']['merged_pairs']}, masked {r['bie']['n_masked']}, "
           f"sigma_tail {r['bie']['sigma_tail']}, lambda_min "
           f"{r['bie']['lambda_min']:.3e}; grid components "
           f"{r['grid']}; worst rel {r['worst_rel']:.4f}, max|lambda-1| "
           f"{r['max_abs_lambda_minus_1']:.4f}"]
    (OUT / "parity.md").write_text("\n".join(md) + "\n")
    print("wrote", OUT / "parity.json", flush=True)


if __name__ == "__main__":
    main()
