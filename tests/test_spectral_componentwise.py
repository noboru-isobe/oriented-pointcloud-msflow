import torch


def test_p3_lobpcg_parity_with_full_svd():
    """P3 feasibility pin (raw tables in results/p3_lobpcg): the
    warm-started Gram-free LOBPCG (scipy, LinearOperator matvec --
    the Gram matrix is never formed) reproduces the full-SVD near-null
    basis at machine precision on the evolving annulus: principal
    angle < 1e-3 deg, smallest singular values and the gap ratio the
    rank machinery consumes agree to < 1e-9. Measured speedup 8-10x
    (NOT pinned -- hardware dependent). Integration stays gated on the
    review's parity/fallback protocol."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from d1a_mm_shadow import make_config
    from d2b0_timelevel_shadow import snapshot
    from p3_lobpcg_feasibility import run_trajectory
    from src.torch.oriented_varifold.mass import (
        compute_recommended_params)

    v0, _, _ = snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    v, _, _ = snapshot(0.5)
    rows = run_trajectory("annulus", v, make_config(2e-5, delta, tau),
                          1, 3)
    for r in rows:
        assert r["principal_angle_deg"] < 1e-3
        assert r["sval_rel_err"] < 1e-9
        assert abs(r["gap_ratio_full"] - r["gap_ratio_lobpcg"]) < 1e-9


def test_p1_decision_table_pinned():
    """P1.1 pins (committed JSON, regenerated fail-closed): the paper
    two-ellipse geometry, the trace-effect floors, the C=1
    C2 ~ C2S invariant where it holds, and its measured VIOLATION on
    the star (a spectral implementation effect, not componentwise
    theory -- flagged for the long-run comparison).

    Headline decomposition (rel one-step displacement change):
      - trace effect (C1->C2) dominates everywhere: ellipse 7.6%,
        flower 45%, star 59%, paper pair 72%; circle stationary;
      - componentwise compatibility PROPER (C2->C2S at C=2, paper
        pair): 0.23% -- NUMERICALLY SMALL AT THE PRESENT ONE-STEP
        RESOLUTION (tolerance-stability in pair_tight_tol), on the
        SYMMETRIC first step where the perimeter gradient barely
        excites exchange directions; the componentwise correction
        remains STRUCTURALLY NECESSARY for excluding
        component-exchange modes (0C-1 algebra unchanged);
      - C=1 invariant: ellipse 1.0% (holds), star 18.2% -- a FINITE-
        DISCRETE implementation sensitivity whose classification
        (decaying finite-N nullspace tilt vs persistent formulation
        difference) is the c2s_refinement N-sweep's job;
      - the M-velocity switch is the LARGEST SEQUENTIAL INCREMENT in
        the chosen ablation chain (37.5% on the star) -- effects
        interact nonlinearly, so no independent 'main effect' is
        claimed -- renamed production_metric_bundle."""
    import json
    from pathlib import Path as _P
    p = _P(__file__).parent.parent / "results" / "p1_production" \
        / "p1_one_step_comparison.json"
    d = json.loads(p.read_text())

    # paper geometry pin
    te = d["two_ellipses_paper"]
    assert te["n_points"] == 256 and te["rank"] == 2

    # fail-closed: every chain row valid on every shape
    for name in ("circle", "ellipse2to1", "flower", "star",
                 "two_ellipses_paper"):
        for c in d["meta"]["chain"]:
            assert d[name][f"diag_{c}"]["valid"], (name, c)

    # circle stationary: no discriminating signal
    assert d["circle"]["diag_C2"]["max_abs_s"] < 1e-12

    # trace-effect floors (the regeneration decision)
    assert d["ellipse2to1"]["trace"]["rel_max_ds"] > 0.05
    assert d["flower"]["trace"]["rel_max_ds"] > 0.3
    assert d["star"]["trace"]["rel_max_ds"] > 0.4
    assert te["trace"]["rel_max_ds"] > 0.5

    # componentwise compatibility proper (C=2) is tiny on the
    # symmetric first step
    assert te["spectral_compat_only"]["rel_max_ds"] < 0.01

    # C=1 invariant: holds on the ellipse, VIOLATED on the star
    assert d["ellipse2to1"]["spectral_compat_only"]["rel_max_ds"] < 0.03
    assert d["star"]["spectral_compat_only"]["rel_max_ds"] > 0.1

    # frozen initial perimeter identical across configs (energy sanity)
    for name in ("ellipse2to1", "star", "two_ellipses_paper"):
        for label in ("trace", "spectral_compat_only",
                      "production_metric_bundle"):
            assert d[name][label]["frozen_P0_rel_diff"] < 1e-12


def test_p12_prepatch_measurements_pinned():
    """P1.2 pre-patch pins (committed JSON):
      (i)   the pair's 0.23% C2->C2S difference is TOLERANCE-STABLE
            (identical at optimizer_tol = 1e-10) -- a resolved small
            number, not optimizer noise;
      (ii)  the C=1 C2 ~ C2S violation DECAYS under N refinement
            (star 18.2% -> 1.5% -> 0.6%; flower 7.1% -> 0.4% -> 0.14%;
            compat-row angle 2.9 deg -> 0.4 deg): a finite-N nullspace
            tilt at sharp tips -- the bordered and global formulations
            share their discrete limit; NOT a persistent formulation
            difference;
      (iii) the PRODUCTION rank mode (state_machine, fixed
            rank_gap_min = 30) REFUSES to initialize on the paper pair
            (first evidence weak_gap, full-SVD gap ~ 15): recorded
            honestly -- the resolution (adaptive noise-floor rule /
            certified seed / different start) is a reviewed decision,
            never a silent oracle fallback."""
    import json
    from pathlib import Path as _P
    p = _P(__file__).parent.parent / "results" / "p1_production" \
        / "p1_one_step_comparison.json"
    d = json.loads(p.read_text())

    assert abs(d["pair_tight_tol"]["rel_max_ds"]
               - d["two_ellipses_paper"]["spectral_compat_only"]
               ["rel_max_ds"]) < 1e-6

    ref = d["c2s_refinement"]
    for shape in ("flower", "star"):
        seq = [ref[f"{shape}_n{n}"]["rel_max_ds"]
               for n in (128, 256, 512)]
        assert seq[0] > seq[1] > seq[2]          # monotone decay
        assert seq[2] < 0.25 * seq[0]            # strong decay
        angles = [ref[f"{shape}_n{n}"]["compat_row_angle_deg"]
                  for n in (128, 256, 512)]
        assert angles[0] > angles[2]             # tilt shrinks
        assert all(ref[f"{shape}_n{n}"]["valid"]
                   for n in (128, 256, 512))

    pf = d["pair_state_machine_preflight"]
    assert pf["initialized"] is False
    assert "weak_gap" in pf["error"]


def test_condition_numbers_pinned():
    """P1.2 diagnostics (committed JSON): conditioning of the three
    realizations of the singular BEM solve.
      - the RAW weighted operator is ill-conditioned everywhere
        (kappa 50-290, s_min ~ 1/N);
      - the BORDERED matrix is uniformly well-conditioned (kappa 2-9:
        the 0C-5 spectral gap IS the conditioning guarantee);
      - the legacy GLOBAL projection matches bordered at C = 1 (its
        one row removes the only near-null mode up to the finite-N
        tilt) but at C = 2 it CANNOT remove the second near-null mode:
        kappa = 110 on the paper pair, ~13x worse than bordered and
        growing with N -- the hidden ill-conditioning that is the
        numerical face of the exchange mode. (Historically masked by
        the exterior-trace bug, which made the legacy system LOOK
        well-conditioned.)"""
    import json
    from pathlib import Path as _P
    p = _P(__file__).parent.parent / "results" / "p1_production" \
        / "p1_one_step_comparison.json"
    cn = json.loads(p.read_text())["condition_numbers"]

    for name, c in cn.items():
        assert c["raw"]["kappa"] > 20, name
        assert c["bordered"]["kappa"] < 20, name
        if c["C"] == 1:
            # global ~ bordered at C = 1
            assert c["global_projected"]["kappa"] \
                < 3 * c["bordered"]["kappa"], name
    pair = cn["two_ellipses_paper"]
    assert pair["C"] == 2
    assert pair["global_projected"]["kappa"] \
        > 5 * pair["bordered"]["kappa"]


def test_p12_rank_initialization_tracks():
    """P1.2 rank-init decision (review): certified seed +
    AdaptiveRankMachine handoff, plus the unseeded adaptive track.
    Measured on the paper pair (the same first frame the FIXED-
    threshold state machine refuses as weak_gap):
      - Track S: the integer certificate C0=2 (provenance recorded; no
        component labels, no connectivity) is accepted only after the
        spectrum itself reads c_abs == c_gap == 2 and the phase
        constant sits in the candidate span; the machine's floor/basis
        seed from the full SVD; subsequent frames hold rank 2 through
        clean and weak evidence.
      - Track A: the UNSEEDED adaptive machine initializes on its own
        (the ~15 gap satisfies the kappa_sep = 10 self-consistency
        rule that the fixed threshold 30 failed) and the two tracks
        agree BITWISE over the pinned window.
      - A WRONG certificate (C0=3) is rejected against the spectrum --
        the certificate can never overrule the operator."""
    import pytest
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"
                            / "experiments"))
    from p1_production_comparison import make_cfg, shapes
    from src.torch.oriented_varifold.mass import (
        compute_recommended_params)
    from src.torch.solver.mm_step import MMStepper
    from src.torch.transport.bem_wasserstein import (
        AmbiguousComponentRankError)

    v, rank = shapes()["two_ellipses_paper"]
    delta, tau = compute_recommended_params(v.positions)

    def _run(cert, n_steps=2):
        cfg = make_cfg("C3", delta, tau, rank)
        cfg.bem_rank_mode = "adaptive_state_machine"
        cfg.initial_rank_certificate = cert
        if cert is not None:
            cfg.initial_rank_certificate_provenance = \
                "paper generator: two disjoint loops at t=0"
        st = MMStepper(cfg)
        vv, decisions = v, []
        for _ in range(n_steps):
            r = st.step(vv)
            vv = r.varifold
            decisions.append(st._last_rank_decision)
        return vv, decisions

    vS, dS = _run(2)
    assert dS[0].action == "seeded_certificate" and dS[0].rank == 2
    assert dS[1].rank == 2

    vA, dA = _run(None)
    assert dA[0].rank == 2                      # self-initialized
    assert torch.equal(vA.positions, vS.positions)   # bitwise parity

    # review hygiene: FULLY label-free -- oracle rank stripped from the
    # config too, so a future refactor leaking component_rank into the
    # deferred path is caught
    cfg = make_cfg("C3", delta, tau, rank)
    cfg.bem_rank_mode = "adaptive_state_machine"
    cfg.bem_component_rank = None
    cfg.initial_rank_certificate = None
    st = MMStepper(cfg)
    vv = v
    for _ in range(2):
        vv = st.step(vv).varifold
    assert st._last_rank_decision.rank == 2
    assert torch.equal(vv.positions, vS.positions)

    with pytest.raises(AmbiguousComponentRankError,
                       match="cannot overrule"):
        _run(3, n_steps=1)
    with pytest.raises(ValueError, match="bem_rank_max"):
        _run(7, n_steps=1)          # outside [1, rank_max]

    # provenance lands in run metadata (config asdict is the metadata
    # channel every artifact generator uses)
    import dataclasses
    cfg = make_cfg("C3", delta, tau, rank)
    cfg.initial_rank_certificate = 2
    cfg.initial_rank_certificate_provenance = \
        "paper generator: two disjoint loops at t=0"
    meta = dataclasses.asdict(cfg)
    assert meta["initial_rank_certificate"] == 2
    assert "two disjoint loops" in \
        meta["initial_rank_certificate_provenance"]


def test_cu_rowspace_gate_failure_path():
    """P1.2 safety (review): the C_u continuation gate must actually
    STOP on an abnormal rotation, and its measured angle is telemetry.
    A same/near-identical row space passes (angle << 30 deg, recorded);
    an artificially rotated C_full raises. Scope note pinned: the gate
    compares only shape-matched consecutive frames -- rank changes and
    deletions reset it (support gates cover those)."""
    import pytest
    from p1_production_comparison import make_cfg, shapes
    from src.torch.oriented_varifold.mass import (
        compute_recommended_params)
    from src.torch.solver.mm_step import MMStepper
    from src.torch.transport.bem_wasserstein import (
        AmbiguousComponentRankError)

    v, rank = shapes()["two_ellipses_paper"]
    delta, tau = compute_recommended_params(v.positions)
    cfg = make_cfg("C3", delta, tau, rank)
    cfg.bem_rank_mode = "adaptive_state_machine"
    st = MMStepper(cfg)
    vv = v
    for _ in range(2):
        vv = st.step(vv).varifold
    # pass path: consecutive frames measured, small angle recorded
    assert st._last_cu_rowspace_angle_deg is not None
    assert st._last_cu_rowspace_angle_deg < 5.0

    # failure path: rotate the stored previous row space by 90 deg in
    # the plane of its own rows' orthogonal complement
    prev = st._prev_spectral_C_full
    Q, _ = torch.linalg.qr(torch.randn(prev.shape[1], prev.shape[1],
                                       dtype=prev.dtype))
    st._prev_spectral_C_full = prev @ Q      # generically ~90 deg
    with pytest.raises(AmbiguousComponentRankError,
                       match="continuation broken"):
        st.step(vv)


def test_pair_projected_gradient_closure():
    """The optimizer-independent closure of the pair's 0.23% (review):
    tight-run diagnostics REVEAL that both C2 and C2S stop on the same
    trust-region breakdown ('bad approximation') at relgrad ~ 1.1e-7 --
    so 'tolerance-stable' was the wrong frame; the requested 1e-10 was
    never reached. The projected INITIAL gradients at y = 0 (no
    optimizer anywhere) close it properly: norms agree to 5e-4
    relative and the physical gradient directions to cos = 0.999986 --
    the symmetric initial perimeter gradient only weakly excites the
    component-exchange correction."""
    import json
    from pathlib import Path as _P
    p = _P(__file__).parent.parent / "results" / "p1_production" \
        / "p1_one_step_comparison.json"
    d = json.loads(p.read_text())

    tt = d["pair_tight_tol"]
    for kind in ("C2", "C2S"):
        dg = tt["diagnostics"][kind]
        assert dg["objective_decreased"]
        assert dg["relative_gradient_norm"] < 1e-6
        # honest record: the trust-region stalls before 1e-10
        assert dg["converged"] is False

    pg = d["pair_projected_gradient"]
    assert pg["rel_norm_diff"] < 1e-3
    assert pg["phys_direction_cos"] > 0.9999
