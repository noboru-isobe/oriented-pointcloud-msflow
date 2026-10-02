"""The events of the two-ellipses run never read the order in which the
particles are stored.

Where an order along a loop is needed (merging criterion, reconnection,
polygon functionals of the tolerances, checks of the removal) it is
derived from the state by certify_loop_orders. Pinned here on stored
states of the production run: the results are unchanged under random
permutations of the particles inside each loop.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import two_ellipses_benchmark as teb  # noqa: E402
from l1_quotient_shadow import window_certificate  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.perimeter.contact_complex import (  # noqa: E402
    derived_loop_permutation,
)
from src.torch.solver.ghost_compression import (  # noqa: E402
    conservative_compression,
)

DT = torch.float64
STATES = ROOT / "results/two_ellipses/two_ellipses_bie_states.pt"
THR = ROOT / "results/two_ellipses/l1_calibration/l1_thresholds.json"
STEP_MERGE, STEP_SPLICE = 2025, 2029
TOL = 1e-12

pytestmark = pytest.mark.skipif(
    not (STATES.exists() and THR.exists()),
    reason="stored production states not available")


@pytest.fixture(scope="module")
def setup():
    torch.set_default_dtype(DT)
    v0, m1, _, _ = teb.build_cloud(0.0)
    delta, tau = map(float, compute_recommended_params(v0.positions))
    ck = torch.load(STATES, weights_only=False)
    thr = json.loads(THR.read_text())
    return ck, m1, delta, tau, thr


def shuffled(pos, ang, m1, seed):
    g = torch.Generator().manual_seed(seed)
    n0 = int(m1.sum())
    p = torch.cat([torch.randperm(n0, generator=g),
                   n0 + torch.randperm(pos.shape[0] - n0, generator=g)])
    return pos[p], ang[p]


def canon(x):
    """Rows sorted lexicographically (a point set, whatever its order)."""
    key = x[:, 0].double() * 1e3 + x[:, 1].double()
    return x[key.argsort()]


def flat(rec):
    out = {}
    for k, v in rec.items():
        if isinstance(v, (tuple, list)):
            for i, w in enumerate(v):
                out[f"{k}{i}"] = w
        else:
            out[k] = v
    return out


def test_derived_permutation_is_storage_independent(setup):
    ck, m1, *_ = setup
    pos, ang = ck[STEP_MERGE]["positions"], ck[STEP_MERGE]["angles"]
    lab = (~m1).long()
    ref = pos[derived_loop_permutation(pos, lab)]
    for seed in (1, 2, 3):
        ps, _ = shuffled(pos, ang, m1, seed)
        assert not torch.equal(ps, pos)
        got = ps[derived_loop_permutation(ps, lab)]
        assert torch.equal(got, ref)


@pytest.mark.parametrize("step", [STEP_MERGE, STEP_SPLICE])
def test_merging_criterion_is_storage_independent(setup, step):
    ck, m1, delta, tau, thr = setup
    pos, ang = ck[step]["positions"], ck[step]["angles"]
    rec0, ok0, fail0 = window_certificate(pos, ang, m1, delta, tau, thr)
    assert ok0, fail0
    f0 = flat(rec0)
    for seed in (1, 2, 3):
        ps, as_ = shuffled(pos, ang, m1, seed)
        rec, ok, fail = window_certificate(ps, as_, m1, delta, tau, thr)
        assert ok == ok0 and fail == fail0
        f = flat(rec)
        assert f.keys() == f0.keys()
        for k in f0:
            if isinstance(f0[k], bool) or f0[k] is None:
                assert f[k] == f0[k], k
            else:
                assert abs(f[k] - f0[k]) <= TOL * max(1.0, abs(f0[k])), k


def test_polygon_functionals_are_storage_independent(setup):
    ck, m1, *_ = setup
    pos, ang = ck[STEP_MERGE]["positions"], ck[STEP_MERGE]["angles"]
    for sel in (m1, ~m1):
        p = pos[sel]
        g = torch.Generator().manual_seed(5)
        q = p[torch.randperm(p.shape[0], generator=g)]
        assert abs(teb.signed_area_order(p)
                   - teb.signed_area_order(q)) <= TOL
        assert teb.signed_area_order(p) > 0
        for a, b in zip(teb.centroid_order(p), teb.centroid_order(q)):
            assert abs(a - b) <= TOL
        for a, b in zip(teb.polygon_moments_order(p),
                        teb.polygon_moments_order(q)):
            assert abs(a - b) <= TOL


def test_reconnection_is_storage_independent(setup):
    ck, m1, delta, tau, thr = setup
    pos, ang = ck[STEP_SPLICE]["positions"], ck[STEP_SPLICE]["angles"]
    sp0, err0, _, _ = teb.certified_splice(pos, ang, m1, delta, tau,
                                           thr, 0.08)
    assert err0 is None, err0
    # the stored production splice (storage order = curve order) is
    # reproduced as a point set
    old = ck[f"spliced_{STEP_SPLICE}"]["positions"]
    assert sp0["positions"].shape == old.shape
    assert (canon(sp0["positions"]) - canon(old)).abs().max() < 1e-10
    for seed in (1, 2, 3):
        ps, as_ = shuffled(pos, ang, m1, seed)
        sp, err, _, _ = teb.certified_splice(ps, as_, m1, delta, tau,
                                             thr, 0.08)
        assert err is None, err
        assert sp["n_removed"] == sp0["n_removed"]
        assert sp["n_bridge"] == sp0["n_bridge"]
        assert (canon(sp["positions"])
                - canon(sp0["positions"])).abs().max() <= TOL
        # same polygon, possibly from another starting vertex
        P, P0 = sp["positions"], sp0["positions"]
        j = int((P0 - P[0]).norm(dim=1).argmin())
        assert (P - P0.roll(-j, 0)).abs().max() <= TOL


def test_removal_is_storage_independent():
    """Concentric triple: the removal map and its checks give the same
    loop when the particles are stored in a random order."""
    import math
    torch.set_default_dtype(DT)

    def circ(R, n, inward=False):
        t = torch.arange(n, dtype=DT) * 2 * math.pi / n
        pos = torch.stack([R * t.cos(), R * t.sin()], 1)
        ang = t + (math.pi if inward else 0.0)
        return pos, ang
    parts = [circ(0.9055, 256), circ(0.35, 90), circ(0.3745, 141, True)]
    pos = torch.cat([p for p, _ in parts])
    ang = torch.cat([a for _, a in parts])
    lab = torch.cat([torch.full((p.shape[0],), i, dtype=torch.long)
                     for i, (p, _) in enumerate(parts)])

    def run(pos, ang, lab):
        nor = torch.stack([ang.cos(), ang.sin()], 1)
        return conservative_compression(pos, ang, nor, lab == 0,
                                        lab == 1, lab == 2)
    ref = run(pos, ang, lab)
    assert ref["certificates"]["all"]
    g = torch.Generator().manual_seed(7)
    p = torch.randperm(pos.shape[0], generator=g)
    got = run(pos[p], ang[p], lab[p])
    assert got["certificates"]["all"]
    assert abs(got["scale"] - ref["scale"]) <= TOL
    assert (canon(got["positions"])
            - canon(ref["positions"])).abs().max() <= TOL
