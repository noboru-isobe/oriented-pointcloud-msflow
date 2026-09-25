"""P1-1 pins: order-free neck telemetry (reviewer program section 4).

rigid rotation / particle permutation / sampling refinement /
varicose seed / asymmetric lobes / lobe-pair non-misselection /
A_L + A_R machine identity.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"
                       / "experiments"))

from neck_telemetry import (  # noqa: E402
    facing_pairs,
    lobe_areas,
    neck_width,
)
from src.torch.shapes.dumbbell import build_capsule_dumbbell  # noqa

DT = torch.float64
H_REF = 2 * math.pi / 256
A0 = math.pi * 0.8


def cloud(**kw):
    torch.set_default_dtype(DT)
    p = dict(R_l=0.55, R_r=0.55, L=1.0, w=0.29, h=H_REF,
             area_target=A0)
    p.update(kw)
    c = build_capsule_dumbbell(**p)
    pos, ang = c["positions"], c["angles"]
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    m = (pos.roll(-1, 0) - pos.roll(1, 0)).norm(dim=1) / 2
    return c, pos, nor, m


class TestNeckWidth:
    def test_matches_generator_width(self):
        c, pos, nor, m = cloud()
        r = neck_width(pos, nor, m, H_REF)
        assert r["status"] == "ok"
        assert abs(r["w_fit"] - c["w"]) < 0.02 * c["w"], (r, c["w"])
        assert abs(r["w_point"] - c["w"]) < 0.02 * c["w"]

    def test_rotation_invariance(self):
        c, pos, nor, m = cloud()
        al = 0.61
        R = torch.tensor([[math.cos(al), -math.sin(al)],
                          [math.sin(al), math.cos(al)]], dtype=DT)
        r0 = neck_width(pos, nor, m, H_REF)
        r1 = neck_width(pos @ R.T, nor @ R.T, m, H_REF,
                        axis=(math.cos(al), math.sin(al)))
        assert abs(r1["w_fit"] - r0["w_fit"]) < 1e-10

    def test_permutation_invariance(self):
        c, pos, nor, m = cloud()
        g = torch.Generator().manual_seed(11)
        perm = torch.randperm(pos.shape[0], generator=g)
        r0 = neck_width(pos, nor, m, H_REF)
        r1 = neck_width(pos[perm], nor[perm], m[perm], H_REF)
        assert abs(r1["w_fit"] - r0["w_fit"]) < 1e-10

    def test_refinement(self):
        _, pos, nor, m = cloud()
        c2, pos2, nor2, m2 = cloud(h=H_REF / 2)
        r1 = neck_width(pos, nor, m, H_REF)
        r2 = neck_width(pos2, nor2, m2, H_REF / 2)
        assert abs(r1["w_fit"] - r2["w_fit"]) < 5e-3

    def test_varicose_seed_recovered(self):
        # displace the neck walls inward by eps cos(kx) (smooth
        # cutoff): w_fit must report w - 2 eps at the waist
        c, pos, nor, m = cloud(L=2.0)
        eps, k = 0.01, math.pi
        x = pos[:, 0]
        cut = torch.exp(-(x / (c["L"] / 2)) ** 4)
        wall = (x.abs() < c["L"] / 2) & (pos[:, 1].abs()
                                         < c["w"] / 2 + H_REF)
        disp = torch.zeros_like(pos)
        disp[:, 1] = -torch.sign(pos[:, 1]) * eps \
            * torch.cos(k * x) * cut
        pos_s = pos + disp * wall[:, None].to(DT)
        r = neck_width(pos_s, nor, m, H_REF)
        assert abs(r["w_fit"] - (c["w"] - 2 * eps)) < 0.15 * eps, \
            (r["w_fit"], c["w"] - 2 * eps)

    def test_asymmetric_lobes(self):
        c, pos, nor, m = cloud(R_l=0.35, R_r=0.7, L=2.0)
        r = neck_width(pos, nor, m, H_REF)
        assert r["status"] == "ok"
        assert abs(r["w_fit"] - c["w"]) < 0.03 * c["w"]

    def test_no_lobe_pair_misselection(self):
        # antipodal walls of a LOBE are anti-parallel too; the ROI +
        # crossing-component selection must keep them out
        c, pos, nor, m = cloud(L=2.0)
        # ROI wide enough to CONTAIN the lobes: the estimator must
        # still report the neck, not the lobe diameter
        r = neck_width(pos, nor, m, H_REF, roi_half=3.0)
        assert r["status"] == "ok"
        assert r["w_fit"] < 2.0 * c["w"]
        # sanity of the negative: antipodal lobe pairs DO exist in
        # the wide ROI (top/bottom of a lobe are anti-parallel)
        e = torch.tensor([1.0, 0.0], dtype=DT)
        roi = (pos @ e).abs() <= 3.0
        II, JJ = facing_pairs(pos, nor, H_REF, roi)
        gaps = ((pos[JJ] - pos[II]) * nor[II]).sum(-1).abs()
        assert float(gaps.max()) > 3.0 * c["w"]   # lobe pairs present


class TestLobeAreas:
    def test_identity_machine_precision(self):
        _, pos, nor, m = cloud()
        r = lobe_areas(pos, nor, m)
        assert abs(r["identity_residual"]) < 1e-14

    def test_split_symmetry_and_total(self):
        c, pos, nor, m = cloud()
        r = lobe_areas(pos, nor, m)
        assert abs(r["A_L"] - r["A_R"]) < 1e-3 * r["A_curr"]
        # total consistent with the shoelace area at quadrature order
        assert abs(r["A_curr"] - c["area"]) / c["area"] < 2e-3

    def test_asymmetric_split(self):
        c, pos, nor, m = cloud(R_l=0.35, R_r=0.7, L=2.0)
        r = lobe_areas(pos, nor, m)
        assert r["A_R"] > 2.0 * r["A_L"]
