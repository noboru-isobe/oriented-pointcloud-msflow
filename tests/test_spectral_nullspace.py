"""0C-2a regressions: the spectral nullspace structure of the interior trace.

Cheap, coarse-resolution versions of scripts/experiments/bem_spectral_audit.py
(which holds the conventions, the full geometry suite, and the refinement
tables). Thresholds here are set from the measured table with roughly a 2x
margin -- they are regression pins, not pass criteria for Gate B.

What is pinned:

  * on noncircular geometries the max-gap detector reads off the number of
    PHASE components C (not the number of boundary components M), the
    spectral gap is clear, and the left near-nullspace aligns with the
    phase-indicator space;
  * those quantities converge under refinement (guards against a threshold
    that merely happens to pass at one resolution);
  * the left and right nullspaces are NOT interchangeable: on an annulus the
    right null density is uniform on the outer ring only, sitting at
    arccos sqrt(L_out/L_total) from the indicator space, while the left one
    IS the indicator. A symmetric treatment Q^T A Q would conflate them.
"""

import math

import pytest
import torch

from scripts.experiments.bem_spectral_audit import (
    audit_case,
    principal_angles_deg,
    spectral_decomposition,
)


# measured at h=0.06 (audit table, results/spectral_0c2a): worst gap 2.41e-2,
# worst theta_max 2.16 deg over these cases
@pytest.mark.parametrize("name", [
    "ellipse",
    "eccentric_annulus",
    "two_unequal_ellipses",
    "nonconcentric_two_holes",
])
def test_detector_gap_and_left_alignment_noncircular(name):
    r = audit_case(name, spacing=0.06)
    assert r["detected_C"] == r["C"], (r["detected_C"], r["C"], r["sigma_low"])
    assert r["gap"] < 0.05
    assert r["theta_max_deg"] < 4.0
    assert r["sigma_Cp1"] > 0.1          # non-null part bounded away from zero


def test_flower_gap_clear_but_ratio_threshold_is_geometry_dependent():
    """The flower's sigma_{C+1} ~ 0.19 (vs ~ 0.49 for disks), so any fixed
    gap-ratio threshold is geometry-dependent -- measured at h=0.015 the
    flower sits at 1.28e-2 while disks are at ~1.5e-3. The *separation* is
    still a factor ~16 even at h=0.06. This pins the structure without
    pretending a universal ratio exists; rank selection for unknown C (0C-4)
    must combine the ratio with absolute residuals.
    """
    r = audit_case("flower", spacing=0.06)
    assert r["detected_C"] == 1
    assert r["gap"] < 0.12               # measured 6.31e-2
    assert r["sigma_Cp1"] > 0.15         # measured 0.194, stable in h
    assert r["theta_max_deg"] < 12.0     # measured 7.36 deg


def test_refinement_convergence_eccentric_annulus():
    """Coarse -> fine must improve everything; a single-resolution pass is
    weak evidence (the trace-side bug hid behind exactly such coincidences).
    """
    coarse = audit_case("eccentric_annulus", spacing=0.08)
    fine = audit_case("eccentric_annulus", spacing=0.04)
    assert coarse["detected_C"] == fine["detected_C"] == 1
    assert fine["sigma_C"] < coarse["sigma_C"]
    assert fine["gap"] < coarse["gap"]
    assert fine["theta_max_deg"] < coarse["theta_max_deg"]
    assert fine["res_L_abs"] < coarse["res_L_abs"]
    assert fine["res_R_abs"] < coarse["res_R_abs"]


def test_left_and_right_nullspaces_are_not_interchangeable():
    """Concentric annulus, R_out=1, R_in=0.5. The left null vector is the
    phase indicator (compatibility: one condition over BOTH rings). The right
    null vector is the density whose single-layer potential is constant on
    the material -- for a CIRCULAR outer boundary that is uniform on the
    outer ring only, so its angle to the indicator is
    arccos sqrt(L_out/(L_out+L_in)) = arccos sqrt(2/3) = 35.26 deg. (For a
    noncircular outer boundary the right null density is the non-uniform
    equilibrium density and no closed form applies.) Measured: left 0.66 deg,
    right 35.21 deg at h=0.04.
    """
    d = spectral_decomposition("concentric_annulus", spacing=0.04)
    left = principal_angles_deg(d["U0"], d["QB"]).max().item()
    right = principal_angles_deg(d["V0"], d["QB"]).max().item()

    assert left < 2.0
    assert right > 25.0
    theory = math.degrees(math.acos(math.sqrt(2.0 / 3.0)))
    assert right == pytest.approx(theory, abs=2.0)


def test_annulus_needs_one_condition_two_disks_need_two():
    """Same M=2, different C -- the distinction point-cloud homology cannot
    make (F12/F13). The spectrum must: annulus C=1, two disks C=2.
    """
    ann = audit_case("concentric_annulus", spacing=0.06)
    two = audit_case("two_disks", spacing=0.06)
    assert (ann["M"], ann["C"], ann["detected_C"]) == (2, 1, 1)
    assert (two["M"], two["C"], two["detected_C"]) == (2, 2, 2)
