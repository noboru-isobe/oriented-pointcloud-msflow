# BIE vs grid static parity

grid 1024^2, fill 0.04, box (-2,2)^2; BIE eps = 0.1 ell, K=3; wall 107 s

## A exact modes

| case | BIE vs exact | grid vs exact | grid vs BIE |
|---|---|---|---|
| disk k=1 | -0.0068 | -0.0067 | +0.0001 |
| disk k=2 | -0.0097 | -0.0066 | +0.0032 |
| disk k=3 | -0.0128 | -0.0083 | +0.0046 |
| disk k=5 | -0.0196 | -0.0124 | +0.0073 |
| disk k=8 | -0.0311 | -0.0232 | +0.0082 |
| annulus radial | -0.0021 | -0.0073 | -0.0052 |

## B separated Gram parity (per-loop Fourier modes k <= 6; random particle-scale directions in the last column)

| fixture | worst |W_g/W_b - 1| | max|lambda-1| | rel Fro | sigma_tail | lambda_min | random: worst / max|lambda-1| |
|---|---|---|---|---|---|---|
| ellipse | 0.0076 | 0.0079 | 0.0024 | 6.68e-01 | 1.929e+00 | 0.165 / 0.246 |
| two_ellipses_gap010 | 0.0077 | 0.0107 | 0.0062 | 3.99e-01 | 7.480e-01 | 0.153 / 0.256 |
| annulus_disk | 0.0159 | 0.0161 | 0.0040 | 1.76e-01 | 3.075e+00 | 0.128 / 0.236 |

## C merge window (report only)

two ellipses at gap 0.8 ell = 0.0204: BIE merged [(0, 1, 0.020361313480691257)], masked 26, sigma_tail [0.048667990551278205], lambda_min 7.516e-01; grid components {'n_components': 1, 'n_compat': 1}; worst rel 0.0137, max|lambda-1| 0.1439
