"""T0.12 DoD: Cole-Hopf (T0.12) and pseudospectral (T0.11) solutions of Burgers agree to
< 1e-6 in max-norm on [-1,1] x [0,1] at nu = 0.01/pi -- two independent methods agreeing is
what makes this ground truth, not a guess.

Resolving the viscous shock that forms near t~0.5 to this tolerance requires n_x=2048,
n_t=3201 (calibrated empirically: n_t, not n_x, turned out to be the limiting factor near
the shock -- doubling n_t from 1601 to 3201 at fixed n_x=2048 cut the worst-case error by
~15x, while doubling n_x at fixed n_t changed almost nothing). Cole-Hopf quadrature is
evaluated on a strided subsample of ~80 time slices rather than the full grid (comparing
every point would need a [n_x*n_t, n_quad] intermediate array, several GB) -- still "a
grid" per the DoD, and it includes the worst-case time (t~0.51, near shock formation) found
during calibration.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qapinn.reference.colehopf import cached_burgers_colehopf
from qapinn.reference.spectral import solve_burgers_spectral

NU = 0.01 / math.pi


@pytest.mark.slow
def test_colehopf_matches_pseudospectral():
    n_x, n_t = 2048, 3201
    t_final = 1.0

    x, t, u_spectral = solve_burgers_spectral(
        NU, n_x, n_t, t_final, lambda xx: -np.sin(math.pi * xx)
    )

    stride = max(1, n_t // 80)
    idxs = list(range(0, n_t, stride))
    if idxs[-1] != n_t - 1:
        idxs.append(n_t - 1)
    t_sample = t[idxs]

    xx = np.broadcast_to(x[None, :], (len(idxs), n_x))
    tt = np.broadcast_to(t_sample[:, None], (len(idxs), n_x))

    u_colehopf = cached_burgers_colehopf(xx, tt, NU, n_quad=200)

    max_err = np.max(np.abs(u_spectral[idxs] - u_colehopf))
    assert max_err < 1e-6, f"max error {max_err}"
