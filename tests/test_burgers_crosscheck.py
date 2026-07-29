"""T0.12 DoD: Cole-Hopf (T0.12) and pseudospectral (T0.11) solutions of Burgers agree to
< 1e-6 in max-norm on [-1,1] x [0,1] at nu = 0.01/pi -- two independent methods agreeing is
what makes this ground truth, not a guess.

Resolving the viscous shock that forms near t~0.5 to this tolerance requires n_x=2048,
n_t=3201 (calibrated empirically: n_t, not n_x, turned out to be the limiting factor near
the shock -- doubling n_t from 1601 to 3201 at fixed n_x=2048 cut the worst-case error by
~15x, while doubling n_x at fixed n_t changed almost nothing). Cole-Hopf quadrature is
evaluated on a strided subsample of ~80 time slices rather than the full grid here, kept
as-is for this test's own runtime -- but note `burgers_colehopf` itself is now CHUNKED
(T1.12: `_compute_metrics`/`reference_solution` calls it on the FULL eval_grid at training
time, e.g. 1024x1024 points for Burgers' 2-D domain, which OOM'd materialising the full
[M, n_quad] array before chunking was added -- see `test_burgers_colehopf_chunking_matches_unchunked` below).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qapinn.reference.colehopf import burgers_colehopf, cached_burgers_colehopf
from qapinn.reference.spectral import solve_burgers_spectral

NU = 0.01 / math.pi


def test_burgers_colehopf_chunking_matches_unchunked():
    # M > colehopf._CHUNK_SIZE forces the chunked loop to run over >= 2 chunks; verify it
    # gives IDENTICAL results to evaluating well below the chunk size in one shot (the
    # underlying per-point formula is unchanged, only the loop batching is new -- T1.12).
    import qapinn.reference.colehopf as colehopf_mod

    rng = np.random.default_rng(0)
    n = colehopf_mod._CHUNK_SIZE + 137  # spans a chunk boundary
    x = rng.uniform(-1.0, 1.0, size=n)
    t = rng.uniform(0.01, 1.0, size=n)

    u_chunked = burgers_colehopf(x, t, NU, n_quad=64)
    u_pointwise = np.concatenate(
        [burgers_colehopf(x[i : i + 5000], t[i : i + 5000], NU, n_quad=64) for i in range(0, n, 5000)]
    )
    assert np.allclose(u_chunked, u_pointwise, atol=1e-12)


def test_burgers_colehopf_handles_full_eval_grid_scale_without_oom():
    # The exact scenario that OOM'd before chunking: a full 2-D eval_grid's worth of
    # points (T1.12's smoke-test sweep across all four PDEs first surfaced this).
    n_eval = 256  # smaller than the real n_eval=1024 (still ~65k points, well past 20k chunk)
    x = np.linspace(-1.0, 1.0, n_eval, endpoint=False)
    t = np.linspace(0.0, 1.0, n_eval, endpoint=False)
    xx, tt = np.meshgrid(x, t, indexing="ij")

    u = burgers_colehopf(xx.reshape(-1), tt.reshape(-1), NU, n_quad=200)
    assert u.shape == (n_eval * n_eval,)
    assert np.all(np.isfinite(u))


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
