"""T1.4 DoD: synthetic error field e = 0.5 sin(pi x) + 0.2 sin(15 pi x) gives exactly two
peaks (positive-frequency side) at omega=pi and omega=15pi with amplitude ratio 0.5:0.2 to
1%. Parseval: numpy's unnormalised FFT convention gives sum|e_hat|^2 = N * sum|e|^2 (verified
directly before writing this test), not sum|e_hat|^2 == sum|e|^2 as the phase doc's wording
implies -- tested against the actual, correct convention rather than the imprecise wording.
"""
from __future__ import annotations

import math

import numpy as np

from qapinn.xai.spectral_error import (
    error_trajectory,
    per_frequency_error,
    save_spectral_error_checkpoint,
    steps_to_tolerance_per_mode,
)


def test_per_frequency_error_two_peaks_correct_ratio():
    # domain length 2 (not 1): the fundamental frequency is then 2*pi/L = pi, so BOTH
    # pi and 15*pi land exactly on FFT bins with zero spectral leakage (domain length 1
    # would put pi at a half-integer cycle count -- the same phenomenon pinned in T0.9).
    n = 512
    length = 2.0
    dx = length / n
    x = np.arange(n) * dx
    e = 0.5 * np.sin(math.pi * x) + 0.2 * np.sin(15 * math.pi * x)

    omega, mag = per_frequency_error(e, np.zeros_like(e), grid_shape=(n,), dx=(dx,))

    idx_pi = int(np.argmin(np.abs(omega - math.pi)))
    idx_15pi = int(np.argmin(np.abs(omega - 15 * math.pi)))

    # these two (plus their negative-frequency mirrors) must be the dominant peaks
    sorted_idx = np.argsort(mag)[::-1]
    top_four_omegas = {round(abs(omega[i]), 3) for i in sorted_idx[:4]}
    assert top_four_omegas == {round(math.pi, 3), round(15 * math.pi, 3)}

    ratio = mag[idx_pi] / mag[idx_15pi]
    expected_ratio = 0.5 / 0.2
    assert abs(ratio - expected_ratio) / expected_ratio < 0.01


def test_per_frequency_error_parseval():
    n = 256
    dx = 1.0 / n
    x = np.arange(n) * dx
    e = 0.5 * np.sin(math.pi * x) + 0.2 * np.sin(15 * math.pi * x) + 0.1 * np.cos(7 * math.pi * x)

    _omega, mag = per_frequency_error(e, np.zeros_like(e), grid_shape=(n,), dx=(dx,))

    lhs = np.sum(mag**2)
    rhs = n * np.sum(e**2)  # numpy's unnormalised FFT convention: sum|X_k|^2 = N*sum|x_n|^2
    assert abs(lhs - rhs) / rhs < 1e-10


def test_per_frequency_error_2d_radial_binning_shape():
    n0, n1 = 32, 32
    dx = (1.0 / n0, 1.0 / n1)
    x0 = np.arange(n0) * dx[0]
    x1 = np.arange(n1) * dx[1]
    X0, X1 = np.meshgrid(x0, x1, indexing="ij")
    e = np.sin(3 * math.pi * X0) * np.sin(math.pi * X1)

    omega, binned = per_frequency_error(e, np.zeros_like(e), grid_shape=(n0, n1), dx=dx, radial_bins=64)

    assert omega.shape == (64,)
    assert binned.shape == (64,)
    assert np.all(binned >= 0.0)


def test_per_frequency_error_2d_unbinned_matches_grid_shape():
    n0, n1 = 16, 16
    dx = (1.0 / n0, 1.0 / n1)
    e = np.random.default_rng(0).standard_normal((n0, n1))

    omega_grid_2d, mag_grid = per_frequency_error(e, np.zeros_like(e), grid_shape=(n0, n1), dx=dx, radial_bins=None)

    assert omega_grid_2d.shape == (n0, n1)
    assert mag_grid.shape == (n0, n1)


def test_save_and_assemble_error_trajectory(tmp_path):
    n = 64
    dx = 1.0 / n
    x = np.arange(n) * dx

    for step, amp in [(0, 1.0), (100, 0.3), (200, 0.05)]:
        e = amp * np.sin(math.pi * x) + 0.2 * np.sin(15 * math.pi * x)
        save_spectral_error_checkpoint(e, np.zeros_like(e), grid_shape=(n,), dx=(dx,), step=step, run_dir=tmp_path)

    traj = error_trajectory(tmp_path)
    assert traj.shape[0] == 3


def test_steps_to_tolerance_per_mode_quantifies_staircase(tmp_path):
    # domain length 2, as above, so pi/15*pi land exactly on bins (no leakage) and the
    # per-checkpoint magnitude scales purely with amp_lo/amp_hi.
    n = 64
    length = 2.0
    dx = length / n
    x = np.arange(n) * dx

    # low mode (pi) decays fast; high mode (15pi) barely decays -- a toy staircase
    schedule = [
        (0, 1.0, 1.0),
        (100, 0.05, 0.9),
        (200, 0.01, 0.8),
        (300, 0.005, 0.7),
    ]
    for step, amp_lo, amp_hi in schedule:
        e = amp_lo * np.sin(math.pi * x) + amp_hi * np.sin(15 * math.pi * x)
        save_spectral_error_checkpoint(e, np.zeros_like(e), grid_shape=(n,), dx=(dx,), step=step, run_dir=tmp_path)

    traj = error_trajectory(tmp_path)
    data = np.load(tmp_path / "xai" / "specerr.npz")
    omega, steps = data["omega"], data["steps"]

    result = steps_to_tolerance_per_mode(traj, omega, steps, target_freqs=[math.pi, 15 * math.pi], tol=0.1)

    lo_key = min(result.keys(), key=lambda k: abs(k - math.pi))
    hi_key = min(result.keys(), key=lambda k: abs(k - 15 * math.pi))

    assert result[lo_key] == 100  # first step where 0.05 < 0.1*1.0
    assert result[hi_key] == -1  # 0.7 never drops below 0.1*1.0=0.1
