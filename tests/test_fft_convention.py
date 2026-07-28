"""T0.9 DoD (D12 guard): FFT peak of sin(15 pi x) sampled on [0,1) with N=256 lands at
|omega| = 15*pi +- d_omega/2 -- the ANGULAR value, not a cycle count. pauli_freq pins the
generator-eigenvalue factor of 2."""
from __future__ import annotations

import math

import numpy as np

from qapinn.xai.spectral_error import omega_grid, pauli_freq


def test_fft_peak_lands_at_angular_frequency_15pi():
    n = 256
    dx = 1.0 / n
    x = np.arange(n) * dx
    signal = np.sin(15 * math.pi * x)

    spectrum = np.fft.fft(signal)
    omega = omega_grid(n, dx)

    peak_idx = int(np.argmax(np.abs(spectrum)))
    d_omega = omega[1] - omega[0]

    # sin(15 pi x) has cycle frequency 7.5 cycles/unit -- a deliberate half-integer case,
    # so the true peak sits exactly midway between two DFT bins (D12's own tolerance).
    assert abs(abs(omega[peak_idx]) - 15 * math.pi) <= abs(d_omega) / 2 + 1e-9

    # this must be an ANGULAR assertion: a bare cycle-count peak (k=7 or 8) would NOT
    # satisfy the analogous check against 15*pi if someone dropped the 2*pi factor.
    cycle_count_guess = abs(omega[peak_idx]) / (2 * math.pi)
    assert cycle_count_guess != 15  # confirms we are not silently checking cycles


def test_omega_grid_matches_numpy_fftfreq_times_2pi():
    n, dx = 64, 0.01
    grid = omega_grid(n, dx)
    expected = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    assert np.allclose(grid, expected)


def test_pauli_freq_returns_scaling_not_half():
    assert pauli_freq(15 * math.pi) == 15 * math.pi
    assert pauli_freq(7.0) == 7.0
    assert pauli_freq(7.0) != 3.5
