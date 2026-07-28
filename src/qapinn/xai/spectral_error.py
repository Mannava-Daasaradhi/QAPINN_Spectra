"""Spectral error utilities. Only `omega_grid` + `pauli_freq` for now (D12 guard, T0.9);
T1.4 adds the per-frequency error trajectory.

D12: angular frequency omega, complex exponential convention e^{i omega x}. sin(15 pi x)
has omega = 15*pi ~= 47.12, NOT k=7.5 cycles. Mixing conventions is the single most likely
silent bug in this project -- every frequency axis in this codebase uses this convention.
"""
from __future__ import annotations

import numpy as np


def omega_grid(n: int, dx: float) -> np.ndarray:
    """Angular-frequency grid matching numpy.fft.fft output ordering."""
    return 2.0 * np.pi * np.fft.fftfreq(n, d=dx)


def pauli_freq(omega_scaling: float) -> float:
    """Encoding gate RZ(omega*x) = exp(-i omega x Z / 2) has generator eigenvalues
    +-omega/2, so eigenvalue DIFFERENCES (the realised frequencies) are {-omega, 0,
    +omega} -- the scaling itself, NOT scaling/2. This factor-of-2 gap between "generator
    eigenvalue" and "realised frequency" is the second most likely silent bug in this
    project (01_CONVENTIONS.md §2)."""
    return omega_scaling
