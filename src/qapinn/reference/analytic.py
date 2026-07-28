"""Analytic reference solutions for P1, P2, P4 (01_CONVENTIONS.md §5, T0.10).

These wrap the same closed forms as each PDE's `exact()`, operating on plain numpy arrays
for the reference API. `heat_exact` additionally supports a general mode list (not just the
two seeded modes {1, 15}) via separation of variables, so it can serve general initial data.
"""
from __future__ import annotations

import numpy as np


def poisson_exact(x: np.ndarray, alpha: float) -> np.ndarray:
    """u*(x) = sin(pi x) + alpha sin(15 pi x)."""
    return np.sin(np.pi * x) + alpha * np.sin(15 * np.pi * x)


def heat_exact(
    x: np.ndarray, t: np.ndarray, nu: float, modes: list[tuple[int, float]]
) -> np.ndarray:
    """Separation-of-variables solution for u_t = nu*u_xx on [0,1] with u(0,t)=u(1,t)=0:

        u(x,t) = sum_j w_j * sin(n_j*pi*x) * exp(-nu*(n_j*pi)^2*t)

    modes: list of (n, weight) pairs. The seeded P2 instance is modes=[(1,1.0),(15,alpha)].
    """
    shape = np.broadcast_shapes(np.shape(x), np.shape(t))
    result = np.zeros(shape, dtype=np.float64)
    for n, w in modes:
        result = result + w * np.sin(n * np.pi * x) * np.exp(-nu * (n * np.pi) ** 2 * t)
    return result


def helmholtz_exact(x: np.ndarray, y: np.ndarray, a1: float, a2: float) -> np.ndarray:
    """u*(x,y) = sin(a1*pi*x) * sin(a2*pi*y)."""
    return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y)
