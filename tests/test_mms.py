"""T0.11 DoD: method of manufactured solutions on solve_burgers_spectral shows spectral
convergence in space (error drops >= 2 orders of magnitude, n_x: 32 -> 64) and 4th-order
convergence in time (observed order >= 3.8 as dt halves).

Manufactured solution: u*(x,t) = exp(-a*sin(pi*x/2)^2) * exp(-t), a periodic (period 2),
C-infinity bump with a controllable number of significant Fourier harmonics -- chosen (and
empirically checked) so n_x=32 sits well above machine-precision error while n_x=64 is
already deep into round-off, giving a clean, non-trivial spectral-convergence signal.
"""
from __future__ import annotations

import math

import numpy as np

from qapinn.reference.spectral import solve_burgers_spectral

NU = 0.01 / math.pi
A = 6.0  # bump sharpness, calibrated empirically (see commit message / derivation notes)


def _u_exact(x: np.ndarray, t: float) -> np.ndarray:
    return np.exp(-A * np.sin(math.pi * x / 2) ** 2) * np.exp(-t)


def _forcing(x: np.ndarray, t: float) -> np.ndarray:
    """Hand-derived F = u_t + u*u_x - nu*u_xx for u* above, via the chain rule on
    u* = env(x) * exp(-t) with env = exp(-A*sin(pi x/2)^2)."""
    s = np.sin(math.pi * x / 2)
    c = np.cos(math.pi * x / 2)
    env = np.exp(-A * s**2)
    u = env * np.exp(-t)
    u_t = -u
    dlog_env_dx = -A * math.pi * s * c  # d/dx[log env] = -A * 2*s*c * (pi/2)
    u_x = u * dlog_env_dx
    d2log_env_dx2 = -A * math.pi * (c * c - s * s) * (math.pi / 2)
    u_xx = u * (d2log_env_dx2 + dlog_env_dx**2)
    return u_t + u * u_x - NU * u_xx


def _max_error(n_x: int, n_t: int, t_final: float) -> float:
    x, t, u = solve_burgers_spectral(
        NU, n_x, n_t, t_final, lambda xx: _u_exact(xx, 0.0), forcing=_forcing
    )
    return float(np.max(np.abs(u[-1] - _u_exact(x, t[-1]))))


def test_spectral_convergence_in_space():
    t_final = 0.2
    n_t = 400  # fine in time so temporal error doesn't contaminate the spatial signal

    err_32 = _max_error(n_x=32, n_t=n_t, t_final=t_final)
    err_64 = _max_error(n_x=64, n_t=n_t, t_final=t_final)

    assert err_64 < err_32 * 1e-2, f"err_32={err_32}, err_64={err_64}"


def test_fourth_order_convergence_in_time():
    t_final = 0.2
    n_x = 64  # fine enough in space that spatial error doesn't contaminate the temporal signal

    err_dt = _max_error(n_x=n_x, n_t=21, t_final=t_final)  # dt = t_final/20
    err_dt_half = _max_error(n_x=n_x, n_t=41, t_final=t_final)  # dt = t_final/40

    observed_order = math.log(err_dt / err_dt_half) / math.log(2.0)
    assert observed_order >= 3.8, f"observed order {observed_order} (err_dt={err_dt}, err_dt_half={err_dt_half})"
