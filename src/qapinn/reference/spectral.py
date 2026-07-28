"""Fourier pseudospectral reference solver for Burgers (T0.11).

Independent ground truth for P3 (cross-checked against Cole-Hopf, T0.12) and the vehicle
for the MMS convergence test. Domain [-1,1) is treated as periodic (period 2): for the
physical P3 instance (u0 = -sin(pi x), zero Dirichlet at x=+-1) this is exact, not an
approximation -- the PDE preserves the odd symmetry of that initial condition, and
periodicity (x=1 identified with x=-1) plus antisymmetry forces u(+-1,t)=0 for all t.

Spatial discretisation: FFT differentiation with the 2/3 dealiasing rule on the nonlinear
term. Time stepping: classic RK4 with an integrating factor (exact treatment of the linear
viscous term nu*u_xx, applied per-step in local time so exponents stay small).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from qapinn.xai.spectral_error import omega_grid


def solve_burgers_spectral(
    nu: float,
    n_x: int,
    n_t: int,
    t_final: float,
    u0: Callable[[np.ndarray], np.ndarray],
    forcing: Callable[[np.ndarray, float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve u_t + u*u_x = nu*u_xx (+ forcing) on the periodic domain [-1,1).

    n_t is the number of output time points (including t=0), so u has shape [n_t, n_x] --
    n_t - 1 RK4 steps of size dt = t_final/(n_t-1) are taken internally.

    returns (x, t, u) with u of shape [n_t, n_x].
    """
    if n_t < 2:
        raise ValueError(f"n_t must be >= 2 (need at least the initial condition plus one step), got {n_t}")

    dx = 2.0 / n_x
    x = -1.0 + dx * np.arange(n_x)
    omega = omega_grid(n_x, dx)  # angular wavenumber grid (D12 convention)
    omega2 = omega**2

    omega_nyquist = np.abs(omega).max()
    dealias = np.abs(omega) <= (2.0 / 3.0) * omega_nyquist

    dt = t_final / (n_t - 1)
    linear_eigs = -nu * omega2  # <= 0, real

    def nonlinear_hat(u_hat: np.ndarray, t: float) -> np.ndarray:
        u_phys = np.fft.ifft(u_hat).real
        u_x_phys = np.fft.ifft(1j * omega * u_hat).real
        rhs_hat = -np.fft.fft(u_phys * u_x_phys) * dealias
        if forcing is not None:
            rhs_hat = rhs_hat + np.fft.fft(forcing(x, t))
        return rhs_hat

    def f_local(tau: float, w: np.ndarray, t_n: float) -> np.ndarray:
        """dw/dtau where w(tau) = exp(-L*tau) * u_hat(t_n+tau); w(0) = u_hat(t_n) exactly."""
        factor_fwd = np.exp(linear_eigs * tau)
        factor_bwd = np.exp(-linear_eigs * tau)
        u_hat = factor_fwd * w
        return factor_bwd * nonlinear_hat(u_hat, t_n + tau)

    u_hat = np.fft.fft(u0(x))
    u_out = np.zeros((n_t, n_x), dtype=np.float64)
    t_out = np.zeros(n_t, dtype=np.float64)
    u_out[0] = np.fft.ifft(u_hat).real

    for step in range(n_t - 1):
        t_n = step * dt
        w0 = u_hat  # w(0) == u_hat(t_n) in local time

        k1 = f_local(0.0, w0, t_n)
        k2 = f_local(dt / 2.0, w0 + (dt / 2.0) * k1, t_n)
        k3 = f_local(dt / 2.0, w0 + (dt / 2.0) * k2, t_n)
        k4 = f_local(dt, w0 + dt * k3, t_n)

        w_next = w0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u_hat = np.exp(linear_eigs * dt) * w_next

        t_out[step + 1] = t_n + dt
        u_out[step + 1] = np.fft.ifft(u_hat).real

    return x, t_out, u_out
