"""Cole-Hopf exact Burgers solution via Gauss-Hermite quadrature (T0.12).

For u_t + u*u_x = nu*u_xx with u(x,0) = -sin(pi x), the Cole-Hopf transform u = -2*nu*phi_x/phi
(phi solving the heat equation) gives the closed form:

    u(x,t) = - I1(x,t) / I2(x,t)
    I1(x,t) = int sin(pi(x-eta)) * exp[-cos(pi(x-eta))/(2 pi nu) - eta^2/(4 nu t)] deta
    I2(x,t) = int             exp[-cos(pi(x-eta))/(2 pi nu) - eta^2/(4 nu t)] deta

(integrals over the real line). Substituting eta = sqrt(4*nu*t)*xi turns the eta^2/(4 nu t)
factor into exactly the Gauss-Hermite weight exp(-xi^2), so Gauss-Hermite quadrature is exact
for the polynomial part and highly accurate overall -- far faster per point than
scipy.integrate.quad while retaining high-order accuracy, since the kernel truly is Gaussian.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import roots_hermite

# D7: never materialise the full [M, n_quad] intermediate array for M in the millions (a
# full eval_grid(1024) on Burgers' 2-D (x,t) domain is ~1e6 points -- at n_quad=200 that's
# several GB and OOMs). Chunked instead, matching the same pattern used for NTK Jacobians
# and residual evaluation elsewhere. This is pure numpy (no autograd), so a much larger
# chunk than the D7 autograd-chunking convention (e.g. 64) is fine.
_CHUNK_SIZE = 20_000


def burgers_colehopf(x: np.ndarray, t: np.ndarray, nu: float, n_quad: int = 200) -> np.ndarray:
    """Exact solution of u_t + u*u_x = nu*u_xx, u(x,0) = -sin(pi x), on the real line.

    x, t broadcast together. t <= 0 returns the initial condition directly (the eta =
    sqrt(4*nu*t)*xi substitution is singular at t=0).
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    shape = np.broadcast_shapes(x.shape, t.shape)
    x_b = np.broadcast_to(x, shape).reshape(-1)
    t_b = np.broadcast_to(t, shape).reshape(-1)

    # scipy.special.roots_hermite (not numpy.polynomial.hermite.hermgauss, which becomes
    # numerically unstable -- NaN nodes/weights -- above n_quad ~ 300-500) -- weight exp(-xi^2)
    nodes, weights = roots_hermite(n_quad)

    result = np.where(t_b <= 0.0, -np.sin(np.pi * x_b), 0.0)
    idx = np.nonzero(t_b > 0.0)[0]

    for start in range(0, len(idx), _CHUNK_SIZE):
        chunk_idx = idx[start : start + _CHUNK_SIZE]
        x_m = x_b[chunk_idx]  # [m]
        t_m = t_b[chunk_idx]  # [m]
        shift = np.sqrt(4.0 * nu * t_m)[:, None] * nodes[None, :]  # [m, n_quad]
        arg = x_m[:, None] - shift  # [m, n_quad]

        exponent = -np.cos(np.pi * arg) / (2.0 * np.pi * nu)  # [m, n_quad]
        # subtract the per-row max before exponentiating (log-sum-exp style stabilisation);
        # this common factor cancels exactly in the numerator/denominator ratio below.
        exponent = exponent - np.max(exponent, axis=1, keepdims=True)
        kernel = np.exp(exponent)

        numerator = np.sum(weights[None, :] * np.sin(np.pi * arg) * kernel, axis=1)
        denominator = np.sum(weights[None, :] * kernel, axis=1)
        result[chunk_idx] = -numerator / denominator

    return result.reshape(shape)


def cached_burgers_colehopf(
    x: np.ndarray,
    t: np.ndarray,
    nu: float,
    n_quad: int = 200,
    cache_path: Path | str = Path("results/reference/burgers_ref.npz"),
) -> np.ndarray:
    """Same as burgers_colehopf, but caches to disk -- this quadrature is expensive and,
    for a fixed (x, t, nu, n_quad), never changes (T0.12)."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        cached = np.load(cache_path)
        if (
            cached["x"].shape == np.shape(x)
            and cached["t"].shape == np.shape(t)
            and np.array_equal(cached["x"], x)
            and np.array_equal(cached["t"], t)
            and float(cached["nu"]) == nu
            and int(cached["n_quad"]) == n_quad
        ):
            return cached["u"]

    u = burgers_colehopf(x, t, nu, n_quad=n_quad)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, x=x, t=t, nu=nu, n_quad=n_quad, u=u)
    return u
