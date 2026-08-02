"""SMCD Part 1: target spectrum (Prop. 2, project.md Section 5). T2.8.

Two paths to a `TargetSpectrum`:
  analytic_spectrum(pde, eps)   -- closed form, exact by construction. Defined for P1
                                    (Poisson), P2 (Heat), P4 (Helmholtz); returns None for
                                    P3 (Burgers, nonlinear, no closed-form solution).
  empirical_spectrum(pde, eps)  -- FFT / sine-projection of the reference solution
                                    (T0.13). The only path for P3; a cross-check
                                    everywhere else (DoD item 3).

Convention: frequencies are SIGNED angular frequencies (D12), matching circuits.py's
frequencies() and 07_circuit_fourier_spectrum.md. For a 1-D real problem on a Dirichlet
eigenbasis this collapses to reporting only the positive representative of each conjugate
pair, with weight = the REAL sine-series coefficient itself (not the halved
complex-exponential coefficient) -- this is what makes P1's weights come out as exactly
{1, alpha}, not {0.5, alpha/2}. For d>1 (Helmholtz) there is no such fold: all 2^d sign
combinations are reported separately, since a real product of sines genuinely has that
many independent complex modes (verified: sin(a1*pi*x)*sin(a2*pi*y) expands into exactly
4 equal-magnitude terms at (+-a1*pi, +-a2*pi), matching the DoD's "four points").

Energy weight (project.md Section 5, the definition that makes P2 a quantitative negative
control): steady problems use w(omega) = |u_hat(omega)|; time-dependent problems use the
TIME-INTEGRATED amplitude w(omega) = ||u_hat(omega, .)||_{L2(0,T)}, computed here via the
exact closed-form integral of |c(omega)|^2 * exp(-2*nu*omega^2*t) (analytic path) or a
per-timestep sine projection followed by trapezoid-rule L2 integration (empirical path).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qapinn.pdes.base import PDE
from qapinn.reference import reference_solution
from qapinn.xai.spectral_error import omega_grid

_K_MAX_1D = 32


@dataclass(frozen=True)
class TargetSpectrum:
    omega: np.ndarray  # [M, d] angular frequencies (d = number of SPATIAL dims)
    weight: np.ndarray  # [M] energy weight
    eps: float
    support: np.ndarray  # [M] bool mask, the set S_eps
    K: float  # max |omega| within the support
    delta: float  # min nonzero spacing within the support
    source: str  # "analytic" | "empirical"


def _finalize(omega: np.ndarray, weight: np.ndarray, eps: float, source: str) -> TargetSpectrum:
    """Builds support/K/delta from raw (omega, weight) arrays. `delta` is the minimum
    nonzero pairwise Euclidean distance between distinct points IN the support -- a plain,
    dimension-agnostic reading of "min nonzero spacing within the support". This is a
    descriptive statistic of the discovered spectrum; it is not necessarily the same
    quantity Algorithm 1 (T2.9) chooses as its own circuit-design frequency unit (e.g. for
    P1's two-point support {pi, 15*pi}, the pairwise gap is 14*pi, but the natural
    ternary-scaling base frequency for circuit design is pi, the two points' common
    divisor -- a different, algorithm-specific quantity T2.9 must derive itself).
    """
    max_w = float(weight.max()) if weight.size else 0.0
    support = weight > eps * max_w
    omega_supp = omega[support]

    if omega_supp.shape[0] == 0:
        K = 0.0
        delta = 0.0
    else:
        mags = np.linalg.norm(omega_supp, axis=1)
        K = float(mags.max())
        if omega_supp.shape[0] == 1:
            delta = 0.0
        else:
            diffs = omega_supp[:, None, :] - omega_supp[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            np.fill_diagonal(dists, np.inf)
            delta = float(dists.min())

    return TargetSpectrum(omega=omega, weight=weight, eps=eps, support=support, K=K, delta=delta, source=source)


# --- analytic path -------------------------------------------------------------------


def analytic_spectrum(pde: PDE, eps: float) -> TargetSpectrum | None:
    if pde.name == "poisson":
        return _analytic_poisson(pde, eps)
    if pde.name == "heat":
        return _analytic_heat(pde, eps)
    if pde.name == "helmholtz":
        return _analytic_helmholtz(pde, eps)
    return None  # burgers: nonlinear, no closed form -- caller falls back to empirical_spectrum


def _analytic_poisson(pde: PDE, eps: float) -> TargetSpectrum:
    alpha = pde.params["alpha"]
    omega = np.array([[math.pi], [15.0 * math.pi]])
    weight = np.array([1.0, abs(alpha)])
    return _finalize(omega, weight, eps, "analytic")


def _l2_time_weight(decay: float, T: float) -> float:
    """||exp(-decay*t)||_{L2(0,T)} = sqrt(integral_0^T exp(-2*decay*t) dt)."""
    if decay < 1e-12:
        return math.sqrt(T)
    return math.sqrt((1.0 - math.exp(-2.0 * decay * T)) / (2.0 * decay))


def _analytic_heat(pde: PDE, eps: float) -> TargetSpectrum:
    alpha = pde.params["alpha"]
    nu = pde.params["nu"]
    T = pde.domain.bounds[pde.domain.time_axis][1]
    modes = [(math.pi, 1.0), (15.0 * math.pi, alpha)]

    omega_list, weight_list = [], []
    for k, c in modes:
        l2 = _l2_time_weight(nu * k**2, T)
        omega_list.append([k])
        weight_list.append(abs(c) * l2)
    return _finalize(np.array(omega_list), np.array(weight_list), eps, "analytic")


def _analytic_helmholtz(pde: PDE, eps: float) -> TargetSpectrum:
    a1 = pde.params["a1"]
    a2 = pde.params["a2"]
    signs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    omega = np.array([[s1 * a1 * math.pi, s2 * a2 * math.pi] for s1, s2 in signs])
    weight = np.full(4, 0.25)
    return _finalize(omega, weight, eps, "analytic")


# --- empirical path (FFT / sine-projection of the reference solution) ----------------


def empirical_spectrum(pde: PDE, eps: float, n_grid: int = 256) -> TargetSpectrum:
    if pde.domain.time_axis is None:
        return _empirical_steady(pde, eps, n_grid)
    return _empirical_time_dependent(pde, eps, n_grid)


def _empirical_steady(pde: PDE, eps: float, n_grid: int) -> TargetSpectrum:
    grid = pde.eval_grid(n_grid).numpy()
    u = reference_solution(pde, grid)

    if pde.dim == 1:
        return _sine_project_1d(u, grid[:, 0], pde.domain.bounds[0], eps)
    if pde.dim == 2:
        return _fft_2d(u, n_grid, pde.domain.bounds, eps)
    raise NotImplementedError(f"empirical_spectrum: unsupported steady pde.dim={pde.dim}")


def _sine_project_1d(
    u: np.ndarray, x: np.ndarray, bounds: tuple[float, float], eps: float, k_max: int = _K_MAX_1D
) -> TargetSpectrum:
    """Exact-eigenbasis sine projection, not a raw FFT: on a length-1 Dirichlet domain
    (P1) omega=pi sits exactly halfway between FFT bins (the worst-case leakage tie found
    in T1.5) -- projecting directly onto sin(k*pi*(x-lo)/L) has zero leakage by
    construction, for any k, regardless of domain length.
    """
    lo, hi = bounds
    length = hi - lo
    n = len(x)
    ks = np.arange(1, k_max + 1)
    basis = np.sin(np.outer(ks, x - lo) * math.pi / length)  # [k_max, n]
    coeffs = (2.0 / n) * (basis @ u)
    omega = (ks * math.pi / length).reshape(-1, 1)
    return _finalize(omega, np.abs(coeffs), eps, "empirical")


def _fft_2d(u: np.ndarray, n_grid: int, bounds: tuple, eps: float) -> TargetSpectrum:
    u_grid = u.reshape(n_grid, n_grid)
    dx = (bounds[0][1] - bounds[0][0]) / n_grid
    dy = (bounds[1][1] - bounds[1][0]) / n_grid
    omega_x = omega_grid(n_grid, dx)
    omega_y = omega_grid(n_grid, dy)

    u_hat = np.fft.fft2(u_grid)
    weight = np.abs(u_hat) / u_grid.size  # discrete -> continuous Fourier coefficient

    ox, oy = np.meshgrid(omega_x, omega_y, indexing="ij")
    omega = np.stack([ox.ravel(), oy.ravel()], axis=-1)
    return _finalize(omega, weight.ravel(), eps, "empirical")


def _empirical_time_dependent(pde: PDE, eps: float, n_grid: int, k_max: int = _K_MAX_1D) -> TargetSpectrum:
    if pde.dim != 2 or pde.domain.time_axis != 1:
        raise NotImplementedError(
            f"empirical_spectrum: time-dependent path assumes dim=2, time_axis=1 "
            f"(x then t); got dim={pde.dim}, time_axis={pde.domain.time_axis}"
        )
    lo_x, hi_x = pde.domain.bounds[0]
    lo_t, hi_t = pde.domain.bounds[1]
    length = hi_x - lo_x

    x = np.arange(n_grid) * (length / n_grid) + lo_x  # matches eval_grid's half-open convention
    # Graded (quadratic-near-lo_t) time grid, NOT uniform. A fast-decaying high-frequency
    # mode's decay timescale can be orders of magnitude shorter than T (P2's k=15*pi mode:
    # timescale ~0.0045 against T=1) -- a uniform grid under-resolves this badly (verified:
    # n_grid=256 uniform + rectangle rule gives 22% error on that mode's L2(0,T) weight,
    # nowhere near the empirical-vs-analytic 1% DoD). Clustering samples near t=lo_t
    # (standard graded-mesh technique for initial-layer-like transients) fixes this
    # generically -- it does not assume a specific (e.g. exponential) decay law, so it
    # also applies to P3 (Burgers), which has no known closed-form decay rate.
    j = np.arange(n_grid)
    t = lo_t + (hi_t - lo_t) * (j / (n_grid - 1)) ** 2

    xx, tt = np.meshgrid(x, t, indexing="ij")
    grid = np.stack([xx.ravel(), tt.ravel()], axis=-1)
    u = reference_solution(pde, grid).reshape(n_grid, n_grid)  # [x_idx, t_idx]

    ks = np.arange(1, k_max + 1)
    basis = np.sin(np.outer(ks, x - lo_x) * math.pi / length)  # [k_max, n_grid]
    coeffs_t = (2.0 / n_grid) * (basis @ u)  # [k_max, n_grid(t)], c_k(t_j)
    l2_over_t = np.sqrt(np.trapezoid(coeffs_t**2, t, axis=1))  # trapezoid rule, graded grid

    omega = (ks * math.pi / length).reshape(-1, 1)
    return _finalize(omega, l2_over_t, eps, "empirical")
