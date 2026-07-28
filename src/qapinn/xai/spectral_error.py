"""Spectral error utilities (D12 guard, T0.9; per-frequency error trajectory, T1.4).

D12: angular frequency omega, complex exponential convention e^{i omega x}. sin(15 pi x)
has omega = 15*pi ~= 47.12, NOT k=7.5 cycles. Mixing conventions is the single most likely
silent bug in this project -- every frequency axis in this codebase uses this convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_RADIAL_BINS = 64


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


def per_frequency_error(
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    grid_shape: tuple[int, ...],
    dx: tuple[float, ...],
    radial_bins: int | None = DEFAULT_RADIAL_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (omega, |e_hat(omega)|) for the error field e = u_pred - u_exact.

    1-D: full FFT of the error on the uniform grid.
    2-D: FFT2, then RADIAL BINNING of |e_hat| into `radial_bins` bins of |omega| (default
         64), because the heatmap axis must stay 1-D. Pass radial_bins=None to instead get
         the UNBINNED 2-D (omega magnitude grid, |e_hat| grid) pair -- P4's four discrete
         target frequencies matter individually, not radially averaged away.
    """
    error = (np.asarray(u_pred) - np.asarray(u_exact)).reshape(grid_shape)
    ndim = len(grid_shape)

    if ndim == 1:
        omega = omega_grid(grid_shape[0], dx[0])
        e_hat = np.fft.fft(error)
        return omega, np.abs(e_hat)

    if ndim == 2:
        omega0 = omega_grid(grid_shape[0], dx[0])
        omega1 = omega_grid(grid_shape[1], dx[1])
        e_hat = np.fft.fft2(error)
        mag = np.abs(e_hat)

        omega0_grid, omega1_grid = np.meshgrid(omega0, omega1, indexing="ij")
        omega_radial = np.sqrt(omega0_grid**2 + omega1_grid**2)

        if radial_bins is None:
            return omega_radial, mag

        max_r = omega_radial.max()
        bin_edges = np.linspace(0.0, max_r, radial_bins + 1)
        bin_idx = np.clip(np.digitize(omega_radial.ravel(), bin_edges) - 1, 0, radial_bins - 1)

        sums = np.bincount(bin_idx, weights=mag.ravel(), minlength=radial_bins)
        counts = np.bincount(bin_idx, minlength=radial_bins)
        binned = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        return bin_centers, binned

    raise ValueError(f"per_frequency_error only supports 1-D or 2-D grids, got ndim={ndim}")


def save_spectral_error_checkpoint(
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    grid_shape: tuple[int, ...],
    dx: tuple[float, ...],
    step: int,
    run_dir: Path | str,
    radial_bins: int | None = DEFAULT_RADIAL_BINS,
) -> np.ndarray:
    """Computes per_frequency_error for this checkpoint and appends it to
    xai/specerr.npz's accumulated (omega, errors[n_checkpoints,n_freq], steps) arrays --
    conventions §8 specifies ONE specerr.npz per run holding the full trajectory, not one
    file per checkpoint. Returns |e_hat(omega)| for this checkpoint."""
    omega, mag = per_frequency_error(u_pred, u_exact, grid_shape, dx, radial_bins=radial_bins)

    out_dir = Path(run_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "specerr.npz"

    if path.is_file():
        existing = np.load(path)
        steps = np.concatenate([existing["steps"], [step]])
        errors = np.vstack([existing["errors"], mag[None, :]])
    else:
        steps = np.array([step])
        errors = mag[None, :]

    np.savez(path, omega=omega, errors=errors, steps=steps)
    return mag


def error_trajectory(run_dir: Path | str) -> np.ndarray:
    """[n_checkpoints, n_freq] assembled from xai/specerr.npz's accumulated per-checkpoint
    arrays."""
    path = Path(run_dir) / "xai" / "specerr.npz"
    return np.load(path)["errors"]


def steps_to_tolerance_per_mode(
    traj: np.ndarray,
    omega: np.ndarray,
    steps: np.ndarray,
    target_freqs: list[float] | None = None,
    tol: float = 0.1,
) -> dict[float, int]:
    """For each target mode, the first checkpoint STEP (not index -- `steps` maps
    checkpoint index -> training step) where |e_hat(omega)| falls below tol times its
    initial (first-checkpoint) value. -1 if never reached. target_freqs defaults to every
    bin in `omega` when not given (the phase doc's signature doesn't take target_freqs
    explicitly, but the caller needs to specify actual frequencies of interest -- e.g.
    {pi, 15*pi} for P1 -- to get a meaningful, non-exhaustive result, and needs `steps` to
    report real training-step numbers rather than opaque checkpoint indices)."""
    if target_freqs is None:
        target_freqs = omega.tolist()

    result: dict[float, int] = {}
    for f in target_freqs:
        idx = int(np.argmin(np.abs(omega - f)))
        initial = traj[0, idx]
        threshold = tol * initial
        step_found = -1
        for i in range(traj.shape[0]):
            if traj[i, idx] < threshold:
                step_found = int(steps[i])
                break
        result[float(omega[idx])] = step_found
    return result
