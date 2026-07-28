"""Figure generation entry points. T1.3 starts this file with `ntk_spectrum`;
T5.1 fills in the remaining ~11 figures and a `--regenerate-all` driver.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch

from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE
from qapinn.viz.style import family_color, save_figure
from qapinn.xai.ntk import ntk, probe_set, spectrum_stats


def make_ntk_spectrum_figure(
    models: dict[str, PINNModel],
    pde: PDE,
    output: str = "residual",
    run_ids: list[str] | None = None,
    omega_band: tuple[float, float] | None = None,
) -> dict[str, float]:
    """models: {family_name: model}. Renders log-log eigenvalue-vs-index for each family
    on the same axes. omega_band: (lo, hi) index range to shade as the encoded band Omega
    -- Phase 2 supplies the real value; this is the shading hook called for in T1.3.

    Returns {family_name: decay_exponent}.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    decay_exponents: dict[str, float] = {}
    for family, model in models.items():
        probe_x = probe_set(pde)
        K = ntk(model, pde, probe_x, output=output, group="all")
        stats = spectrum_stats(K)
        eigs = np.array(stats["eigenvalues"])
        eigs_positive = eigs[eigs > 0]
        idx = np.arange(1, len(eigs_positive) + 1)
        ax.loglog(idx, eigs_positive, label=family, color=family_color(family))
        decay_exponents[family] = stats["decay_exponent"]

    if omega_band is not None:
        ax.axvspan(omega_band[0], omega_band[1], color="grey", alpha=0.2, label=r"encoded band $\Omega$")

    ax.set_xlabel("index $i$")
    ax.set_ylabel(r"eigenvalue $\lambda_i$")
    ax.legend()
    fig.tight_layout()

    save_figure(fig, "ntk_spectrum", run_ids=run_ids)
    plt.close(fig)
    return decay_exponents


def _sine_mode_amplitudes(u_pred: np.ndarray, u_exact: np.ndarray, x: np.ndarray, k_max: int) -> np.ndarray:
    """|c_k| for k=1..k_max, where e = u_pred - u_exact is projected onto sin(k*pi*x).

    P1's exact solution sin(pi x) + alpha*sin(15 pi x) is built from the EXACT eigenfunctions
    of -u''=f on [0,1] with u(0)=u(1)=0 -- sin(k pi x) for integer k. A raw FFT over this
    domain (length 1) puts every such mode exactly halfway between two FFT bins (omega=k*pi
    sits at a half-integer multiple of the fundamental 2*pi/L), which is the worst possible
    spectral-leakage case (T0.9's own DoD tolerates this: "peak at 15pi +- domega/2"). Verified
    directly: reading off the nearest FFT bin for pi/15pi gives a noisy, non-monotonic signal
    (the bin either straddles the DC term or thrashes by 10x between adjacent checkpoints),
    unusable for steps_to_tolerance_per_mode. Projecting directly onto sin(k pi x) instead
    -- exact for this eigenbasis, zero leakage by construction -- gives a clean, ~monotonic
    per-mode amplitude. Used only for the T1.5 P1 gate figure/metric; T1.4's general FFT-based
    per_frequency_error is untouched and still used for xai/specerr.npz."""
    ks = np.arange(1, k_max + 1)
    basis = np.sin(np.outer(ks, x) * np.pi)  # [k_max, n]
    e = u_pred - u_exact
    coeffs = (2.0 / len(x)) * (basis @ e)
    return np.abs(coeffs)


def build_staircase_data(
    pde_name: str = "poisson",
    model_name: str = "c_mlp",
    alpha: float = 0.3,
    seeds: tuple[int, ...] = (0, 1, 2),
    overrides: dict | None = None,
    k_max: int = 20,
) -> dict:
    """Trains `model_name` on `pde_name` (with alpha overridden) for the full configured
    step count across `seeds`, then for each run's saved checkpoints (T0.19), reloads and
    computes the per-mode error trajectory against the reference solution via direct
    projection onto sin(k*pi*x), k=1..k_max (see `_sine_mode_amplitudes`) -- exact for P1's
    Dirichlet eigenbasis, unlike a leakage-prone FFT on this domain. Also writes the
    FFT-based xai/specerr.npz per run (T1.4 contract, still valid/general-purpose). Returns
    the median-over-seeds sine-mode trajectory plus its (omega=k*pi, steps, run_ids)."""
    import qapinn.models as models_pkg
    import qapinn.pdes as pdes_pkg
    import qapinn.reference as reference_pkg
    from qapinn.config import load_config
    from qapinn.train.checkpoint import load_checkpoint
    from qapinn.train.loop import train
    from qapinn.xai.spectral_error import save_spectral_error_checkpoint

    all_overrides = {"pde.params.alpha": alpha}
    if overrides:
        all_overrides.update(overrides)

    per_seed_trajectories = []
    run_ids = []
    steps_common = None
    omega = np.arange(1, k_max + 1) * math.pi

    for seed in seeds:
        cfg = load_config(pde_name, model_name, seed=seed, overrides=all_overrides)
        result = train(cfg)
        run_ids.append(result.run_id)

        pde = pdes_pkg.build(result.cfg.pde)
        n_eval = result.cfg.pde.n_eval
        eval_grid = pde.eval_grid(n_eval)
        x = eval_grid.numpy().reshape(-1)
        u_ref = reference_pkg.reference_solution(pde, eval_grid.numpy()).reshape(-1)
        dx = (1.0 / n_eval,)

        ckpt_dir = result.run_dir / "checkpoints"
        ckpt_files = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))

        steps = []
        sine_traj = []
        for ckpt_path in ckpt_files:
            step = int(ckpt_path.stem.split("_")[1])
            model = models_pkg.build(result.cfg.model, input_dim=pde.dim)
            load_checkpoint(model, ckpt_path)
            with torch.no_grad():
                u_pred = pde.apply_hard_bc(eval_grid, model(eval_grid)).numpy().reshape(-1)
            save_spectral_error_checkpoint(
                u_pred, u_ref, grid_shape=(n_eval,), dx=dx, step=step, run_dir=result.run_dir
            )
            steps.append(step)
            sine_traj.append(_sine_mode_amplitudes(u_pred, u_ref, x, k_max))

        if steps_common is None:
            steps_common = np.array(steps)
        per_seed_trajectories.append(np.stack(sine_traj, axis=0))  # [n_checkpoints, k_max]

    stacked = np.stack(per_seed_trajectories, axis=0)  # [n_seeds, n_checkpoints, k_max]
    median_traj = np.median(stacked, axis=0)  # [n_checkpoints, k_max]

    return {"median_traj": median_traj, "omega": omega, "steps": steps_common, "run_ids": run_ids}


def make_staircase_figure(
    median_traj: np.ndarray,
    omega: np.ndarray,
    steps: np.ndarray,
    freq_range: tuple[float, float] = (0.0, 60.0),
    run_ids: list[str] | None = None,
) -> None:
    """Heatmap |e_hat(omega,t)|: omega on the vertical axis, training step on the
    horizontal (log scale), log10 colour scale."""
    mask = (omega >= freq_range[0]) & (omega <= freq_range[1])
    order = np.argsort(omega[mask])
    omega_sorted = omega[mask][order]
    traj_sub = median_traj[:, mask][:, order]  # [n_checkpoints, n_freq_sub]

    steps_plot = np.where(steps <= 0, 1, steps)  # step=0 has no log; plot it at x=1
    log_traj = np.log10(np.clip(traj_sub, 1e-300, None)).T  # [n_freq_sub, n_checkpoints]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    mesh = ax.pcolormesh(steps_plot, omega_sorted, log_traj, shading="auto", cmap="viridis")
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\omega$")
    fig.colorbar(mesh, ax=ax, label=r"$\log_{10}|\hat{e}(\omega)|$")
    fig.tight_layout()

    save_figure(fig, "staircase_cmlp_p1", run_ids=run_ids)
    plt.close(fig)


if __name__ == "__main__":
    print("scripts/make_figures.py: no CLI driver yet (arrives in T5.1); import and call directly.")
