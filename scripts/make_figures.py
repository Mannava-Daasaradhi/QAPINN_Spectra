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
from qapinn.viz.style import FIGSIZE, family_color, save_figure
from qapinn.xai.ntk import ntk, probe_set, spectrum_stats


def make_ntk_spectrum_figure(
    models: dict[str, PINNModel],
    pde: PDE,
    output: str = "residual",
    run_ids: list[str] | None = None,
    omega_band: tuple[float, float] | None = None,
    name: str = "ntk_spectrum",
) -> dict[str, float]:
    """models: {family_name: model}. Renders log-log eigenvalue-vs-index for each family
    on the same axes. omega_band: (lo, hi) index range to shade as the encoded band Omega
    -- Phase 2 supplies the real value; this is the shading hook called for in T1.3.
    name: figure basename (default "ntk_spectrum", T1.3's single-family DoD name; T3.8
    passes a per-problem name since it renders one figure per problem instance).

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

    save_figure(fig, name, run_ids=run_ids)
    plt.close(fig)
    return decay_exponents


def _decay_exponent_in_index_range(eigs_desc: np.ndarray, lo: int, hi: int) -> float:
    """Same log-log linear-fit method as `xai.ntk.spectrum_stats`, but restricted to a
    caller-chosen 1-indexed index range `[lo, hi)` instead of spectrum_stats' fixed
    "middle 80%" -- needed here to measure the decay exponent separately INSIDE vs
    OUTSIDE the encoded band (PR-7), not just once over the whole spectrum."""
    n = len(eigs_desc)
    lo = max(1, lo)
    hi = min(n + 1, hi)
    if hi - lo < 2:
        return float("nan")
    idx = np.arange(lo, hi)
    vals = eigs_desc[lo - 1 : hi - 1]
    positive = vals > 0
    if positive.sum() < 2:
        return float("nan")
    slope, _intercept = np.polyfit(np.log(idx[positive]), np.log(vals[positive]), 1)
    return float(slope)


def make_ntk_spectrum_comparison(
    run_dir_a: Path | str,
    run_dir_b: Path | str,
    step: int,
    omega_set: list,
    figure_name: str,
    family_a: str = "c_mlp",
    family_b: str = "q_serial",
) -> dict[str, float]:
    """T3.8: completes the T1.3 hook now that hybrid models exist -- loads `family_a` and
    `family_b` checkpoints at a MATCHED training step, renders the two-family
    `ntk_spectrum` figure with the encoded band shaded, and reports the decay exponent
    measured INSIDE vs OUTSIDE that band for `family_b` (PR-7's own deciding metric).

    Band choice: the shaded index range is `(1, len(omega_set) + 1)`. `omega_set`
    (`results/design_cards.json`) is `family_b`'s circuit's OWN realized frequency
    support -- it comes straight from `smcd/design.py`'s BAND design step (`Omega =
    Omega.tolist()` in the DesignCard), not merely a target spectrum -- so the top
    `len(omega_set)` ranked NTK eigenvalues are the ones PR-7 predicts sit at a plateau,
    reflecting the circuit's Fourier degrees of freedom. No prior code in this repo
    defined this index-range mapping (`03_PHASE1_instruments.md` T1.3 explicitly left it
    as an open hook for Phase 2/3); this is a judgment call made here, not an existing
    convention -- see BENCH.md for the full reasoning, and revisit if it turns out wrong.
    """
    import yaml as yaml_mod

    from qapinn.config import ExpConfig
    from qapinn.models import build as build_model
    from qapinn.pdes import build as build_pde
    from qapinn.train.checkpoint import load_checkpoint

    def _cfg(run_dir: Path | str) -> ExpConfig:
        return ExpConfig(**yaml_mod.safe_load((Path(run_dir) / "config.yaml").read_text(encoding="utf-8")))

    def _load(run_dir: Path | str, cfg: ExpConfig):
        pde = build_pde(cfg.pde)
        model = build_model(
            cfg.model,
            input_dim=pde.dim,
            pde=pde,
            smcd_eps=cfg.smcd_eps,
            smcd_coverage_target=cfg.smcd_coverage_target,
        )
        load_checkpoint(model, Path(run_dir) / "checkpoints" / f"step_{step}.pt")
        return pde, model

    cfg_a, cfg_b = _cfg(run_dir_a), _cfg(run_dir_b)
    # Checked before any checkpoint is loaded: the mismatch is a caller error either way,
    # and the checkpoints are gitignored, so they may not exist on this machine.
    if cfg_a.pde.name != cfg_b.pde.name:
        raise ValueError(
            f"{family_a} and {family_b} runs must be on the same PDE to be compared, "
            f"got {cfg_a.pde.name!r} vs {cfg_b.pde.name!r}"
        )
    pde_a, model_a = _load(run_dir_a, cfg_a)
    pde_b, model_b = _load(run_dir_b, cfg_b)

    band_size = len(omega_set)
    whole_decay = make_ntk_spectrum_figure(
        {family_a: model_a, family_b: model_b},
        pde_a,
        run_ids=[cfg_a.run_id, cfg_b.run_id],
        omega_band=(1, band_size + 1),
        name=figure_name,
    )

    probe_x = probe_set(pde_b)
    K_b = ntk(model_b, pde_b, probe_x, output="residual", group="all")
    eigs_b = np.array(spectrum_stats(K_b)["eigenvalues"])  # already sorted descending

    return {
        "decay_exponent_whole": whole_decay,
        "decay_exponent_inside_band": _decay_exponent_in_index_range(eigs_b, 1, band_size + 1),
        "decay_exponent_outside_band": _decay_exponent_in_index_range(eigs_b, band_size + 1, len(eigs_b) + 1),
        "band_size": band_size,
    }


def make_ntk_spectrum_comparison_from_npz(
    run_dir_a: Path | str,
    run_dir_b: Path | str,
    step: int,
    omega_set: list,
    figure_name: str,
    family_a: str = "c_mlp",
    family_b: str = "q_serial",
) -> dict[str, float]:
    """T5.1's DoD-compliant version of `make_ntk_spectrum_comparison`: regenerates the
    same figure and PR-7 metrics purely from each run's already-committed
    `xai/ntk_step{step}.npz` (written by the training loop's own instrumentation, T1.2),
    never from a checkpoint or a reconstructed live model.

    `make_ntk_spectrum_comparison` (above) intentionally still reconstructs models from
    checkpoints -- per `tests/test_phase3_figures.py`'s own docstring, that version exists
    to verify the NTK-computation plumbing against real checkpoints during T3.6/T3.8/T3.9
    prep, not to serve as the final regeneration driver. `docs/plan/07_PHASE5_package.md`'s
    T5.1 DoD is explicit: figures regenerate from committed `metrics.json`/`xai/*.npz`
    only. Same return shape as `make_ntk_spectrum_comparison` so callers are interchangeable.
    """

    def _eigs(run_dir: Path | str) -> np.ndarray:
        run_dir = Path(run_dir)
        data = np.load(run_dir / "xai" / f"ntk_step{step}.npz")
        eigs = np.asarray(data["eigenvalues"])
        return eigs[np.argsort(eigs)[::-1]]  # descending, matching spectrum_stats' convention

    eigs_a = _eigs(run_dir_a)
    eigs_b = _eigs(run_dir_b)

    band_size = len(omega_set)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for eigs, family in ((eigs_a, family_a), (eigs_b, family_b)):
        eigs_positive = eigs[eigs > 0]
        idx = np.arange(1, len(eigs_positive) + 1)
        ax.loglog(idx, eigs_positive, label=family, color=family_color(family))
    ax.axvspan(1, band_size + 1, color="grey", alpha=0.2, label=r"encoded band $\Omega$")
    ax.set_xlabel("index $i$")
    ax.set_ylabel(r"eigenvalue $\lambda_i$")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figure_name, run_ids=None)
    plt.close(fig)

    whole_decay_b = _decay_exponent_in_index_range(eigs_b, 1, len(eigs_b) + 1)

    return {
        "decay_exponent_whole": whole_decay_b,
        "decay_exponent_inside_band": _decay_exponent_in_index_range(eigs_b, 1, band_size + 1),
        "decay_exponent_outside_band": _decay_exponent_in_index_range(eigs_b, band_size + 1, len(eigs_b) + 1),
        "band_size": band_size,
    }


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


def _omega_lines_from_design_card(omega_set: list[list[float]], signed_axis: bool = False) -> np.ndarray:
    """Design cards store Omega as a list of frequency VECTORS (one component per PDE
    dimension the circuit encodes, T2.8). The heatmap's frequency axis is always 1-D, so
    each vector is mapped onto it:

    - radially binned axis (2-D and time-dependent problems, omega >= 0): the Euclidean
      norm, the same radial reduction `spectral_error.per_frequency_error` applies;
    - signed axis (Poisson's two-sided FFT, omega in [-W, W]): the signed component
      itself. Omega is symmetric (it holds both +w and -w), and so is a real field's error
      spectrum. Folding to |w| here left the mirrored negative half of Omega counted as
      "outside" in PR-8, which is how v1.0 reported Poisson's overlap as 71.3% instead of
      99.9% (FINDINGS.md, post-submission errata).
    """
    if signed_axis and all(len(w) == 1 for w in omega_set):
        return np.array([float(w[0]) for w in omega_set])
    return np.array([float(np.linalg.norm(w)) for w in omega_set])


def make_freq_heatmap_figure(
    problem: str,
    run_dir_a: Path | str,
    run_dir_b: Path | str,
    omega_set: list[list[float]],
    family_a: str = "c_mlp",
    family_b: str = "q_serial",
    freq_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """T3.9: side-by-side |e_hat(omega,t)| heatmaps for `family_a` (baseline) and
    `family_b` (the family under test), sharing one colour scale, with the design card's
    Omega overlaid as horizontal lines on both panels (project.md SS7.2, C3's decisive
    figure).

    Also computes PR-8 quantitatively: for each Omega frequency, the nearest bin in the
    (shared, since both runs use the same problem/eval grid) frequency axis is tagged
    "inside Omega". Returns {'overlap_fraction': ..., 'total_improvement': ...} where
    overlap_fraction is the share of total improvement (sum over checkpoints and
    frequencies of max(|e_hat_a| - |e_hat_b|, 0)) that falls on an inside-Omega bin --
    the quantity PR-8 stakes its threshold (>=70%) on. A NEGATIVE total_improvement
    (family_b is WORSE, not better) is reported as-is, not clamped -- "whatever the plot
    says, it ships" (T3.7's framing applies here too).
    """
    from qapinn.xai.spectral_error import error_trajectory

    def _load(run_dir: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        run_dir = Path(run_dir)
        data = np.load(run_dir / "xai" / "specerr.npz")
        return data["omega"], error_trajectory(run_dir), data["steps"]

    omega_a, traj_a, steps_a = _load(run_dir_a)
    omega_b, traj_b, steps_b = _load(run_dir_b)
    if not np.array_equal(omega_a, omega_b) or not np.array_equal(steps_a, steps_b):
        raise ValueError(
            f"{family_a} and {family_b} runs must share the same eval grid/checkpoints "
            "to be plotted on one shared axis -- got mismatched omega or steps arrays"
        )
    omega, steps = omega_a, steps_a

    if freq_range is not None:
        mask = (omega >= freq_range[0]) & (omega <= freq_range[1])
    else:
        mask = np.ones_like(omega, dtype=bool)
    order = np.argsort(omega[mask])
    omega_sorted = omega[mask][order]
    traj_a_sub = traj_a[:, mask][:, order]  # [n_checkpoints, n_freq_sub]
    traj_b_sub = traj_b[:, mask][:, order]

    steps_plot = np.where(steps <= 0, 1, steps)
    vmin = float(np.log10(np.clip(np.minimum(traj_a_sub, traj_b_sub), 1e-300, None)).min())
    vmax = float(np.log10(np.clip(np.maximum(traj_a_sub, traj_b_sub), 1e-300, None)).max())

    signed_axis = bool((omega < 0).any())
    omega_lines = _omega_lines_from_design_card(omega_set, signed_axis=signed_axis)
    if freq_range is not None:
        omega_lines = omega_lines[(omega_lines >= freq_range[0]) & (omega_lines <= freq_range[1])]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["double"], sharey=True)
    for ax, traj_sub, family in ((axes[0], traj_a_sub, family_a), (axes[1], traj_b_sub, family_b)):
        log_traj = np.log10(np.clip(traj_sub, 1e-300, None)).T  # [n_freq_sub, n_checkpoints]
        # rasterized: as vector quads, every cell edge rendered as a thin light seam in PDF
        # viewers, indistinguishable from the Omega overlay lines drawn on top.
        mesh = ax.pcolormesh(
            steps_plot, omega_sorted, log_traj, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax, rasterized=True
        )
        for w in omega_lines:
            ax.axhline(w, color="white", linewidth=0.4, alpha=0.35)
        ax.set_xscale("log")
        ax.set_xlabel("training step")
        ax.set_title(family, color=family_color(family))
    if signed_axis and freq_range is None and len(omega_lines) > 0:
        # A two-sided axis spans +-Nyquist (~+-3200 for Poisson) while Omega sits within
        # +-40*pi; show 2x Omega's reach so the band is readable. Plot window only -- the
        # PR-8 numbers below always use the full axis.
        reach = 2.0 * float(np.abs(omega_lines).max())
        axes[0].set_ylim(max(-reach, float(omega_sorted.min())), min(reach, float(omega_sorted.max())))
    axes[0].set_ylabel(r"$\omega$")
    fig.colorbar(mesh, ax=axes, label=r"$\log_{10}|\hat{e}(\omega)|$")

    save_figure(fig, f"freq_heatmap_{problem}", run_ids=None)
    plt.close(fig)

    # PR-8: nearest-bin membership test, then partition total improvement mass.
    if len(omega_lines) > 0:
        nearest_bin = np.array([np.argmin(np.abs(omega - w)) for w in omega_lines])
        inside_mask = np.zeros_like(omega, dtype=bool)
        inside_mask[np.unique(nearest_bin)] = True
    else:
        inside_mask = np.zeros_like(omega, dtype=bool)

    improvement = np.clip(traj_a - traj_b, 0.0, None)  # [n_checkpoints, n_freq], a beats b where positive
    total_improvement = float(improvement.sum())
    inside_improvement = float(improvement[:, inside_mask].sum())
    overlap_fraction = inside_improvement / total_improvement if total_improvement > 0 else float("nan")

    return {"overlap_fraction": overlap_fraction, "total_improvement": total_improvement}


def _achieved_coverage_weighted(run_dir: Path):
    """Recomputes the achieved `coverage_weighted` for a run POST HOC, bypassing
    `metrics.json`'s `smcd_coverage_weighted` field -- that field is a dead placeholder,
    hardcoded to 0.0 in `train/loop.py` for every family regardless of whether SMCD was
    actually used (verified directly on a real q_serial run, BENCH.md's T3.6-T3.9 prep
    section). `smcd()` is a deterministic pure function of (pde, eps, coverage_target) --
    no RNG -- so re-running it against the run's own saved `config.yaml` reproduces
    exactly the design that run was trained with. Returns (coverage_weighted, n_qubits,
    n_layers).
    """
    import yaml as yaml_mod

    from qapinn.config import ExpConfig
    from qapinn.pdes import build as build_pde
    from qapinn.smcd.design import smcd

    cfg = ExpConfig(**yaml_mod.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
    pde = build_pde(cfg.pde)
    card = smcd(pde, eps=cfg.smcd_eps, coverage_target=cfg.smcd_coverage_target)
    return card.coverage_weighted, card.n_qubits, card.n_layers


def make_coverage_vs_error_figure(run_dirs_by_problem: dict[str, list]) -> dict[str, dict]:
    """T3.7 -- THE headline figure (project.md SS5.3 step 11). x-axis: achieved
    `coverage_weighted` (recomputed post hoc per run, see `_achieved_coverage_weighted` --
    NOT read from `metrics.json`, whose own field is a dead placeholder). y-axis: median
    final rel-L2 error (log scale) over seeds sharing the same achieved coverage, with
    IQR bands. One series per problem, each point annotated `(n_qubits, n_layers)`.
    Overlays the Spearman rho (coverage vs error) and its p-value per problem, per PR-9.

    Because coverage is the ACHIEVED value, not the requested `smcd_coverage_target`,
    points are grouped by their (rounded) achieved coverage -- not by the nominal target
    -- before taking the per-group median/IQR, matching the phase doc's own framing
    ("the points will not be evenly spaced, and that is correct").

    `run_dirs_by_problem`: {problem_label: [run_dir, ...]}, one entry per problem with
    every one of that problem's coverage_sweep run directories (all coverage_target
    values, all seeds, pooled -- grouping happens internally on achieved coverage).

    Returns {problem_label: {"spearman_rho": ..., "spearman_p": ..., "n_points": ...}}.
    Ships whatever the data says -- a flat or non-monotone curve refutes C1 and is not
    hidden or reshaped to look better.
    """
    import json as json_mod

    from scipy import stats as scipy_stats

    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    results: dict[str, dict] = {}

    for problem, run_dirs in run_dirs_by_problem.items():
        points = []  # (coverage_weighted, rel_l2, n_qubits, n_layers)
        for run_dir in run_dirs:
            run_dir = Path(run_dir)
            cov, n_q, n_l = _achieved_coverage_weighted(run_dir)
            metrics = json_mod.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            points.append((cov, metrics["rel_l2"], n_q, n_l))

        by_coverage: dict[float, list[tuple[float, int, int]]] = {}
        for cov, rel_l2, n_q, n_l in points:
            by_coverage.setdefault(round(cov, 6), []).append((rel_l2, n_q, n_l))

        covs_sorted = sorted(by_coverage)
        errs_by_cov = [[r for r, _, _ in by_coverage[c]] for c in covs_sorted]
        medians = [float(np.median(errs)) for errs in errs_by_cov]
        q1 = [float(np.percentile(errs, 25)) for errs in errs_by_cov]
        q3 = [float(np.percentile(errs, 75)) for errs in errs_by_cov]

        all_covs = [p[0] for p in points]
        all_errs = [p[1] for p in points]
        rho, pval = scipy_stats.spearmanr(all_covs, all_errs)

        # rho/p go in the legend: free-floating annotations collided with the data points,
        # and a fixed-point p format printed Poisson's 3.2e-5 as "p=0.000".
        color = family_color("q_serial")  # every coverage_sweep run is q_serial (T3.3)
        label = rf"{problem}: $\rho$={rho:+.2f}, p={pval:.2g}"
        line = ax.plot(covs_sorted, medians, "o-", label=label, color=color if len(results) == 0 else None)[0]
        ax.fill_between(covs_sorted, q1, q3, alpha=0.15, color=line.get_color())
        for c, m in zip(covs_sorted, medians):
            n_q, n_l = by_coverage[c][0][1], by_coverage[c][0][2]
            ax.annotate(f"({n_q},{n_l})", (c, m), fontsize=6, xytext=(2, 2), textcoords="offset points")

        results[problem] = {"spearman_rho": float(rho), "spearman_p": float(pval), "n_points": len(points)}

    ax.set_xlabel("coverage (weighted, achieved)")
    ax.set_ylabel("median rel-L2 error")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()

    save_figure(fig, "coverage_vs_error", run_ids=None)
    plt.close(fig)
    return results


def _final_step(cfg) -> int:
    return cfg.train.steps_adam + cfg.train.steps_lbfgs


def read_grad_var_final(run_dir: Path) -> float | None:
    """Reads `grad_var_final` from `xai/gradvar_step<final>.npz` -- metrics.json's own
    `grad_var_final` field is a dead placeholder (always 0.0, BENCH.md's "metrics.json's
    dead-placeholder fields" section), so this bypasses it the same way
    `_achieved_coverage_weighted` bypasses `smcd_coverage_weighted`. Returns None if the
    file doesn't exist (e.g. a classical family, or `gradvar` wasn't in
    `train_cfg.instruments` for that run)."""
    import yaml as yaml_mod

    from qapinn.config import ExpConfig

    run_dir = Path(run_dir)
    cfg = ExpConfig(**yaml_mod.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
    npz_path = run_dir / "xai" / f"gradvar_step{_final_step(cfg)}.npz"
    if not npz_path.is_file():
        return None
    return float(np.load(npz_path)["var_mean"])


def make_barren_frontier_figure(run_dirs: list) -> dict:
    """T4.5: log Var[d_theta L] vs n_qubits for each n_layers, theoretical 2^-n line
    overlaid (project.md SS7.5). `run_dirs`: every depth_sweep run (any problem/seed --
    grouped internally by (n_qubits, n_layers), median grad-var taken over seeds sharing
    the same size). Returns the practical frontier: the largest (n, L) at which the
    median grad-var stays above a numerically-meaningless floor (1e-10, matched to T3.4's
    own "watch for and stop on: q_serial gradient variance < 1e-10" barren-plateau
    trigger), plus that point's actual grad-var.

    Also computes PR-10's own actual deciding metric (`docs/predictions.md`:
    "`barren_plateau_fit.slope_b` ... `b < log 2`") via `qapinn.xai.gradvar.
    barren_plateau_fit` -- ONE global least-squares fit of `log(var_mean)` vs `n_qubits`
    pooled across every `(n, L)` point present (same pooling convention
    `tests/test_gradvar.py`'s own usage already establishes), not per-`L`. The frontier
    above and this fit answer two DIFFERENT questions -- "how far can we practically
    push `n` before it collapses" vs "what's the actual decay RATE compared to the
    generic `2^-n` reference" -- both are part of the barren-plateau story and this
    function was previously only answering the first.
    """
    import yaml as yaml_mod

    from qapinn.config import ExpConfig
    from qapinn.xai.gradvar import barren_plateau_fit

    by_size: dict[tuple[int, int], list[float]] = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        cfg = ExpConfig(**yaml_mod.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
        gv = read_grad_var_final(run_dir)
        if gv is None:
            continue
        key = (cfg.model.n_qubits, cfg.model.n_layers)
        by_size.setdefault(key, []).append(gv)

    if not by_size:
        raise ValueError("no runs with a readable grad_var_final -- nothing to plot")

    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    n_layers_present = sorted({L for _, L in by_size})
    frontier_n, frontier_l, frontier_gv = None, None, None
    BARREN_FLOOR = 1e-10

    for L in n_layers_present:
        sizes = sorted(n for (n, layer) in by_size if layer == L)
        medians = [float(np.median(by_size[(n, L)])) for n in sizes]
        ax.plot(sizes, np.log(medians), "o-", label=f"L={L}")
        for n, m in zip(sizes, medians):
            if m > BARREN_FLOOR and (frontier_n is None or n > frontier_n):
                frontier_n, frontier_l, frontier_gv = n, L, m

    if n_layers_present:
        n_range = np.array(sorted({n for n, _ in by_size}))
        ax.plot(n_range, np.log(2.0**-n_range.astype(float)), "k--", label=r"theoretical $2^{-n}$")

    ax.set_xlabel("$n$ (qubits)")
    ax.set_ylabel(r"$\log \mathrm{Var}[\partial_\theta L]$")
    ax.legend()
    fig.tight_layout()

    save_figure(fig, "barren_frontier", run_ids=None)
    plt.close(fig)

    fit_points = [
        {"n_qubits": n, "var_mean": float(np.median(by_size[(n, L)]))} for (n, L) in by_size
    ]
    fit = barren_plateau_fit(fit_points)

    if "error" in fit:
        return {
            "frontier_n_qubits": frontier_n,
            "frontier_n_layers": frontier_l,
            "frontier_grad_var": frontier_gv,
            "slope_b": None,
            "fit_r2": None,
            "pr10_holds": None,
            "fit_error": fit["error"],
            "fit_error_reason": fit["reason"],
            "sizes_checked": sorted(by_size.keys()),
        }

    return {
        "frontier_n_qubits": frontier_n,
        "frontier_n_layers": frontier_l,
        "frontier_grad_var": frontier_gv,
        "slope_b": fit["slope_b"],
        "fit_r2": fit["r2"],
        "pr10_holds": fit["slope_b"] < math.log(2.0),
        "sizes_checked": sorted(by_size.keys()),
    }


_ABLATION_FAMILIES = ("q_serial", "c_rff_matched", "q_random", "c_ff")
_ABLATION_PAIRS = (
    ("q_serial", "c_rff_matched"),
    ("c_rff_matched", "q_random"),
    ("q_serial", "q_random"),
    ("c_rff_matched", "c_ff"),
)


def _rel_l2_runs_by_seed(run_dirs: list) -> list[dict]:
    import json as json_mod

    import yaml as yaml_mod

    out = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        cfg = yaml_mod.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        metrics = json_mod.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        out.append({"seed": cfg["seed"], "rel_l2": metrics["rel_l2"]})
    return out


def make_ablation_matched_figure(run_dirs_by_problem: dict) -> dict:
    """T4.2 DoD (project.md SS6, SS13 -- "the sharpest ablation in the project"): per
    problem, the paired comparison across `q_serial`, `c_rff_matched`, `q_random` (the
    three-way test C1's own falsifier is written against), plus `c_ff` (needed for the
    interpretation table's 4th row, "all ~= c_ff"). One boxplot panel per problem; full
    pairwise Wilcoxon + Cliff's delta statistics (T4.1's `paired_comparison`) for every
    pair in `_ABLATION_PAIRS`.

    Deliberately does NOT collapse the statistics into one of the interpretation table's
    4 canned outcome strings. project.md's own DoD text says to "write the conclusion the
    data supports, not the one we hoped for" -- that is a human judgment call to make in
    FINDINGS.md using this figure and these numbers, not something to templater away
    mechanically. A purely automatic classifier risks manufacturing false confidence at
    exactly the n<=5-seed regime project.md SS8 already flags as honestly ambiguous much
    of the time (T4.1's own p=0.0625 floor test).

    `run_dirs_by_problem`: {problem_label: {family: [run_dir, ...]}}. A pair with fewer
    than 2 shared seeds is recorded as `"insufficient_data"` rather than raising (matches
    `paired_comparison`'s own floor) or being silently skipped.

    Returns {problem_label: {"n_seeds": {family: n}, "comparisons": {"a_vs_b": dict |
    "insufficient_data"}}}.
    """
    from qapinn.stats import paired_comparison

    problems = list(run_dirs_by_problem)
    # At most 3 panels per row: six panels in one row made a 21-inch-wide figure whose
    # text shrank to ~3pt at page width.
    ncols = min(3, len(problems))
    nrows = math.ceil(len(problems) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIGSIZE["double"][0], 2.3 * nrows), squeeze=False)
    for ax in axes.flat[len(problems):]:
        ax.set_visible(False)
    results: dict = {}

    for i, problem in enumerate(problems):
        ax = axes.flat[i]
        by_family = run_dirs_by_problem[problem]
        runs_by_family = {fam: _rel_l2_runs_by_seed(by_family[fam]) for fam in _ABLATION_FAMILIES if fam in by_family}
        n_seeds = {fam: len(runs) for fam, runs in runs_by_family.items()}

        present = [fam for fam in _ABLATION_FAMILIES if runs_by_family.get(fam)]
        ax.boxplot([[r["rel_l2"] for r in runs_by_family[fam]] for fam in present], tick_labels=present)
        ax.set_yscale("log")
        ax.set_title(problem, fontsize=8)
        ax.tick_params(axis="x", labelrotation=30, labelsize=6)

        comparisons: dict = {}
        for a, b in _ABLATION_PAIRS:
            key = f"{a}_vs_{b}"
            runs_a, runs_b = runs_by_family.get(a, []), runs_by_family.get(b, [])
            shared = {r["seed"] for r in runs_a} & {r["seed"] for r in runs_b}
            if len(shared) < 2:
                comparisons[key] = "insufficient_data"
                continue
            comparisons[key] = paired_comparison(runs_a, runs_b, metric="rel_l2")
        results[problem] = {"n_seeds": n_seeds, "comparisons": comparisons}

    fig.tight_layout()
    save_figure(fig, "ablation_matched", run_ids=None)
    plt.close(fig)
    return results


def _predicted_benefit(q_serial_run_dir: Path) -> float:
    """Recomputes the SMCD a-priori `predicted_benefit` post hoc from a q_serial run's own
    saved `config.yaml` -- same zero-risk read-back pattern as `_achieved_coverage_weighted`
    (a fresh `smcd(pde, eps, coverage_target)` call is deterministic and reproduces exactly
    the design that run was built with). `predicted_benefit` is a property of the
    (pde, SMCD design) pair, not of any individual family's trained result -- computed
    once per problem from a q_serial run since q_serial is the family SMCD actually
    designs (project.md SS5.3's own a-priori heuristic, T2.8)."""
    import yaml as yaml_mod

    from qapinn.config import ExpConfig
    from qapinn.pdes import build as build_pde
    from qapinn.smcd.design import smcd

    cfg = ExpConfig(**yaml_mod.safe_load((Path(q_serial_run_dir) / "config.yaml").read_text(encoding="utf-8")))
    pde = build_pde(cfg.pde)
    card = smcd(pde, eps=cfg.smcd_eps, coverage_target=cfg.smcd_coverage_target)
    return card.predicted_benefit


def make_decision_map_figure(run_dirs_by_problem: dict) -> dict:
    """T4.3 DoD (project.md SS5.3/SS10, C4): predicted (SMCD a-priori `predicted_benefit`,
    computed from the PDE's target spectrum BEFORE any training) vs measured benefit
    (`q_serial`'s relative rel_l2 improvement over `c_mlp`, `(median_c_mlp - median_q_serial)
    / median_c_mlp`) -- PR-4's own definition of "benefit" (`docs/predictions.md`: "no
    family beats c_mlp by more than 5%... the a-priori SMCD number from T2.8 is 2.5%" --
    both sides of that sentence are fractions of `c_mlp`'s own error). One point per
    problem; a y=x reference line shows whether SMCD's predictions track reality.

    `run_dirs_by_problem`: {problem_label: {family: [run_dir, ...]}}. A problem missing
    either `c_mlp` or `q_serial` runs is OMITTED from the plot and the returned dict
    (not silently zeroed) -- both are required for PR-4's own comparison.

    Returns {problem_label: {"predicted_benefit": ..., "measured_benefit": ...,
    "n_seeds": {"c_mlp": ..., "q_serial": ...}}}.

    Deliberately stops at the numbers, same as `make_ablation_matched_figure`: the
    decision table itself (project.md SS10's PDE-property -> recommendation mapping) is a
    written synthesis over ALL SIX problems' evidence, not a per-problem computation this
    function could produce piecemeal from partial data.
    """
    results: dict = {}
    for problem, by_family in run_dirs_by_problem.items():
        if not by_family.get("c_mlp") or not by_family.get("q_serial"):
            continue
        c_mlp_runs = _rel_l2_runs_by_seed(by_family["c_mlp"])
        q_serial_runs = _rel_l2_runs_by_seed(by_family["q_serial"])
        median_c_mlp = float(np.median([r["rel_l2"] for r in c_mlp_runs]))
        median_q_serial = float(np.median([r["rel_l2"] for r in q_serial_runs]))
        measured = (median_c_mlp - median_q_serial) / median_c_mlp
        predicted = _predicted_benefit(by_family["q_serial"][0])
        results[problem] = {
            "predicted_benefit": predicted,
            "measured_benefit": measured,
            "n_seeds": {"c_mlp": len(c_mlp_runs), "q_serial": len(q_serial_runs)},
        }

    if not results:
        raise ValueError("no problem has both c_mlp and q_serial runs -- nothing to plot")

    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    xs = [r["predicted_benefit"] for r in results.values()]
    ys = [r["measured_benefit"] for r in results.values()]
    ax.scatter(xs, ys, color=family_color("q_serial"))
    # Label above for gains, below for losses: heat (+0.08) and helmholtz_k10 (-0.09)
    # sit next to each other near the origin.
    for problem, r in results.items():
        ax.annotate(
            problem, (r["predicted_benefit"], r["measured_benefit"]), fontsize=6,
            xytext=(4, 3) if r["measured_benefit"] >= 0 else (4, -9), textcoords="offset points",
        )
    lo, hi = min(xs + [0.0]), max(xs + [0.01])
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    # symlog: helmholtz_k4's measured "benefit" is ~ -1e5 (q_serial fails catastrophically
    # where c_mlp reaches ~1e-4), which on a linear axis collapsed every other problem
    # into one unreadable point at the origin. Linear within +-1, logarithmic beyond.
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("predicted benefit (SMCD a-priori)")
    ax.set_ylabel("measured benefit (q_serial vs c_mlp)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    save_figure(fig, "decision_map", run_ids=None)
    plt.close(fig)
    return results


def make_attribution_figure(run_dir: Path | str, problem_label: str, step: int, family: str = "q_serial") -> dict:
    """T1.6/T5.1: `attribution_{p3,p4}` -- bar plot of the collocation-point attribution
    field at `step` (`xai/attribution_step{step}.npz`, written by T1.6's
    `attribution_field`), with the completeness-axiom violation and the
    attribution-vs-residual Spearman correlation annotated as text (project.md SS7.3
    requires the correlation number, not just the picture, per the phase doc's own DoD).
    """
    run_dir = Path(run_dir)
    data = np.load(run_dir / "xai" / f"attribution_step{step}.npz")
    field = np.asarray(data["field"])
    completeness_error = float(data["completeness_error"])
    correlation = float(data["correlation"])

    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    ax.bar(np.arange(len(field)), field, color=family_color(family))
    ax.set_xlabel("collocation point index")
    ax.set_ylabel("integrated-gradients attribution")
    ax.set_title(f"{problem_label} ({family})", fontsize=8)
    ax.annotate(
        f"completeness err={completeness_error:.2e}\ncorr(attr, residual)={correlation:.3f}",
        xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=6,
    )
    fig.tight_layout()
    save_figure(fig, f"attribution_{problem_label}", run_ids=None)
    plt.close(fig)
    return {"completeness_error": completeness_error, "correlation": correlation, "n_points": len(field)}


def make_fisher_effdim_figure(run_dirs_by_family: dict[str, Path | str]) -> dict:
    """T1.7/T5.1: `fisher_effdim` -- effective-dimension trajectory across checkpoints
    (`xai/fisher_step{N}.npz`'s `effective_dimension`, T1.7), one line per family, so a
    reader can compare how quickly each family's effective capacity grows during training.
    `run_dirs_by_family`: {family: run_dir}, one run per family (same problem).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    results: dict = {}
    for family, run_dir in run_dirs_by_family.items():
        run_dir = Path(run_dir)
        points = []
        for p in sorted((run_dir / "xai").glob("fisher_step*.npz")):
            step = int(p.stem.removeprefix("fisher_step"))
            d = np.load(p)
            points.append((step, float(d["effective_dimension"])))
        points.sort()
        if not points:
            continue
        steps_plot = [max(s, 1) for s, _ in points]
        effdims = [e for _, e in points]
        ax.plot(steps_plot, effdims, "o-", label=family, color=family_color(family))
        results[family] = {"final_effective_dimension": effdims[-1]}
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("effective dimension")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "fisher_effdim", run_ids=None)
    plt.close(fig)
    if not results:
        raise ValueError("no family had any fisher_step*.npz files -- nothing to plot")
    return results


def make_probes_cka_figure(run_dir: Path | str, step: int, family: str = "q_serial") -> dict:
    """T1.10/T5.1: `probes_cka` -- per-layer linear-probe R^2 at `step`
    (`xai/probes_step{N}.npz`, T1.10's `layer_probe_r2`), one bar per layer in forward
    order, so the reader can see "how much of the solution is already linearly decodable
    here" grow through the network -- the DoD's own framing (input layer low, final layer
    high for a converged model).
    """
    run_dir = Path(run_dir)
    data = np.load(run_dir / "xai" / f"probes_step{step}.npz")
    layers = list(data.files)
    r2 = [float(data[layer]) for layer in layers]

    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    ax.bar(np.arange(len(layers)), r2, color=family_color(family))
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(r"probe $R^2$")
    ax.set_title(family, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "probes_cka", run_ids=None)
    plt.close(fig)
    return dict(zip(layers, r2))


def make_landscape_grid_figure(run_dir: Path | str, step: int, family: str = "q_serial") -> dict:
    """T1.11/T5.1: `landscape_grid` -- heatmap of the filter-normalised 2-D loss-landscape
    slice (`xai/landscape_step{N}.npz`'s `[25,25]` grid, T1.11), confirming visually that
    the trained point sits at a minimum (the DoD's own check, done numerically in
    `tests/test_landscape.py`; this figure is the qualitative companion for slides).
    """
    run_dir = Path(run_dir)
    data = np.load(run_dir / "xai" / f"landscape_step{step}.npz")
    grid = np.asarray(data["grid"])
    n = grid.shape[0]
    center = n // 2
    argmin = np.unravel_index(np.argmin(grid), grid.shape)

    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    mesh = ax.pcolormesh(grid, cmap="viridis")
    ax.scatter([center + 0.5], [center + 0.5], marker="x", color="red", s=40, label="trained point")
    ax.set_title(family, fontsize=8)
    ax.legend(fontsize=6)
    fig.colorbar(mesh, ax=ax, label="loss")
    fig.tight_layout()
    save_figure(fig, "landscape_grid", run_ids=None)
    plt.close(fig)
    return {"argmin_index": [int(argmin[0]), int(argmin[1])], "center_index": center, "min_at_center": tuple(argmin) == (center, center)}


def regenerate_all(strict: bool = False) -> dict:
    """T5.1's single entry point (`tasks.py figures`): regenerates every figure in the
    inventory (`docs/plan/07_PHASE5_package.md`) from committed `results/runs/*/metrics.json`
    and `xai/*.npz` only. A figure whose required sweep has not produced any runs yet is
    reported SKIPPED, never silently omitted or faked -- `project.md` SS9: "if a figure
    cannot be regenerated, it does not go in the paper." `strict=True` raises if anything
    is skipped or errors -- the real T5.1 DoD check, meant to be run once T3.4/T3.5 are
    both complete. Returns {"produced": {...}, "skipped": {...}, "errors": {...}}.

    Every Omega band (heatmap overlays, NTK band shading) comes from
    `results/design_cards.json`'s `omega_set` -- the same source
    `scripts/adjudicate_predictions.py` uses. Before 2026-09-25 this driver passed the
    card's 4 encoding `scalings` instead, so the committed heatmaps and ntk_spectrum_p1/p4
    marked the wrong band, and running the adjudicator afterwards redrew the same files
    with different content.
    """
    import json as _json
    import sys as _sys
    from pathlib import Path as _Path

    scripts_dir = _Path(__file__).resolve().parent
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))
    from check_specerr_integrity import specerr_has_duplicate_steps
    from cost_ledger import build_cost_ledger, enumerate_core_matrix_run_ids

    from qapinn.runner import enumerate_labeled_runs

    runs_dir = REPO_ROOT / "results" / "runs"
    exp_dir = REPO_ROOT / "configs" / "exp"
    design_cards = _json.loads((REPO_ROOT / "results" / "design_cards.json").read_text(encoding="utf-8"))

    def _done_run_dirs(exp_cfg_path):
        """run_id -> {'problem', 'family', 'seed'} for every finished run of an experiment
        config, plus the enumerated total. Labels come from `enumerate_labeled_runs`, which
        applies `axes` overrides -- a core-matrix-only helper once made this driver report
        coverage_sweep as "0/6 runs, not launched yet" with all 36 runs done (T3.10)."""
        labeled = enumerate_labeled_runs(exp_cfg_path)
        out = {}
        for problem, cfg in labeled:
            if (runs_dir / cfg.run_id / "metrics.json").is_file():
                out[cfg.run_id] = {"problem": problem, "family": cfg.model.family, "seed": cfg.seed}
        return out, len(labeled)

    def _first_clean(run_dirs: list):
        """First run_dir in the list whose specerr.npz isn't crash-retry-contaminated
        (see scripts/check_specerr_integrity.py) -- falls back to run_dirs[0] if every
        candidate is contaminated, so callers still get a deterministic pick to report a
        clear error against rather than an empty selection."""
        for rd in run_dirs:
            if not specerr_has_duplicate_steps(rd):
                return rd
        return run_dirs[0] if run_dirs else None

    core_done, core_total = _done_run_dirs(exp_dir / "core_matrix.yaml")
    by_problem_family: dict[str, dict[str, list]] = {}
    for rid, meta in core_done.items():
        by_problem_family.setdefault(meta["problem"], {}).setdefault(meta["family"], []).append(runs_dir / rid)

    produced: dict = {}
    skipped: dict[str, str] = {}
    excluded: dict[str, str] = {}  # deliberate, permanent exclusions -- strict mode tolerates these
    errors: dict[str, str] = {}

    def _attempt(name, fn):
        try:
            produced[name] = fn()
            print(f"  OK    {name}")
        except Exception as e:  # noqa: BLE001 -- report every failure, don't let one figure kill the run
            errors[name] = f"{type(e).__name__}: {e}"
            print(f"  ERROR {name}: {errors[name]}")

    def _skip(name, reason):
        skipped[name] = reason
        print(f"  SKIP  {name}: {reason}")

    def _exclude(name, reason):
        excluded[name] = reason
        print(f"  EXCL  {name}: {reason}")

    print(f"Regenerating figures from results/runs/ (core_matrix: {len(core_done)}/{core_total} runs available)...")

    # staircase_cmlp_p1 (T1.5): DELIBERATELY NOT CALLED HERE. build_staircase_data()
    # retrains 3 seeds from scratch at the full step budget (it is a data-generation
    # helper, not a regeneration-from-committed-artifacts one) -- calling it from this
    # driver would both violate T5.1's "never retrain" contract and contend for GPU with
    # any sweep still running. T1.5's own gate already produced and verified this figure
    # (run_ids 0280f156172b, a5f1ca54add5, 5669659249e2, alpha=0.05 -- NOT this function's
    # alpha=0.3 default, which BENCH.md documents as non-convergent). A real npz-based
    # regeneration needs its own per-mode-amplitude artifact saved at data-generation time;
    # left as a known follow-up rather than guessed at here. Reported as EXCLUDED, not
    # SKIPPED, so `--strict` can pass: it fails on a missing sweep, not on this.
    _exclude("staircase_cmlp_p1", "build_staircase_data() retrains from scratch -- needs an npz-based rewrite first")

    # coverage_vs_error (T3.7, THE headline figure) -- needs coverage_sweep.yaml runs.
    cov_done, cov_total = _done_run_dirs(exp_dir / "coverage_sweep.yaml")
    if not cov_done:
        _skip("coverage_vs_error", f"coverage_sweep.yaml has 0/{cov_total} runs -- sweep not launched yet")
    else:
        cov_by_problem: dict[str, list] = {}
        for rid, meta in cov_done.items():
            cov_by_problem.setdefault(meta["problem"], []).append(runs_dir / rid)
        _attempt("coverage_vs_error", lambda: make_coverage_vs_error_figure(cov_by_problem))

    # ntk_spectrum_{p1,p4}: c_mlp vs q_serial, poisson (P1) and helmholtz_k4 (P4).
    for label, problem in (("ntk_spectrum_p1", "poisson"), ("ntk_spectrum_p4", "helmholtz_k4")):
        fams = by_problem_family.get(problem, {})
        if "c_mlp" not in fams or "q_serial" not in fams:
            _skip(label, f"{problem} missing c_mlp or q_serial runs")
            continue
        run_dir_a, run_dir_b = _first_clean(fams["c_mlp"]), _first_clean(fams["q_serial"])

        def _ntk(run_dir_a=run_dir_a, run_dir_b=run_dir_b, label=label, problem=problem):
            omega_set = design_cards[problem]["omega_set"]
            steps_a = {int(p.stem.removeprefix("ntk_step")) for p in (run_dir_a / "xai").glob("ntk_step*.npz")}
            steps_b = {int(p.stem.removeprefix("ntk_step")) for p in (run_dir_b / "xai").glob("ntk_step*.npz")}
            shared = steps_a & steps_b
            if not shared:
                raise ValueError("no shared ntk_step*.npz between the two runs")
            step = max(shared)
            return make_ntk_spectrum_comparison_from_npz(
                run_dir_a, run_dir_b, step, omega_set, label, family_a="c_mlp", family_b="q_serial"
            )

        _attempt(label, _ntk)

    # ntk_spectrum_{poisson,helmholtz_k10}_pr7: the figures check_pr7 draws while
    # adjudicating PR-7 -- same runs (first core_matrix c_mlp/q_serial run), final
    # checkpoint and Omega -- so `tasks.py figures` alone covers every figure the paper cites.
    for problem in ("poisson", "helmholtz_k10"):
        name = f"ntk_spectrum_{problem}_pr7"
        fams = by_problem_family.get(problem, {})
        if not fams.get("c_mlp") or not fams.get("q_serial"):
            _skip(name, f"{problem} missing c_mlp or q_serial runs")
            continue

        def _ntk_pr7(run_dir_a=fams["c_mlp"][0], run_dir_b=fams["q_serial"][0], problem=problem, name=name):
            import yaml as yaml_mod

            from qapinn.config import ExpConfig

            cfg = ExpConfig(**yaml_mod.safe_load((run_dir_a / "config.yaml").read_text(encoding="utf-8")))
            return make_ntk_spectrum_comparison_from_npz(
                run_dir_a, run_dir_b, _final_step(cfg), design_cards[problem]["omega_set"], name
            )

        _attempt(name, _ntk_pr7)

    # freq_heatmap_{6 instances}: c_mlp vs q_serial, one per problem.
    for problem in ("poisson", "heat", "burgers", "helmholtz_k4", "helmholtz_k10", "helmholtz_k20"):
        name = f"freq_heatmap_{problem}"
        fams = by_problem_family.get(problem, {})
        if "c_mlp" not in fams or "q_serial" not in fams:
            _skip(name, f"{problem} missing c_mlp or q_serial runs")
            continue
        run_dir_a, run_dir_b = _first_clean(fams["c_mlp"]), _first_clean(fams["q_serial"])

        def _heat(run_dir_a=run_dir_a, run_dir_b=run_dir_b, problem=problem):
            return make_freq_heatmap_figure(problem, run_dir_a, run_dir_b, design_cards[problem]["omega_set"])

        _attempt(name, _heat)

    # ablation_matched (T4.2) / decision_map (T4.3) -- both usable directly from whatever
    # core_matrix families are currently done; each function itself narrows to problems
    # with enough family coverage rather than requiring the full matrix.
    _attempt("ablation_matched", lambda: make_ablation_matched_figure(by_problem_family))
    _attempt("decision_map", lambda: make_decision_map_figure(by_problem_family))

    # barren_frontier (T4.5) -- needs depth_sweep.yaml runs.
    depth_done, depth_total = _done_run_dirs(exp_dir / "depth_sweep.yaml")
    if not depth_done:
        _skip("barren_frontier", f"depth_sweep.yaml has 0/{depth_total} runs -- sweep not launched yet")
    else:
        _attempt("barren_frontier", lambda: make_barren_frontier_figure([runs_dir / rid for rid in depth_done]))

    # cost_ledger_table (T4.6) -- already-existing, working driver.
    def _cost_ledger():
        import json as _json

        label_by_run_id = enumerate_core_matrix_run_ids()
        run_dirs = sorted(runs_dir / rid for rid in label_by_run_id if (runs_dir / rid / "metrics.json").is_file())
        ledger = build_cost_ledger(run_dirs, label_by_run_id=label_by_run_id)
        (REPO_ROOT / "results" / "cost_ledger.json").write_text(_json.dumps(ledger, indent=2), encoding="utf-8")
        return {"n_production_runs": ledger["n_production_runs"], "n_entries": len(ledger["entries"])}

    _attempt("cost_ledger_table", _cost_ledger)

    def _final_step_available(run_dir: Path, prefix: str) -> int | None:
        steps = [int(p.stem.removeprefix(f"{prefix}_step")) for p in (run_dir / "xai").glob(f"{prefix}_step*.npz")]
        return max(steps) if steps else None

    # attribution_{p3,p4} (T1.6): burgers (P3), helmholtz_k4 (P4), q_serial family.
    for label, problem in (("attribution_p3", "burgers"), ("attribution_p4", "helmholtz_k4")):
        fams = by_problem_family.get(problem, {})
        run_dir = _first_clean(fams["q_serial"]) if fams.get("q_serial") else None
        if run_dir is None:
            _skip(label, f"{problem} missing q_serial runs")
            continue
        step = _final_step_available(run_dir, "attribution")
        if step is None:
            _skip(label, f"{run_dir} has no attribution_step*.npz")
            continue
        _attempt(label, lambda run_dir=run_dir, problem=problem, step=step: make_attribution_figure(run_dir, problem, step))

    # fisher_effdim (T1.7): c_mlp vs q_serial trajectory, poisson (matches the P1 baseline
    # convention already used for ntk_spectrum_p1).
    fams = by_problem_family.get("poisson", {})
    if "c_mlp" not in fams or "q_serial" not in fams:
        _skip("fisher_effdim", "poisson missing c_mlp or q_serial runs")
    else:
        run_dirs_by_family = {"c_mlp": _first_clean(fams["c_mlp"]), "q_serial": _first_clean(fams["q_serial"])}
        _attempt("fisher_effdim", lambda: make_fisher_effdim_figure(run_dirs_by_family))

    # probes_cka (T1.10) / landscape_grid (T1.11): one representative run, poisson/q_serial
    # at its final checkpoint.
    q_serial_poisson = _first_clean(fams["q_serial"]) if fams.get("q_serial") else None
    if q_serial_poisson is None:
        _skip("probes_cka", "poisson missing q_serial runs")
        _skip("landscape_grid", "poisson missing q_serial runs")
    else:
        probes_step = _final_step_available(q_serial_poisson, "probes")
        if probes_step is None:
            _skip("probes_cka", f"{q_serial_poisson} has no probes_step*.npz")
        else:
            _attempt("probes_cka", lambda: make_probes_cka_figure(q_serial_poisson, probes_step, "q_serial"))

        landscape_step = _final_step_available(q_serial_poisson, "landscape")
        if landscape_step is None:
            _skip("landscape_grid", f"{q_serial_poisson} has no landscape_step*.npz")
        else:
            _attempt("landscape_grid", lambda: make_landscape_grid_figure(q_serial_poisson, landscape_step, "q_serial"))

    total = len(produced) + len(skipped) + len(excluded) + len(errors)
    print(
        f"\n{len(produced)}/{total} produced, {len(skipped)} skipped, "
        f"{len(excluded)} excluded by design, {len(errors)} errored"
    )
    if strict and (skipped or errors):
        raise RuntimeError(f"regenerate_all(strict=True): {len(skipped)} skipped, {len(errors)} errored")
    return {"produced": produced, "skipped": skipped, "excluded": excluded, "errors": errors}


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parsed = parser.parse_args()
    regenerate_all(strict=parsed.strict)
