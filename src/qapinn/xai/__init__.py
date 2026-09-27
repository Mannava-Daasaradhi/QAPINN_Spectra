"""XAI instrument registry wired into the training loop (T1.12, project.md Section 7).

`run_instruments` is called at every checkpoint in `cfg.train.checkpoints`; it writes one
`xai/<name>_step<N>.npz` per instrument, except 'specerr' which (per T1.4's own
established contract) accumulates every checkpoint into a single `xai/specerr.npz` rather
than one file per step. Expensive instruments (gradvar, landscape) only actually run at the
FINAL checkpoint -- which instruments count as "expensive" is a config default
(`cfg.train.instruments_final_only`), not a hardcoded set here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import qapinn.reference as reference_pkg
from qapinn.config import ExpConfig
from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE
from qapinn.xai.attribution import (
    attribution_residual_correlation,
    completeness_error,
    integrated_gradients,
)
from qapinn.xai.drift import encoder_drift
from qapinn.xai.fisher import effective_dimension, empirical_fisher
from qapinn.xai.gradvar import gradient_variance
from qapinn.xai.landscape import loss_landscape_slice
from qapinn.xai.ntk import block_mass, probe_set, save_ntk_report
from qapinn.xai.probes import collect_layer_outputs, layer_probe_r2
from qapinn.xai.spectral_error import save_spectral_error_checkpoint

# Per-checkpoint instrumentation budget -- deliberately smaller than each instrument's own
# "one-off report" defaults (e.g. T1.2's probe_set default of 512, T1.6's attribution_field
# default grid_n=128), since these run at EVERY checkpoint, not once.
_PROBE_SIZE = 128
_ATTRIBUTION_GRID_N = 32
_ATTRIBUTION_N_STEPS = 16
_LANDSCAPE_GRID_N = 25
_LANDSCAPE_BATCH = 128
_GRADVAR_N_SAMPLES = 20
_PROBES_GRID_N = 32

# 01_CONVENTIONS.md SS10: --smoke reduces steps/POINTS, not just steps -- these budgets are
# themselves "points" (probe/grid sizes) and were left at their full-run values regardless
# of smoke=True until T2.17, which made NTK/Fisher's parameter-shift Jacobians (quadratic-
# ish in probe size x quantum-param count) the dominant fixed per-checkpoint cost across
# the 42-combo smoke matrix even after steps_adam/steps_lbfgs were already cut.
_PROBE_SIZE_SMOKE = 16
_ATTRIBUTION_GRID_N_SMOKE = 8
_ATTRIBUTION_N_STEPS_SMOKE = 4
_LANDSCAPE_GRID_N_SMOKE = 9
_LANDSCAPE_BATCH_SMOKE = 32
_GRADVAR_N_SAMPLES_SMOKE = 5
_PROBES_GRID_N_SMOKE = 8

_NTK_K0_CACHE: dict[str, torch.Tensor] = {}


def _is_final_checkpoint(cfg: ExpConfig, step: int) -> bool:
    total_steps = cfg.train.steps_adam + cfg.train.steps_lbfgs
    return step >= total_steps


def _device(model: PINNModel) -> torch.device:
    return next(model.parameters()).device


def _run_ntk(model: PINNModel, pde: PDE, cfg: ExpConfig, step: int, run_dir, smoke: bool = False) -> None:
    probe_x = probe_set(pde, n_probe=_PROBE_SIZE_SMOKE if smoke else _PROBE_SIZE).to(_device(model))
    key = str(run_dir)
    K_0 = _NTK_K0_CACHE.get(key)
    result = save_ntk_report(model, pde, probe_x, step, run_dir, K_0=K_0)
    if key not in _NTK_K0_CACHE:
        _NTK_K0_CACHE[key] = result["K"].detach().clone()


_BLOCK_MASS_PROBE_SEED = 0  # fixed, NOT step-seeded (unlike _run_landscape): block_mass
# is an NTK-family instrument (project.md SS7.1's per-loss-block eigenvalue mass), and
# T1.2's own established convention for this family is a probe set held FIXED across
# checkpoints ("the SAME probe set is used for every family/checkpoint/PDE-instance
# comparison" -- see probe_set()'s docstring) so that a change in trace_rr/trace_bb across
# steps reflects the model's evolving NTK, not a different sample of probe points.


def _run_block_mass(model: PINNModel, pde: PDE, cfg: ExpConfig, step: int, run_dir, smoke: bool = False) -> None:
    """T3.6 (D6): per-loss-block NTK eigenvalue mass (trace_rr, trace_bb, trace_rb), only
    meaningful under bc_mode='soft' -- see block_mass()'s own docstring. Uses the SAME
    sample_collocation/sample_boundary_only calls train/loop.py's own soft-BC loss uses
    (pde.sample_boundary_only, not sample_boundary -- matches the boundary-only points the
    soft BC loss term is actually computed against, T2.x's own choice, not a new one made
    here)."""
    n = _PROBE_SIZE_SMOKE if smoke else _PROBE_SIZE
    device = _device(model)
    gen = torch.Generator().manual_seed(_BLOCK_MASS_PROBE_SEED)
    probe_r = pde.sample_collocation(n, gen).to(device)
    probe_b = pde.sample_boundary_only(n, gen).to(device)
    result = block_mass(model, pde, probe_r, probe_b)

    out_dir = Path(run_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"block_mass_step{step}.npz",
        trace_rr=result["trace_rr"],
        trace_bb=result["trace_bb"],
        trace_rb=result["trace_rb"] if result["trace_rb"] is not None else np.nan,
    )


def _run_specerr(model: PINNModel, pde: PDE, cfg: ExpConfig, step: int, run_dir, smoke: bool = False) -> None:
    n_eval = cfg.pde.n_eval
    eval_grid = pde.eval_grid(n_eval).to(_device(model))
    with torch.no_grad():
        if cfg.train.bc_mode.startswith("hard"):
            u_pred = pde.apply_hard_bc(eval_grid, model(eval_grid))
        else:
            u_pred = model(eval_grid)
    u_pred_np = u_pred.detach().cpu().numpy().reshape(-1)
    u_ref = reference_pkg.reference_solution(pde, eval_grid.detach().cpu().numpy()).reshape(-1)

    grid_shape = (n_eval,) * pde.dim
    dx = tuple((hi - lo) / n_eval for lo, hi in pde.domain.bounds)
    save_spectral_error_checkpoint(u_pred_np, u_ref, grid_shape=grid_shape, dx=dx, step=step, run_dir=run_dir)


def _run_attribution(model: PINNModel, pde: PDE, cfg: ExpConfig, step: int, run_dir, smoke: bool = False) -> None:
    grid_n = _ATTRIBUTION_GRID_N_SMOKE if smoke else _ATTRIBUTION_GRID_N
    n_steps = _ATTRIBUTION_N_STEPS_SMOKE if smoke else _ATTRIBUTION_N_STEPS
    grid = pde.eval_grid(grid_n).to(_device(model))
    attrs = integrated_gradients(model, pde, grid, n_steps=n_steps, target="residual")
    field = attrs.detach().abs().sum(dim=-1).cpu().numpy()

    probe_n = min(32, grid.shape[0])
    completeness = completeness_error(model, pde, grid[:probe_n], attrs[:probe_n], target="residual")

    grid_grad = grid.clone().requires_grad_(True)
    u_full = pde.apply_hard_bc(grid_grad, model(grid_grad))
    residual_field = pde.residual(grid_grad, u_full).detach().abs().cpu().numpy().reshape(-1)
    correlation = attribution_residual_correlation(field, residual_field)

    out_dir = Path(run_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"attribution_step{step}.npz",
        field=field,
        completeness_error=completeness,
        correlation=correlation,
    )


def _run_fisher(model: PINNModel, pde: PDE, cfg: ExpConfig, step: int, run_dir, smoke: bool = False) -> None:
    probe_size = _PROBE_SIZE_SMOKE if smoke else _PROBE_SIZE
    x = pde.sample_collocation(probe_size, torch.Generator().manual_seed(0)).to(_device(model))
    F = empirical_fisher(model, pde, x, group="all", output="residual")
    F_np = F.detach().cpu().numpy()
    eigs = np.linalg.eigvalsh(F_np) if F_np.ndim == 2 else F_np

    d_eff = effective_dimension(eigs, n_data=max(3, cfg.pde.n_collocation))

    out_dir = Path(run_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"fisher_step{step}.npz", eigenvalues=eigs, effective_dimension=d_eff)


def _run_drift(model: PINNModel, pde: PDE, cfg: ExpConfig, step: int, run_dir, smoke: bool = False) -> None:
    report = encoder_drift(model)
    if not report:
        return  # c_mlp has no frequency structure at all -- nothing to report (xai/drift.py)

    out_dir = Path(run_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"drift_step{step}.npz",
        frobenius=report["frobenius"],
        realised_omega=np.asarray(report["realised_omega"]),
        coverage_now=report["coverage_now"],
    )


def _run_gradvar(model: PINNModel, pde: PDE, cfg: ExpConfig, step: int, run_dir, smoke: bool = False) -> None:
    n_samples = _GRADVAR_N_SAMPLES_SMOKE if smoke else _GRADVAR_N_SAMPLES
    gv = gradient_variance(model, pde, n_samples=n_samples, group="quantum")

    out_dir = Path(run_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"gradvar_step{step}.npz",
        var_mean=gv["var_mean"],
        var_per_param=np.asarray(gv["var_per_param"]),
        n_qubits=gv["n_qubits"] if gv["n_qubits"] is not None else -1,
        n_layers=gv["n_layers"] if gv["n_layers"] is not None else -1,
    )


def _run_probes(model: PINNModel, pde: PDE, cfg: ExpConfig, step: int, run_dir, smoke: bool = False) -> None:
    grid = pde.eval_grid(_PROBES_GRID_N_SMOKE if smoke else _PROBES_GRID_N).to(_device(model))
    layer_outputs = collect_layer_outputs(model, grid)
    r2 = layer_probe_r2(model, pde, grid, layer_outputs)

    out_dir = Path(run_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"probes_step{step}.npz", **r2)


def _run_landscape(model: PINNModel, pde: PDE, cfg: ExpConfig, step: int, run_dir, smoke: bool = False) -> None:
    device = _device(model)
    batch_size = _LANDSCAPE_BATCH_SMOKE if smoke else _LANDSCAPE_BATCH
    grid_n = _LANDSCAPE_GRID_N_SMOKE if smoke else _LANDSCAPE_GRID_N
    gen = torch.Generator().manual_seed(step)
    batch = {"x_r": pde.sample_collocation(batch_size, gen).to(device)}
    landscape_gen = torch.Generator(device=device).manual_seed(step)
    grid = loss_landscape_slice(model, pde, batch, cfg.train, grid_n=grid_n, span=1.0, gen=landscape_gen)

    out_dir = Path(run_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"landscape_step{step}.npz", grid=grid)


INSTRUMENTS = {
    "ntk": _run_ntk,
    "specerr": _run_specerr,
    "attribution": _run_attribution,
    "fisher": _run_fisher,
    "drift": _run_drift,
    "gradvar": _run_gradvar,
    "probes": _run_probes,
    "landscape": _run_landscape,
    "block_mass": _run_block_mass,
}


def run_instruments(
    model: PINNModel,
    pde: PDE,
    cfg: ExpConfig,
    step: int,
    out_dir,
    which: tuple[str, ...] | None = None,
    smoke: bool = False,
) -> None:
    """Runs each named instrument in `which` (default `cfg.train.instruments`) at this
    checkpoint. `out_dir` is the RUN directory (each instrument writes into its own
    `out_dir/xai/<name>_step<N>.npz`, except 'specerr'; see module docstring). Instruments
    listed in `cfg.train.instruments_final_only` only actually run when this is the LAST
    checkpoint of the run (`step >= steps_adam + steps_lbfgs`). `smoke=True` shrinks each
    instrument's own probe/grid size (01_CONVENTIONS.md SS10: smoke reduces steps/points);
    default False preserves every existing caller's exact behaviour unchanged."""
    if which is None:
        which = cfg.train.instruments

    is_final = _is_final_checkpoint(cfg, step)

    for name in which:
        if name not in INSTRUMENTS:
            raise ValueError(f"unknown instrument {name!r}; expected one of {sorted(INSTRUMENTS)}")
        if name in cfg.train.instruments_final_only and not is_final:
            continue
        INSTRUMENTS[name](model, pde, cfg, step, out_dir, smoke=smoke)
