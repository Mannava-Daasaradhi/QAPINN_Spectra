"""Training loop: Adam (cosine schedule) -> LBFGS (strong_wolfe) -> final evaluation
(01_CONVENTIONS.md §8, T0.18).
"""
from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

import qapinn
import qapinn.models as models_pkg
import qapinn.pdes as pdes_pkg
import qapinn.reference as reference_pkg
from qapinn.config import ExpConfig
from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE
from qapinn.seeding import set_global_seed
from qapinn.train.losses import pinn_loss

RESULTS_ROOT = Path("results/runs")

# Populated by Phase 1 (T1.12): each hook is called at every checkpoint as
# hook(model, pde, eval_grid, step, run_dir). Empty in Phase 0 -- "for now the hook list
# is empty" (02_PHASE0_foundations.md, T0.18).
_XAI_HOOKS: list[Callable[[PINNModel, PDE, Tensor, int, Path], None]] = []


def register_xai_hook(hook: Callable[[PINNModel, PDE, Tensor, int, Path], None]) -> None:
    _XAI_HOOKS.append(hook)


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, float]
    history: pd.DataFrame = field(repr=False)


def _grad_norm(model: PINNModel) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total**0.5


_RESIDUAL_CHUNK_SIZE = 4096


def _residual_sq_sum_chunked(model: PINNModel, pde: PDE, eval_grid: Tensor, bc_mode: str) -> float:
    """Sum of squared residuals, processed in chunks so second-derivative autograd graphs
    are never materialised over the full eval grid at once. n_eval is a PER-AXIS
    resolution (T0.6): a 2-D problem's eval_grid(1024) is ~1e6 points, and computing d2
    over all of them simultaneously under deterministic algorithms can exhaust VRAM (D7
    already establishes this chunking pattern for NTK Jacobians)."""
    total_sq = 0.0
    n = eval_grid.shape[0]
    for start in range(0, n, _RESIDUAL_CHUNK_SIZE):
        chunk = eval_grid[start : start + _RESIDUAL_CHUNK_SIZE].clone().requires_grad_(True)
        if bc_mode == "hard":
            u = pde.apply_hard_bc(chunk, model(chunk))
        else:
            u = model(chunk)
        residual = pde.residual(chunk, u)
        total_sq += residual.detach().pow(2).sum().item()
    return total_sq


def _compute_metrics(model: PINNModel, pde: PDE, eval_grid: Tensor, bc_mode: str) -> dict[str, float]:
    with torch.no_grad():
        if bc_mode == "hard":
            u_pred = pde.apply_hard_bc(eval_grid, model(eval_grid))
        else:
            u_pred = model(eval_grid)
    u_pred_np = u_pred.detach().cpu().numpy().reshape(-1)

    grid_np = eval_grid.detach().cpu().numpy()
    u_ref = reference_pkg.reference_solution(pde, grid_np).reshape(-1)

    diff = u_pred_np - u_ref
    ref_norm = np.linalg.norm(u_ref)
    rel_l2 = float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0 else float(np.linalg.norm(diff))
    l_inf = float(np.max(np.abs(diff)))

    residual_norm = float(_residual_sq_sum_chunked(model, pde, eval_grid, bc_mode) ** 0.5)

    return {"rel_l2": rel_l2, "l_inf": l_inf, "residual_norm": residual_norm}


def train(cfg: ExpConfig, *, smoke: bool = False) -> RunResult:
    t_start = time.time()

    if smoke:
        pde_cfg = dataclasses.replace(cfg.pde, n_collocation=256)
        train_cfg = dataclasses.replace(cfg.train, steps_adam=50, steps_lbfgs=10, checkpoints=(0, -1))
        cfg = dataclasses.replace(cfg, pde=pde_cfg, train=train_cfg)

    pde_cfg = cfg.pde
    train_cfg = cfg.train

    gen = set_global_seed(cfg.seed)
    device = qapinn.device

    pde = pdes_pkg.build(pde_cfg)
    model = models_pkg.build(cfg.model, input_dim=pde.dim, gen=gen).to(device)

    design_card = None  # Phase 2 (T2.9) builds this for quantum families

    run_id = cfg.run_id
    run_dir = RESULTS_ROOT / run_id
    (run_dir / "xai").mkdir(parents=True, exist_ok=True)

    numeric_checkpoints = {c for c in train_cfg.checkpoints if c != -1}
    final_checkpoint_requested = -1 in train_cfg.checkpoints

    history_rows: list[dict[str, float]] = []
    checkpoint_rel_l2: list[tuple[int, float]] = []

    batch_cache: dict[str, Tensor | None] = {}

    def get_batch(step: int) -> dict[str, Tensor | None]:
        if step % 100 == 0 or "x_r" not in batch_cache:
            x_r = pde.sample_collocation(pde_cfg.n_collocation, gen).to(device)
            x_r.requires_grad_(True)
            batch_cache["x_r"] = x_r
            if train_cfg.bc_mode == "soft":
                x_bc = pde.sample_boundary_only(pde_cfg.n_boundary, gen).to(device)
                batch_cache["x_bc"] = x_bc
                x_ic = pde.sample_initial(pde_cfg.n_boundary, gen)
                batch_cache["x_ic"] = x_ic.to(device) if x_ic is not None else None
        return dict(batch_cache)

    def run_checkpoint(step: int) -> None:
        eval_grid = pde.eval_grid(pde_cfg.n_eval).to(device)
        m = _compute_metrics(model, pde, eval_grid, train_cfg.bc_mode)
        checkpoint_rel_l2.append((step, m["rel_l2"]))
        for hook in _XAI_HOOKS:
            hook(model, pde, eval_grid, step, run_dir)

    run_checkpoint(0)

    if train_cfg.lr_schedule != "cosine":
        raise ValueError(f"unsupported lr_schedule {train_cfg.lr_schedule!r} (only 'cosine' is implemented)")

    global_step = 0
    if train_cfg.steps_adam > 0:
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=train_cfg.steps_adam)

        for _ in range(train_cfg.steps_adam):
            batch = get_batch(global_step)
            optimizer_adam.zero_grad()
            loss, _terms = pinn_loss(model, pde, batch, train_cfg)
            loss.backward()
            grad_norm = _grad_norm(model)
            optimizer_adam.step()
            history_rows.append(
                {"step": global_step, "loss": loss.item(), "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm}
            )
            scheduler.step()
            global_step += 1
            if global_step in numeric_checkpoints:
                run_checkpoint(global_step)

    if train_cfg.steps_lbfgs > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=train_cfg.lr, max_iter=1, line_search_fn="strong_wolfe"
        )

        for _ in range(train_cfg.steps_lbfgs):
            batch = get_batch(global_step)

            def closure(batch=batch):
                optimizer_lbfgs.zero_grad()
                loss, _terms = pinn_loss(model, pde, batch, train_cfg)
                loss.backward()
                return loss

            loss = optimizer_lbfgs.step(closure)
            grad_norm = _grad_norm(model)
            history_rows.append(
                {"step": global_step, "loss": float(loss.item()), "lr": train_cfg.lr, "grad_norm": grad_norm}
            )
            global_step += 1
            if global_step in numeric_checkpoints:
                run_checkpoint(global_step)

    if final_checkpoint_requested:
        run_checkpoint(global_step)

    def first_step_below(tol: float) -> int:
        for step, rel_l2 in checkpoint_rel_l2:
            if rel_l2 < tol:
                return step
        return -1

    wall_clock_s = time.time() - t_start
    # checkpoint_rel_l2 only tracked rel_l2 (cheap to keep for steps_to_tol); recompute
    # the full metrics dict once more here so l_inf / residual_norm are also on record
    final_eval_grid = pde.eval_grid(pde_cfg.n_eval).to(device)
    final_metrics = _compute_metrics(model, pde, final_eval_grid, train_cfg.bc_mode)

    metrics = {
        "rel_l2": final_metrics["rel_l2"],
        "l_inf": final_metrics["l_inf"],
        "residual_norm": final_metrics["residual_norm"],
        "steps_to_tol_1e-2": first_step_below(1e-2),
        "steps_to_tol_1e-3": first_step_below(1e-3),
        "n_params": model.n_params(),
        "wall_clock_s": wall_clock_s,
        "circuit_evals": 0,
        "grad_var_final": 0.0,
        "ntk_cond": 0.0,
        "ntk_decay_exponent": 0.0,
        "smcd_coverage": 0.0,
        "smcd_coverage_weighted": 0.0,
        "encoder_drift": 0.0,
    }

    history = pd.DataFrame(history_rows, columns=["step", "loss", "lr", "grad_norm"])

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    with (run_dir / "design_card.json").open("w", encoding="utf-8") as f:
        json.dump(design_card, f)

    with (run_dir / "provenance.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"run_id": run_id, "seed": cfg.seed, "device": str(device), "wall_clock_s": wall_clock_s},
            f,
            indent=2,
        )

    history.to_parquet(run_dir / "history.parquet")

    return RunResult(run_id=run_id, run_dir=run_dir, metrics=metrics, history=history)
