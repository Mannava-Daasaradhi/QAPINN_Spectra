"""Loss assembly: hard + soft BC (01_CONVENTIONS.md §5, D6, T0.17)."""
from __future__ import annotations

import torch
from torch import Tensor

from qapinn.config import TrainConfig
from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE


def pinn_loss(
    model: PINNModel, pde: PDE, batch: dict[str, Tensor], cfg: TrainConfig
) -> tuple[Tensor, dict[str, float]]:
    """batch keys: 'x_r' (collocation points, requires_grad=True) always; 'x_bc' and
    'x_ic' additionally for bc_mode == 'soft' (x_ic absent/None for steady PDEs -- P1, P4).

    bc_mode == 'hard': u = pde.apply_hard_bc(x, model(x)); loss = lambda_r * MSE(residual).
    bc_mode == 'soft': u = model(x) raw; loss = lambda_r*MSE(res) + lambda_b*MSE(bc)
        (+ lambda_i*MSE(ic) if the PDE has an initial condition).

    Returns the scalar loss and a dict of the individual (detached) terms for logging.
    """
    x_r = batch["x_r"]

    if cfg.bc_mode == "hard":
        u_r = pde.apply_hard_bc(x_r, model(x_r))
        residual = pde.residual(x_r, u_r)
        residual_loss = torch.mean(residual**2)
        loss = cfg.lambda_residual * residual_loss
        terms = {"residual": residual_loss.detach().item(), "loss": loss.detach().item()}
        return loss, terms

    if cfg.bc_mode == "soft":
        u_r = model(x_r)
        residual = pde.residual(x_r, u_r)
        residual_loss = torch.mean(residual**2)
        loss = cfg.lambda_residual * residual_loss
        terms = {"residual": residual_loss.detach().item()}

        x_bc = batch.get("x_bc")
        if x_bc is not None and x_bc.shape[0] > 0:
            u_bc = model(x_bc)
            bc_loss = torch.mean(pde.boundary_residual(x_bc, u_bc) ** 2)
            loss = loss + cfg.lambda_bc * bc_loss
            terms["bc"] = bc_loss.detach().item()

        x_ic = batch.get("x_ic")
        if x_ic is not None and x_ic.shape[0] > 0:
            u_ic = model(x_ic)
            ic_loss = torch.mean(pde.boundary_residual(x_ic, u_ic) ** 2)
            loss = loss + cfg.lambda_ic * ic_loss
            terms["ic"] = ic_loss.detach().item()

        terms["loss"] = loss.detach().item()
        return loss, terms

    raise ValueError(f"unknown bc_mode {cfg.bc_mode!r} (expected 'hard' or 'soft')")
