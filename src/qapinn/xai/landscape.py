"""Loss landscape slices (T1.11, project.md Section 7.6, Li et al. 2018).

Filter-normalised (per-parameter-tensor, matching the phase doc's "normalised per-layer"
wording) 2-D random-direction slices around the model's current parameters. Reuses
train/losses.py's pinn_loss directly rather than re-deriving the training objective, so
the landscape always reflects the SAME loss the model was actually trained on.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from qapinn.config import TrainConfig
from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE
from qapinn.train.losses import pinn_loss

DEFAULT_GRID_N = 25
DEFAULT_SPAN = 1.0


def _filter_normalised_direction(params: list[torch.nn.Parameter], gen: torch.Generator | None) -> list[Tensor]:
    """One random direction, same shapes as `params`, each tensor rescaled to the norm of
    its corresponding trained parameter (Li et al.'s filter normalisation, applied per
    whole parameter tensor rather than per individual filter/neuron)."""
    direction = []
    for p in params:
        d = torch.randn(p.shape, generator=gen, dtype=p.dtype, device=p.device)
        d_norm = d.norm()
        p_norm = p.detach().norm()
        if d_norm > 0 and p_norm > 0:
            d = d * (p_norm / d_norm)
        direction.append(d)
    return direction


def loss_landscape_slice(
    model: PINNModel,
    pde: PDE,
    batch: dict[str, Tensor],
    cfg: TrainConfig,
    grid_n: int = DEFAULT_GRID_N,
    span: float = DEFAULT_SPAN,
    gen: torch.Generator | None = None,
) -> np.ndarray:
    """Loss on a `grid_n` x `grid_n` grid of (alpha, beta) in `linspace(-span, span,
    grid_n)`^2, evaluating `theta* + alpha*d1 + beta*d2` for two independent
    filter-normalised random directions d1, d2. Restores the model's original parameters
    before returning. Returns `[grid_n, grid_n]` (row=alpha, col=beta)."""
    params = list(model.parameters())
    theta_star = [p.detach().clone() for p in params]

    d1 = _filter_normalised_direction(params, gen)
    d2 = _filter_normalised_direction(params, gen)

    alphas = np.linspace(-span, span, grid_n)
    betas = np.linspace(-span, span, grid_n)
    losses = np.zeros((grid_n, grid_n))

    try:
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                with torch.no_grad():
                    for p, t0, dd1, dd2 in zip(params, theta_star, d1, d2):
                        p.copy_(t0 + a * dd1 + b * dd2)

                batch_local = dict(batch)
                batch_local["x_r"] = batch["x_r"].detach().clone().requires_grad_(True)
                loss, _terms = pinn_loss(model, pde, batch_local, cfg)
                losses[i, j] = float(loss.detach().item())
    finally:
        with torch.no_grad():
            for p, t0 in zip(params, theta_star):
                p.copy_(t0)

    return losses
