"""T1.11 DoD: produces a [25,25] array whose minimum sits at the centre (the trained
point) for a converged model.
"""
from __future__ import annotations

import numpy as np
import torch

from qapinn.config import TrainConfig
from qapinn.models.mlp import MLPPINN
from qapinn.pdes.poisson import Poisson
from qapinn.train.losses import pinn_loss
from qapinn.xai.landscape import loss_landscape_slice


def _train_briefly(model, pde, cfg, steps=1500, lr=1e-3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x_r = pde.sample_collocation(128, gen).requires_grad_(True)
        loss, _ = pinn_loss(model, pde, {"x_r": x_r}, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def test_loss_landscape_minimum_at_centre_for_converged_model():
    torch.manual_seed(0)
    pde = Poisson(alpha=0.0)
    model = MLPPINN(input_dim=1, widths=(16, 16))
    cfg = TrainConfig(bc_mode="hard")
    _train_briefly(model, pde, cfg, steps=1500)

    batch = {"x_r": pde.sample_collocation(128, torch.Generator().manual_seed(1))}
    grid = loss_landscape_slice(model, pde, batch, cfg, grid_n=25, span=1.0, gen=torch.Generator().manual_seed(2))

    assert grid.shape == (25, 25)
    min_idx = np.unravel_index(np.argmin(grid), grid.shape)
    assert min_idx == (12, 12)


def test_loss_landscape_restores_model_parameters():
    torch.manual_seed(1)
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=1, widths=(8, 8))
    cfg = TrainConfig(bc_mode="hard")

    params_before = [p.detach().clone() for p in model.parameters()]
    batch = {"x_r": pde.sample_collocation(32, torch.Generator().manual_seed(2))}
    loss_landscape_slice(model, pde, batch, cfg, grid_n=5, span=1.0, gen=torch.Generator().manual_seed(3))

    for p_before, p_after in zip(params_before, model.parameters()):
        assert torch.equal(p_before, p_after)


def test_loss_landscape_shape_matches_grid_n():
    torch.manual_seed(2)
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=1, widths=(4,))
    cfg = TrainConfig(bc_mode="hard")

    batch = {"x_r": pde.sample_collocation(16, torch.Generator().manual_seed(0))}
    grid = loss_landscape_slice(model, pde, batch, cfg, grid_n=7, span=0.5, gen=torch.Generator().manual_seed(0))
    assert grid.shape == (7, 7)
    assert np.all(np.isfinite(grid))
