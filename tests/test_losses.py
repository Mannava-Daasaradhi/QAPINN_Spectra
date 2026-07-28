"""T0.17 DoD: with bc_mode='hard' on P1, feeding a model that exactly returns (u*-B)/D gives
a loss below 1e-10."""
from __future__ import annotations

from torch import nn

from qapinn.config import TrainConfig
from qapinn.models.mlp import MLPPINN
from qapinn.pdes.heat import Heat
from qapinn.pdes.poisson import Poisson
from qapinn.seeding import set_global_seed
from qapinn.train.losses import pinn_loss


class _ExactRawOutput(nn.Module):
    """forward returns exactly (u* - B)/D, so apply_hard_bc reproduces u* exactly."""

    def __init__(self, pde):
        super().__init__()
        self.pde = pde

    def forward(self, x):
        return (self.pde.exact(x) - self.pde.bc_lift(x)) / self.pde.bc_mask(x)


def test_hard_bc_loss_near_zero_for_exact_model():
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(0)
    x_r = pde.sample_collocation(512, gen)
    x_r.requires_grad_(True)

    model = _ExactRawOutput(pde)
    cfg = TrainConfig(bc_mode="hard")

    loss, terms = pinn_loss(model, pde, {"x_r": x_r}, cfg)

    assert loss.item() < 1e-10
    assert terms["residual"] < 1e-10


def test_hard_bc_loss_nonzero_for_wrong_model():
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(0)
    x_r = pde.sample_collocation(512, gen)
    x_r.requires_grad_(True)

    model = nn.Linear(1, 1)  # arbitrary, does not solve the PDE
    cfg = TrainConfig(bc_mode="hard")

    loss, terms = pinn_loss(model, pde, {"x_r": x_r}, cfg)

    assert loss.item() > 1e-6
    assert set(terms.keys()) == {"residual", "loss"}


def test_soft_bc_includes_residual_bc_and_ic_terms_for_heat():
    pde = Heat(alpha=0.3, nu=0.05)
    gen = set_global_seed(0)
    batch = {
        "x_r": pde.sample_collocation(256, gen).requires_grad_(True),
        "x_bc": pde.sample_boundary_only(64, gen),
        "x_ic": pde.sample_initial(64, gen),
    }
    # a plain nn.Linear has zero curvature (its second derivative doesn't depend on x at
    # all, which torch.autograd.grad rejects as "not used in the graph") -- MLPPINN's tanh
    # nonlinearity gives genuine curvature for the residual's second-derivative term.
    model = MLPPINN(input_dim=2, widths=(8, 8))
    cfg = TrainConfig(bc_mode="soft")

    loss, terms = pinn_loss(model, pde, batch, cfg)

    assert set(terms.keys()) == {"residual", "bc", "ic", "loss"}
    assert loss.item() == terms["loss"]


def test_soft_bc_omits_ic_term_for_steady_pde():
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(0)
    batch = {
        "x_r": pde.sample_collocation(256, gen).requires_grad_(True),
        "x_bc": pde.sample_boundary_only(64, gen),
        "x_ic": pde.sample_initial(64, gen),  # None for a steady PDE
    }
    model = MLPPINN(input_dim=1, widths=(8,))
    cfg = TrainConfig(bc_mode="soft")

    _loss, terms = pinn_loss(model, pde, batch, cfg)

    assert set(terms.keys()) == {"residual", "bc", "loss"}


def test_exact_solution_gives_near_zero_soft_bc_loss_too():
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(0)
    batch = {
        "x_r": pde.sample_collocation(512, gen).requires_grad_(True),
        "x_bc": pde.sample_boundary_only(128, gen),
        "x_ic": None,
    }

    class _ExactSoft(nn.Module):
        def forward(self, x):
            return pde.exact(x)

    cfg = TrainConfig(bc_mode="soft")
    loss, terms = pinn_loss(_ExactSoft(), pde, batch, cfg)

    assert loss.item() < 1e-10
    assert terms["residual"] < 1e-10
    assert terms["bc"] < 1e-10


def test_unknown_bc_mode_raises():
    import pytest

    pde = Poisson(alpha=0.3)
    gen = set_global_seed(0)
    x_r = pde.sample_collocation(16, gen).requires_grad_(True)
    model = nn.Linear(1, 1)
    cfg = TrainConfig(bc_mode="not_a_real_mode")

    with pytest.raises(ValueError):
        pinn_loss(model, pde, {"x_r": x_r}, cfg)
