"""T1.6 DoD: on a linear model, IG attributions equal (x - baseline) . w exactly;
completeness_error < 1e-6 for n_steps=256.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from qapinn.models.base import PINNModel
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson
from qapinn.xai.attribution import (
    attribution_field,
    attribution_residual_correlation,
    completeness_error,
    integrated_gradients,
)


class _LinearModel(PINNModel):
    """u = w.x + b, an exactly linear map -- IG's path-integral gradient is constant
    (== w) along the whole baseline->x segment, so IG must recover (x-baseline).w exactly
    regardless of the number of Riemann-sum steps."""

    def __init__(self, input_dim):
        super().__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.head(x)

    def param_groups(self):
        return {"classical": list(self.parameters()), "quantum": []}


def test_integrated_gradients_linear_model_exact():
    torch.manual_seed(0)
    pde = Poisson(alpha=0.3)
    model = _LinearModel(input_dim=1)

    x = torch.linspace(0.0, 1.0, 11).reshape(-1, 1)
    baseline = torch.full_like(x, 0.5)

    attrs = integrated_gradients(model, pde, x, baseline=baseline, n_steps=8, target="u")

    w = model.head.weight.detach()  # [1, d]
    expected = (x - baseline) * w.reshape(1, -1)
    assert torch.allclose(attrs, expected, atol=1e-6)


def test_completeness_error_small_for_linear_model():
    torch.manual_seed(1)
    pde = Poisson(alpha=0.3)
    model = _LinearModel(input_dim=1)

    x = torch.linspace(0.0, 1.0, 11).reshape(-1, 1)
    attrs = integrated_gradients(model, pde, x, n_steps=256, target="u")

    err = completeness_error(model, pde, x, attrs, target="u")
    assert err < 1e-6


def test_integrated_gradients_default_baseline_is_domain_centre():
    pde = Poisson(alpha=0.0)
    model = _LinearModel(input_dim=1)
    x = torch.tensor([[0.0], [1.0]])

    attrs = integrated_gradients(model, pde, x, n_steps=4, target="u")
    w = model.head.weight.detach().reshape(1, -1)
    expected = (x - 0.5) * w
    assert torch.allclose(attrs, expected, atol=1e-6)


def test_integrated_gradients_residual_target_runs_and_has_right_shape():
    # target='residual' needs third-order autograd through diffops.laplacian (create_graph
    # threaded at every level) -- this exercises that it doesn't error, not exact values.
    pde = Poisson(alpha=0.3)
    model = _LinearModel(input_dim=1)
    x = torch.linspace(0.1, 0.9, 6).reshape(-1, 1)

    attrs = integrated_gradients(model, pde, x, n_steps=4, target="residual")
    assert attrs.shape == x.shape
    assert torch.isfinite(attrs).all()


def test_attribution_field_shape_matches_eval_grid():
    pde = Poisson(alpha=0.3)
    model = _LinearModel(input_dim=1)

    field = attribution_field(model, pde, grid_n=16, n_steps=4, target="u")
    assert field.shape == (16,)
    assert np.all(np.isfinite(field))


def test_attribution_field_shape_matches_eval_grid_2d():
    pde = Helmholtz(k=4.0, a1=1.0, a2=1.0)
    model = _LinearModel(input_dim=2)

    field = attribution_field(model, pde, grid_n=8, n_steps=4, target="u")
    assert field.shape == (8, 8)


def test_attribution_residual_correlation_perfect_match():
    rng = np.random.default_rng(0)
    residual_field = rng.random((32,))
    # a monotone (here, identical) transform of residual_field must give correlation 1.0
    attrs_field = residual_field * 3.0 + 1.0

    corr = attribution_residual_correlation(attrs_field, residual_field)
    assert abs(corr - 1.0) < 1e-10


def test_attribution_residual_correlation_uncorrelated_near_zero():
    rng = np.random.default_rng(0)
    a = rng.permutation(200).astype(float)
    b = rng.permutation(200).astype(float)

    corr = attribution_residual_correlation(a, b)
    assert abs(corr) < 0.2


def test_completeness_error_reported_for_nonlinear_model_not_hidden():
    # IG is known to be unstable for nonlinear models -- this just asserts the function
    # returns a finite, non-negative number (it is reported, not asserted small).
    from qapinn.models.mlp import MLPPINN

    pde = Poisson(alpha=0.3)
    torch.manual_seed(2)
    model = MLPPINN(input_dim=1, widths=(8, 8))
    x = torch.linspace(0.1, 0.9, 5).reshape(-1, 1)

    attrs = integrated_gradients(model, pde, x, n_steps=16, target="u")
    err = completeness_error(model, pde, x, attrs, target="u")
    assert math.isfinite(err)
    assert err >= 0.0
