"""T1.10 DoD: probe R^2 on the *input* layer for P1 is low (a linear function of x cannot
represent sin(15 pi x)), and R^2 on the final pre-head layer is high (>0.95) for a trained
model. cka(X, X) == 1.
"""
from __future__ import annotations

import torch

from qapinn.models.mlp import MLPPINN
from qapinn.pdes.poisson import Poisson
from qapinn.xai.probes import cka, collect_layer_outputs, layer_probe_r2


def _train_briefly(model, pde, steps=1500, lr=1e-3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x = pde.sample_collocation(256, gen).requires_grad_(True)
        u = pde.apply_hard_bc(x, model(x))
        residual = pde.residual(x, u)
        loss = (residual**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def test_layer_probe_r2_input_low_final_hidden_high_for_trained_model():
    torch.manual_seed(0)
    pde = Poisson(alpha=0.0)  # pure sin(pi x), easy/fast to fit well
    model = MLPPINN(input_dim=1, widths=(32, 32))
    _train_briefly(model, pde, steps=1500)

    grid = pde.eval_grid(256)
    layer_outputs = collect_layer_outputs(model, grid)
    r2 = layer_probe_r2(model, pde, grid, layer_outputs)

    assert r2["input"] < 0.5  # a linear map of x cannot represent sin(pi x)'s curvature well
    final_hidden_key = "activations.1"  # last Tanh, right before the head (layers.2)
    assert final_hidden_key in layer_outputs
    assert r2[final_hidden_key] > 0.95


def test_layer_probe_r2_input_layer_low_for_high_frequency_target():
    # alpha=0.3 (has the 15*pi component) makes the input-layer failure even more obvious:
    # a linear function of x is a poor fit for sin(15 pi x) regardless of training.
    torch.manual_seed(1)
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=1, widths=(16, 16))

    grid = pde.eval_grid(256)
    layer_outputs = collect_layer_outputs(model, grid)
    r2 = layer_probe_r2(model, pde, grid, layer_outputs)

    assert r2["input"] < 0.3


def test_cka_self_similarity_is_one():
    torch.manual_seed(2)
    X = torch.randn(50, 8)
    assert abs(cka(X, X) - 1.0) < 1e-6


def test_cka_orthogonal_representations_near_zero():
    torch.manual_seed(3)
    n = 200
    X = torch.randn(n, 5)
    # Y built from a totally independent random source -> near-zero alignment
    Y = torch.randn(n, 5)
    val = cka(X, Y)
    assert -1e-6 <= val < 0.3


def test_collect_layer_outputs_includes_input_and_named_submodules():
    model = MLPPINN(input_dim=1, widths=(4, 4))
    x = torch.linspace(0.0, 1.0, 10).reshape(-1, 1)
    outputs = collect_layer_outputs(model, x)

    assert "input" in outputs
    assert torch.equal(outputs["input"], x)
    assert "layers.0" in outputs
    assert "activations.0" in outputs
    assert "layers.2" in outputs  # the head (1 -> 4 -> 4 -> 1: layers 0,1,2)
