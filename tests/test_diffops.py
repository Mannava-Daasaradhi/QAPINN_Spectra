"""T0.7 DoD: d1/d2/laplacian match closed forms on u = sin(3x)cos(2y) to 1e-10; create_graph
really flows to a leaf parameter through d2 (not just through x)."""
from __future__ import annotations

import torch

from qapinn.pdes.diffops import d1, d2, laplacian


def _sample_xy(n: int = 37) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(n, 2, dtype=torch.get_default_dtype(), generator=gen) * 2.0 - 1.0
    x.requires_grad_(True)
    return x


def test_d1_matches_closed_form():
    x = _sample_xy()
    u = torch.sin(3 * x[:, 0:1]) * torch.cos(2 * x[:, 1:2])

    du_dx = d1(u, x, 0)
    expected = 3 * torch.cos(3 * x[:, 0:1]) * torch.cos(2 * x[:, 1:2])

    assert torch.allclose(du_dx, expected, atol=1e-10)


def test_d2_matches_closed_form():
    x = _sample_xy()
    u = torch.sin(3 * x[:, 0:1]) * torch.cos(2 * x[:, 1:2])

    d2xx = d2(u, x, 0, 0)
    expected = -9 * torch.sin(3 * x[:, 0:1]) * torch.cos(2 * x[:, 1:2])

    assert torch.allclose(d2xx, expected, atol=1e-10)


def test_laplacian_matches_closed_form():
    x = _sample_xy()
    u = torch.sin(3 * x[:, 0:1]) * torch.cos(2 * x[:, 1:2])

    lap = laplacian(u, x)
    expected = -13.0 * u

    assert torch.allclose(lap, expected, atol=1e-10)


def test_laplacian_default_dims_is_all_dims():
    x = _sample_xy()
    u = torch.sin(3 * x[:, 0:1]) * torch.cos(2 * x[:, 1:2])

    lap_default = laplacian(u, x)
    lap_explicit = laplacian(u, x, dims=(0, 1))
    assert torch.equal(lap_default, lap_explicit)


def test_create_graph_flows_to_parameter_through_d2():
    x = _sample_xy()
    a = torch.tensor(2.0, requires_grad=True)
    u = a * torch.sin(3 * x[:, 0:1]) * torch.cos(2 * x[:, 1:2])

    d2xx = d2(u, x, 0, 0)
    loss = d2xx.sum()
    loss.backward()

    assert a.grad is not None
    expected_grad = (-9 * torch.sin(3 * x[:, 0:1]) * torch.cos(2 * x[:, 1:2])).sum()
    assert torch.allclose(a.grad, expected_grad, atol=1e-8)
