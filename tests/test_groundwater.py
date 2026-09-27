"""v2 showcase problem: steady groundwater flow under canal-irrigated farmland."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch

from qapinn.config import load_config
from qapinn.pdes import build
from qapinn.pdes.groundwater import Groundwater, waterlogged_length_m
from qapinn.runner import enumerate_runs


def _pde() -> Groundwater:
    return build(load_config("groundwater", "c_mlp").pde)


def _finite_difference_head(pde: Groundwater, n: int = 20001) -> tuple[np.ndarray, np.ndarray]:
    """Second-order finite differences for T h'' = -R, independent of the closed form."""
    X = np.linspace(0.0, pde.params["length_m"], n)
    dx = X[1] - X[0]
    A = sp.diags([np.ones(n - 3), -2.0 * np.ones(n - 2), np.ones(n - 3)], [-1, 0, 1]) / dx**2
    rhs = -pde.recharge_m_per_day(X[1:-1]) / pde.params["transmissivity"]
    rhs[0] -= pde.params["head_left"] / dx**2
    rhs[-1] -= pde.params["head_right"] / dx**2
    interior = spl.spsolve(A.tocsc(), rhs)
    return X, np.concatenate([[pde.params["head_left"]], interior, [pde.params["head_right"]]])


def test_exact_head_matches_finite_differences():
    pde = _pde()
    X, h_fd = _finite_difference_head(pde)
    assert np.abs(pde.head_m(X) - h_fd).max() < 1e-6  # metres


def test_exact_head_satisfies_the_pinn_residual_and_boundaries():
    pde = _pde()
    x = torch.linspace(0.02, 0.98, 257, dtype=torch.float64).unsqueeze(-1).requires_grad_(True)
    X = x.detach().numpy()[:, 0] * pde.params["length_m"]
    h = pde.head_m(X)
    # u'' by finite differences on x (the exact solution is a numpy function)
    dx = 1e-4
    upp = (pde.head_m((X / pde.params["length_m"] + dx) * pde.params["length_m"]) - 2 * h
           + pde.head_m((X / pde.params["length_m"] - dx) * pde.params["length_m"])) / dx**2
    f = pde.forcing(x.detach()).numpy()[:, 0]
    np.testing.assert_allclose(-upp, f, rtol=1e-4, atol=1e-3)
    ends = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    np.testing.assert_allclose(pde.exact(ends).numpy()[:, 0], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(pde.apply_hard_bc(ends, torch.ones(2, 1, dtype=torch.float64)).numpy()[:, 0], [1.0, 0.0])


def test_scenario_numbers():
    """The numbers the showcase quotes: peak water table and waterlogged length."""
    pde = _pde()
    X = np.linspace(0.0, 2000.0, 20001)
    h = pde.head_m(X)
    assert 8.4 < h.max() < 8.6
    assert 700 < waterlogged_length_m(X, h, threshold_m=7.5) < 725


def test_waterlogged_length_interpolates_crossings():
    X = np.array([0.0, 1.0, 2.0, 3.0])
    assert waterlogged_length_m(X, np.array([0.0, 2.0, 2.0, 0.0]), threshold_m=1.0) == 2.0


def test_groundwater_is_sweepable_by_label(tmp_path):
    spec = tmp_path / "gw.yaml"
    spec.write_text("problems: [groundwater]\nfamilies: [c_mlp]\nseeds: [0]\n", encoding="utf-8")
    (cfg,) = enumerate_runs(spec)
    assert cfg.pde.name == "groundwater"
