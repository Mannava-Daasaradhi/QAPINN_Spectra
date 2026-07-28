"""T0.6 DoD: dummy PDE subclass instantiates; eval_grid shape correct; sample_collocation
reproducible under a fixed generator."""
from __future__ import annotations

import torch

from qapinn.pdes.base import PDE, Domain
from qapinn.seeding import set_global_seed


class _DummyPDEBase(PDE):
    """Minimal concrete PDE: physics methods are stubs, only sampling/grid geometry
    is under test here (T0.6 is about the base class, not the physics -- that's T0.8)."""

    def residual(self, x, u):
        return u

    def forcing(self, x):
        return torch.zeros_like(x[:, :1])

    def exact(self, x):
        return torch.zeros_like(x[:, :1])

    def bc_lift(self, x):
        return torch.zeros_like(x[:, :1])

    def bc_mask(self, x):
        return torch.ones_like(x[:, :1])

    def boundary_residual(self, x_b, u_b):
        return u_b

    def symbol(self, omega):
        return (omega**2).sum(axis=-1)


class _Dummy1D(_DummyPDEBase):
    def __init__(self):
        self.name = "dummy1d"
        self.domain = Domain(bounds=((0.0, 1.0),), names=("x",), time_axis=None)
        self.dim = 1
        self.params = {}


class _Dummy2DSteady(_DummyPDEBase):
    def __init__(self):
        self.name = "dummy2d"
        self.domain = Domain(bounds=((0.0, 1.0), (0.0, 1.0)), names=("x", "y"), time_axis=None)
        self.dim = 2
        self.params = {}


class _Dummy2DTime(_DummyPDEBase):
    def __init__(self):
        self.name = "dummy2d_time"
        self.domain = Domain(bounds=((0.0, 1.0), (0.0, 1.0)), names=("x", "t"), time_axis=1)
        self.dim = 2
        self.params = {}


def test_dummy_pde_instantiates():
    pde = _Dummy1D()
    assert pde.dim == 1
    assert pde.name == "dummy1d"


def test_eval_grid_shape_2d():
    pde = _Dummy2DSteady()
    grid = pde.eval_grid(64)
    assert grid.shape == (4096, 2)


def test_eval_grid_excludes_right_endpoint():
    pde = _Dummy1D()
    grid = pde.eval_grid(8)
    assert grid.shape == (8, 1)
    assert torch.allclose(grid[0, 0], torch.tensor(0.0))
    assert grid[:, 0].max().item() < 1.0
    # uniform spacing
    diffs = grid[1:, 0] - grid[:-1, 0]
    assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-12)


def test_sample_collocation_reproducible_and_in_bounds():
    pde = _Dummy1D()
    gen1 = set_global_seed(0)
    a = pde.sample_collocation(100, gen1)
    gen2 = set_global_seed(0)
    b = pde.sample_collocation(100, gen2)

    assert torch.equal(a, b)
    assert a.shape == (100, 1)
    assert (a >= 0.0).all() and (a <= 1.0).all()


def test_sample_boundary_1d_hits_both_endpoints():
    pde = _Dummy1D()
    gen = set_global_seed(0)
    pts = pde.sample_boundary(2000, gen)
    assert set(torch.unique(pts).tolist()) == {0.0, 1.0}


def test_sample_boundary_time_axis_excludes_final_time():
    pde = _Dummy2DTime()
    gen = set_global_seed(0)
    pts = pde.sample_boundary(4000, gen)

    on_x_edge = (pts[:, 0] == 0.0) | (pts[:, 0] == 1.0)
    on_ic = pts[:, 1] == 0.0
    assert bool((on_x_edge | on_ic).all())
    assert not bool((pts[:, 1] == 1.0).any())


def test_sample_boundary_2d_steady_hits_all_four_edges():
    pde = _Dummy2DSteady()
    gen = set_global_seed(0)
    pts = pde.sample_boundary(4000, gen)

    on_x_edge = (pts[:, 0] == 0.0) | (pts[:, 0] == 1.0)
    on_y_edge = (pts[:, 1] == 0.0) | (pts[:, 1] == 1.0)
    assert bool((on_x_edge | on_y_edge).all())


def test_apply_hard_bc_matches_lift_plus_mask_times_n():
    pde = _Dummy1D()
    x = torch.tensor([[0.0], [0.5], [1.0]])
    n_out = torch.tensor([[5.0], [5.0], [5.0]])
    # dummy: bc_lift=0, bc_mask=1 everywhere -> apply_hard_bc == n_out
    result = pde.apply_hard_bc(x, n_out)
    assert torch.allclose(result, n_out)


def test_target_spectrum_analytic_defaults_to_none():
    pde = _Dummy1D()
    assert pde.target_spectrum_analytic(eps=1e-3) is None


def test_domain_rejects_mismatched_bounds_and_names():
    import pytest

    with pytest.raises(ValueError):
        Domain(bounds=((0.0, 1.0), (0.0, 1.0)), names=("x",), time_axis=None)


def test_domain_rejects_bad_bound_order():
    import pytest

    with pytest.raises(ValueError):
        Domain(bounds=((1.0, 0.0),), names=("x",), time_axis=None)
