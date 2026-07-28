"""T0.13 DoD: reference_solution dispatches to the right closed-form/quadrature per PDE and
hits the cache on a second call with the same (pde, grid) -- checked via file mtime."""
from __future__ import annotations

import math

import numpy as np
import pytest

import qapinn.reference as reference_mod
from qapinn.pdes.burgers import Burgers
from qapinn.pdes.heat import Heat
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson
from qapinn.reference.analytic import heat_exact, helmholtz_exact, poisson_exact


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_mod, "_CACHE_DIR", tmp_path)


def test_reference_solution_poisson_dispatch():
    pde = Poisson(alpha=0.3)
    grid = np.linspace(0.0, 1.0, 50).reshape(-1, 1)

    u = reference_mod.reference_solution(pde, grid)
    expected = poisson_exact(grid[:, 0], 0.3)
    assert np.allclose(u, expected)


def test_reference_solution_heat_dispatch():
    pde = Heat(alpha=0.3, nu=0.05)
    rng = np.random.default_rng(1)
    grid = np.stack([rng.uniform(0, 1, 50), rng.uniform(0, 1, 50)], axis=1)

    u = reference_mod.reference_solution(pde, grid)
    expected = heat_exact(grid[:, 0], grid[:, 1], 0.05, modes=[(1, 1.0), (15, 0.3)])
    assert np.allclose(u, expected)


def test_reference_solution_helmholtz_dispatch():
    pde = Helmholtz(k=10.0, a1=3.0, a2=1.0)
    rng = np.random.default_rng(2)
    grid = np.stack([rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50)], axis=1)

    u = reference_mod.reference_solution(pde, grid)
    expected = helmholtz_exact(grid[:, 0], grid[:, 1], 3.0, 1.0)
    assert np.allclose(u, expected)


def test_reference_solution_burgers_dispatch_uses_colehopf():
    pde = Burgers()
    grid = np.array([[0.5, 0.1], [0.0, 0.0], [-0.5, 0.2]])

    u = reference_mod.reference_solution(pde, grid)

    # t=0 row must equal the exact IC -sin(pi x) (Cole-Hopf's own boundary case)
    assert math.isclose(u[1], -math.sin(math.pi * 0.0), abs_tol=1e-10)


def test_reference_solution_caches_second_call(tmp_path):
    pde = Poisson(alpha=0.3)
    grid = np.linspace(0.0, 1.0, 50).reshape(-1, 1)

    u1 = reference_mod.reference_solution(pde, grid)
    cache_files = list(tmp_path.glob("*.npz"))
    assert len(cache_files) == 1
    mtime_1 = cache_files[0].stat().st_mtime_ns

    u2 = reference_mod.reference_solution(pde, grid)
    mtime_2 = cache_files[0].stat().st_mtime_ns

    assert mtime_1 == mtime_2  # second call read the cache, never rewrote the file
    assert np.array_equal(u1, u2)


def test_reference_solution_different_grid_misses_cache(tmp_path):
    pde = Poisson(alpha=0.3)
    grid_a = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    grid_b = np.linspace(0.0, 1.0, 64).reshape(-1, 1)

    reference_mod.reference_solution(pde, grid_a)
    reference_mod.reference_solution(pde, grid_b)

    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_reference_solution_unknown_pde_raises():
    class _NotAPDE:
        name = "mystery"

    with pytest.raises(TypeError):
        reference_mod.reference_solution(_NotAPDE(), np.zeros((5, 1)))
