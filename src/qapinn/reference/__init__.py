"""Reference solution dispatch + caching (T0.13).

Dispatches to the closed-form analytic solution where available (P1, P2, P4) or the cached
Cole-Hopf quadrature (P3). Caches to results/reference/<pde_name>_<grid_hash>.npz -- distinct
from colehopf.py's own fixed-name cache, which is specifically the T0.12 validation artifact;
this one is a generic per-(pde, grid) cache for arbitrary downstream callers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from qapinn.pdes.base import PDE
from qapinn.pdes.burgers import Burgers
from qapinn.pdes.heat import Heat
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson
from qapinn.reference.analytic import heat_exact, helmholtz_exact, poisson_exact
from qapinn.reference.colehopf import cached_burgers_colehopf

_CACHE_DIR = Path("results/reference")


def _grid_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(grid).tobytes()).hexdigest()[:12]


def reference_solution(pde: PDE, grid: np.ndarray) -> np.ndarray:
    """Ground truth for `pde` on `grid` ([N, d] numpy array of physical coordinates)."""
    grid = np.asarray(grid, dtype=np.float64)
    cache_path = _CACHE_DIR / f"{pde.name}_{_grid_hash(grid)}.npz"

    if cache_path.is_file():
        return np.load(cache_path)["u"]

    if isinstance(pde, Poisson):
        u = poisson_exact(grid[:, 0], pde.params["alpha"])
    elif isinstance(pde, Heat):
        u = heat_exact(
            grid[:, 0],
            grid[:, 1],
            pde.params["nu"],
            modes=[(1, 1.0), (15, pde.params["alpha"])],
        )
    elif isinstance(pde, Helmholtz):
        u = helmholtz_exact(grid[:, 0], grid[:, 1], pde.params["a1"], pde.params["a2"])
    elif isinstance(pde, Burgers):
        u = cached_burgers_colehopf(grid[:, 0], grid[:, 1], pde.params["nu"])
    else:
        raise TypeError(f"no reference solution dispatch registered for PDE {pde.name!r}")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, u=u)
    return u
