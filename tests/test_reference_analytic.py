"""T0.10 DoD: analytic reference wrappers agree with pde.exact to 1e-12 on random points."""
from __future__ import annotations

import numpy as np
import torch

from qapinn.pdes.heat import Heat
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson
from qapinn.reference.analytic import heat_exact, helmholtz_exact, poisson_exact


def test_poisson_exact_matches_pde_exact():
    rng = np.random.default_rng(0)
    x_np = rng.uniform(0.0, 1.0, size=200)
    alpha = 0.3

    ref = poisson_exact(x_np, alpha)

    pde = Poisson(alpha=alpha)
    x_t = torch.tensor(x_np, dtype=torch.get_default_dtype()).reshape(-1, 1)
    expected = pde.exact(x_t).detach().numpy().reshape(-1)

    assert np.allclose(ref, expected, atol=1e-12)


def test_heat_exact_matches_pde_exact_on_seeded_modes():
    rng = np.random.default_rng(1)
    x_np = rng.uniform(0.0, 1.0, size=200)
    t_np = rng.uniform(0.0, 1.0, size=200)
    alpha, nu = 0.3, 0.05

    ref = heat_exact(x_np, t_np, nu, modes=[(1, 1.0), (15, alpha)])

    pde = Heat(alpha=alpha, nu=nu)
    x_t = torch.tensor(np.stack([x_np, t_np], axis=1), dtype=torch.get_default_dtype())
    expected = pde.exact(x_t).detach().numpy().reshape(-1)

    assert np.allclose(ref, expected, atol=1e-12)


def test_heat_exact_supports_general_modes_beyond_seeded_pair():
    # 3-mode general initial data, not expressible via the PDE class (seeded to modes
    # 1 and 15 only) -- should still satisfy separation of variables at t=0.
    x_np = np.linspace(0.0, 1.0, 50, endpoint=False)
    nu = 0.05
    modes = [(1, 1.0), (3, 0.5), (7, 0.2)]

    u0 = heat_exact(x_np, np.zeros_like(x_np), nu, modes=modes)
    expected_ic = sum(w * np.sin(n * np.pi * x_np) for n, w in modes)

    assert np.allclose(u0, expected_ic, atol=1e-12)


def test_helmholtz_exact_matches_pde_exact():
    rng = np.random.default_rng(2)
    x_np = rng.uniform(-1.0, 1.0, size=200)
    y_np = rng.uniform(-1.0, 1.0, size=200)
    k, a1, a2 = 10.0, 3.0, 1.0

    ref = helmholtz_exact(x_np, y_np, a1, a2)

    pde = Helmholtz(k=k, a1=a1, a2=a2)
    xy_t = torch.tensor(np.stack([x_np, y_np], axis=1), dtype=torch.get_default_dtype())
    expected = pde.exact(xy_t).detach().numpy().reshape(-1)

    assert np.allclose(ref, expected, atol=1e-12)
