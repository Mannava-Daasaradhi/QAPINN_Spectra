"""T0.8 DoD:
1. Exact solutions satisfy their own residual to 1e-8 (P1, P2, P4). P3 (Burgers) has no
   closed-form solution before T0.12's Cole-Hopf solver exists, so its residual FORMULA
   is instead checked against a hand-derived closed form for an arbitrary manufactured
   test function -- still catches sign errors, the stated purpose of this check.
2. apply_hard_bc reproduces the boundary/initial data exactly, for all four PDEs.
3. forcing matches the analytically derived f, for P1, P2, P4.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import yaml

from qapinn.config import CONFIG_ROOT, PDEConfig
from qapinn.pdes import build
from qapinn.pdes.burgers import Burgers
from qapinn.pdes.heat import Heat
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson
from qapinn.seeding import set_global_seed


def _interior_points(pde, n, gen):
    x = pde.sample_collocation(n, gen)
    x.requires_grad_(True)
    return x


# ---------------------------------------------------------------------------
# 1. Exact solution satisfies its own residual
# ---------------------------------------------------------------------------


def test_poisson_exact_satisfies_residual():
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(0)
    x = _interior_points(pde, 256, gen)
    res = pde.residual(x, pde.exact(x))
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-8)


def test_heat_exact_satisfies_residual():
    pde = Heat(alpha=0.3, nu=0.05)
    gen = set_global_seed(0)
    x = _interior_points(pde, 256, gen)
    res = pde.residual(x, pde.exact(x))
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-8)


@pytest.mark.parametrize("k,a1,a2", [(4.0, 1.0, 1.0), (10.0, 3.0, 1.0), (20.0, 6.0, 2.0)])
def test_helmholtz_exact_satisfies_residual(k, a1, a2):
    pde = Helmholtz(k=k, a1=a1, a2=a2)
    gen = set_global_seed(0)
    x = _interior_points(pde, 256, gen)
    res = pde.residual(x, pde.exact(x))
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-8)


def test_burgers_residual_formula_sign_correct_via_manufactured_function():
    pde = Burgers()
    nu = pde.params["nu"]
    gen = set_global_seed(0)
    x = _interior_points(pde, 256, gen)
    xx = x[:, 0:1]
    tt = x[:, 1:2]

    u = torch.sin(math.pi * xx) * torch.exp(-tt)  # arbitrary smooth test function
    res = pde.residual(x, u)

    u_t_expected = -torch.sin(math.pi * xx) * torch.exp(-tt)
    u_x_expected = math.pi * torch.cos(math.pi * xx) * torch.exp(-tt)
    u_xx_expected = -(math.pi**2) * torch.sin(math.pi * xx) * torch.exp(-tt)
    expected = u_t_expected + u * u_x_expected - nu * u_xx_expected

    assert torch.allclose(res, expected, atol=1e-8)


def test_burgers_exact_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        Burgers().exact(torch.zeros(4, 2))


def test_burgers_symbol_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        Burgers().symbol(np.zeros((4, 2)))


# ---------------------------------------------------------------------------
# 2. apply_hard_bc reproduces boundary/initial data exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pde_factory",
    [
        lambda: Poisson(alpha=0.3),
        lambda: Heat(alpha=0.3, nu=0.05),
        lambda: Burgers(),
        lambda: Helmholtz(k=10.0, a1=3.0, a2=1.0),
    ],
    ids=["poisson", "heat", "burgers", "helmholtz"],
)
def test_apply_hard_bc_reproduces_constrained_data(pde_factory):
    pde = pde_factory()
    gen = set_global_seed(0)
    x_b = pde.sample_boundary(2000, gen)
    n_out = torch.randn(2000, 1, generator=gen)

    result = pde.apply_hard_bc(x_b, n_out)
    expected = pde.bc_lift(x_b)

    assert torch.allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. forcing matches the analytically derived f, for P1, P2, P4
# ---------------------------------------------------------------------------


def test_poisson_forcing_matches_analytic():
    pde = Poisson(alpha=0.3)
    x = torch.linspace(0.0, 1.0, 200).reshape(-1, 1)
    f = pde.forcing(x)
    expected = (math.pi**2) * torch.sin(math.pi * x) + 0.3 * (15 * math.pi) ** 2 * torch.sin(
        15 * math.pi * x
    )
    assert torch.allclose(f, expected, atol=1e-10)


def test_heat_forcing_is_zero():
    pde = Heat(alpha=0.3, nu=0.05)
    x = torch.rand(50, 2)
    f = pde.forcing(x)
    assert torch.allclose(f, torch.zeros_like(f))


def test_helmholtz_forcing_matches_analytic():
    k, a1, a2 = 10.0, 3.0, 1.0
    pde = Helmholtz(k=k, a1=a1, a2=a2)
    x = torch.rand(50, 2) * 2 - 1
    f = pde.forcing(x)
    u_star = torch.sin(a1 * math.pi * x[:, 0:1]) * torch.sin(a2 * math.pi * x[:, 1:2])
    expected = (k**2 - math.pi**2 * (a1**2 + a2**2)) * u_star
    assert torch.allclose(f, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Factory + real config-file wiring
# ---------------------------------------------------------------------------


def test_factory_builds_all_four_pdes():
    for name, params in [
        ("poisson", {"alpha": 0.3}),
        ("heat", {"alpha": 0.3, "nu": 0.05}),
        ("burgers", {}),
        ("helmholtz", {"k": 10.0, "a1": 3.0, "a2": 1.0}),
    ]:
        cfg = PDEConfig(name=name, params=params)
        pde = build(cfg)
        assert pde.name == name


def test_factory_rejects_unknown_pde():
    cfg = PDEConfig(name="not_a_pde", params={})
    with pytest.raises(ValueError):
        build(cfg)


def test_real_pde_yaml_files_parse_correctly():
    for name in ("poisson", "heat", "burgers", "helmholtz"):
        with (CONFIG_ROOT / "pde" / f"{name}.yaml").open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = PDEConfig(**data)
        assert cfg.name == name
        build(cfg)  # must construct without error


def test_helmholtz_yaml_carries_default_k10_triple():
    with (CONFIG_ROOT / "pde" / "helmholtz.yaml").open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["params"]["k"] == 10.0
    assert data["params"]["a1"] == 3
    assert data["params"]["a2"] == 1


def test_helmholtz_resonance_table():
    """Pins the three (k, a1, a2) triples from 01_CONVENTIONS.md §5."""
    for a1, a2, expected_pi2 in [(1.0, 1.0, 19.7392), (3.0, 1.0, 98.6960), (6.0, 2.0, 394.7842)]:
        assert math.pi**2 * (a1**2 + a2**2) == pytest.approx(expected_pi2, abs=1e-3)
