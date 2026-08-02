"""T2.8 DoD: tests/test_symbol.py asserts
1. P1 analytic support is exactly {pi, 15*pi} with weights {1, alpha}.
2. P4 (k=10) analytic support is exactly the four points (+-3*pi, +-1*pi).
3. Analytic and empirical agree for P1, P2, P4 -- same support, weights matching to 1%.
4. P2's high-mode weight ratio is ~= 0.025 (assert within 20%).
"""
from __future__ import annotations

import math

import numpy as np

from qapinn.pdes.heat import Heat
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson
from qapinn.smcd.symbol import analytic_spectrum, empirical_spectrum

EPS = 0.05


def _sorted_support(spec):
    omega_supp = spec.omega[spec.support]
    weight_supp = spec.weight[spec.support]
    order = np.lexsort(omega_supp.T[::-1])
    return omega_supp[order], weight_supp[order]


def test_p1_analytic_support_and_weights():
    pde = Poisson(alpha=0.3)
    spec = analytic_spectrum(pde, EPS)
    assert spec is not None
    omega_supp, weight_supp = _sorted_support(spec)
    assert omega_supp.shape == (2, 1)
    np.testing.assert_allclose(omega_supp.ravel(), [math.pi, 15.0 * math.pi], atol=1e-9)
    np.testing.assert_allclose(weight_supp, [1.0, 0.3], atol=1e-9)


def test_p4_analytic_support_is_four_points():
    pde = Helmholtz(k=10.0, a1=3.0, a2=1.0)
    spec = analytic_spectrum(pde, EPS)
    assert spec is not None
    omega_supp, weight_supp = _sorted_support(spec)
    assert omega_supp.shape == (4, 2)
    expected = np.array(
        [[s1 * 3.0 * math.pi, s2 * 1.0 * math.pi] for s1 in (-1, 1) for s2 in (-1, 1)]
    )
    expected = expected[np.lexsort(expected.T[::-1])]
    np.testing.assert_allclose(omega_supp, expected, atol=1e-9)
    np.testing.assert_allclose(weight_supp, np.full(4, 0.25), atol=1e-9)


def _assert_analytic_empirical_agree(pde, n_grid):
    analytic = analytic_spectrum(pde, EPS)
    empirical = empirical_spectrum(pde, EPS, n_grid=n_grid)
    assert analytic is not None

    a_omega, a_weight = _sorted_support(analytic)
    e_omega, e_weight = _sorted_support(empirical)

    assert a_omega.shape == e_omega.shape, (a_omega, e_omega)
    np.testing.assert_allclose(e_omega, a_omega, atol=1e-6)
    np.testing.assert_allclose(e_weight, a_weight, rtol=0.01)


def test_p1_analytic_empirical_agree():
    _assert_analytic_empirical_agree(Poisson(alpha=0.3), n_grid=256)


def test_p2_analytic_empirical_agree():
    _assert_analytic_empirical_agree(Heat(alpha=0.3, nu=0.05), n_grid=256)


def test_p4_analytic_empirical_agree():
    _assert_analytic_empirical_agree(Helmholtz(k=10.0, a1=3.0, a2=1.0), n_grid=64)


def test_p2_high_mode_weight_ratio():
    pde = Heat(alpha=0.3, nu=0.05)
    spec = analytic_spectrum(pde, EPS)
    assert spec is not None
    ratio = spec.weight[1] / spec.weight[0]  # w(15*pi) / w(pi)
    assert abs(ratio - 0.025) / 0.025 < 0.20, f"ratio={ratio}"


def test_p3_burgers_has_no_analytic_spectrum():
    from qapinn.pdes.burgers import Burgers

    assert analytic_spectrum(Burgers(), EPS) is None
