"""T2.11 DoD: tests/test_noise.py -- shot noise has the right empirical variance over
10 000 draws (within 5%); depolarizing with p=0 is the identity; gradients flow through
both.
"""
from __future__ import annotations

import torch

from qapinn.models.noise import GlobalDepolarizing, ShotNoise

N_DRAWS = 10_000


def test_shot_noise_empirical_variance():
    n_shots = 1024
    module = ShotNoise(n_shots=n_shots)
    gen = torch.Generator().manual_seed(0)

    for expval_value in (0.0, 0.5, -0.7, 0.95):
        expval = torch.full((N_DRAWS,), expval_value)
        out = module(expval, generator=gen)
        empirical_var = (out - expval_value).var().item()
        expected_var = (1.0 - expval_value**2) / n_shots
        rel_err = abs(empirical_var - expected_var) / expected_var
        assert rel_err < 0.05, f"expval={expval_value}: rel_err={rel_err}"


def test_shot_noise_n_shots_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        ShotNoise(n_shots=0)


def test_shot_noise_gradient_flows_unmodified():
    module = ShotNoise(n_shots=1024)
    gen = torch.Generator().manual_seed(1)
    expval = torch.tensor([0.3, -0.2, 0.8], requires_grad=True)
    out = module(expval, generator=gen)
    (grad,) = torch.autograd.grad(out.sum(), expval)
    # straight-through: the noise term is detached, so d(out)/d(expval) == 1 exactly
    torch.testing.assert_close(grad, torch.ones_like(expval))


def test_depolarizing_p_zero_is_identity():
    module = GlobalDepolarizing(p=0.0, m=5)
    expval = torch.tensor([0.1, -0.5, 0.99, -1.0, 0.0])
    out = module(expval)
    torch.testing.assert_close(out, expval)


def test_depolarizing_scales_correctly():
    p, m = 0.02, 3
    module = GlobalDepolarizing(p=p, m=m)
    expval = torch.tensor([0.5, -0.3])
    out = module(expval)
    expected = ((1.0 - p) ** m) * expval
    torch.testing.assert_close(out, expected)


def test_depolarizing_gradient_flows():
    module = GlobalDepolarizing(p=0.05, m=2)
    expval = torch.tensor([0.4, -0.6], requires_grad=True)
    out = module(expval)
    (grad,) = torch.autograd.grad(out.sum(), expval)
    expected_grad = torch.full_like(expval, (1.0 - 0.05) ** 2)
    torch.testing.assert_close(grad, expected_grad)


def test_depolarizing_rejects_invalid_params():
    import pytest

    with pytest.raises(ValueError):
        GlobalDepolarizing(p=1.5, m=1)
    with pytest.raises(ValueError):
        GlobalDepolarizing(p=0.1, m=-1)


def test_both_classes_are_labelled_surrogates():
    assert "surrogate" in ShotNoise.__doc__.lower() or "surrogate" in noise_module_doc().lower()
    assert "surrogate" in GlobalDepolarizing.__doc__.lower() or "surrogate" in noise_module_doc().lower()


def noise_module_doc() -> str:
    import qapinn.models.noise as noise_mod

    return noise_mod.__doc__ or ""
