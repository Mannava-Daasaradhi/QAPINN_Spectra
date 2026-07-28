"""T0.14 DoD: match_param_count('c_mlp', target=3000) yields a model whose n_params() is
within 10% of 3000; an impossible target raises."""
from __future__ import annotations

import pytest

import qapinn.models as models_pkg
from qapinn.models.base import match_param_count


def test_match_param_count_c_mlp_within_tolerance():
    cfg = match_param_count("c_mlp", target=3000)
    model = models_pkg.build(cfg, input_dim=1)
    n = model.n_params()
    assert abs(n - 3000) <= 0.10 * 3000, f"n_params={n}"


def test_match_param_count_respects_input_dim():
    cfg = match_param_count("c_mlp", target=3000, input_dim=2)
    model = models_pkg.build(cfg, input_dim=2)
    n = model.n_params()
    assert abs(n - 3000) <= 0.10 * 3000, f"n_params={n}"


def test_match_param_count_impossible_target_raises():
    with pytest.raises(ValueError):
        match_param_count("c_mlp", target=1)


def test_match_param_count_unsupported_family_raises():
    with pytest.raises(ValueError):
        match_param_count("not_a_real_family", target=3000)
