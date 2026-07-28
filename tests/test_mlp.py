"""T0.15 DoD: c_mlp forward on [B,d] gives [B,1]; param_groups()['quantum'] == []."""
from __future__ import annotations

import pytest
import torch

from qapinn.models.mlp import MLPPINN


def test_mlp_forward_shape():
    model = MLPPINN(input_dim=2, widths=(16, 16))
    x = torch.rand(10, 2)
    out = model(x)
    assert out.shape == (10, 1)


def test_mlp_param_groups_quantum_empty():
    model = MLPPINN(input_dim=1, widths=(8,))
    groups = model.param_groups()
    assert groups["quantum"] == []
    assert len(groups["classical"]) > 0
    assert set(groups.keys()) == {"classical", "quantum"}


def test_mlp_realised_frequencies_is_none():
    model = MLPPINN(input_dim=1, widths=(8,))
    assert model.realised_frequencies() is None


def test_mlp_n_params_matches_manual_count():
    model = MLPPINN(input_dim=1, widths=(4, 4))
    # sizes: 1 -> 4 -> 4 -> 1, each linear layer contributes (in+1)*out params (bias included)
    expected = (1 + 1) * 4 + (4 + 1) * 4 + (4 + 1) * 1
    assert model.n_params() == expected


def test_mlp_rejects_non_tanh_activation():
    with pytest.raises(ValueError):
        MLPPINN(input_dim=1, widths=(8,), activation="relu")


def test_mlp_xavier_normal_init_not_all_zero():
    model = MLPPINN(input_dim=1, widths=(8, 8))
    for layer in model.layers:
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
