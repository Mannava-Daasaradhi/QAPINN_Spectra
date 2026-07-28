"""T0.16 DoD: c_rff_matched with B=[[pi],[15pi]] can least-squares-fit sin(pi x)+0.3 sin(15pi x)
to rel-L2 < 1e-8 -- the matched basis spans the target exactly (classical dry-run of Prop. 3)."""
from __future__ import annotations

import math

import numpy as np
import torch

from qapinn.models.fourier_features import FourierFeatures, make_c_ff, make_c_rff_matched


def test_c_rff_matched_spans_target_exactly():
    frequencies = [math.pi, 15 * math.pi]
    model = make_c_rff_matched(frequencies)

    x = torch.linspace(0.0, 1.0, 2000, dtype=torch.get_default_dtype()).reshape(-1, 1)
    target = torch.sin(math.pi * x) + 0.3 * torch.sin(15 * math.pi * x)

    with torch.no_grad():
        features = model.features(x)  # [N, 4]: sin(pi x), sin(15 pi x), cos(pi x), cos(15 pi x)
        design = torch.cat([features, torch.ones_like(x)], dim=-1)
        solution = torch.linalg.lstsq(design, target).solution
        fitted = design @ solution

    rel_l2 = torch.linalg.norm(fitted - target) / torch.linalg.norm(target)
    assert rel_l2.item() < 1e-8


def test_c_rff_matched_realised_frequencies_returns_B_rows():
    frequencies = [math.pi, 15 * math.pi]
    model = make_c_rff_matched(frequencies)

    freqs = model.realised_frequencies()
    assert freqs.shape == (2, 1)
    assert np.allclose(sorted(freqs.flatten()), sorted(frequencies))


def test_fourier_features_use_angular_frequency_directly_no_extra_2pi():
    # B holds ANGULAR frequencies directly (D12): sin(omega*x) at x=0.5 should be exactly
    # sin(omega*0.5) -- a direct check that no extra 2*pi factor sneaks into the forward pass.
    omega = 4.0
    features = FourierFeatures(torch.tensor([[omega]], dtype=torch.get_default_dtype()))
    x = torch.tensor([[0.5]], dtype=torch.get_default_dtype())

    out = features(x)

    expected = torch.tensor([[math.sin(omega * 0.5), math.cos(omega * 0.5)]])
    assert torch.allclose(out, expected, atol=1e-10)


def test_c_ff_forward_shape_and_param_groups():
    gen = torch.Generator().manual_seed(0)
    model = make_c_ff(input_dim=1, n_features=32, ff_sigma=10.0, gen=gen)

    x = torch.rand(20, 1)
    out = model(x)
    assert out.shape == (20, 1)

    groups = model.param_groups()
    assert groups["quantum"] == []
    assert len(groups["classical"]) > 0


def test_c_ff_b_is_buffer_not_parameter():
    model = make_c_ff(input_dim=1, n_features=8, ff_sigma=1.0)

    param_names = {name for name, _ in model.named_parameters()}
    assert "features.B" not in param_names

    buffer_names = {name for name, _ in model.named_buffers()}
    assert "features.B" in buffer_names


def test_c_ff_reproducible_with_generator():
    gen1 = torch.Generator().manual_seed(0)
    model1 = make_c_ff(input_dim=1, n_features=16, ff_sigma=5.0, gen=gen1)

    gen2 = torch.Generator().manual_seed(0)
    model2 = make_c_ff(input_dim=1, n_features=16, ff_sigma=5.0, gen=gen2)

    assert torch.equal(model1.features.B, model2.features.B)
