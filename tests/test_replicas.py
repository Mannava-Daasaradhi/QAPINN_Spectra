"""v2 (2026-09-28): SerialHybrid with n_replicas copies of the designed circuit."""
from __future__ import annotations

import numpy as np
import torch

from qapinn.config import load_config
from qapinn.models import build
from qapinn.models.hybrid import SerialHybrid
from qapinn.pdes import build as build_pde


def _poisson_model(n_replicas: int | None, seed: int = 0) -> SerialHybrid:
    cfg = load_config("poisson", "q_serial", overrides={"model.n_replicas": n_replicas} if n_replicas else None)
    torch.manual_seed(seed)
    return build(cfg.model, input_dim=1, pde=build_pde(cfg.pde))


def test_one_replica_is_the_v1_model():
    """n_replicas unset and n_replicas=1 build the same parameters and the same output."""
    v1, one = _poisson_model(None), _poisson_model(1)
    assert [n for n, _ in v1.named_parameters()] == [n for n, _ in one.named_parameters()]
    x = torch.linspace(0, 1, 17).unsqueeze(-1)
    assert torch.equal(v1(x), one(x))


def test_replicas_keep_the_frequency_set_and_add_parameters():
    v1, four = _poisson_model(None), _poisson_model(4)
    np.testing.assert_array_equal(v1.realised_frequencies(), four.realised_frequencies())
    n_theta = v1.circuit.theta.numel()
    assert four.n_params() == v1.n_params() + 3 * n_theta + 3  # 3 more circuits, 3 more head weights
    assert four(torch.rand(8, 1)).shape == (8, 1)


def test_replicas_output_has_no_energy_outside_omega():
    """Mixing copies of one circuit cannot create frequencies the design did not reach."""
    model = _poisson_model(4)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 1.0)
        model.encoder.A.copy_(torch.eye(1))
        model.encoder.b.zero_()
        n = 4096
        x = torch.arange(n, dtype=torch.float64).unsqueeze(-1) / n * 2.0  # period 2 covers every omega = k*pi
        y = model.to(torch.float64)(x).squeeze(-1).numpy()
    spectrum = np.abs(np.fft.rfft(y)) / n
    omega_bins = {round(abs(w) / np.pi) for w in model.realised_frequencies().ravel()}
    outside = [spectrum[k] for k in range(len(spectrum)) if k not in omega_bins]
    assert max(outside) < 1e-8
