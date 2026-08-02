"""T2.14 DoD: for P1, c_rff_matched.realised_frequencies() contains {pi, 15*pi}; its
parameter count is matched to q_serial within 10%.

Also closes out T1.13's waived Phase-1-gate criterion 3 (BENCH.md): c_rff_matched must run
error-free on all four PDEs -- previously heat/burgers/helmholtz failed with a matmul
shape-mismatch RuntimeError since c_rff_matched.yaml hardcoded a 1-D frequency list
against 2-D input.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import qapinn.models as models_pkg
from qapinn.config import ModelConfig
from qapinn.pdes.burgers import Burgers
from qapinn.pdes.heat import Heat
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson

ALL_PDES = [
    Poisson(alpha=0.05),
    Heat(alpha=0.3, nu=0.05),
    Helmholtz(k=10.0, a1=3.0, a2=1.0),
    Burgers(),
]


def test_p1_realised_frequencies_contains_target():
    pde = Poisson(alpha=0.05)
    model = models_pkg.build(ModelConfig(family="c_rff_matched"), input_dim=pde.dim, pde=pde)
    freqs = model.realised_frequencies().ravel()

    assert np.any(np.isclose(freqs, math.pi, atol=1e-6))
    assert np.any(np.isclose(freqs, 15.0 * math.pi, atol=1e-6))


def test_p1_param_count_matches_q_serial_within_10_percent():
    pde = Poisson(alpha=0.05)
    c_rff_matched = models_pkg.build(ModelConfig(family="c_rff_matched"), input_dim=pde.dim, pde=pde)
    q_serial = models_pkg.build(ModelConfig(family="q_serial"), input_dim=pde.dim, pde=pde)

    rel_diff = abs(c_rff_matched.n_params() - q_serial.n_params()) / q_serial.n_params()
    assert rel_diff <= 0.10, f"rel_diff={rel_diff}"


def test_c_rff_matched_runs_error_free_on_all_four_pdes():
    for pde in ALL_PDES:
        model = models_pkg.build(ModelConfig(family="c_rff_matched"), input_dim=pde.dim, pde=pde)
        out = model(torch.zeros(4, pde.dim))
        assert out.shape == (4, 1), pde.name


def test_explicit_frequencies_path_still_works():
    """Backward-compat: cfg.frequencies (pre-Phase-2 explicit-list path, T0.16) is still
    supported when no pde is given."""
    cfg = ModelConfig(family="c_rff_matched", frequencies=(math.pi, 15.0 * math.pi))
    model = models_pkg.build(cfg, input_dim=1)
    freqs = model.realised_frequencies().ravel()
    np.testing.assert_allclose(sorted(freqs), sorted([math.pi, 15.0 * math.pi]))
