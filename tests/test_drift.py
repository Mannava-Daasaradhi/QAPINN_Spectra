"""T1.8 DoD: with A frozen at identity, frobenius == 0 and realised_omega == Omega. After
manually perturbing A by a known scale factor s, realised_omega == s*Omega.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from qapinn.models.base import PINNModel
from qapinn.models.fourier_features import make_c_ff
from qapinn.models.mlp import MLPPINN
from qapinn.xai.drift import encoder_drift


class _DummyEncoderModel(PINNModel):
    """Minimal stand-in for a Phase-2 quantum re-uploading model's affine encoder (D3):
    E(x) = A @ x, Omega fixed. Just enough structure to test encoder_drift() before the
    real quantum model exists -- exposes the same `encoder_A` / `encoder_omega` duck-typed
    protocol the real model will use."""

    def __init__(self, omega: torch.Tensor):
        super().__init__()
        d = omega.shape[-1]
        self.encoder_A = nn.Parameter(torch.eye(d))
        self.register_buffer("encoder_omega", omega)
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        z = x @ self.encoder_A.T
        return self.head(z)

    def param_groups(self):
        return {"classical": [], "quantum": [self.encoder_A]}

    def realised_frequencies(self):
        return (self.encoder_omega.detach() @ self.encoder_A.detach()).cpu().numpy()


def test_encoder_drift_classical_mlp_returns_empty():
    model = MLPPINN(input_dim=1, widths=(4,))
    assert encoder_drift(model) == {}


def test_encoder_drift_fourier_feature_family_reports_zero_drift_by_construction():
    model = make_c_ff(input_dim=1, n_features=5, ff_sigma=1.0, gen=torch.Generator().manual_seed(0))
    report = encoder_drift(model)

    assert report["frobenius"] == 0.0
    assert np.allclose(report["realised_omega"], model.realised_frequencies())
    assert np.isnan(report["coverage_now"])


def test_encoder_drift_identity_A_gives_zero_frobenius_and_matches_omega():
    omega = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    model = _DummyEncoderModel(omega)

    report = encoder_drift(model)
    assert report["frobenius"] == 0.0
    assert np.allclose(report["realised_omega"], omega.numpy())


def test_encoder_drift_perturbed_A_scales_realised_omega():
    omega = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    model = _DummyEncoderModel(omega)

    s = 3.5
    with torch.no_grad():
        model.encoder_A.copy_(s * torch.eye(2))

    report = encoder_drift(model)
    expected_frobenius = float(np.linalg.norm((s - 1.0) * np.eye(2), ord="fro"))
    assert abs(report["frobenius"] - expected_frobenius) < 1e-10
    assert np.allclose(report["realised_omega"], s * omega.numpy(), atol=1e-6)


def test_encoder_drift_coverage_with_target_spectrum():
    omega = torch.tensor([[1.0], [2.0], [3.0]])
    model = _DummyEncoderModel(omega)

    target = np.array([[1.0], [2.0], [99.0]])  # 2 of 3 targets are reachable at A=I
    report = encoder_drift(model, target_omega=target)
    assert abs(report["coverage_now"] - 2.0 / 3.0) < 1e-10
