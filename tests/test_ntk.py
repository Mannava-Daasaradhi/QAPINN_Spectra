"""T1.2 DoD:
1. Analytic check: for a linear model u = w.phi(x) with fixed features phi, NTK == phi(x)phi(x')^T
   exactly (to 1e-10).
2. Prop. 4 identity (CORRECTED -- see xai/ntk.py's module docstring for the derivation):
   Theta_hyb == Theta_cl + Theta_q to 1e-10 on a dummy two-group model.
3. PSD: Theta_cl and Theta_q have no eigenvalue below -1e-10.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from qapinn.models.base import PINNModel
from qapinn.models.fourier_features import FourierFeatures
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson
from qapinn.seeding import set_global_seed
from qapinn.xai.ntk import (
    block_mass,
    jacobian,
    ntk,
    ntk_blocks,
    ntk_drift,
    probe_set,
    save_ntk_report,
    spectrum_stats,
)


class _LinearFeatureModel(PINNModel):
    """u = w . phi(x), phi fixed (no bias) -- the textbook linear-model NTK case."""

    def __init__(self, B):
        super().__init__()
        self.features = FourierFeatures(B)
        self.head = nn.Linear(2 * self.features.n_features, 1, bias=False)

    def forward(self, x):
        return self.head(self.features(x))

    def param_groups(self):
        return {"classical": list(self.parameters()), "quantum": []}


class _TwoGroupModel(PINNModel):
    """Dummy two-parameter-group classical model for the Prop. 4 test (the real hybrid
    model arrives in Phase 2)."""

    def __init__(self, input_dim=1):
        super().__init__()
        self.a = nn.Linear(input_dim, 8)
        self.b = nn.Linear(8, 1)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x)))

    def param_groups(self):
        return {"classical": list(self.a.parameters()), "quantum": list(self.b.parameters())}


def test_ntk_analytic_check_linear_model():
    B = torch.tensor([[math.pi], [15 * math.pi], [3.0]])
    model = _LinearFeatureModel(B)
    pde = Poisson(alpha=0.3)

    x_probe = torch.linspace(0.0, 1.0, 20).reshape(-1, 1)
    K = ntk(model, pde, x_probe, output="u", group="all")

    with torch.no_grad():
        phi = model.features(x_probe)
    K_expected = phi @ phi.T

    assert torch.allclose(K, K_expected, atol=1e-10)


def test_jacobian_u_matches_manual_gradient():
    B = torch.tensor([[math.pi], [3.0]])
    model = _LinearFeatureModel(B)
    pde = Poisson(alpha=0.3)
    x_probe = torch.tensor([[0.2], [0.7]])

    J = jacobian(model, pde, x_probe, output="u", group="all")

    with torch.no_grad():
        phi = model.features(x_probe)
    assert torch.allclose(J, phi, atol=1e-10)


def test_prop4_identity_corrected():
    model = _TwoGroupModel()
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(0)
    x_probe = pde.sample_collocation(16, gen)
    x_probe.requires_grad_(False)

    blocks = ntk_blocks(model, pde, x_probe)

    assert torch.allclose(blocks["hyb"], blocks["cl"] + blocks["q"], atol=1e-10)


def test_prop4_identity_real_hybrid_model():
    """T2.18 gate: this identity was only ever exercised against _TwoGroupModel above (a
    dummy stand-in noted at T1.2 as "the real hybrid model arrives in Phase 2") --
    re-run now that SerialHybrid exists. This is not purely definitional the way it looks
    (theta_hyb = theta_cl + theta_q by construction in ntk_blocks): the dummy model's
    "quantum" group was plain nn.Linear, exercised by ordinary autograd only. A real
    SerialHybrid's quantum group backpropagates through ReuploadCircuit's parameter-shift
    custom backward (pshift.py, T2.7) -- this confirms jacobian()'s per-group
    autograd.grad extraction still produces correctly shaped, non-degenerate rows through
    that different differentiation path, not just that the addition is self-consistent."""
    from qapinn.models.hybrid import SerialHybrid

    model = SerialHybrid(n_qubits=2, n_layers=1, scalings=[[1.0, 2.0]], wire_to_dim=(0, 0), input_dim=1)
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(7)
    x_probe = pde.sample_collocation(16, gen)
    x_probe.requires_grad_(False)

    blocks = ntk_blocks(model, pde, x_probe)

    assert torch.allclose(blocks["hyb"], blocks["cl"] + blocks["q"], atol=1e-10)
    assert blocks["cl"].shape == (16, 16) and blocks["q"].shape == (16, 16)
    assert blocks["cl"].abs().sum() > 0  # encoder + head params actually contribute
    assert blocks["q"].abs().sum() > 0  # quantum circuit params actually contribute


def test_ntk_blocks_psd():
    model = _TwoGroupModel()
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(1)
    x_probe = pde.sample_collocation(16, gen)

    blocks = ntk_blocks(model, pde, x_probe)

    eigs_cl = np.linalg.eigvalsh(blocks["cl"].detach().numpy())
    eigs_q = np.linalg.eigvalsh(blocks["q"].detach().numpy())

    assert eigs_cl.min() > -1e-10
    assert eigs_q.min() > -1e-10


def test_jacobian_residual_matches_classical_backward_sum():
    """Cross-check: sum of the per-point residual Jacobian rows equals the classical
    .backward() gradient of the summed residual -- confirms the per-point autograd.grad
    loop (used for output='residual'/'bc' since torch.func can't compose with pde.residual's
    classical create_graph=True second derivatives) is extracting genuine per-sample rows,
    not some aggregated or duplicated quantity."""
    model = _TwoGroupModel()
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(2)
    x_probe = pde.sample_collocation(10, gen)

    J = jacobian(model, pde, x_probe, output="residual", group="all")

    x_grad = x_probe.clone().requires_grad_(True)
    u = pde.apply_hard_bc(x_grad, model(x_grad))
    residual = pde.residual(x_grad, u)
    model.zero_grad()
    residual.sum().backward()
    manual_grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])

    assert torch.allclose(J.sum(dim=0), manual_grad, atol=1e-9)


def test_jacobian_residual_chunking_matches_unchunked():
    model = _TwoGroupModel(input_dim=2)
    pde = Helmholtz(k=10.0, a1=3.0, a2=1.0)
    gen = set_global_seed(3)
    x_probe = pde.sample_collocation(20, gen)

    J_chunk_small = jacobian(model, pde, x_probe, output="residual", group="all", chunk=4)
    J_chunk_large = jacobian(model, pde, x_probe, output="residual", group="all", chunk=64)

    assert torch.allclose(J_chunk_small, J_chunk_large, atol=1e-12)


def test_spectrum_stats_shape_and_signs():
    model = _TwoGroupModel()
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(4)
    x_probe = pde.sample_collocation(32, gen)

    K = ntk(model, pde, x_probe, output="residual", group="all")
    stats = spectrum_stats(K)

    assert len(stats["eigenvalues"]) == 32
    assert stats["trace"] > 0
    assert stats["condition_number"] >= 1.0
    assert 0.0 < stats["effective_rank"] <= 32.0


def test_block_mass_returns_traces():
    model = _TwoGroupModel()
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(5)
    probe_r = pde.sample_collocation(16, gen)
    probe_b = pde.sample_boundary_only(16, gen)

    result = block_mass(model, pde, probe_r, probe_b)

    assert result["trace_rr"] > 0
    assert result["trace_bb"] > 0


def test_ntk_drift_zero_for_identical_kernels():
    model = _TwoGroupModel()
    pde = Poisson(alpha=0.3)
    gen = set_global_seed(6)
    x_probe = pde.sample_collocation(16, gen)

    K = ntk(model, pde, x_probe, output="residual", group="all")
    assert ntk_drift(K, K) == 0.0


def test_jacobian_invalid_output_raises():
    import pytest

    model = _TwoGroupModel()
    pde = Poisson(alpha=0.3)
    x_probe = torch.rand(4, 1)
    with pytest.raises(ValueError):
        jacobian(model, pde, x_probe, output="not_a_real_output")


def test_jacobian_invalid_group_raises():
    import pytest

    model = _TwoGroupModel()
    pde = Poisson(alpha=0.3)
    x_probe = torch.rand(4, 1)
    with pytest.raises(ValueError):
        jacobian(model, pde, x_probe, output="u", group="not_a_real_group")


def test_probe_set_deterministic_and_correct_size():
    pde = Poisson(alpha=0.3)
    p1 = probe_set(pde, n_probe=64)
    p2 = probe_set(pde, n_probe=64)

    assert p1.shape == (64, 1)
    assert torch.equal(p1, p2)


def test_probe_set_works_for_2d_pde():
    pde = Helmholtz(k=10.0, a1=3.0, a2=1.0)
    p = probe_set(pde, n_probe=64)
    assert p.shape == (64, 2)


def test_save_ntk_report_writes_npz_and_computes_drift(tmp_path):
    model = _TwoGroupModel()
    pde = Poisson(alpha=0.3)
    probe_x = probe_set(pde, n_probe=32)

    report_0 = save_ntk_report(model, pde, probe_x, step=0, run_dir=tmp_path)
    assert (tmp_path / "xai" / "ntk_step0.npz").is_file()
    assert report_0["drift"] == 0.0

    with torch.no_grad():
        model.a.weight += 0.1  # perturb so step-1's NTK differs from step-0's

    report_1 = save_ntk_report(model, pde, probe_x, step=1, run_dir=tmp_path, K_0=report_0["K"])
    assert (tmp_path / "xai" / "ntk_step1.npz").is_file()
    assert report_1["drift"] > 0.0

    saved = np.load(tmp_path / "xai" / "ntk_step1.npz")
    assert len(saved["eigenvalues"]) == 32
    assert float(saved["drift"]) == report_1["drift"]
