"""T1.7 DoD: for a linear model with k independent features, d_eff recovers k within 10%
as n_data -> large. Assert the Fisher eigenvalue spectrum is non-negative.

effective_dimension's formula: the phase doc's literal "2 * sum log(1+c*lambda)/log(c)"
converges to 2k (not k) as n_data -> large for k equal eigenvalues (verified numerically
before writing this test) -- see xai/fisher.py's module docstring for the derivation of
the corrected, factor-of-2-free formula used here.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from qapinn.models.base import PINNModel
from qapinn.models.mlp import MLPPINN
from qapinn.pdes.poisson import Poisson
from qapinn.xai.fisher import FISHER_DENSE_THRESHOLD, effective_dimension, empirical_fisher


class _SineFeatureModel(PINNModel):
    """u = w . [sqrt(2) sin(1 pi x), ..., sqrt(2) sin(k pi x)] -- a linear head over a
    FIXED orthonormal (on [0,1]) sine basis. With output='u', g_b = features(x_b)
    (independent of w), so F = (1/B) sum_b phi(x_b)phi(x_b)^T -> identity_k as the sample
    grid densifies, giving k eigenvalues that all -> 1 after mean-normalisation -- the
    textbook "k independent features" case the DoD asks for."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.head = nn.Linear(k, 1, bias=False)

    def _features(self, x):
        ks = torch.arange(1, self.k + 1, dtype=x.dtype)
        return math.sqrt(2.0) * torch.sin(ks * math.pi * x)

    def forward(self, x):
        return self.head(self._features(x))

    def param_groups(self):
        return {"classical": list(self.parameters()), "quantum": []}


def test_effective_dimension_recovers_k_for_linear_model():
    k = 5
    model = _SineFeatureModel(k)
    pde = Poisson(alpha=0.3)
    x = pde.eval_grid(2000)

    F = empirical_fisher(model, pde, x, output="u")
    assert F.shape == (k, k)

    eigs = np.linalg.eigvalsh(F.detach().numpy())
    assert np.all(eigs >= -1e-8)

    d_eff = effective_dimension(eigs, n_data=10**12)
    assert abs(d_eff - k) / k < 0.10


def test_effective_dimension_matches_k_for_various_feature_counts():
    pde = Poisson(alpha=0.3)
    for k in (2, 8):
        model = _SineFeatureModel(k)
        x = pde.eval_grid(2000)
        F = empirical_fisher(model, pde, x, output="u")
        eigs = np.linalg.eigvalsh(F.detach().numpy())
        d_eff = effective_dimension(eigs, n_data=10**12)
        assert abs(d_eff - k) / k < 0.10


def test_empirical_fisher_eigenvalues_non_negative_nonlinear_model():
    torch.manual_seed(0)
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=1, widths=(8, 8))
    x = pde.sample_collocation(64, torch.Generator().manual_seed(0))

    F = empirical_fisher(model, pde, x, output="residual")
    eigs = np.linalg.eigvalsh(F.detach().numpy())
    assert np.all(eigs >= -1e-6)


def test_empirical_fisher_large_N_uses_svd_path_and_matches_dense():
    # Force the SVD (eigenvalue-only) path by lowering the threshold, and cross-check its
    # eigenvalues against the exact dense (J^T @ J / B) path on the same small model --
    # the two computation strategies must agree.
    import qapinn.xai.fisher as fisher_mod

    torch.manual_seed(1)
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=1, widths=(8, 8))
    x = pde.sample_collocation(96, torch.Generator().manual_seed(1))

    F_dense = empirical_fisher(model, pde, x, output="residual")
    dense_eigs = np.sort(np.linalg.eigvalsh(F_dense.detach().numpy()))[::-1]

    original_threshold = fisher_mod.FISHER_DENSE_THRESHOLD
    try:
        fisher_mod.FISHER_DENSE_THRESHOLD = 1  # force the SVD path for this small model
        eigs_svd = empirical_fisher(model, pde, x, output="residual")
    finally:
        fisher_mod.FISHER_DENSE_THRESHOLD = original_threshold

    assert eigs_svd.dim() == 1
    svd_eigs_sorted = np.sort(eigs_svd.detach().numpy())[::-1]
    n_common = min(len(dense_eigs), len(svd_eigs_sorted))
    assert np.allclose(dense_eigs[:n_common], svd_eigs_sorted[:n_common], atol=1e-4, rtol=1e-3)
    assert np.all(svd_eigs_sorted >= -1e-6)


def test_empirical_fisher_shape_group_selection():
    torch.manual_seed(2)
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=1, widths=(8,))
    x = pde.sample_collocation(32, torch.Generator().manual_seed(2))

    n_params = model.n_params()
    F = empirical_fisher(model, pde, x, group="all", output="residual")
    assert F.shape == (n_params, n_params)


def test_effective_dimension_default_threshold_is_fisher_dense_threshold():
    assert FISHER_DENSE_THRESHOLD == 4000
