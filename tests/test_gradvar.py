"""T1.9 DoD: on a classical MLP the variance is roughly n-independent (slope ~= 0),
confirming the instrument does not manufacture a plateau where none exists.
"""
from __future__ import annotations

import torch

from qapinn.models.mlp import MLPPINN
from qapinn.pdes.poisson import Poisson
from qapinn.xai.gradvar import barren_plateau_fit, gradient_variance


def test_gradient_variance_classical_mlp_slope_near_zero():
    pde = Poisson(alpha=0.3)
    widths = [4, 8, 16, 32]
    results = []
    for w in widths:
        torch.manual_seed(0)
        model = MLPPINN(input_dim=1, widths=(w, w))
        gv = gradient_variance(model, pde, n_samples=40, group="classical")
        assert gv["n_qubits"] is None  # classical family, no circuit structure
        assert gv["var_mean"] >= 0.0
        results.append({"n_qubits": w, "var_mean": gv["var_mean"]})

    fit = barren_plateau_fit(results)
    # true barren-plateau reference is slope_b = log(2) ~= 0.693; a classical MLP at these
    # scales must sit nowhere near that -- assert well below half of it.
    assert abs(fit["slope_b"]) < 0.35


def test_gradient_variance_deterministic_given_fixed_seeds():
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=1, widths=(8,))
    gv1 = gradient_variance(model, pde, n_samples=20, group="classical")
    gv2 = gradient_variance(model, pde, n_samples=20, group="classical")
    assert gv1["var_mean"] == gv2["var_mean"]


def test_gradient_variance_empty_group_returns_zero():
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=1, widths=(4,))
    gv = gradient_variance(model, pde, n_samples=10, group="quantum")
    assert gv["var_mean"] == 0.0
    assert gv["var_per_param"] == []


def test_barren_plateau_fit_recovers_known_slope():
    import numpy as np

    rng = np.random.default_rng(0)
    ns = np.array([2, 4, 6, 8, 10, 12])
    true_b = 0.693
    var_means = np.exp(-true_b * ns) * np.exp(rng.normal(0, 0.01, size=ns.shape))
    results = [{"n_qubits": int(n), "var_mean": float(v)} for n, v in zip(ns, var_means)]

    fit = barren_plateau_fit(results)
    assert abs(fit["slope_b"] - true_b) < 0.05
    assert fit["r2"] > 0.99


def test_barren_plateau_fit_rejects_constant_n_qubits():
    # a single n_qubits value repeated across every (n_layers) point is exactly
    # depth_sweep.yaml's current grid -- the fit's independent variable is constant, so it
    # must refuse to report a slope rather than return a np.polyfit solver artifact.
    results = [{"n_qubits": 6, "var_mean": v} for v in (1.0, 0.5, 0.25, 0.1)]
    fit = barren_plateau_fit(results)
    assert fit["error"] == "degenerate_fit"
    assert "slope_b" not in fit
    assert fit["n_qubits_seen"] == [6.0]
