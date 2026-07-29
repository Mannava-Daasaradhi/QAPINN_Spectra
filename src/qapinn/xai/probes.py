"""Layerwise linear probes + CKA (T1.10, project.md Section 7.6).

Ridge regression is implemented in closed form (no new dependency) rather than pulling in
scikit-learn for one small, well-understood linear-algebra step.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

import qapinn.reference as reference_pkg
from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE

DEFAULT_RIDGE_ALPHA = 1e-6


def collect_layer_outputs(model: PINNModel, x: Tensor) -> dict[str, Tensor]:
    """One forward pass, capturing every named submodule's output via
    register_forward_hook, plus the raw input itself under the key 'input' (the DoD's
    "input layer" -- x before any transformation at all)."""
    outputs: dict[str, Tensor] = {"input": x.detach()}
    handles = []

    def _make_hook(name: str):
        def hook(_module, _inp, out):
            outputs[name] = out.detach()

        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    return outputs


def _ridge_r2(X: np.ndarray, y: np.ndarray, alpha: float = DEFAULT_RIDGE_ALPHA) -> float:
    """In-sample R^2 of a ridge regression of y on X, with an unregularised intercept."""
    X_ = np.concatenate([X, np.ones((X.shape[0], 1))], axis=-1)
    n_features = X_.shape[1]
    reg = alpha * np.eye(n_features)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(X_.T @ X_ + reg, X_.T @ y)
    y_pred = X_ @ w
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def layer_probe_r2(model: PINNModel, pde: PDE, grid: Tensor, layer_outputs: dict[str, Tensor]) -> dict[str, float]:
    """Ridge-regress u* on each layer's activations; R^2 per layer answers 'how much of
    the solution is already linearly decodable here?' Ground truth via
    qapinn.reference.reference_solution, NOT pde.exact() directly -- Burgers has no closed
    form and raises NotImplementedError from exact() by design (T0.13)."""
    u_exact = reference_pkg.reference_solution(pde, grid.detach().cpu().numpy()).reshape(-1)
    result = {}
    for name, activ in layer_outputs.items():
        X = activ.detach().cpu().numpy().reshape(activ.shape[0], -1)
        result[name] = _ridge_r2(X, u_exact)
    return result


def cka(X: Tensor, Y: Tensor) -> float:
    """Linear centred kernel alignment (Kornblith et al. 2019) between two
    representations, X:[B,p], Y:[B,q]."""
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    num = torch.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    denom = torch.linalg.norm(Xc.T @ Xc, ord="fro") * torch.linalg.norm(Yc.T @ Yc, ord="fro")
    return float((num / denom).item())
