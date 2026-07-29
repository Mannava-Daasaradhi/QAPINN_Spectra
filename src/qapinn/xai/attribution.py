"""Integrated Gradients / collocation-point attribution (T1.6, project.md Section 7.3).

target: 'u' | 'residual' | 'bc', matching xai/ntk.py's jacobian() output convention --
'u' is the raw, unmasked model output (no hard-BC ansatz applied); 'residual' applies
apply_hard_bc first then pde.residual (needs create_graph=True through the SECOND-order
Laplacian, diffops.py); 'bc' is pde.boundary_residual on the raw output. Differentiating
'residual' or 'bc' w.r.t. x for IG is therefore a THIRD-order autograd chain through the
network -- valid because diffops.d1/d2 already use create_graph=True at every level
(01_CONVENTIONS.md Section 6), so the graph connecting the residual back to x survives an
extra outer torch.autograd.grad call.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import Tensor

from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE

DEFAULT_N_STEPS = 64
DEFAULT_CHUNK = 128


def _eval_target(model: PINNModel, pde: PDE, x: Tensor, target: str) -> Tensor:
    if target == "u":
        return model(x)
    if target == "bc":
        return pde.boundary_residual(x, model(x))
    if target == "residual":
        u_full = pde.apply_hard_bc(x, model(x))
        return pde.residual(x, u_full)
    raise ValueError(f"unknown target {target!r}; expected 'u', 'residual', or 'bc'")


def _domain_centre(pde: PDE, x: Tensor) -> Tensor:
    bounds = torch.tensor(pde.domain.bounds, dtype=x.dtype, device=x.device)  # [d,2]
    centre = bounds.mean(dim=-1)  # [d]
    return centre.unsqueeze(0).expand(x.shape[0], -1).clone()


def integrated_gradients(
    model: PINNModel,
    pde: PDE,
    x: Tensor,
    baseline: Tensor | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    target: str = "residual",
    chunk: int = DEFAULT_CHUNK,
) -> Tensor:
    """IG of the target scalar w.r.t. input coordinates. Baseline defaults to the domain
    centre. Returns [B, d] attributions."""
    if baseline is None:
        baseline = _domain_centre(pde, x)

    B = x.shape[0]
    out = torch.zeros_like(x)

    for start in range(0, B, chunk):
        end = min(start + chunk, B)
        x_c = x[start:end]
        b_c = baseline[start:end]

        grad_sum = torch.zeros_like(x_c)
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            x_interp = (b_c + alpha * (x_c - b_c)).clone().requires_grad_(True)
            F = _eval_target(model, pde, x_interp, target)
            (grad,) = torch.autograd.grad(F.sum(), x_interp, create_graph=False, retain_graph=False)
            grad_sum = grad_sum + grad

        out[start:end] = (x_c - b_c) * (grad_sum / n_steps)

    return out


def attribution_field(model: PINNModel, pde: PDE, grid_n: int = 128, **ig_kwargs) -> np.ndarray:
    """Sum of |attribution| across input dims on pde.eval_grid(grid_n), reshaped to the
    grid's natural [grid_n]*d shape."""
    device = next(model.parameters()).device
    grid = pde.eval_grid(grid_n).to(device)
    attrs = integrated_gradients(model, pde, grid, **ig_kwargs)
    field = attrs.detach().abs().sum(dim=-1).cpu().numpy()
    return field.reshape((grid_n,) * pde.dim)


def completeness_error(model: PINNModel, pde: PDE, x: Tensor, attrs: Tensor, target: str = "residual") -> float:
    """IG's completeness axiom: sum_i attrs_i ~= F(x) - F(baseline). Returns the relative
    violation, averaged over the batch. `target` must match whatever `attrs` was computed
    with (not exposed as a batch tensor, since the axiom is evaluated per-example against
    a scalar F). IG is known to be unstable -- we REPORT this number, we do not hide it."""
    baseline = _domain_centre(pde, x)
    x_ = x.clone().requires_grad_(True)
    baseline_ = baseline.clone().requires_grad_(True)
    F_x = _eval_target(model, pde, x_, target).detach().squeeze(-1)
    F_base = _eval_target(model, pde, baseline_, target).detach().squeeze(-1)

    diff = F_x - F_base
    attr_sum = attrs.sum(dim=-1)
    violation = (attr_sum - diff).abs()
    denom = diff.abs()
    rel = torch.where(denom > 1e-12, violation / denom, violation)
    return float(rel.mean().item())


def attribution_residual_correlation(attrs_field: np.ndarray, residual_field: np.ndarray) -> float:
    """Spearman correlation between the (flattened) attribution field and the true
    residual-magnitude field (project.md Section 7.3: report the correlation, not just
    the picture)."""
    corr, _p = spearmanr(attrs_field.reshape(-1), residual_field.reshape(-1))
    return float(corr)
