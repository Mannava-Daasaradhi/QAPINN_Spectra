"""Gradient variance / barren-plateau tracking (T1.9, project.md Section 7.5).

`gradient_variance` re-initialises `group`'s parameters `n_samples` times and measures the
variance of the residual-loss gradient on a FIXED batch -- fixed once per call so
re-initialisation noise, not batch noise, is what gets measured. No real quantum circuit
family exists yet (Phase 2, T2.x), so re-initialisation uses a generic heuristic (Xavier-
normal for >=2-D weight tensors, matching every classical family's own init scheme;
uniform(-pi, pi) for 1-D tensors, matching the standard quantum-circuit-angle convention)
rather than each model's own (currently nonexistent, for quantum) init routine.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE

DEFAULT_BATCH_SIZE = 256
DEFAULT_BATCH_SEED = 0


def _select_params(model: PINNModel, group: str) -> list[nn.Parameter]:
    if group == "all":
        return list(model.parameters())
    groups = model.param_groups()
    if group not in ("classical", "quantum"):
        raise ValueError(f"unknown group {group!r}; expected 'all', 'classical', or 'quantum'")
    return groups[group]


def _reinit_param(p: nn.Parameter, gen: torch.Generator) -> None:
    with torch.no_grad():
        if p.dim() >= 2:
            nn.init.xavier_normal_(p, generator=gen)
        else:
            p.uniform_(-math.pi, math.pi, generator=gen)


def gradient_variance(
    model: PINNModel, pde: PDE, n_samples: int = 100, group: str = "quantum"
) -> dict:
    """Re-initialise `group`'s parameters `n_samples` times from the init distribution,
    compute grad_theta L each time on a FIXED batch (L = mean squared PDE residual, hard
    BC applied). Returns {'var_mean': mean over coordinates of Var[d_theta L],
    'var_per_param': [...], 'n_qubits': n, 'n_layers': L} -- n_qubits/n_layers are read
    from the model if present (quantum families, Phase 2), else None (classical)."""
    params = _select_params(model, group)
    data_gen = torch.Generator().manual_seed(DEFAULT_BATCH_SEED)
    x = pde.sample_collocation(DEFAULT_BATCH_SIZE, data_gen)

    if not params:
        return {"var_mean": 0.0, "var_per_param": [], "n_qubits": getattr(model, "n_qubits", None),
                "n_layers": getattr(model, "n_layers", None)}

    reinit_gen = torch.Generator().manual_seed(DEFAULT_BATCH_SEED + 1)
    grads_per_sample = []
    for _ in range(n_samples):
        for p in params:
            _reinit_param(p, reinit_gen)

        x_c = x.clone().requires_grad_(True)
        u_full = pde.apply_hard_bc(x_c, model(x_c))
        residual = pde.residual(x_c, u_full)
        loss = (residual**2).mean()

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        flat = torch.cat(
            [g.reshape(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, params)]
        )
        grads_per_sample.append(flat)

    G = torch.stack(grads_per_sample, dim=0)  # [n_samples, N]
    var_per_param = G.var(dim=0, unbiased=True)

    return {
        "var_mean": float(var_per_param.mean().item()),
        "var_per_param": var_per_param.detach().cpu().numpy().tolist(),
        "n_qubits": getattr(model, "n_qubits", None),
        "n_layers": getattr(model, "n_layers", None),
    }


def barren_plateau_fit(results: list[dict]) -> dict:
    """Fit log(var_mean) = a - b*n_qubits by least squares. Returns {'slope_b', 'r2'}.
    Theoretical prediction for a global-cost, deep random circuit is b = log(2) (Var ~
    2^-n); we compare our local-observable, shallow circuits against it -- the expectation
    is to sit well ABOVE that line (small or near-zero b), which is the point of the
    cost-ledger section."""
    ns = np.array([r["n_qubits"] for r in results], dtype=float)
    var_means = np.array([r["var_mean"] for r in results], dtype=float)
    log_var = np.log(np.clip(var_means, 1e-300, None))

    slope, intercept = np.polyfit(ns, log_var, 1)  # log_var ~= intercept + slope*n
    b = -slope
    pred = intercept + slope * ns
    ss_res = float(np.sum((log_var - pred) ** 2))
    ss_tot = float(np.sum((log_var - log_var.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"slope_b": float(b), "r2": r2}
