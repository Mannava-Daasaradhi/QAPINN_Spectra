"""Empirical NTK for a PINN (T1.2). Primary XAI instrument (project.md §7.1).

Two differentiation strategies, chosen by `output`:
- output='u': torch.func.jacrev + vmap over probe points (fully functorch-composable;
  'u' needs no PDE x-derivatives, exactly the fast path D7 calls for).
- output='residual'/'bc': per-probe-point classical torch.autograd.grad, looping over
  probe points (chunked, D7) and reusing ONE forward pass through pde.residual() /
  pde.boundary_residual() via retain_graph=True. torch.func transforms categorically
  disallow classical autograd.grad / .requires_grad_() calls inside a transformed
  function (verified directly: RuntimeError even for a single untransformed point via
  plain jacrev, no vmap involved) -- and pde.residual() needs create_graph=True
  second-order autograd internally for PINN training (diffops.py, T0.7). Reimplementing
  every PDE's residual formula a second time in terms of nested torch.func.jacrev calls
  would duplicate physics in two places with a real risk of the two silently diverging.
  This path is chunked for memory and its correctness is cross-checked in tests against
  the classical training-time residual computation on the same points.

Prop. 4 correction: the phase doc states Theta_hyb == Theta_cl + Theta_q + 2*sym(Theta_cross).
This is NOT correct in general -- verified both analytically (index summation over a
block-concatenated Jacobian) and numerically (random matrices, matched and mismatched
column counts): Theta_hyb == Theta_cl + Theta_q EXACTLY, with no cross contribution,
because Theta_hyb = J_full @ J_full.T where J_full = [J_cl | J_q] is a horizontal
concatenation along the PARAMETER axis, and column-block concatenation makes cross terms
vanish identically in the Gram matrix (every entry is a sum over parameter columns, and
each column belongs to exactly one group). 'cross' is retained below purely as an optional
diagnostic (parameter-space alignment between the two Jacobian blocks) -- it does not
enter Theta_hyb's decomposition, and is left as None when the two groups have different
parameter counts (J_cl @ J_q.T is only shape-valid when they match).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE

DEFAULT_CHUNK = 64
PROBE_SIZE = 512


def _group_params(model: PINNModel, group: str) -> list[torch.nn.Parameter]:
    if group == "all":
        return list(model.parameters())
    groups = model.param_groups()
    if group not in ("classical", "quantum"):
        raise ValueError(f"unknown group {group!r}; expected 'all', 'classical', or 'quantum'")
    return groups[group]


def _group_named_params(model: PINNModel, group: str) -> dict[str, torch.nn.Parameter]:
    all_named = dict(model.named_parameters())
    if group == "all":
        return all_named
    selected_ids = {id(p) for p in _group_params(model, group)}
    return {name: p for name, p in all_named.items() if id(p) in selected_ids}


def _jacobian_u_functorch(model: PINNModel, probe_x: Tensor, group: str) -> Tensor:
    group_named = _group_named_params(model, group)
    other_named = {k: v.detach() for k, v in model.named_parameters() if k not in group_named}
    P = probe_x.shape[0]

    if not group_named:
        return torch.zeros(P, 0, dtype=probe_x.dtype, device=probe_x.device)

    def u_fn(params_subset: dict, x_single: Tensor) -> Tensor:
        full = {**other_named, **params_subset}
        u = torch.func.functional_call(model, full, (x_single.unsqueeze(0),))
        return u.squeeze()

    jac_fn = torch.func.jacrev(u_fn, argnums=0)
    J_dict = torch.func.vmap(jac_fn, in_dims=(None, 0))(group_named, probe_x)
    flat_parts = [J_dict[name].reshape(P, -1) for name in group_named]
    return torch.cat(flat_parts, dim=-1)


def _jacobian_pde_classical(
    model: PINNModel, pde: PDE, probe_x: Tensor, output: str, group: str, chunk: int
) -> Tensor:
    params = _group_params(model, group)
    P = probe_x.shape[0]
    n_params = sum(p.numel() for p in params)
    J = torch.zeros(P, n_params, dtype=probe_x.dtype, device=probe_x.device)
    if n_params == 0:
        return J

    for start in range(0, P, chunk):
        end = min(start + chunk, P)
        x_chunk = probe_x[start:end].clone().requires_grad_(True)
        u_raw = model(x_chunk)
        if output == "bc":
            target = pde.boundary_residual(x_chunk, u_raw)
        else:
            u_full = pde.apply_hard_bc(x_chunk, u_raw)
            target = pde.residual(x_chunk, u_full)

        for local_i in range(end - start):
            grads = torch.autograd.grad(
                target[local_i], params, retain_graph=True, create_graph=False, allow_unused=True
            )
            flat = torch.cat(
                [
                    g.reshape(-1) if g is not None else torch.zeros(p.numel(), dtype=probe_x.dtype)
                    for g, p in zip(grads, params)
                ]
            )
            J[start + local_i] = flat

    return J


def jacobian(
    model: PINNModel, pde: PDE, probe_x: Tensor, output: str, group: str = "all", chunk: int = DEFAULT_CHUNK
) -> Tensor:
    """output: 'u' | 'residual' | 'bc'. group: 'all' | 'classical' | 'quantum'
    (uses model.param_groups()). returns J: [P, N_params_in_group]."""
    if output == "u":
        return _jacobian_u_functorch(model, probe_x, group)
    if output in ("residual", "bc"):
        return _jacobian_pde_classical(model, pde, probe_x, output, group, chunk)
    raise ValueError(f"unknown output {output!r}; expected 'u', 'residual', or 'bc'")


def ntk(model: PINNModel, pde: PDE, probe_x: Tensor, output: str = "residual", group: str = "all") -> Tensor:
    J = jacobian(model, pde, probe_x, output=output, group=group)
    return J @ J.T


def ntk_blocks(model: PINNModel, pde: PDE, probe_x: Tensor) -> dict[str, Tensor | None]:
    """{'hyb': Theta, 'cl': J_cl J_cl^T, 'q': J_q J_q^T, 'cross': J_cl J_q^T or None}.
    Theta ('hyb') == cl + q EXACTLY (see module docstring re: the phase doc's erroneous
    +2*sym(cross) term)."""
    J_cl = jacobian(model, pde, probe_x, output="residual", group="classical")
    J_q = jacobian(model, pde, probe_x, output="residual", group="quantum")

    theta_cl = J_cl @ J_cl.T
    theta_q = J_q @ J_q.T if J_q.shape[-1] > 0 else torch.zeros_like(theta_cl)
    theta_hyb = theta_cl + theta_q

    cross = J_cl @ J_q.T if (J_cl.shape[-1] > 0 and J_cl.shape[-1] == J_q.shape[-1]) else None

    return {"hyb": theta_hyb, "cl": theta_cl, "q": theta_q, "cross": cross}


def spectrum_stats(K: Tensor) -> dict:
    """{'eigenvalues': [...], 'decay_exponent': float, 'condition_number': float,
    'trace': float, 'effective_rank': float}. decay_exponent: slope of a least-squares
    line fit to log(lambda_i) vs log(i) over the middle 80% of the (sorted descending)
    index range (drop the first and last 10%)."""
    K_np = K.detach().cpu().numpy()
    eigs = np.linalg.eigvalsh((K_np + K_np.T) / 2.0)
    eigs = np.sort(eigs)[::-1]  # descending
    eigs_clipped = np.clip(eigs, a_min=0.0, a_max=None)

    n = len(eigs)
    lo = max(1, int(0.1 * n))
    hi = min(n, int(0.9 * n))
    idx = np.arange(lo, hi) + 1  # 1-indexed for log(i)
    vals = eigs_clipped[lo:hi]
    positive = vals > 0
    if positive.sum() >= 2:
        slope, _intercept = np.polyfit(np.log(idx[positive]), np.log(vals[positive]), 1)
        decay_exponent = float(slope)
    else:
        decay_exponent = float("nan")

    trace = float(np.sum(eigs))
    max_eig = float(eigs[0]) if n > 0 else 0.0
    min_eig_positive = float(eigs_clipped[eigs_clipped > 0].min()) if np.any(eigs_clipped > 0) else float("nan")
    condition_number = max_eig / min_eig_positive if min_eig_positive not in (0.0, float("nan")) else float("inf")
    effective_rank = float((trace**2) / np.sum(eigs**2)) if np.sum(eigs**2) > 0 else 0.0

    return {
        "eigenvalues": eigs.tolist(),
        "decay_exponent": decay_exponent,
        "condition_number": condition_number,
        "trace": trace,
        "effective_rank": effective_rank,
    }


def block_mass(model: PINNModel, pde: PDE, probe_r: Tensor, probe_b: Tensor) -> dict:
    """Per-loss-block eigenvalue mass: trace(Theta_rr), trace(Theta_bb), trace(Theta_rb).
    Only meaningful under bc_mode='soft' (D6): the residual and boundary losses only both
    exist as separate terms in that mode."""
    J_r = jacobian(model, pde, probe_r, output="residual", group="all")
    J_b = jacobian(model, pde, probe_b, output="bc", group="all")

    theta_rr = J_r @ J_r.T
    theta_bb = J_b @ J_b.T
    theta_rb = J_r @ J_b.T

    return {
        "trace_rr": float(torch.trace(theta_rr).item()),
        "trace_bb": float(torch.trace(theta_bb).item()),
        "trace_rb": float(torch.trace(theta_rb).item()) if theta_rb.shape[0] == theta_rb.shape[1] else None,
    }


def ntk_drift(K_t: Tensor, K_0: Tensor) -> float:
    """||Theta_t - Theta_0||_F / ||Theta_0||_F."""
    diff_norm = torch.linalg.norm(K_t - K_0, ord="fro")
    base_norm = torch.linalg.norm(K_0, ord="fro")
    return float((diff_norm / base_norm).item())
