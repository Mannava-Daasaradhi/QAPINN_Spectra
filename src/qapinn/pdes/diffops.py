"""Autograd differential operators (01_CONVENTIONS.md §6).

Never use finite differences anywhere in this project -- all three helpers use
torch.autograd.grad(..., create_graph=True, retain_graph=True) so higher-order derivatives
and parameter gradients both survive.
"""
from __future__ import annotations

import torch
from torch import Tensor


def d1(u: Tensor, x: Tensor, i: int) -> Tensor:
    """∂u/∂x_i. u:[B,1], x:[B,d] with requires_grad=True -> [B,1]."""
    (grad_x,) = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )
    return grad_x[:, i : i + 1]


def d2(u: Tensor, x: Tensor, i: int, j: int) -> Tensor:
    """∂²u/∂x_i∂x_j -> [B,1]"""
    du_di = d1(u, x, i)
    (grad_x,) = torch.autograd.grad(
        du_di, x, grad_outputs=torch.ones_like(du_di), create_graph=True, retain_graph=True
    )
    return grad_x[:, j : j + 1]


def laplacian(u: Tensor, x: Tensor, dims: tuple[int, ...] | None = None) -> Tensor:
    """Σ_i ∂²u/∂x_i² over dims (default: all spatial dims)."""
    if dims is None:
        dims = tuple(range(x.shape[1]))
    total: Tensor | None = None
    for i in dims:
        term = d2(u, x, i, i)
        total = term if total is None else total + term
    return total
