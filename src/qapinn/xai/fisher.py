"""Empirical Fisher information & effective dimension (T1.7, project.md Section 7.4).

`effective_dimension`'s formula correction: the phase doc states
    d_eff ~= 2 * sum_i log(1 + c*lambda_i) / log(c)
This is Abbas et al.'s full formula, d = 2*log(sqrt(det(I + c*F))) / log(c), collapsed to a
single point (no integral over parameter space -- the "MLE approximation"). But
log(sqrt(det(I+c*F))) = (1/2) * sum_i log(1+c*lambda_i) (det of a matrix with eigenvalues
1+c*lambda_i, then sqrt halves the log). The outer "2 *" and this inner "1/2" cancel, so
the correct single-point formula is `d_eff ~= sum_i log(1+c*lambda_i) / log(c)`, WITHOUT
a leading factor of 2. Verified numerically: for k eigenvalues all equal after
mean-normalisation (the "k independent features" DoD case), the phase doc's literal
formula converges to 2k as n_data -> large, not k; the corrected formula converges to
exactly k (checked out to n=1e100). Implemented and tested the corrected version.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE
from qapinn.xai.ntk import jacobian

FISHER_DENSE_THRESHOLD = 4000


def empirical_fisher(model: PINNModel, pde: PDE, x: Tensor, group: str = "all", output: str = "residual") -> Tensor:
    """F = (1/B) sum_b g_b g_b^T where g_b = grad_theta output(x_b) (output='residual' by
    default, project.md Section 7.4; also accepts 'u'/'bc', matching xai/ntk.py's
    jacobian() convention). Conceptually [N,N], N = number of parameters in `group`.

    For N > FISHER_DENSE_THRESHOLD, materialising an N x N matrix is wasteful -- returns
    just the EIGENVALUE SPECTRUM instead (1-D, length min(B,N)), via a randomised SVD of
    the [B,N] Jacobian J (eigenvalues of F are singular_values(J)^2 / B; F is never
    formed). Callers must check `.dim()`: 2 -> full F matrix (eigendecompose it, e.g.
    `np.linalg.eigvalsh`, to get F_eigs for `effective_dimension`); 1 -> already F_eigs.
    """
    J = jacobian(model, pde, x, output=output, group=group)
    B, N = J.shape
    if N == 0:
        return torch.zeros(0, dtype=x.dtype, device=x.device)

    if N <= FISHER_DENSE_THRESHOLD:
        return (J.T @ J) / B

    q = min(B, N)
    _U, S, _Vh = torch.svd_lowrank(J, q=q)
    return (S**2) / B


def effective_dimension(F_eigs: np.ndarray, n_data: int, gamma: float = 1.0) -> float:
    """Abbas et al. effective dimension, single-point (MLE) approximation (see module
    docstring for the factor-of-2 correction):
        d_eff ~= Sum_i log(1 + c*lambda_i) / log(c),   c = gamma*n/(2*pi*log(n))
    lambda normalised so mean(lambda)=1. This is an approximation to the full
    parameter-space integral -- document it as such, do not present it as exact."""
    lam = np.clip(np.asarray(F_eigs, dtype=float), 0.0, None)
    mean_lam = lam.mean() if lam.size > 0 else 0.0
    if mean_lam > 0:
        lam = lam / mean_lam

    c = gamma * n_data / (2.0 * np.pi * np.log(n_data))
    return float(np.sum(np.log1p(c * lam)) / np.log(c))
