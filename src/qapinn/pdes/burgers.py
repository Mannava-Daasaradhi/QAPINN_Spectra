"""P3 -- Burgers: u_t + u*u_x = (0.01/pi) u_xx on [-1,1]x[0,1] (01_CONVENTIONS.md §5).

Nonlinear PDE: no closed-form exact() or symbol(). Ground truth comes from
qapinn.reference.reference_solution (T0.13), which dispatches to the Cole-Hopf quadrature
solution (T0.12) for this PDE, cross-checked against the pseudospectral solver (T0.11).
Target spectrum (T2.8) falls back to the empirical path (FFT of the reference solution) --
`target_spectrum_analytic` is left at the base-class default (returns None).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from qapinn.pdes.base import PDE, Domain
from qapinn.pdes.diffops import d1, d2


class Burgers(PDE):
    """u(x,0) = -sin(pi x); u(+-1,t) = 0; nu default 0.01/pi."""

    def __init__(self, nu: float = 0.01 / math.pi):
        self.name = "burgers"
        self.domain = Domain(bounds=((-1.0, 1.0), (0.0, 1.0)), names=("x", "t"), time_axis=1)
        self.dim = 2
        self.params = {"nu": nu}

    def residual(self, x: Tensor, u: Tensor) -> Tensor:
        nu = self.params["nu"]
        u_t = d1(u, x, 1)
        u_x = d1(u, x, 0)
        u_xx = d2(u, x, 0, 0)
        return u_t + u * u_x - nu * u_xx - self.forcing(x)

    def forcing(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x[:, 0:1])

    def exact(self, x: Tensor) -> Tensor:
        raise NotImplementedError(
            "Burgers has no closed-form exact() for general (x,t); use "
            "qapinn.reference.reference_solution(pde, grid) (T0.13), which dispatches to "
            "the Cole-Hopf quadrature solution (T0.12) for this PDE."
        )

    def bc_lift(self, x: Tensor) -> Tensor:
        xx = x[:, 0:1]
        return -torch.sin(math.pi * xx)

    def bc_mask(self, x: Tensor) -> Tensor:
        xx = x[:, 0:1]
        tt = x[:, 1:2]
        return tt * (1.0 - xx**2)

    def boundary_residual(self, x_b: Tensor, u_b: Tensor) -> Tensor:
        return u_b - self.bc_lift(x_b)

    def symbol(self, omega: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Burgers is nonlinear; no closed-form symbol exists (01_CONVENTIONS.md §5). "
            "SMCD (T2.8) uses the empirical target-spectrum path instead."
        )
