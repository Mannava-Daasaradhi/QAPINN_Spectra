"""P2 -- Heat (NEGATIVE CONTROL): u_t = nu * u_xx on [0,1]x[0,1] (01_CONVENTIONS.md §5).

nu * (15*pi)^2 ~= 111, so the high-frequency mode is e^{-111} at t=1 -- dead. SMCD (T2.8)
must predict "no benefit" for this PDE before any training happens (contribution C4).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from qapinn.pdes.base import PDE, Domain
from qapinn.pdes.diffops import d1, d2


class Heat(PDE):
    """u(x,0) = sin(pi x) + alpha sin(15 pi x); u(0,t) = u(1,t) = 0; nu default 0.05."""

    def __init__(self, alpha: float = 0.3, nu: float = 0.05):
        self.name = "heat"
        self.domain = Domain(bounds=((0.0, 1.0), (0.0, 1.0)), names=("x", "t"), time_axis=1)
        self.dim = 2
        self.params = {"alpha": alpha, "nu": nu}

    def residual(self, x: Tensor, u: Tensor) -> Tensor:
        nu = self.params["nu"]
        u_t = d1(u, x, 1)
        u_xx = d2(u, x, 0, 0)
        return u_t - nu * u_xx - self.forcing(x)

    def forcing(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x[:, 0:1])

    def exact(self, x: Tensor) -> Tensor:
        alpha = self.params["alpha"]
        nu = self.params["nu"]
        xx = x[:, 0:1]
        tt = x[:, 1:2]
        return torch.sin(math.pi * xx) * torch.exp(-nu * math.pi**2 * tt) + alpha * torch.sin(
            15 * math.pi * xx
        ) * torch.exp(-nu * (15 * math.pi) ** 2 * tt)

    def bc_lift(self, x: Tensor) -> Tensor:
        alpha = self.params["alpha"]
        xx = x[:, 0:1]
        return torch.sin(math.pi * xx) + alpha * torch.sin(15 * math.pi * xx)

    def bc_mask(self, x: Tensor) -> Tensor:
        xx = x[:, 0:1]
        tt = x[:, 1:2]
        return tt * xx * (1.0 - xx)

    def boundary_residual(self, x_b: Tensor, u_b: Tensor) -> Tensor:
        return u_b - self.bc_lift(x_b)

    def symbol(self, omega: np.ndarray) -> np.ndarray:
        """Spatial symbol of the diffusion operator nu*d^2/dx^2 acting on e^{i omega x}:
        nu*(i omega)^2 = -nu*omega^2. Combined with u_t = symbol(omega)*u_hat in Fourier
        space this gives the per-mode decay rate (real, negative); T2.8 uses its
        magnitude to predict "no benefit" for modes with fast decay."""
        nu = self.params["nu"]
        return (-nu * omega[:, 0] ** 2).astype(np.complex128)
