"""P1 -- Poisson: -u'' = f on [0,1], u(0)=u(1)=0 (01_CONVENTIONS.md §5)."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from qapinn.pdes.base import PDE, Domain
from qapinn.pdes.diffops import laplacian


class Poisson(PDE):
    """u*(x) = sin(pi x) + alpha sin(15 pi x); f(x) = pi^2 sin(pi x) + alpha (15pi)^2 sin(15pi x)."""

    def __init__(self, alpha: float = 0.3):
        self.name = "poisson"
        self.domain = Domain(bounds=((0.0, 1.0),), names=("x",), time_axis=None)
        self.dim = 1
        self.params = {"alpha": alpha}

    def residual(self, x: Tensor, u: Tensor) -> Tensor:
        return -laplacian(u, x) - self.forcing(x)

    def forcing(self, x: Tensor) -> Tensor:
        alpha = self.params["alpha"]
        xx = x[:, 0:1]
        return (math.pi**2) * torch.sin(math.pi * xx) + alpha * (15 * math.pi) ** 2 * torch.sin(
            15 * math.pi * xx
        )

    def exact(self, x: Tensor) -> Tensor:
        alpha = self.params["alpha"]
        xx = x[:, 0:1]
        return torch.sin(math.pi * xx) + alpha * torch.sin(15 * math.pi * xx)

    def bc_lift(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x[:, 0:1])

    def bc_mask(self, x: Tensor) -> Tensor:
        xx = x[:, 0:1]
        return xx * (1.0 - xx)

    def boundary_residual(self, x_b: Tensor, u_b: Tensor) -> Tensor:
        return u_b - self.bc_lift(x_b)

    def symbol(self, omega: np.ndarray) -> np.ndarray:
        """-u'' <-> omega^2 * u_hat."""
        return (omega[:, 0] ** 2).astype(np.complex128)
