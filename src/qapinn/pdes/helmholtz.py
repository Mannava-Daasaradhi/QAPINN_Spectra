"""P4 -- Helmholtz (HERO RESULT): Delta u + k^2 u = f on [-1,1]^2, u=0 on boundary
(01_CONVENTIONS.md §5). Requires cross-frequency terms omega_x +- omega_y -> entangler
ring_cz (Algorithm 1 step 7, T2.9)."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from qapinn.pdes.base import PDE, Domain
from qapinn.pdes.diffops import laplacian


class Helmholtz(PDE):
    """u*(x,y) = sin(a1 pi x) sin(a2 pi y); f = (k^2 - pi^2(a1^2+a2^2)) u*.

    (k, a1, a2) triples used across the matrix: (4,1,1), (10,3,1), (20,6,2) -- chosen so
    pi^2(a1^2+a2^2) sits near resonance with k^2.
    """

    def __init__(self, k: float, a1: float, a2: float):
        self.name = "helmholtz"
        self.domain = Domain(bounds=((-1.0, 1.0), (-1.0, 1.0)), names=("x", "y"), time_axis=None)
        self.dim = 2
        self.params = {"k": k, "a1": a1, "a2": a2}

    def residual(self, x: Tensor, u: Tensor) -> Tensor:
        k = self.params["k"]
        return laplacian(u, x) + (k**2) * u - self.forcing(x)

    def forcing(self, x: Tensor) -> Tensor:
        k = self.params["k"]
        a1 = self.params["a1"]
        a2 = self.params["a2"]
        u_star = self.exact(x)
        return (k**2 - math.pi**2 * (a1**2 + a2**2)) * u_star

    def exact(self, x: Tensor) -> Tensor:
        a1 = self.params["a1"]
        a2 = self.params["a2"]
        xx = x[:, 0:1]
        yy = x[:, 1:2]
        return torch.sin(a1 * math.pi * xx) * torch.sin(a2 * math.pi * yy)

    def bc_lift(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x[:, 0:1])

    def bc_mask(self, x: Tensor) -> Tensor:
        xx = x[:, 0:1]
        yy = x[:, 1:2]
        return (1.0 - xx**2) * (1.0 - yy**2)

    def boundary_residual(self, x_b: Tensor, u_b: Tensor) -> Tensor:
        return u_b - self.bc_lift(x_b)

    def symbol(self, omega: np.ndarray) -> np.ndarray:
        """Delta u + k^2 u <-> (k^2 - |omega|^2) u_hat."""
        k = self.params["k"]
        omega_sq = (omega**2).sum(axis=1)
        return (k**2 - omega_sq).astype(np.complex128)
