"""v2 showcase -- steady groundwater flow under canal-irrigated farmland (2026-09-28).

A confined aquifer lies between two rivers a distance L apart. Water enters it from
background recharge R0 (rain and irrigation return flow) and from seepage out of
n_canals canals running parallel to the rivers at equal spacing. Dupuit's steady 1-D
equation for the hydraulic head h (water-table height above datum, metres) is

    T h''(X) = -R(X),   0 <= X <= L,   h(0) = head_left,   h(L) = head_right,

with R(X) = R0 + sum_j q / (sigma sqrt(2 pi)) exp(-(X - c_j)^2 / (2 sigma^2)): each canal
leaks q m^3/day per metre of its length into a strip of width ~sigma around its centre
c_j = (j + 1/2) L / n_canals. On x = X / L the PINN solves -u'' = f with
f(x) = (L^2 / T) R(x L) and u = h in metres, the same interface as P1 (Poisson) with
non-zero boundary heads.

The exact solution is closed form: sigma (z Phi(z) + phi(z)), z = (X - c) / sigma, has
second derivative phi(z) / sigma, so every term of R integrates twice exactly.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from qapinn.pdes.base import PDE, Domain
from qapinn.pdes.diffops import laplacian

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


class Groundwater(PDE):
    def __init__(
        self,
        length_m: float = 2000.0,
        transmissivity: float = 100.0,
        recharge: float = 4.0e-4,
        canal_seepage: float = 0.4,
        n_canals: float = 6.0,
        canal_width_m: float = 30.0,
        head_left: float = 1.0,
        head_right: float = 0.0,
    ):
        if length_m <= 0 or transmissivity <= 0 or canal_width_m <= 0:
            raise ValueError("length_m, transmissivity and canal_width_m must be positive")
        if n_canals < 0 or n_canals != int(n_canals):
            raise ValueError(f"n_canals must be a non-negative whole number, got {n_canals}")
        self.name = "groundwater"
        self.domain = Domain(bounds=((0.0, 1.0),), names=("x",), time_axis=None)
        self.dim = 1
        self.params = {
            "length_m": length_m,
            "transmissivity": transmissivity,
            "recharge": recharge,
            "canal_seepage": canal_seepage,
            "n_canals": n_canals,
            "canal_width_m": canal_width_m,
            "head_left": head_left,
            "head_right": head_right,
        }

    # --- physical quantities (X in metres) --------------------------------
    def canal_centres_m(self) -> np.ndarray:
        n, L = int(self.params["n_canals"]), self.params["length_m"]
        return (np.arange(n) + 0.5) * L / n

    def recharge_m_per_day(self, X: np.ndarray) -> np.ndarray:
        p = self.params
        X = np.asarray(X, dtype=np.float64)
        total = np.full_like(X, p["recharge"])
        for c in self.canal_centres_m():
            z = (X - c) / p["canal_width_m"]
            total = total + p["canal_seepage"] / (p["canal_width_m"] * _SQRT2PI) * np.exp(-0.5 * z * z)
        return total

    def head_m(self, X: np.ndarray) -> np.ndarray:
        """Exact head h(X) in metres (closed form, see module docstring)."""
        from scipy.special import ndtr  # standard normal CDF

        p = self.params
        X = np.asarray(X, dtype=np.float64)
        sigma, T, L = p["canal_width_m"], p["transmissivity"], p["length_m"]

        def particular(Xv: np.ndarray) -> np.ndarray:
            # g'' = -R / T
            g = -p["recharge"] * Xv * Xv / 2.0
            for c in self.canal_centres_m():
                z = (Xv - c) / sigma
                g = g - p["canal_seepage"] * sigma * (z * ndtr(z) + np.exp(-0.5 * z * z) / _SQRT2PI)
            return g / T

        g0, gL = particular(np.array(0.0)), particular(np.array(L))
        a = p["head_left"] - g0
        b = (p["head_right"] - gL - a) / L
        return particular(X) + a + b * X

    # --- PDE interface on x = X / L ---------------------------------------
    def residual(self, x: Tensor, u: Tensor) -> Tensor:
        return -laplacian(u, x) - self.forcing(x)

    def forcing(self, x: Tensor) -> Tensor:
        p = self.params
        L, sigma = p["length_m"], p["canal_width_m"]
        X = x[:, 0:1] * L
        rate = torch.full_like(X, p["recharge"])
        for c in self.canal_centres_m():
            z = (X - float(c)) / sigma
            rate = rate + p["canal_seepage"] / (sigma * _SQRT2PI) * torch.exp(-0.5 * z * z)
        return (L * L / p["transmissivity"]) * rate

    def exact(self, x: Tensor) -> Tensor:
        X = x[:, 0:1].detach().cpu().double().numpy() * self.params["length_m"]
        return torch.as_tensor(self.head_m(X), dtype=x.dtype, device=x.device)

    def bc_lift(self, x: Tensor) -> Tensor:
        p = self.params
        xx = x[:, 0:1]
        return p["head_left"] + (p["head_right"] - p["head_left"]) * xx

    def bc_mask(self, x: Tensor) -> Tensor:
        xx = x[:, 0:1]
        return xx * (1.0 - xx)

    def boundary_residual(self, x_b: Tensor, u_b: Tensor) -> Tensor:
        return u_b - self.bc_lift(x_b)

    def symbol(self, omega: np.ndarray) -> np.ndarray:
        """-u'' <-> omega^2 * u_hat."""
        return (omega[:, 0] ** 2).astype(np.complex128)


def waterlogged_length_m(X: np.ndarray, head: np.ndarray, threshold_m: float) -> float:
    """Length of field (metres) where the water table is above `threshold_m`, by linear
    interpolation of the crossings on a sorted grid X."""
    X = np.asarray(X, dtype=np.float64)
    above = np.asarray(head, dtype=np.float64) - threshold_m
    total = 0.0
    for i in range(len(X) - 1):
        a, b = above[i], above[i + 1]
        dx = X[i + 1] - X[i]
        if a >= 0 and b >= 0:
            total += dx
        elif a >= 0 or b >= 0:
            total += dx * max(a, b) / (abs(a) + abs(b))
    return total
