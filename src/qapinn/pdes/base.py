"""PDE abstract base + Domain (01_CONVENTIONS.md §5)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Domain:
    bounds: tuple[tuple[float, float], ...]  # ((x_lo,x_hi), (t_lo,t_hi))
    names: tuple[str, ...]  # ("x",) | ("x","t") | ("x","y")
    time_axis: int | None  # index of the time coordinate, None if steady

    def __post_init__(self) -> None:
        if len(self.bounds) != len(self.names):
            raise ValueError(
                f"Domain.bounds has {len(self.bounds)} entries but names has "
                f"{len(self.names)}: {self.bounds!r} vs {self.names!r}"
            )
        for lo, hi in self.bounds:
            if not lo < hi:
                raise ValueError(f"Domain bound must satisfy lo < hi, got ({lo}, {hi})")
        if self.time_axis is not None and not (0 <= self.time_axis < len(self.names)):
            raise ValueError(
                f"time_axis={self.time_axis} out of range for {len(self.names)} dims"
            )


class PDE(ABC):
    """Abstract base for the four PDE instances (P1-P4). See 01_CONVENTIONS.md §5.

    Concrete subclasses set `name`, `domain`, `dim`, `params` in `__init__`.
    """

    name: str
    domain: Domain
    dim: int
    params: dict[str, float]

    # --- physics -------------------------------------------------------
    @abstractmethod
    def residual(self, x: Tensor, u: Tensor) -> Tensor:
        """x:[B,d] (requires_grad=True), u:[B,1] -> residual:[B,1].
        Uses diffops helpers; must call them with create_graph=True."""

    @abstractmethod
    def forcing(self, x: Tensor) -> Tensor:
        """[B,d] -> [B,1]"""

    @abstractmethod
    def exact(self, x: Tensor) -> Tensor:
        """[B,d] -> [B,1] ground truth"""

    # --- hard boundary ansatz  u = B(x) + D(x) * N(x) -------------------
    @abstractmethod
    def bc_lift(self, x: Tensor) -> Tensor:
        """B(x): [B,d] -> [B,1]"""

    @abstractmethod
    def bc_mask(self, x: Tensor) -> Tensor:
        """D(x): [B,d] -> [B,1], vanishes on the constrained set"""

    def apply_hard_bc(self, x: Tensor, n_out: Tensor) -> Tensor:
        return self.bc_lift(x) + self.bc_mask(x) * n_out

    # --- soft-BC variant (D6) -------------------------------------------
    @abstractmethod
    def boundary_residual(self, x_b: Tensor, u_b: Tensor) -> Tensor:
        """x_b:[B,d], u_b:[B,1] -> [B,1]"""

    # --- SMCD -----------------------------------------------------------
    @abstractmethod
    def symbol(self, omega: np.ndarray) -> np.ndarray:
        """Fourier symbol sigma(omega) of the linear (or linearised) operator.
        omega: [M,d] angular frequencies -> [M] complex."""

    def target_spectrum_analytic(self, eps: float) -> TargetSpectrum | None:  # noqa: F821
        """Return None if no closed form exists; SMCD then falls back to the empirical
        path (from the reference solution). Overridden per-PDE in T2.8; TargetSpectrum
        is defined in qapinn.smcd.card (T2.10)."""
        return None

    # --- sampling --------------------------------------------------------
    def sample_collocation(self, n: int, gen: torch.Generator) -> Tensor:
        """Uniform samples in the domain box, drawn from the explicit generator. [n, d]."""
        dtype = torch.get_default_dtype()
        lo = torch.tensor([b[0] for b in self.domain.bounds], dtype=dtype)
        hi = torch.tensor([b[1] for b in self.domain.bounds], dtype=dtype)
        u = torch.rand((n, self.dim), generator=gen, dtype=dtype)
        return lo + u * (hi - lo)

    def _sample_from_faces(
        self, n: int, gen: torch.Generator, faces: list[tuple[int, float]]
    ) -> Tensor:
        dtype = torch.get_default_dtype()
        lo_vec = torch.tensor([b[0] for b in self.domain.bounds], dtype=dtype)
        hi_vec = torch.tensor([b[1] for b in self.domain.bounds], dtype=dtype)
        u = torch.rand((n, self.dim), generator=gen, dtype=dtype)
        x = lo_vec + u * (hi_vec - lo_vec)

        face_idx = torch.randint(0, len(faces), (n,), generator=gen)
        for i, (axis, value) in enumerate(faces):
            mask = face_idx == i
            x[mask, axis] = value
        return x

    def _all_faces(self) -> list[tuple[int, float]]:
        faces: list[tuple[int, float]] = []
        for axis, (lo, hi) in enumerate(self.domain.bounds):
            if axis == self.domain.time_axis:
                faces.append((axis, lo))
            else:
                faces.append((axis, lo))
                faces.append((axis, hi))
        return faces

    def sample_boundary(self, n: int, gen: torch.Generator) -> Tensor:
        """Uniform samples on the constrained set: both endpoints of every non-time axis
        (the spatial boundary), plus only the lower endpoint of the time axis if one
        exists (the initial condition; the upper/final time is never constrained). This
        single rule reproduces the constrained set of all four PDE instances (P1: both
        endpoints of x; P2/P3: both endpoints of x + t=0; P4: all four edges). [n, d]."""
        return self._sample_from_faces(n, gen, self._all_faces())

    def sample_boundary_only(self, n: int, gen: torch.Generator) -> Tensor:
        """Pure spatial-boundary points: both endpoints of every non-time axis, EXCLUDING
        the initial-condition slice. Needed to keep the BC and IC loss blocks distinct in
        soft-BC mode (D6 -- the NTK loss-imbalance study, T3.6, treats residual/BC/IC as
        three separate blocks). [n, d]."""
        faces = [(axis, v) for axis, v in self._all_faces() if axis != self.domain.time_axis]
        return self._sample_from_faces(n, gen, faces)

    def sample_initial(self, n: int, gen: torch.Generator) -> Tensor | None:
        """Pure initial-condition points (time axis fixed at its lower bound), or None if
        the PDE is steady (no time axis -- there is no IC term for P1/P4). [n, d]."""
        if self.domain.time_axis is None:
            return None
        lo, _ = self.domain.bounds[self.domain.time_axis]
        faces = [(self.domain.time_axis, lo)]
        return self._sample_from_faces(n, gen, faces)

    def eval_grid(self, n: int) -> Tensor:
        """Deterministic tensor-product uniform grid, [n**d, d]. Excludes the duplicate
        endpoint on every axis (the grid FFT-based spectral analysis assumes: n samples
        over a half-open interval [lo, hi), matching numpy.fft's implicit periodicity)."""
        dtype = torch.get_default_dtype()
        axes = []
        for lo, hi in self.domain.bounds:
            step = (hi - lo) / n
            axes.append(torch.arange(n, dtype=dtype) * step + lo)
        grids = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([g.reshape(-1) for g in grids], dim=-1)
