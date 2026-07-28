"""PINNModel abstract base + size matching (01_CONVENTIONS.md §7)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor, nn

from qapinn.config import ModelConfig


class PINNModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """[B,d] -> [B,1]. This is the RAW network output N(x), BEFORE the hard-BC ansatz.
        The training loop applies pde.apply_hard_bc()."""

    @abstractmethod
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Keys are exactly {'classical', 'quantum'}. 'quantum' is [] for classical
        families. Required by the NTK block decomposition (D7) -- Prop. 4 is tested
        through this."""

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def realised_frequencies(self) -> np.ndarray | None:
        """Angular frequencies the model can currently represent, in PHYSICAL coordinates.
        Fourier-feature families: 2*pi*B rows. Quantum families: Omega*A where A is the
        affine encoder matrix (D3). None for c_mlp (the base-class default). Recomputed on
        demand -- it drifts during training."""
        return None


def match_param_count(family: str, target: int, tol: float = 0.10, **kw) -> ModelConfig:
    """Binary-search the MLP width (or n_features) so n_params lands within tol of target.
    Raises if unreachable. The target is set by the SMCD-designed q_serial model, so every
    family is matched TO the quantum model, not the other way round.

    Searches over REAL constructed models (via qapinn.models.build), not an estimated
    parameter-count formula, so the result can never drift from what the family actually
    builds. kw: input_dim (default 1), n_hidden_layers (default 3, matching
    ModelConfig.widths' default length).
    """
    import qapinn.models as models_pkg  # local import: avoids a base.py <-> models cycle

    input_dim = kw.get("input_dim", 1)
    n_hidden_layers = kw.get("n_hidden_layers", 3)

    def _cfg_for(width: int) -> ModelConfig:
        if family == "c_mlp":
            return ModelConfig(family=family, widths=(width,) * n_hidden_layers)
        if family in ("c_ff", "c_rff_matched"):
            return ModelConfig(family=family, n_features=width)
        raise ValueError(f"match_param_count: unsupported family {family!r}")

    def _n_params(width: int) -> int:
        model = models_pkg.build(_cfg_for(width), input_dim=input_dim)
        return model.n_params()

    lo, hi = 1, 1
    while _n_params(hi) < target:
        hi *= 2
        if hi > 10**7:
            raise ValueError(f"target {target} unreachable for family {family!r} (width search exceeded 1e7)")

    while lo < hi:
        mid = (lo + hi) // 2
        if _n_params(mid) < target:
            lo = mid + 1
        else:
            hi = mid

    candidates = range(max(1, lo - 2), lo + 3)
    best_width = min(candidates, key=lambda w: abs(_n_params(w) - target))
    n = _n_params(best_width)
    if abs(n - target) > tol * target:
        raise ValueError(
            f"target {target} unreachable for family {family!r} within tol={tol} "
            f"(closest: width={best_width}, n_params={n})"
        )

    return _cfg_for(best_width)
