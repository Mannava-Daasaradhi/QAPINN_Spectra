"""Model registry / factory (01_CONVENTIONS.md §7)."""
from __future__ import annotations

from qapinn.config import ModelConfig
from qapinn.models.base import PINNModel, match_param_count
from qapinn.models.mlp import MLPPINN

_REGISTRY: dict[str, type[PINNModel]] = {
    "c_mlp": MLPPINN,
}


def build(cfg: ModelConfig, input_dim: int) -> PINNModel:
    """Instantiate the model family named in cfg.family. input_dim comes from the
    associated PDE (pde.dim), since ModelConfig does not itself carry dimensionality."""
    if cfg.family == "c_mlp":
        return MLPPINN(input_dim=input_dim, widths=cfg.widths, activation=cfg.activation)
    raise ValueError(f"unknown/unregistered model family {cfg.family!r}; registered: {sorted(_REGISTRY)}")


__all__ = ["MLPPINN", "PINNModel", "build", "match_param_count"]
