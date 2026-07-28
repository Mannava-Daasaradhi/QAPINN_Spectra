"""Model registry / factory (01_CONVENTIONS.md §7)."""
from __future__ import annotations

import torch

from qapinn.config import ModelConfig
from qapinn.models.base import PINNModel, match_param_count
from qapinn.models.fourier_features import FourierFeaturePINN, make_c_ff, make_c_rff_matched
from qapinn.models.mlp import MLPPINN

_REGISTRY: dict[str, type[PINNModel]] = {
    "c_mlp": MLPPINN,
    "c_ff": FourierFeaturePINN,
    "c_rff_matched": FourierFeaturePINN,
}


def build(cfg: ModelConfig, input_dim: int, gen: torch.Generator | None = None) -> PINNModel:
    """Instantiate the model family named in cfg.family. input_dim comes from the
    associated PDE (pde.dim), since ModelConfig does not itself carry dimensionality."""
    if cfg.family == "c_mlp":
        return MLPPINN(input_dim=input_dim, widths=cfg.widths, activation=cfg.activation)
    if cfg.family == "c_ff":
        return make_c_ff(input_dim=input_dim, n_features=cfg.n_features, ff_sigma=cfg.ff_sigma, gen=gen)
    if cfg.family == "c_rff_matched":
        if cfg.frequencies is None:
            raise ValueError(
                "c_rff_matched requires cfg.frequencies to be set (pre-Phase-2 placeholder, "
                "T2.14 wires the real design card)"
            )
        return make_c_rff_matched(cfg.frequencies)
    raise ValueError(f"unknown/unregistered model family {cfg.family!r}; registered: {sorted(_REGISTRY)}")


__all__ = ["MLPPINN", "FourierFeaturePINN", "PINNModel", "build", "match_param_count"]
