"""Model registry / factory (01_CONVENTIONS.md §7)."""
from __future__ import annotations

import torch

from qapinn.config import ModelConfig
from qapinn.models.base import PINNModel, match_param_count
from qapinn.models.fourier_features import FourierFeaturePINN, make_c_ff, make_c_rff_matched
from qapinn.models.hybrid import OctaveEnsemble, ParallelHybrid, SerialHybrid
from qapinn.models.mlp import MLPPINN

_REGISTRY: dict[str, type[PINNModel]] = {
    "c_mlp": MLPPINN,
    "c_ff": FourierFeaturePINN,
    "c_rff_matched": FourierFeaturePINN,
    "q_serial": SerialHybrid,
    "q_random": SerialHybrid,
    "q_parallel": ParallelHybrid,
    "q_octave": OctaveEnsemble,
}

_QUANTUM_FAMILIES = ("q_serial", "q_random", "q_parallel")


def _random_scalings(n_layers: int, n_qubits: int, mode: str, gen: torch.Generator | None) -> torch.Tensor:
    """q_random's ablation (C1's falsifier): SAME n, L, param count as q_serial's SMCD
    design, but scalings that ignore the design entirely -- proves any benefit comes from
    the DESIGN, not just from being quantum."""
    if mode == "unit":
        return torch.ones(n_layers, n_qubits)
    if mode == "random":
        return torch.rand(n_layers, n_qubits, generator=gen) * 2.5 + 0.5  # matches T2.5's draw range
    raise ValueError(f"q_random: unsupported scaling_mode {mode!r}, expected 'random' or 'unit'")


def build(
    cfg: ModelConfig,
    input_dim: int,
    gen: torch.Generator | None = None,
    pde=None,
    smcd_eps: float = 1e-3,
    smcd_coverage_target: float | None = None,
) -> PINNModel:
    """Instantiate the model family named in cfg.family. input_dim comes from the
    associated PDE (pde.dim), since ModelConfig does not itself carry dimensionality.
    `pde` (the actual PDE object, not just its dim) is required for the q_* families,
    which design their circuit against `qapinn.smcd.design.smcd(pde, ...)` -- SMCD needs
    the PDE's own target spectrum, not just a dimension count. `smcd_eps` /
    `smcd_coverage_target` mirror ExpConfig's own same-named fields (pre-placed for this
    by an earlier task, unwired until now).
    """
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

    if cfg.family in _QUANTUM_FAMILIES:
        if pde is None:
            raise ValueError(f"{cfg.family!r} requires `pde` (SMCD designs against the actual PDE)")
        from qapinn.smcd.design import smcd  # local import: avoids a models <-> smcd cycle

        card = smcd(pde, eps=smcd_eps, coverage_target=smcd_coverage_target)
        wire_to_dim = tuple(card.wire_to_dim)
        if cfg.family == "q_serial":
            return SerialHybrid(
                n_qubits=card.n_qubits,
                n_layers=card.n_layers,
                scalings=card.scalings,
                wire_to_dim=wire_to_dim,
                entangler=card.entangler,
                observable=card.observable,
            )
        if cfg.family == "q_random":
            scalings = _random_scalings(card.n_layers, card.n_qubits, cfg.scaling_mode or "random", gen)
            return SerialHybrid(
                n_qubits=card.n_qubits,
                n_layers=card.n_layers,
                scalings=scalings,
                wire_to_dim=wire_to_dim,
                entangler=card.entangler,
                observable=card.observable,
            )
        if cfg.family == "q_parallel":
            return ParallelHybrid(
                n_qubits=card.n_qubits,
                n_layers=card.n_layers,
                scalings=card.scalings,
                wire_to_dim=wire_to_dim,
                entangler=card.entangler,
                observable=card.observable,
                mlp_widths=cfg.widths,
                activation=cfg.activation,
            )

    if cfg.family == "q_octave":
        raise NotImplementedError(
            "q_octave needs T2.15's octave-split design (partitioning a target spectrum "
            "into octaves and constructing per-octave scalings) -- OctaveEnsemble itself "
            "is implemented (T2.12) and directly constructible from a hand-built "
            "circuit_configs list, but build() has nothing to wire it to yet."
        )

    raise ValueError(f"unknown/unregistered model family {cfg.family!r}; registered: {sorted(_REGISTRY)}")


__all__ = [
    "MLPPINN",
    "FourierFeaturePINN",
    "SerialHybrid",
    "ParallelHybrid",
    "OctaveEnsemble",
    "PINNModel",
    "build",
    "match_param_count",
]
