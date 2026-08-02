"""Model registry / factory (01_CONVENTIONS.md §7)."""
from __future__ import annotations

import numpy as np
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


def _pad_time_column(frequencies: np.ndarray, pde) -> np.ndarray:
    """`qapinn.smcd.symbol.TargetSpectrum` deliberately omits the time axis for
    time-dependent PDEs (P2/P3): its weight is already the TIME-INTEGRATED
    ||u_hat(omega,.)||_{L2(0,T)} amplitude (T2.8), so `omega` only has one column per
    SPATIAL dimension. `FourierFeaturePINN`'s `B` must have one column per input
    dimension (`pde.dim`, including time), or `x @ B.T` shape-mismatches (the exact bug
    T1.13 deferred to this task: c_rff_matched previously failed on heat/burgers/
    helmholtz with a 2-D input against a 1-D frequency list). Insert a zero column at
    `pde.domain.time_axis` -- a genuinely zero TEMPORAL frequency for a feature whose
    amplitude already encodes the time dependence some other way, not a placeholder."""
    if frequencies.shape[1] >= pde.dim:
        return frequencies
    time_axis = pde.domain.time_axis
    if time_axis is None:
        raise ValueError(
            f"_pad_time_column: frequencies has {frequencies.shape[1]} columns but "
            f"pde.dim={pde.dim} and the PDE has no time axis to pad -- unexpected "
            f"dimensionality mismatch, not something padding can fix"
        )
    return np.insert(frequencies, time_axis, 0.0, axis=1)


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
        if pde is not None:
            from qapinn.smcd.card import matched_and_padded_frequencies  # avoids a models <-> smcd cycle
            from qapinn.smcd.design import smcd

            card = smcd(pde, eps=smcd_eps, coverage_target=smcd_coverage_target)
            q_serial = SerialHybrid(
                n_qubits=card.n_qubits,
                n_layers=card.n_layers,
                scalings=card.scalings,
                wire_to_dim=tuple(card.wire_to_dim),
                entangler=card.entangler,
                observable=card.observable,
                input_dim=pde.dim,
            )
            # T2.14 DoD: realised_frequencies() must CONTAIN the target support AND the
            # param count must match q_serial within 10% -- the bare target support alone
            # is typically far too small (P1: 2 frequencies -> 5 params vs q_serial's 14),
            # so pad with additional (nearest-to-DC) Omega rows up to q_serial's actual
            # measured n_params(), not an estimated formula (match_param_count's own
            # established principle).
            frequencies = matched_and_padded_frequencies(card, target_param_count=q_serial.n_params())
            frequencies = _pad_time_column(frequencies, pde)
            return make_c_rff_matched(frequencies)
        if cfg.frequencies is None:
            raise ValueError(
                "c_rff_matched requires either `pde` (T2.14: derives B from the real "
                "SMCD design card) or cfg.frequencies (pre-Phase-2 explicit-list path, "
                "still supported for tests that want a hand-picked frequency set)"
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
                input_dim=pde.dim,
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
                input_dim=pde.dim,
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
                input_dim=pde.dim,
            )

    if cfg.family == "q_octave":
        if pde is None:
            raise ValueError("'q_octave' requires `pde` (SMCD designs against the actual PDE)")
        from qapinn.smcd.design import smcd  # local import: avoids a models <-> smcd cycle

        card = smcd(pde, eps=smcd_eps, coverage_target=smcd_coverage_target)
        if card.octave_configs is not None:
            configs = card.octave_configs
        else:
            # No split was needed for this PDE's own SMCD design (T2.15: none of this
            # project's four PDEs naturally trigger a split at the default n_max/L_max --
            # see BENCH.md's T2.15 section) -- degenerate to a single-circuit ensemble,
            # which is mathematically equivalent to q_serial but built through
            # OctaveEnsemble's own API so q_octave is always constructible.
            configs = [
                {
                    "n_qubits": card.n_qubits,
                    "n_layers": card.n_layers,
                    "scalings": card.scalings,
                    "wire_to_dim": card.wire_to_dim,
                    "entangler": card.entangler,
                    "observable": card.observable,
                }
            ]
        return OctaveEnsemble(configs, input_dim=pde.dim)

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
