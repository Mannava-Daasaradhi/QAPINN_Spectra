"""Hybrid classical-quantum PINN model families (T2.12, project.md Sections 5.3/6).

AffineEncoder exposes .A/.b (not private) for xai/drift.py's encoder-drift instrument
(T1.8), which recognises a trainable quantum encoder via the duck-typed `model.encoder_A`
/ `model.encoder_omega` protocol -- SerialHybrid/ParallelHybrid/OctaveEnsemble all expose
that same pair of properties so drift.py (written against a stand-in before any real
quantum model existed) works unmodified against the real thing.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from qapinn.models.base import PINNModel
from qapinn.models.circuits import ReuploadCircuit
from qapinn.models.mlp import MLPPINN


class AffineEncoder(nn.Module):
    """z = A x + b, A initialised to I (D3): identity at init so realised_frequencies()
    exactly matches the raw circuit's own Omega before any training happens (T2.12's own
    DoD). A is [dim, dim] -- square, so ||A - I||_F (drift.py) is always defined."""

    def __init__(self, dim: int):
        super().__init__()
        self.A = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.A.T + self.b


def _realised_frequencies(circuit: ReuploadCircuit, encoder: AffineEncoder) -> np.ndarray:
    """Omega @ A in physical coordinates (D3) -- shared by every hybrid family below."""
    Omega = circuit.frequencies()
    if Omega.ndim == 1:
        Omega = Omega.reshape(-1, 1)
    A = encoder.A.detach().cpu().numpy()
    return Omega @ A


def _omega_buffer(circuit: ReuploadCircuit) -> Tensor:
    Omega = circuit.frequencies()
    if Omega.ndim == 1:
        Omega = Omega.reshape(-1, 1)
    return torch.as_tensor(Omega, dtype=torch.get_default_dtype())


def _circuit_dim(wire_to_dim) -> int:
    return len(set(wire_to_dim))


class SerialHybrid(PINNModel):
    """x -> AffineEncoder -> ReuploadCircuit -> Linear head -> u."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        scalings,
        wire_to_dim: tuple[int, ...],
        entangler: str = "ring_cz",
        observable: str = "z0",
    ):
        super().__init__()
        self.encoder = AffineEncoder(_circuit_dim(wire_to_dim))
        self.circuit = ReuploadCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            scalings=scalings,
            wire_to_dim=wire_to_dim,
            entangler=entangler,
            observable=observable,
        )
        self.head = nn.Linear(1, 1)
        nn.init.ones_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.circuit(self.encoder(x)))

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "classical": list(self.encoder.parameters()) + list(self.head.parameters()),
            "quantum": list(self.circuit.parameters()),
        }

    def realised_frequencies(self) -> np.ndarray | None:
        return _realised_frequencies(self.circuit, self.encoder)

    @property
    def encoder_A(self) -> Tensor:
        return self.encoder.A

    @property
    def encoder_omega(self) -> Tensor:
        return _omega_buffer(self.circuit)


class ParallelHybrid(PINNModel):
    """u = MLP(x) + w * Circuit(AffineEncoder(x)) -- the HQPINN form from the literature."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        scalings,
        wire_to_dim: tuple[int, ...],
        entangler: str = "ring_cz",
        observable: str = "z0",
        mlp_widths: tuple[int, ...] = (32, 32),
        activation: str = "tanh",
    ):
        super().__init__()
        d = _circuit_dim(wire_to_dim)
        self.mlp = MLPPINN(input_dim=d, widths=mlp_widths, activation=activation)
        self.encoder = AffineEncoder(d)
        self.circuit = ReuploadCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            scalings=scalings,
            wire_to_dim=wire_to_dim,
            entangler=entangler,
            observable=observable,
        )
        self.w = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x) + self.w * self.circuit(self.encoder(x))

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "classical": list(self.mlp.parameters()) + list(self.encoder.parameters()) + [self.w],
            "quantum": list(self.circuit.parameters()),
        }

    def realised_frequencies(self) -> np.ndarray | None:
        return _realised_frequencies(self.circuit, self.encoder)

    @property
    def encoder_A(self) -> Tensor:
        return self.encoder.A

    @property
    def encoder_omega(self) -> Tensor:
        return _omega_buffer(self.circuit)


class OctaveEnsemble(PINNModel):
    """u = sum_m w_m * Circuit_m(Enc_m(x)), each circuit matched to one octave of the
    target spectrum (project.md Section 5.4). This class is generic over a LIST of
    per-octave circuit configs -- T2.15 is responsible for actually partitioning a target
    spectrum into octaves and constructing the matching scalings; this class just wires
    whatever list of configs it is given into a weighted-sum ensemble.

    circuit_configs: list of dicts with keys n_qubits, n_layers, scalings, wire_to_dim,
    and optionally entangler ('ring_cz' default) / observable ('z0' default).
    """

    def __init__(self, circuit_configs: list[dict]):
        super().__init__()
        if not circuit_configs:
            raise ValueError("OctaveEnsemble needs at least one circuit config")

        self.encoders = nn.ModuleList()
        self.circuits = nn.ModuleList()
        for cfg in circuit_configs:
            wire_to_dim = tuple(cfg["wire_to_dim"])
            self.encoders.append(AffineEncoder(_circuit_dim(wire_to_dim)))
            self.circuits.append(
                ReuploadCircuit(
                    n_qubits=cfg["n_qubits"],
                    n_layers=cfg["n_layers"],
                    scalings=cfg["scalings"],
                    wire_to_dim=wire_to_dim,
                    entangler=cfg.get("entangler", "ring_cz"),
                    observable=cfg.get("observable", "z0"),
                )
            )
        self.w = nn.Parameter(torch.ones(len(circuit_configs)) / len(circuit_configs))

    def forward(self, x: Tensor) -> Tensor:
        out = 0.0
        for m in range(len(self.circuits)):
            out = out + self.w[m] * self.circuits[m](self.encoders[m](x))
        return out

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        classical = list(self.encoders.parameters()) + [self.w]
        quantum = list(self.circuits.parameters())
        return {"classical": classical, "quantum": quantum}

    def realised_frequencies(self) -> np.ndarray | None:
        parts = [_realised_frequencies(c, e) for c, e in zip(self.circuits, self.encoders)]
        return np.concatenate(parts, axis=0)
