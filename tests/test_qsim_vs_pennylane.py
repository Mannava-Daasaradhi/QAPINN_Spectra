"""T2.5 DoD (hard gate): build the identical circuit in PennyLane (default.qubit, torch
interface) and compare against ReuploadCircuit's own qsim.py-backed forward() on 100
random (x, theta) draws for n in {2,4,6}, L in {1,3,5}, both entangler settings.

Max absolute difference must be < 1e-10. If this fails, the cause is almost always
qubit-ordering/endianness or CZ wire indexing -- fix the simulator, NEVER loosen this
tolerance (PennyLane is the oracle cited in the paper, D2).
"""
from __future__ import annotations

import math

import numpy as np
import pennylane as qml
import pytest
import torch

from qapinn.models.circuits import PREP_ANGLE, ReuploadCircuit

TOLERANCE = 1e-10


def _pennylane_qnode(n_qubits, n_layers, scalings, wire_to_dim, entangler, observable):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(z_sample, theta):
        for q in range(n_qubits):
            qml.RY(PREP_ANGLE, wires=q)
        for l in range(n_layers):
            for q in range(n_qubits):
                angle = scalings[l, q] * z_sample[wire_to_dim[q]]
                qml.RZ(angle, wires=q)
            for q in range(n_qubits):
                qml.RY(theta[l, q, 0], wires=q)
                qml.RZ(theta[l, q, 1], wires=q)
            if entangler == "ring_cz" and n_qubits >= 2:
                # mirrors circuits.py's ReuploadCircuit.forward() exactly, including the
                # n_qubits==2 special case (CZ(0,1) then CZ(1,0) would cancel, T2.6).
                n_cz_pairs = 1 if n_qubits == 2 else n_qubits
                for q in range(n_cz_pairs):
                    qml.CZ(wires=[q, (q + 1) % n_qubits])
        for q in range(n_qubits):
            qml.RY(theta[n_layers, q, 0], wires=q)
            qml.RZ(theta[n_layers, q, 1], wires=q)

        if observable == "z0":
            return qml.expval(qml.PauliZ(0))
        obs = qml.PauliZ(0)
        for q in range(1, n_qubits):
            obs = obs + qml.PauliZ(q)
        return qml.expval(obs * (1.0 / n_qubits))

    return circuit


@pytest.mark.parametrize("n_qubits", [2, 4, 6])
@pytest.mark.parametrize("n_layers", [1, 3, 5])
@pytest.mark.parametrize("entangler", ["none", "ring_cz"])
def test_qsim_matches_pennylane_oracle(n_qubits, n_layers, entangler):
    torch.manual_seed(hash((n_qubits, n_layers, entangler)) % (2**31))
    n_draws = 100
    wire_to_dim = tuple(q % 2 for q in range(n_qubits))
    observable = "z0" if n_qubits % 2 == 0 else "z_mean"

    max_diff = 0.0
    for _ in range(n_draws):
        scalings = (torch.rand(n_layers, n_qubits) * 2.5 + 0.5)
        circuit = ReuploadCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            scalings=scalings,
            wire_to_dim=wire_to_dim,
            entangler=entangler,
            observable=observable,
        )
        theta = torch.randn(n_layers + 1, n_qubits, 2)
        with torch.no_grad():
            circuit.theta.copy_(theta)
        z = torch.rand(2) * 2.0 - 1.0

        with torch.no_grad():
            qsim_out = circuit(z.reshape(1, -1)).squeeze().item()

        pl_qnode = _pennylane_qnode(n_qubits, n_layers, scalings, wire_to_dim, entangler, observable)
        pl_out = pl_qnode(z, theta).item()

        max_diff = max(max_diff, abs(qsim_out - pl_out))

    assert max_diff < TOLERANCE, f"n={n_qubits} L={n_layers} entangler={entangler}: max_diff={max_diff}"
