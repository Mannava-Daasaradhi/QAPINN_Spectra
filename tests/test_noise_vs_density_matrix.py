"""T4.4 DoD: validate the analytic depolarizing surrogate (D8, `GlobalDepolarizing`)
against a real density-matrix simulation. Our surrogate applies ONE global depolarizing
factor `(1-p)^L` to the noiseless expectation value; the physically faithful model is
LOCAL depolarizing noise on every wire after every layer, simulated via PennyLane's
`default.mixed` density-matrix device. The relative discrepancy between the two is
MEASURED and reported here, not hidden -- per the phase doc, if it exceeds 10% every
noise conclusion drawn from the surrogate must be downgraded to qualitative.

Circuit construction mirrors test_qsim_vs_pennylane.py's `_pennylane_qnode` exactly (the
project's own established PennyLane cross-validation pattern, T2.5's hard gate) so the
NOISELESS baseline is provably the same circuit our surrogate assumes.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

from qapinn.models.circuits import PREP_ANGLE
from qapinn.models.noise import GlobalDepolarizing

N_QUBITS = 4
N_LAYERS = 3
N_DRAWS = 200
DEPOL_P = 1e-2  # a physically meaningful per-layer noise rate (matches noise_study.yaml's order of magnitude scaled up 10x -- 1e-3 gives too small a signal to distinguish surrogate error from simulation noise at this qubit count)


def _noiseless_qnode(wire_to_dim, entangler="ring_cz", observable="z0"):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def circuit(z_sample, theta, scalings):
        for q in range(N_QUBITS):
            qml.RY(PREP_ANGLE, wires=q)
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                qml.RZ(scalings[l, q] * z_sample[wire_to_dim[q]], wires=q)
            for q in range(N_QUBITS):
                qml.RY(theta[l, q, 0], wires=q)
                qml.RZ(theta[l, q, 1], wires=q)
            if entangler == "ring_cz":
                for q in range(N_QUBITS):
                    qml.CZ(wires=[q, (q + 1) % N_QUBITS])
        for q in range(N_QUBITS):
            qml.RY(theta[N_LAYERS, q, 0], wires=q)
            qml.RZ(theta[N_LAYERS, q, 1], wires=q)
        return qml.expval(qml.PauliZ(0))

    return circuit


def _noisy_density_matrix_qnode(wire_to_dim, p, entangler="ring_cz", observable="z0"):
    """Identical gate sequence, but on `default.mixed` with a LOCAL DepolarizingChannel
    on every wire immediately after each of the N_LAYERS layers (encoding + trainable +
    entangler) -- the physically faithful noise model our surrogate approximates."""
    dev = qml.device("default.mixed", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def circuit(z_sample, theta, scalings):
        for q in range(N_QUBITS):
            qml.RY(PREP_ANGLE, wires=q)
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                qml.RZ(scalings[l, q] * z_sample[wire_to_dim[q]], wires=q)
            for q in range(N_QUBITS):
                qml.RY(theta[l, q, 0], wires=q)
                qml.RZ(theta[l, q, 1], wires=q)
            if entangler == "ring_cz":
                for q in range(N_QUBITS):
                    qml.CZ(wires=[q, (q + 1) % N_QUBITS])
            for q in range(N_QUBITS):
                qml.DepolarizingChannel(p, wires=q)
        for q in range(N_QUBITS):
            qml.RY(theta[N_LAYERS, q, 0], wires=q)
            qml.RZ(theta[N_LAYERS, q, 1], wires=q)
        return qml.expval(qml.PauliZ(0))

    return circuit


def test_global_depolarizing_surrogate_vs_local_density_matrix_noise():
    torch.manual_seed(0)
    wire_to_dim = tuple(q % 2 for q in range(N_QUBITS))
    noiseless_qnode = _noiseless_qnode(wire_to_dim)
    noisy_qnode = _noisy_density_matrix_qnode(wire_to_dim, DEPOL_P)
    surrogate = GlobalDepolarizing(p=DEPOL_P, m=N_LAYERS)

    rel_errors = []
    for _ in range(N_DRAWS):
        scalings = torch.rand(N_LAYERS, N_QUBITS) * 2.5 + 0.5
        theta = torch.randn(N_LAYERS + 1, N_QUBITS, 2)
        z = torch.rand(2) * 2.0 - 1.0

        clean = noiseless_qnode(z, theta, scalings)
        true_noisy = noisy_qnode(z, theta, scalings).item()
        surrogate_noisy = surrogate(clean).item()

        denom = max(1e-06, abs(true_noisy))
        rel_errors.append(abs(surrogate_noisy - true_noisy) / denom)

    rel_errors = np.array(rel_errors)
    median_rel_error = float(np.median(rel_errors))
    max_rel_error = float(np.max(rel_errors))

    # Reported honestly regardless of outcome (T4.4's own DoD) -- printed so it survives
    # into `pytest -s` / CI logs even though this isn't a hard pass/fail gate on its own.
    print(
        f"\nT4.4 noise-surrogate validation (n={N_QUBITS}, L={N_LAYERS}, p={DEPOL_P}, "
        f"N={N_DRAWS}): median rel. error={median_rel_error:.4f}, max={max_rel_error:.4f}"
    )

    # Sanity bound only (catches a genuine implementation bug, e.g. a sign/scale error)
    # -- NOT the phase doc's 10% "downgrade to qualitative" threshold, which is a
    # scientific judgment call for FINDINGS.md, not something a unit test should silently
    # gate on. 2.0 (200%) is generous: it only fails if the surrogate is wildly,
    # structurally wrong, not merely quantitatively imprecise.
    assert median_rel_error < 2.0
    assert np.all(np.isfinite(rel_errors))
