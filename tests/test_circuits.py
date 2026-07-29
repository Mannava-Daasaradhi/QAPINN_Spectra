"""T2.4 DoD: frequencies() for L=3, ternary Delta=pi, one wire returns exactly the 27
values pi*{-13,...,13}; for linear scalings returns the 7 values pi*{-3,...,3}.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from qapinn.models.circuits import ReuploadCircuit


def _ternary_scalings(n_layers, n_qubits, delta):
    return [[delta * 3**l for _ in range(n_qubits)] for l in range(n_layers)]


def _linear_scalings(n_layers, n_qubits, omega0):
    return [[omega0 for _ in range(n_qubits)] for _ in range(n_layers)]


def test_frequencies_ternary_L3_one_wire_27_values():
    scalings = _ternary_scalings(n_layers=3, n_qubits=1, delta=math.pi)
    circuit = ReuploadCircuit(n_qubits=1, n_layers=3, scalings=scalings, wire_to_dim=(0,))

    omega = circuit.frequencies()
    expected = math.pi * np.arange(-13, 14)
    assert omega.shape == (27,)
    assert np.allclose(np.sort(omega), np.sort(expected), atol=1e-8)


def test_frequencies_linear_L3_one_wire_7_values():
    scalings = _linear_scalings(n_layers=3, n_qubits=1, omega0=math.pi)
    circuit = ReuploadCircuit(n_qubits=1, n_layers=3, scalings=scalings, wire_to_dim=(0,))

    omega = circuit.frequencies()
    expected = math.pi * np.arange(-3, 4)
    assert omega.shape == (7,)
    assert np.allclose(np.sort(omega), np.sort(expected), atol=1e-8)


@pytest.mark.parametrize("L,expected_max_idx", [(1, 1), (2, 4), (4, 40)])
def test_frequencies_ternary_other_L_matches_bijection(L, expected_max_idx):
    scalings = _ternary_scalings(n_layers=L, n_qubits=1, delta=math.pi)
    circuit = ReuploadCircuit(n_qubits=1, n_layers=L, scalings=scalings, wire_to_dim=(0,))
    omega = circuit.frequencies()
    assert omega.shape == (3**L,)
    assert abs(omega.max() - expected_max_idx * math.pi) < 1e-8
    assert abs(omega.min() + expected_max_idx * math.pi) < 1e-8


def test_forward_shape_and_finite():
    scalings = _ternary_scalings(n_layers=2, n_qubits=1, delta=math.pi)
    circuit = ReuploadCircuit(n_qubits=1, n_layers=2, scalings=scalings, wire_to_dim=(0,))
    z = torch.linspace(0.0, 1.0, 8).reshape(-1, 1)
    out = circuit(z)
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()
    assert torch.all(out.abs() <= 1.0 + 1e-8)  # <Z> in [-1,1]


def test_forward_z_mean_observable():
    scalings = _ternary_scalings(n_layers=1, n_qubits=2, delta=math.pi)
    circuit = ReuploadCircuit(
        n_qubits=2, n_layers=1, scalings=scalings, wire_to_dim=(0, 0), observable="z_mean"
    )
    z = torch.linspace(0.0, 1.0, 5).reshape(-1, 1)
    out = circuit(z)
    assert out.shape == (5, 1)
    assert torch.isfinite(out).all()


def test_global_observable_rejected():
    scalings = _ternary_scalings(n_layers=1, n_qubits=1, delta=math.pi)
    with pytest.raises(ValueError):
        ReuploadCircuit(n_qubits=1, n_layers=1, scalings=scalings, wire_to_dim=(0,), observable="z0z1")


def test_invalid_entangler_rejected():
    scalings = _ternary_scalings(n_layers=1, n_qubits=2, delta=math.pi)
    with pytest.raises(ValueError):
        ReuploadCircuit(n_qubits=2, n_layers=1, scalings=scalings, wire_to_dim=(0, 1), entangler="cnot_chain")


def test_scalings_is_a_buffer_not_a_parameter():
    scalings = _ternary_scalings(n_layers=1, n_qubits=1, delta=math.pi)
    circuit = ReuploadCircuit(n_qubits=1, n_layers=1, scalings=scalings, wire_to_dim=(0,))
    param_names = {name for name, _ in circuit.named_parameters()}
    buffer_names = {name for name, _ in circuit.named_buffers()}
    assert "scalings" in buffer_names
    assert "scalings" not in param_names
    assert "theta" in param_names


def test_frequencies_two_dims_no_entangler_only_axis_aligned_terms():
    # 2 wires, 2 dims, entangler='none': every nonzero-frequency term must have AT MOST
    # one nonzero component (no genuine cross terms) -- the design-level counterpart of
    # T2.6's FFT-based check on the actual circuit output.
    scalings = _linear_scalings(n_layers=2, n_qubits=2, omega0=math.pi)
    circuit = ReuploadCircuit(
        n_qubits=2, n_layers=2, scalings=scalings, wire_to_dim=(0, 1), entangler="none"
    )
    omega = circuit.frequencies()
    assert omega.shape[1] == 2
    n_nonzero_components = np.sum(np.abs(omega) > 1e-12, axis=1)
    assert np.all(n_nonzero_components <= 1)


def test_frequencies_two_dims_with_entangler_has_cross_terms():
    scalings = _linear_scalings(n_layers=2, n_qubits=2, omega0=math.pi)
    circuit = ReuploadCircuit(
        n_qubits=2, n_layers=2, scalings=scalings, wire_to_dim=(0, 1), entangler="ring_cz"
    )
    omega = circuit.frequencies()
    n_nonzero_components = np.sum(np.abs(omega) > 1e-12, axis=1)
    assert np.any(n_nonzero_components == 2)  # genuine (omega_x, omega_y) cross terms exist


def test_theta_gradients_flow():
    scalings = _ternary_scalings(n_layers=2, n_qubits=1, delta=math.pi)
    circuit = ReuploadCircuit(n_qubits=1, n_layers=2, scalings=scalings, wire_to_dim=(0,))
    z = torch.linspace(0.0, 1.0, 4).reshape(-1, 1)
    out = circuit(z)
    loss = out.sum()
    loss.backward()
    assert circuit.theta.grad is not None
    assert torch.isfinite(circuit.theta.grad).all()
