"""T2.6 -- HARD GATE. Numerically verifies Prop. 1
(docs/derivations/07_circuit_fourier_spectrum.md) directly on the actual circuit output.

Protocol: the circuit output is an EXACT trigonometric polynomial (Prop. 1), so a
correctly-sampled FFT is exact -- tolerance can be 1e-8, not 1e-2.
1. Ternary scalings, Delta known, L layers, random theta.
2. All frequencies are integer multiples of Delta => f is periodic with period T=2pi/Delta.
3. Sample N = 4*max_index+1 points on [0,T) -- strictly above Nyquist, exactly one period,
   endpoint excluded. Zero spectral leakage by construction.
4. FFT; map bin indices to angular frequency via omega_grid (T0.9).
5. ASSERT: every bin whose omega NOT IN Omega has |coefficient| < 1e-8.
6. ASSERT: at least 80% of bins with omega IN Omega have |coefficient| > 1e-6, averaged
   over 5 random theta draws (a random theta can null a few coefficients by chance).

The expected Omega is computed HERE, independently of circuits.py's own frequencies()
(brute-force itertools.product over signed subset sums) -- this test verifies Prop. 1
against the actual circuit output, not against circuits.py's own bookkeeping, so it must
not reuse that bookkeeping to decide what counts as "in Omega".

This is the project's scientific foundation. Do not proceed past it on a failing test.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

from qapinn.models.circuits import ReuploadCircuit
from qapinn.xai.spectral_error import omega_grid

FFT_LEAK_TOL = 1e-8
FFT_PRESENT_TOL = 1e-6
N_THETA_DRAWS = 5


def _expected_omega_1wire(scalings: list[float]) -> list[float]:
    """Brute-force signed-subset-sum set for ONE wire's per-layer scalings -- an
    independent re-implementation of Prop. 1, not a call into circuits.py."""
    L = len(scalings)
    omega = {sum(m * w for m, w in zip(signs, scalings)) for signs in itertools.product((-1, 0, 1), repeat=L)}
    return sorted(omega)


def _sample_and_fft_1wire(circuit: ReuploadCircuit, delta: float, max_index: int, theta: torch.Tensor):
    N = 4 * max_index + 1
    T = 2 * math.pi / delta
    dx = T / N
    x = (torch.arange(N, dtype=torch.get_default_dtype()) * dx).reshape(-1, 1)

    with torch.no_grad():
        circuit.theta.copy_(theta)
        f = circuit(x).squeeze(-1).numpy()

    omega = omega_grid(N, dx)
    coeffs = np.abs(np.fft.fft(f))
    return omega, coeffs


def _check_support(omega: np.ndarray, coeffs: np.ndarray, expected_omega: list[float], rtol: float = 1e-6):
    """Returns (max_leak, in_support_mask, frac_present) for one (omega,coeffs) draw."""
    expected = np.asarray(expected_omega)
    in_support = np.array([np.any(np.isclose(w, expected, atol=1e-8, rtol=rtol)) for w in omega])

    leak = coeffs[~in_support]
    max_leak = float(leak.max()) if leak.size else 0.0

    present = coeffs[in_support] > FFT_PRESENT_TOL
    frac_present = float(present.mean()) if present.size else 1.0

    return max_leak, frac_present


@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_ternary_1wire_support_matches_exactly(L):
    delta = math.pi
    scalings_list = [delta * 3**l for l in range(L)]
    expected_omega = _expected_omega_1wire(scalings_list)
    assert len(expected_omega) == 3**L

    max_index = round(max(abs(w) for w in expected_omega) / delta)
    scalings = torch.tensor([[s] for s in scalings_list])

    torch.manual_seed(0)
    circuit = ReuploadCircuit(n_qubits=1, n_layers=L, scalings=scalings, wire_to_dim=(0,))

    fracs = []
    for draw in range(N_THETA_DRAWS):
        theta = torch.randn(L + 1, 1, 2) * 0.5 + draw  # vary theta across draws
        omega, coeffs = _sample_and_fft_1wire(circuit, delta, max_index, theta)
        max_leak, frac_present = _check_support(omega, coeffs, expected_omega)
        assert max_leak < FFT_LEAK_TOL, f"L={L} draw={draw}: leakage {max_leak} outside Omega"
        fracs.append(frac_present)

    assert sum(fracs) / len(fracs) >= 0.8, f"L={L}: avg fraction present {sum(fracs)/len(fracs)}"


def test_linear_scalings_support_is_exactly_omega0_times_range():
    L = 3
    omega0 = math.pi
    scalings_list = [omega0] * L
    expected_omega = _expected_omega_1wire(scalings_list)
    assert expected_omega == sorted(omega0 * k for k in range(-L, L + 1))

    max_index = L
    scalings = torch.tensor([[s] for s in scalings_list])

    torch.manual_seed(1)
    circuit = ReuploadCircuit(n_qubits=1, n_layers=L, scalings=scalings, wire_to_dim=(0,))

    fracs = []
    for draw in range(N_THETA_DRAWS):
        theta = torch.randn(L + 1, 1, 2) * 0.5 + draw
        omega, coeffs = _sample_and_fft_1wire(circuit, omega0, max_index, theta)
        max_leak, frac_present = _check_support(omega, coeffs, expected_omega)
        assert max_leak < FFT_LEAK_TOL, f"draw={draw}: leakage {max_leak} outside {{-L..L}}*omega0"
        fracs.append(frac_present)

    assert sum(fracs) / len(fracs) >= 0.8


def _sample_and_fft2_2wire(circuit: ReuploadCircuit, delta_x: float, delta_y: float, max_idx_x: int, max_idx_y: int, theta: torch.Tensor):
    Nx = 4 * max_idx_x + 1
    Ny = 4 * max_idx_y + 1
    Tx = 2 * math.pi / delta_x
    Ty = 2 * math.pi / delta_y
    dx = Tx / Nx
    dy = Ty / Ny
    xs = torch.arange(Nx, dtype=torch.get_default_dtype()) * dx
    ys = torch.arange(Ny, dtype=torch.get_default_dtype()) * dy
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [Nx*Ny, 2]

    with torch.no_grad():
        circuit.theta.copy_(theta)
        f = circuit(grid).squeeze(-1).numpy().reshape(Nx, Ny)

    omega_x = omega_grid(Nx, dx)
    omega_y = omega_grid(Ny, dy)
    coeffs = np.abs(np.fft.fft2(f))
    return omega_x, omega_y, coeffs


def _two_wire_setup(entangler: str):
    # DIFFERENT deltas per wire so their own frequency sets never accidentally coincide
    # at a nonzero value -- makes "which wire produced this bin" unambiguous.
    L = 2
    delta_x, delta_y = math.pi, 5 * math.pi
    scalings_x = [delta_x * 3**l for l in range(L)]
    scalings_y = [delta_y * 3**l for l in range(L)]
    expected_x = _expected_omega_1wire(scalings_x)
    expected_y = _expected_omega_1wire(scalings_y)
    max_idx_x = round(max(abs(w) for w in expected_x) / delta_x)
    max_idx_y = round(max(abs(w) for w in expected_y) / delta_y)

    scalings = torch.tensor([[sx, sy] for sx, sy in zip(scalings_x, scalings_y)])  # [L,2]
    circuit = ReuploadCircuit(
        n_qubits=2, n_layers=L, scalings=scalings, wire_to_dim=(0, 1), entangler=entangler, observable="z0"
    )
    return circuit, delta_x, delta_y, max_idx_x, max_idx_y, expected_x, expected_y


def test_two_wires_no_entangler_no_cross_terms():
    circuit, delta_x, delta_y, max_idx_x, max_idx_y, _expected_x, _expected_y = _two_wire_setup("none")

    torch.manual_seed(2)
    theta = torch.randn(circuit.n_layers + 1, 2, 2) * 0.5
    omega_x, omega_y, coeffs = _sample_and_fft2_2wire(circuit, delta_x, delta_y, max_idx_x, max_idx_y, theta)

    x_nonzero = np.abs(omega_x) > 1e-8
    y_nonzero = np.abs(omega_y) > 1e-8
    cross_mask = x_nonzero[:, None] & y_nonzero[None, :]  # bins with BOTH axes nonzero
    cross_energy = coeffs[cross_mask]

    assert cross_energy.max() < FFT_LEAK_TOL, (
        f"entangler='none' produced a genuine cross term, max={cross_energy.max()} "
        "-- Alg. 1 step 7's 'avoid gratuitous entanglement' would be numerically false"
    )


def test_two_wires_ring_cz_cross_terms_appear():
    circuit, delta_x, delta_y, max_idx_x, max_idx_y, _expected_x, _expected_y = _two_wire_setup("ring_cz")

    torch.manual_seed(3)
    theta = torch.randn(circuit.n_layers + 1, 2, 2) * 0.5
    omega_x, omega_y, coeffs = _sample_and_fft2_2wire(circuit, delta_x, delta_y, max_idx_x, max_idx_y, theta)

    x_nonzero = np.abs(omega_x) > 1e-8
    y_nonzero = np.abs(omega_y) > 1e-8
    cross_mask = x_nonzero[:, None] & y_nonzero[None, :]
    cross_energy = coeffs[cross_mask]

    assert cross_energy.max() > FFT_PRESENT_TOL, "entangler='ring_cz' should produce omega_x +- omega_y cross terms"
