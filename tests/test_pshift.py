"""T2.7 DoD (hard requirement, not a gate): parameter-shift and autodiff must agree to
1e-9 for d/dtheta, d/dx, and d^2/dx^2. This VERIFIES project.md section 3.8's claim that
the PINN residual is exactly computable on hardware, rather than merely asserting it.

n_layers=3 configurations are essential here, not decorative: with n_layers=1 there is at
most one encoding gate per input dimension, so psr_grad2_input's mixed-gate cross terms
(docs/derivations/08_parameter_shift.md) are never exercised and a broken cross-term
implementation would silently pass. n_layers=1 is kept too, as the trivial no-cross-term
sanity check.
"""
from __future__ import annotations

import torch

from qapinn.models.circuits import ReuploadCircuit
from qapinn.models.pshift import psr_grad_input, psr_grad_theta, psr_grad2_input

TOLERANCE = 1e-9
N_DRAWS = 20


def _make_circuit(n_qubits, n_layers, entangler, seed):
    g = torch.Generator().manual_seed(seed)
    scalings = torch.rand(n_layers, n_qubits, generator=g) * 2.0 + 0.5
    wire_to_dim = tuple(q % 2 for q in range(n_qubits))
    observable = "z0" if n_qubits % 2 == 0 else "z_mean"
    circuit = ReuploadCircuit(
        n_qubits=n_qubits,
        n_layers=n_layers,
        scalings=scalings,
        wire_to_dim=wire_to_dim,
        entangler=entangler,
        observable=observable,
    )
    with torch.no_grad():
        circuit.theta.copy_(torch.randn(n_layers + 1, n_qubits, 2, generator=g) * 0.7)
    return circuit


CONFIGS = [
    (2, 1, "none"),
    (2, 1, "ring_cz"),
    (2, 3, "ring_cz"),
    (4, 1, "none"),
    (4, 3, "ring_cz"),
]


def _autodiff_grad_theta(circuit, z, idx):
    f = circuit.evaluate(z, theta=circuit.theta)
    (g,) = torch.autograd.grad(f.sum(), circuit.theta, create_graph=False)
    return g[idx]


def _autodiff_dx_dx2(circuit, z, dim):
    f = circuit(z)
    (g,) = torch.autograd.grad(f.sum(), z, create_graph=True)
    dfdx = g[:, dim]
    (h,) = torch.autograd.grad(dfdx.sum(), z, create_graph=False)
    d2fdx2 = h[:, dim]
    return dfdx.detach(), d2fdx2.detach()


def test_psr_grad_theta_matches_autodiff():
    max_diff = 0.0
    for cfg_i, (n_qubits, n_layers, entangler) in enumerate(CONFIGS):
        circuit = _make_circuit(n_qubits, n_layers, entangler, seed=1000 + cfg_i)
        g = torch.Generator().manual_seed(2000 + cfg_i)
        for draw in range(N_DRAWS):
            z = (torch.rand(1, 2, generator=g) * 2.0 - 1.0).requires_grad_(True)
            idx = (
                int(torch.randint(0, n_layers + 1, (1,), generator=g).item()),
                int(torch.randint(0, n_qubits, (1,), generator=g).item()),
                int(torch.randint(0, 2, (1,), generator=g).item()),
            )
            theta = circuit.theta.detach().clone().requires_grad_(True)
            f = circuit.evaluate(z.detach(), theta=theta)
            (autodiff,) = torch.autograd.grad(f.sum(), theta)
            autodiff_val = autodiff[idx].item()

            shift_val = psr_grad_theta(circuit, z.detach(), circuit.theta, idx).item()
            max_diff = max(max_diff, abs(shift_val - autodiff_val))

    assert max_diff < TOLERANCE, f"psr_grad_theta max_diff={max_diff}"


def test_psr_grad_input_matches_autodiff():
    max_diff = 0.0
    for cfg_i, (n_qubits, n_layers, entangler) in enumerate(CONFIGS):
        circuit = _make_circuit(n_qubits, n_layers, entangler, seed=3000 + cfg_i)
        g = torch.Generator().manual_seed(4000 + cfg_i)
        for draw in range(N_DRAWS):
            for dim in (0, 1):
                z = (torch.rand(1, 2, generator=g) * 2.0 - 1.0).requires_grad_(True)
                dfdx, _ = _autodiff_dx_dx2(circuit, z, dim)

                shift_val = psr_grad_input(circuit, z.detach(), dim).squeeze().item()
                max_diff = max(max_diff, abs(shift_val - dfdx.item()))

    assert max_diff < TOLERANCE, f"psr_grad_input max_diff={max_diff}"


def test_psr_grad2_input_matches_autodiff():
    max_diff = 0.0
    for cfg_i, (n_qubits, n_layers, entangler) in enumerate(CONFIGS):
        circuit = _make_circuit(n_qubits, n_layers, entangler, seed=5000 + cfg_i)
        g = torch.Generator().manual_seed(6000 + cfg_i)
        for draw in range(N_DRAWS):
            for dim in (0, 1):
                z = (torch.rand(1, 2, generator=g) * 2.0 - 1.0).requires_grad_(True)
                _, d2fdx2 = _autodiff_dx_dx2(circuit, z, dim)

                shift_val = psr_grad2_input(circuit, z.detach(), dim).squeeze().item()
                max_diff = max(max_diff, abs(shift_val - d2fdx2.item()))

    assert max_diff < TOLERANCE, f"psr_grad2_input max_diff={max_diff}"
