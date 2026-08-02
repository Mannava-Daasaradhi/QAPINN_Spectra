"""Parameter-shift derivatives (T2.7). Verifies the exact shift rule -- first order
(docs/derivations/08_parameter_shift.md) and its T2.7 second-order extension -- against
autograd, gate for gate. This is what LICENSES project.md section 3.8's claim that the
PINN residual is exactly computable on real quantum hardware, rather than merely
asserting it.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from qapinn.models.circuits import ReuploadCircuit

HALF_PI = math.pi / 2


def psr_grad_theta(circuit: ReuploadCircuit, z: Tensor, theta: Tensor, idx: tuple[int, int, int]) -> Tensor:
    """d f / d theta[idx], exact first-order shift rule: 1/2 [f(theta+pi/2) - f(theta-pi/2)],
    holding z and every other theta entry fixed.
    """
    theta = theta.detach()
    theta_plus = theta.clone()
    theta_plus[idx] = theta_plus[idx] + HALF_PI
    theta_minus = theta.clone()
    theta_minus[idx] = theta_minus[idx] - HALF_PI
    f_plus = circuit.evaluate(z, theta=theta_plus)
    f_minus = circuit.evaluate(z, theta=theta_minus)
    return 0.5 * (f_plus - f_minus)


def _encoding_gates(circuit: ReuploadCircuit, dim: int) -> list[tuple[int, int]]:
    return [
        (l, q)
        for l in range(circuit.n_layers)
        for q in range(circuit.n_qubits)
        if circuit.wire_to_dim[q] == dim
    ]


def psr_grad_input(circuit: ReuploadCircuit, z: Tensor, dim: int) -> Tensor:
    """d f / d z[:, dim], exact first-order shift rule. Re-uploading means z[:, dim] can
    feed MULTIPLE encoding gates (every layer re-encodes it, and multiple wires may share
    a dimension); by the multivariate chain rule this is exactly a sum of per-gate
    shift-rule terms (08_parameter_shift.md) -- no cross terms at first order, since the
    derivative of a sum is the sum of derivatives.
    """
    z = z.detach()
    theta = circuit.theta.detach()
    gates = _encoding_gates(circuit, dim)
    total = None
    for l, q in gates:
        omega = circuit.scalings[l, q]
        f_plus = circuit.evaluate(z, theta=theta, shifts={(l, q): HALF_PI})
        f_minus = circuit.evaluate(z, theta=theta, shifts={(l, q): -HALF_PI})
        term = omega * 0.5 * (f_plus - f_minus)
        total = term if total is None else total + term
    if total is None:
        total = torch.zeros(z.shape[0], 1, dtype=torch.get_default_dtype(), device=z.device)
    return total


def psr_grad2_input(circuit: ReuploadCircuit, z: Tensor, dim: int) -> Tensor:
    """d^2 f / d z[:, dim]^2, exact second-order shift rule (docs/derivations/
    08_parameter_shift.md's T2.7 extension). Because z[:, dim] can feed MULTIPLE encoding
    gates under re-uploading, the multivariate chain rule for phi_i = omega_i * x (each
    phi_i linear in x, so d^2 phi_i / dx^2 = 0) gives

        d^2f/dx^2 = sum_i omega_i^2 d^2f/dphi_i^2 + sum_{i!=j} omega_i omega_j d^2f/(dphi_i dphi_j)

    i.e. a DIAGONAL term per gate (the single-gate 3-point rule) PLUS a MIXED term per
    distinct pair of gates sharing dim (the 4-point rule, exact by the same composability
    argument that gives the first-order rule). Dropping the mixed terms is wrong whenever
    n_layers > 1 or more than one wire reads the same dimension -- i.e. almost always for
    a re-uploading circuit.
    """
    z = z.detach()
    theta = circuit.theta.detach()
    gates = _encoding_gates(circuit, dim)
    if not gates:
        return torch.zeros(z.shape[0], 1, dtype=torch.get_default_dtype(), device=z.device)

    f0 = circuit.evaluate(z, theta=theta)
    total = torch.zeros_like(f0)

    for l, q in gates:
        omega = circuit.scalings[l, q]
        f_plus = circuit.evaluate(z, theta=theta, shifts={(l, q): HALF_PI})
        f_minus = circuit.evaluate(z, theta=theta, shifts={(l, q): -HALF_PI})
        total = total + (omega**2 / 2.0) * (f_plus - 2.0 * f0 + f_minus)

    for i in range(len(gates)):
        li, qi = gates[i]
        omega_i = circuit.scalings[li, qi]
        for j in range(i + 1, len(gates)):
            lj, qj = gates[j]
            omega_j = circuit.scalings[lj, qj]
            f_pp = circuit.evaluate(z, theta=theta, shifts={(li, qi): HALF_PI, (lj, qj): HALF_PI})
            f_pm = circuit.evaluate(z, theta=theta, shifts={(li, qi): HALF_PI, (lj, qj): -HALF_PI})
            f_mp = circuit.evaluate(z, theta=theta, shifts={(li, qi): -HALF_PI, (lj, qj): HALF_PI})
            f_mm = circuit.evaluate(z, theta=theta, shifts={(li, qi): -HALF_PI, (lj, qj): -HALF_PI})
            mixed = 0.25 * (f_pp - f_pm - f_mp + f_mm)
            total = total + 2.0 * omega_i * omega_j * mixed

    return total
