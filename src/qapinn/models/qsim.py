"""Batched statevector simulator (T2.3, D2). Minimal, autograd-native, CUDA-resident.

State shape [B, 2]*n, complex128. Qubit 0 is the FIRST trailing axis (axis 1; axis 0 is
batch). Never builds a 2^n x 2^n matrix -- single/two-qubit gates act as local tensor
contractions (`apply_1q`) or elementwise sign/select operations (`apply_cz`,
`apply_cnot`), matching real statevector-simulator practice.

Autograd safety (D1: second derivatives must flow through the PDE residual, which
differentiates the circuit's *output* twice w.r.t. its *input*): every op here is a
composition of standard differentiable primitives (stack, einsum, elementwise
multiply/exp/cos/sin) -- no `.detach()`, `.item()`, or `.numpy()` anywhere. Probabilities
are computed as `z * conj(z)` (a polynomial in the real/imaginary parts), never
`z.abs()**2` (whose gradient involves `1/|z|`, singular at zero amplitude -- exactly the
kind of second-derivative landmine D1 warns about).
"""
from __future__ import annotations

import torch
from torch import Tensor

_C128 = torch.complex128


def _as_real(theta) -> Tensor:
    if torch.is_tensor(theta):
        return theta
    return torch.as_tensor(theta, dtype=torch.get_default_dtype())


def rz(theta) -> Tensor:
    """RZ(phi) = diag(e^{-i phi/2}, e^{i phi/2}). theta: [] or [B] (real) -> [2,2] or [B,2,2]."""
    theta = _as_real(theta)
    half = (theta / 2).to(_C128)
    zero = torch.zeros_like(half)
    row0 = torch.stack([torch.exp(-1j * half), zero], dim=-1)
    row1 = torch.stack([zero, torch.exp(1j * half)], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def ry(theta) -> Tensor:
    """RY(phi) = [[cos(phi/2), -sin(phi/2)], [sin(phi/2), cos(phi/2)]]."""
    theta = _as_real(theta)
    half = theta / 2
    c = torch.cos(half).to(_C128)
    s = torch.sin(half).to(_C128)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def rx(theta) -> Tensor:
    """RX(phi) = [[cos(phi/2), -i sin(phi/2)], [-i sin(phi/2), cos(phi/2)]]."""
    theta = _as_real(theta)
    half = theta / 2
    c = torch.cos(half).to(_C128)
    neg_i_s = -1j * torch.sin(half).to(_C128)
    row0 = torch.stack([c, neg_i_s], dim=-1)
    row1 = torch.stack([neg_i_s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


class StateVectorSim:
    """Namespace of stateless statevector operations; `state` is always passed explicitly
    and a new state tensor is returned (functional style -- no instance state)."""

    @staticmethod
    def zeros_state(batch: int, n: int, device=None) -> Tensor:
        """|0...0>^{\\otimes n}, batched. Returns [B, 2,...,2] (n trailing axes), complex128."""
        shape = (batch,) + (2,) * n
        state = torch.zeros(shape, dtype=_C128, device=device)
        idx = (slice(None),) + (0,) * n
        state[idx] = 1.0
        return state

    @staticmethod
    def apply_1q(state: Tensor, U: Tensor, wire: int) -> Tensor:
        """U: [2,2] (shared across the batch) or [B,2,2] (per-sample, e.g. an encoding
        gate with per-sample angle omega*x_b). Contracts U against `wire`'s axis without
        ever reshaping/merging the other qubit axes -- einsum's ellipsis handles the
        arbitrary middle dimensions directly."""
        wire_axis = wire + 1
        state_m = torch.movedim(state, wire_axis, -1)  # [..., 2], wire's axis now last
        if U.dim() == 2:
            new_last = torch.einsum("ij,...j->...i", U, state_m)
        elif U.dim() == 3:
            new_last = torch.einsum("bij,b...j->b...i", U, state_m)
        else:
            raise ValueError(f"apply_1q: U must be [2,2] or [B,2,2], got shape {tuple(U.shape)}")
        return torch.movedim(new_last, -1, wire_axis)

    @staticmethod
    def apply_cz(state: Tensor, w0: int, w1: int) -> Tensor:
        """Diagonal: multiplies the |1,1> sub-block (on wires w0,w1) by -1. No matmul --
        mult(i,j) = 1 - 2*i*j for i,j in {0,1} is -1 only when both are 1."""
        axis0, axis1 = w0 + 1, w1 + 1
        shape0 = [1] * state.dim()
        shape0[axis0] = 2
        shape1 = [1] * state.dim()
        shape1[axis1] = 2
        idx0 = torch.tensor([0.0, 1.0], dtype=_C128, device=state.device).reshape(shape0)
        idx1 = torch.tensor([0.0, 1.0], dtype=_C128, device=state.device).reshape(shape1)
        mult = 1.0 - 2.0 * idx0 * idx1
        return state * mult

    @staticmethod
    def apply_cnot(state: Tensor, control: int, target: int) -> Tensor:
        """Flips `target`'s axis (via a full-tensor flip + control-conditioned select, no
        indexing/axis-shift bookkeeping): where control=0 keep the original state, where
        control=1 take the target-flipped version."""
        c_axis, t_axis = control + 1, target + 1
        flipped = torch.flip(state, dims=[t_axis])
        shape_c = [1] * state.dim()
        shape_c[c_axis] = 2
        control_is_1 = torch.tensor([0.0, 1.0], dtype=_C128, device=state.device).reshape(shape_c)
        control_is_0 = 1.0 - control_is_1
        return control_is_0 * state + control_is_1 * flipped

    @staticmethod
    def expval_z(state: Tensor, wire: int) -> Tensor:
        """<Z_wire> = P(wire=0) - P(wire=1), summed over every other qubit axis. [B]."""
        axis = wire + 1
        idx0 = [slice(None)] * state.dim()
        idx0[axis] = 0
        idx1 = [slice(None)] * state.dim()
        idx1[axis] = 1
        amp0 = state[tuple(idx0)]
        amp1 = state[tuple(idx1)]
        batch = state.shape[0]
        p0 = (amp0 * amp0.conj()).real.reshape(batch, -1).sum(dim=1)
        p1 = (amp1 * amp1.conj()).real.reshape(batch, -1).sum(dim=1)
        return p0 - p1

    @staticmethod
    def expval_z_mean(state: Tensor) -> Tensor:
        """(1/n) sum_wire <Z_wire> -- a sum of LOCAL observables (not a single global
        Pauli string spanning all qubits), the only two observable choices this project
        allows (Alg. 1 step 8, T2.4)."""
        n = state.dim() - 1
        total = StateVectorSim.expval_z(state, 0)
        for wire in range(1, n):
            total = total + StateVectorSim.expval_z(state, wire)
        return total / n
