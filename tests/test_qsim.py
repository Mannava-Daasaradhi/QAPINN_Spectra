"""T2.3 DoD:
1. Norm preservation: ||state||^2 == 1 to 1e-12 after a random circuit.
2. Known values: RY(pi/2) on |0> gives <Z> = 0; RX(pi) gives <Z> = -1.
3. Bell state via RY(pi/2) + CNOT gives <Z_0> = <Z_1> = 0 and correlation 1.
4. Second derivatives flow: torch.autograd.grad twice through expval_z w.r.t. an input
   angle returns a finite tensor.
"""
from __future__ import annotations

import math

import torch

from qapinn.models.qsim import StateVectorSim, rx, ry, rz


def test_norm_preserved_after_random_circuit():
    torch.manual_seed(0)
    n = 3
    batch = 5
    state = StateVectorSim.zeros_state(batch, n)

    gates = [rx, ry, rz]
    for _ in range(12):
        wire = torch.randint(0, n, (1,)).item()
        gate_fn = gates[torch.randint(0, 3, (1,)).item()]
        theta = torch.randn(batch) * 3.0
        state = StateVectorSim.apply_1q(state, gate_fn(theta), wire)
    state = StateVectorSim.apply_cz(state, 0, 1)
    state = StateVectorSim.apply_cnot(state, 1, 2)
    state = StateVectorSim.apply_cz(state, 0, 2)

    norm_sq = (state * state.conj()).real.flatten(1).sum(dim=1)
    assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-12)


def test_ry_pi_over_2_gives_zero_expval_z():
    state = StateVectorSim.zeros_state(1, 1)
    state = StateVectorSim.apply_1q(state, ry(torch.tensor(math.pi / 2)), 0)
    ez = StateVectorSim.expval_z(state, 0)
    assert torch.allclose(ez, torch.zeros_like(ez), atol=1e-10)


def test_rx_pi_gives_minus_one_expval_z():
    state = StateVectorSim.zeros_state(1, 1)
    state = StateVectorSim.apply_1q(state, rx(torch.tensor(math.pi)), 0)
    ez = StateVectorSim.expval_z(state, 0)
    assert torch.allclose(ez, -torch.ones_like(ez), atol=1e-10)


def test_bell_state_marginals_zero_and_correlation_one():
    state = StateVectorSim.zeros_state(1, 2)
    state = StateVectorSim.apply_1q(state, ry(torch.tensor(math.pi / 2)), 0)
    state = StateVectorSim.apply_cnot(state, control=0, target=1)

    ez0 = StateVectorSim.expval_z(state, 0)
    ez1 = StateVectorSim.expval_z(state, 1)
    assert torch.allclose(ez0, torch.zeros_like(ez0), atol=1e-10)
    assert torch.allclose(ez1, torch.zeros_like(ez1), atol=1e-10)

    probs = (state * state.conj()).real  # [1,2,2]
    signs = torch.tensor([1.0, -1.0], dtype=probs.dtype)
    ezz = torch.einsum("bij,i,j->b", probs, signs, signs)
    assert torch.allclose(ezz, torch.ones_like(ezz), atol=1e-10)


def test_second_derivatives_flow_through_expval_z():
    # RY(pi/2) -> RZ(x) -> RY(pi/2) -> measure Z: analytically <Z> = -cos(x)
    # (derived in docs/derivations/07_circuit_fourier_spectrum.md's L=1 case), so this is
    # a genuinely x-dependent circuit, not a degenerate all-diagonal one.
    x = torch.tensor(0.7, requires_grad=True)
    state = StateVectorSim.zeros_state(1, 1)
    state = StateVectorSim.apply_1q(state, ry(torch.tensor(math.pi / 2)), 0)
    state = StateVectorSim.apply_1q(state, rz(x), 0)
    state = StateVectorSim.apply_1q(state, ry(torch.tensor(math.pi / 2)), 0)
    ez = StateVectorSim.expval_z(state, 0)

    (grad1,) = torch.autograd.grad(ez.sum(), x, create_graph=True)
    (grad2,) = torch.autograd.grad(grad1.sum(), x, create_graph=True)

    assert torch.isfinite(grad2).all()
    # cross-check against the closed form <Z> = -cos(x): d<Z>/dx = sin(x), d^2<Z>/dx^2 = cos(x)
    assert torch.allclose(ez, -torch.cos(x.detach()), atol=1e-10)
    assert torch.allclose(grad1, torch.sin(x.detach()), atol=1e-8)
    assert torch.allclose(grad2, torch.cos(x.detach()), atol=1e-8)


def test_apply_1q_batched_gate_matches_per_sample_angles():
    # per-sample encoding angles (omega * x_b) must broadcast correctly, not silently
    # share one angle across the batch.
    thetas = torch.tensor([0.0, math.pi / 2, math.pi])
    state = StateVectorSim.zeros_state(3, 1)
    state = StateVectorSim.apply_1q(state, rx(thetas), 0)
    ez = StateVectorSim.expval_z(state, 0)
    expected = torch.tensor([1.0, 0.0, -1.0])
    assert torch.allclose(ez, expected, atol=1e-10)


def test_expval_z_mean_averages_local_observables():
    state = StateVectorSim.zeros_state(1, 2)
    state = StateVectorSim.apply_1q(state, rx(torch.tensor(math.pi)), 0)  # wire 0 -> <Z>=-1
    # wire 1 stays |0> -> <Z>=+1
    mean = StateVectorSim.expval_z_mean(state)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-10)
