"""Data-re-uploading circuit module (T2.4). Layer structure and frequencies() implement
docs/derivations/07_circuit_fourier_spectrum.md's Prop. 1 directly -- do not vary the
layer structure without updating tests/test_circuit_spectrum.py (T2.6).

PREP_ANGLE correction (found during T2.6's own verification, not a T2.4 bug): the phase
doc prescribes RY(pi/2) exactly (equal superposition, a=b=1/sqrt(2)) to avoid the "RZ on
|0> is a global phase" degeneracy. That avoids ONE degeneracy but, verified directly
against the PennyLane oracle (T2.5) and independent of trainable-gate generality (tested
with a full RZ-RY-RZ and even RX-RY-RZ trainable block -- same result both times),
introduces a DIFFERENT one: with a=b EXACTLY, every ternary-scaled frequency whose
balanced-ternary digit m_1=0 (docs/derivations/07_circuit_fourier_spectrum.md Section 4)
-- exactly 1/3 of Omega -- has IDENTICALLY ZERO amplitude for every theta. Perturbing the
prep angle by even 0.1 rad away from pi/2 (or using pi/3) removes the degeneracy entirely
(0 structurally-zero bins, verified numerically). This matters beyond a test threshold:
Prop. 3 ("S_eps subset Omega => representable by a linear head") would be FALSE for any
target spectrum touching one of those dead frequencies under the literal pi/2 prep, so
this is fixed at the source rather than worked around in the test.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from torch import Tensor, nn

from qapinn.models.qsim import StateVectorSim, ry, rz

_SIGNS = (-1, 0, 1)
# NOT pi/2 -- see module docstring. Any angle other than {0, pi/2, pi} (mod pi) avoids
# both known degeneracies; pi/3 is a clean, well away from all three.
PREP_ANGLE = math.pi / 3


class ReuploadCircuit(nn.Module):
    """[B, d] -> [B, 1]. Layer structure (fixed):
        for l in range(L):
            for q in range(n): RZ(scalings[l,q] * z[:, wire_to_dim[q]])   # encoding
            for q in range(n): RY(theta[l,q,0]); RZ(theta[l,q,1])        # trainable
            if entangler == "ring_cz": CZ(q, (q+1) % n) for all q
        final: RY(theta[L,q,0]); RZ(theta[L,q,1]) for all q, then measure the observable
    A RY(PREP_ANGLE) is prepended on every wire before the first encoding gate -- an RZ on
    |0> is a global phase and would produce a constant (x-independent) output, the single
    most common "my circuit has no frequency content" bug (T2.4 phase doc note).
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        scalings,
        wire_to_dim: tuple[int, ...],
        entangler: str = "ring_cz",
        observable: str = "z0",
        init_sigma: float = 0.1,
    ):
        super().__init__()
        if observable not in ("z0", "z_mean"):
            raise ValueError(
                f"observable must be 'z0' or 'z_mean' (local only, Alg. 1 step 8 bans global "
                f"observables); got {observable!r}"
            )
        if entangler not in ("ring_cz", "none"):
            raise ValueError(f"entangler must be 'ring_cz' or 'none'; got {entangler!r}")
        if len(wire_to_dim) != n_qubits:
            raise ValueError(f"wire_to_dim must have length n_qubits={n_qubits}, got {len(wire_to_dim)}")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.wire_to_dim = tuple(wire_to_dim)
        self.entangler = entangler
        self.observable = observable

        scalings_t = torch.as_tensor(scalings, dtype=torch.get_default_dtype())
        if tuple(scalings_t.shape) != (n_layers, n_qubits):
            raise ValueError(
                f"scalings must have shape [n_layers, n_qubits] = [{n_layers},{n_qubits}], "
                f"got {tuple(scalings_t.shape)}"
            )
        # D4: scalings are prescribed by SMCD, not trained -- a buffer, never a Parameter.
        self.register_buffer("scalings", scalings_t)

        # Alg. 1 step 10: small-angle init. L+1 trainable RY/RZ rounds (one after every
        # encoding layer, plus one final round after the loop).
        theta = torch.randn(n_layers + 1, n_qubits, 2) * init_sigma
        self.theta = nn.Parameter(theta)

    def forward(self, z: Tensor) -> Tensor:
        return self.evaluate(z)

    def evaluate(
        self,
        z: Tensor,
        theta: Tensor | None = None,
        shifts: dict[tuple[int, int], float] | None = None,
    ) -> Tensor:
        """The circuit evaluation shared by forward() and pshift.py's exact
        parameter-shift derivatives (T2.7). `theta` defaults to self.theta (this is what
        forward() uses); passing a different tensor evaluates the SAME circuit structure
        at different trainable angles without mutating self.theta (used by
        psr_grad_theta). `shifts` adds a raw phase offset to specific encoding-gate
        INSTANCES, keyed by `(layer, wire)` -- needed because z is shared across every
        re-uploading layer, so shifting z itself would shift every encoding gate that
        reads it at once, not the single gate instance the shift rule needs held fixed
        (used by psr_grad_input / psr_grad2_input).
        """
        if theta is None:
            theta = self.theta
        batch = z.shape[0]
        device = z.device
        state = StateVectorSim.zeros_state(batch, self.n_qubits, device=device)

        prep = ry(torch.tensor(PREP_ANGLE, device=device))
        for q in range(self.n_qubits):
            state = StateVectorSim.apply_1q(state, prep, q)

        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                angle = self.scalings[l, q] * z[:, self.wire_to_dim[q]]
                if shifts is not None and (l, q) in shifts:
                    angle = angle + shifts[(l, q)]
                state = StateVectorSim.apply_1q(state, rz(angle), q)
            for q in range(self.n_qubits):
                state = StateVectorSim.apply_1q(state, ry(theta[l, q, 0]), q)
                state = StateVectorSim.apply_1q(state, rz(theta[l, q, 1]), q)
            if self.entangler == "ring_cz" and self.n_qubits >= 2:
                # For n_qubits==2, "ring" over range(n) applies CZ(0,1) then CZ(1,0) --
                # the SAME pair twice (CZ is symmetric in its two wires), and CZ*CZ=I
                # (diagonal +-1 matrix), so the two calls cancel exactly, silently
                # disabling entanglement entirely (found via T2.6's FFT-based check: no
                # design-level or PennyLane-mirror test catches this, since both replicate
                # the same loop). n>=3 has n distinct cyclic pairs, no such collision.
                n_cz_pairs = 1 if self.n_qubits == 2 else self.n_qubits
                for q in range(n_cz_pairs):
                    state = StateVectorSim.apply_cz(state, q, (q + 1) % self.n_qubits)

        for q in range(self.n_qubits):
            state = StateVectorSim.apply_1q(state, ry(theta[self.n_layers, q, 0]), q)
            state = StateVectorSim.apply_1q(state, rz(theta[self.n_layers, q, 1]), q)

        out = StateVectorSim.expval_z(state, 0) if self.observable == "z0" else StateVectorSim.expval_z_mean(state)
        return out.unsqueeze(-1)

    def frequencies(self) -> np.ndarray:
        """Omega, per Prop. 1 (docs/derivations/07_circuit_fourier_spectrum.md Section 3):
        per input dimension, the signed-subset-sum set of that dimension's wires' scalings
        across all layers, {sum_l m_l*scalings[l,q] : m_l in {-1,0,1}}, combined additively
        across wires sharing a dimension. For d==1 this is a flat [M] array of scalar
        frequencies. For d>1 with `entangler='none'`, wires acting on DIFFERENT dimensions
        do not correlate before the (single-wire) measurement, so only one dimension's
        slot is nonzero per term (a "cross" in frequency space, not a full grid) --
        exactly why entangler='none' shows only its own wire's frequencies (T2.6). With an
        entangler, the full joint (Cartesian-product) set is reachable, returned as [M, d].
        """
        scalings = self.scalings.detach().cpu().numpy()  # [L, n]
        L, n = scalings.shape

        def _wire_reachable_set(q: int) -> set[float]:
            vals = {0.0}
            for l in range(L):
                vals = {v + m * scalings[l, q] for v in vals for m in _SIGNS}
            return vals

        dims = sorted(set(self.wire_to_dim))
        d = len(dims)

        def _dim_reachable_set(dim: int) -> list[float]:
            wires_here = [q for q in range(n) if self.wire_to_dim[q] == dim]
            combined = {0.0}
            for q in wires_here:
                wire_set = _wire_reachable_set(q)
                combined = {a + b for a in combined for b in wire_set}
            return sorted(combined)

        per_dim_sets = [_dim_reachable_set(dim) for dim in dims]

        if d == 1:
            return np.array(per_dim_sets[0], dtype=float)

        if self.entangler == "none":
            omega = []
            for i, dim_set in enumerate(per_dim_sets):
                for val in dim_set:
                    vec = [0.0] * d
                    vec[i] = val
                    omega.append(vec)
            return np.unique(np.array(omega, dtype=float), axis=0)

        return np.array(list(itertools.product(*per_dim_sets)), dtype=float)
