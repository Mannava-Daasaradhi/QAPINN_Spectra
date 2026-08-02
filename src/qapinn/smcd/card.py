"""SMCD Part 3: design card & coverage metric (T2.10, project.md Section 5.3-5.4).

Built AHEAD of T2.9 in task-index order (same pull-forward pattern as T0.14/T0.15):
Algorithm 1 (T2.9, design.py) needs DesignCard and coverage() to exist to construct and
score its own output, so this file lands first even though the phase doc lists T2.9 before
T2.10.

Also houses the D5 depth-rule utility (Algorithm 1 step 4) as its own standalone,
independently-testable function, since tests/test_smcd_depth.py pins its exact table of
(K/Delta -> L) values directly.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from qapinn.smcd.symbol import TargetSpectrum

# Pre-registered "classical reach" frequency (project.md Section 5.3's predicted_benefit
# heuristic): the geometric mean of P1's own two known target frequencies (pi, 15*pi).
# Chosen because T1.5's OWN measured evidence (BENCH.md's T1.5 section: steps_to_tolerance
# ratio 16.2x between pi and 15*pi -- pi converges fast, 15*pi converges 16x slower) shows
# the classical c_mlp baseline's practical "reach" sits clearly BETWEEN these two
# frequencies; a geometric (not arithmetic) mean is the natural midpoint for a phenomenon
# (spectral bias) that is inherently log-frequency-scaled.
#
# PROVISIONAL: project.md Section 5.3 calls for this to be "frozen before Phase 3" via a
# proper NTK-eigenvalue-crossing analysis. T1.5's saved run artifacts only kept
# per-frequency error trajectories (xai/specerr.npz), not NTK eigenspectra (T1.5's own
# run_instruments call didn't include the 'ntk' instrument), so that sharper analysis
# needs a fresh training run and is deferred -- do not treat this value as final.
OMEGA_KNEE = math.pi * math.sqrt(15.0)

_ISCLOSE_ATOL = 1e-9


def d5_depth(K: float, delta: float) -> int:
    """L = ceil(log_3(2*K/delta + 1)) -- the CORRECTED depth rule (D5), derived in
    docs/derivations/07_circuit_fourier_spectrum.md Section 4. project.md Algorithm 1's
    own literal step 4 states L = ceil(log_3(K/delta)) instead, which under-covers K (T2.1
    found this counterexample: P1's K=15*pi, delta=pi gives L=3 under the literal
    formula, but max|Omega| at L=3 is only 13*pi < 15*pi -- 15*pi is NOT reachable). This
    task's own phase-doc table (T2.9) already specifies the corrected formula directly.
    """
    if delta <= 0:
        raise ValueError(f"d5_depth: delta must be positive, got {delta}")
    if K <= 0:
        return 1
    return max(1, math.ceil(math.log(2.0 * K / delta + 1.0, 3.0)))


@dataclass(frozen=True)
class DesignCard:
    pde: str
    eps: float
    target_omega: list
    target_weight: list
    omega_set: list
    coverage: float
    coverage_weighted: float
    predicted_benefit: float
    n_qubits: int
    n_layers: int
    scalings: list
    wire_to_dim: list
    entangler: str
    observable: str
    param_count: int
    predicted_ntk_band: tuple[float, float]
    octave_split: bool
    notes: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "DesignCard":
        d = json.loads(s)
        d["predicted_ntk_band"] = tuple(d["predicted_ntk_band"])
        return cls(**d)


def _match_mask(target_omega: np.ndarray, Omega: np.ndarray, rtol: float) -> np.ndarray:
    """[M] bool: for each target_omega row, whether SOME row of Omega matches it within
    rtol (relative) / _ISCLOSE_ATOL (absolute floor, needed since omega components can be
    exactly 0). Never uses `==` directly (D12: frequencies are floats built via different
    code paths -- a symbol-derived 15*pi and a scaling-product 15*pi differ in the last
    bits)."""
    if target_omega.shape[0] == 0 or Omega.shape[0] == 0:
        return np.zeros(target_omega.shape[0], dtype=bool)
    close = np.isclose(Omega[None, :, :], target_omega[:, None, :], rtol=rtol, atol=_ISCLOSE_ATOL)
    return close.all(axis=2).any(axis=1)


def coverage(S: TargetSpectrum, Omega: np.ndarray, rtol: float = 1e-6) -> tuple[float, float]:
    """(coverage_count, coverage_weighted) = (|S_hat intersect Omega| / |S_hat|,
    sum_{S_hat intersect Omega} w / sum_{S_hat} w), matching within a RELATIVE tolerance
    (never `==`). An empty target support is vacuously fully covered (1.0, 1.0) -- nothing
    is required, so nothing can be missing.
    """
    target_omega = S.omega[S.support]
    target_weight = S.weight[S.support]
    if target_omega.shape[0] == 0:
        return 1.0, 1.0

    matched = _match_mask(target_omega, Omega, rtol)
    count_cov = float(matched.sum()) / float(matched.shape[0])
    total_w = float(target_weight.sum())
    weighted_cov = float(target_weight[matched].sum()) / total_w if total_w > 0 else 1.0
    return count_cov, weighted_cov


def predicted_benefit(
    S: TargetSpectrum, Omega: np.ndarray, omega_knee: float = OMEGA_KNEE, rtol: float = 1e-6
) -> float:
    """Fraction of the target support's energy that lies BOTH above omega_knee (out of
    reach of the classical c_mlp baseline, per T1.5) AND inside Omega (this circuit can
    actually represent it) -- project.md Section 5.3's pre-registered a-priori benefit
    heuristic. 0.0 if the support carries no energy."""
    target_omega = S.omega[S.support]
    target_weight = S.weight[S.support]
    total_w = float(target_weight.sum())
    if target_omega.shape[0] == 0 or total_w <= 0:
        return 0.0

    mags = np.linalg.norm(target_omega, axis=1)
    above_knee = mags > omega_knee
    matched = _match_mask(target_omega, Omega, rtol)
    numer = float(target_weight[above_knee & matched].sum())
    return numer / total_w
