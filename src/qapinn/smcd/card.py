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
    # T2.15: when octave_split is True, the n_qubits/n_layers/scalings/wire_to_dim/
    # entangler/observable fields above still describe the single OVER-BUDGET design
    # (kept for backward compatibility / documentation), while octave_configs holds the
    # REAL list of per-octave circuit configs (each directly consumable by
    # hybrid.OctaveEnsemble) that should actually be built and trained. None when no
    # split was needed.
    octave_configs: list | None = None

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


def matched_target_frequencies(card: DesignCard, rtol: float = 1e-6) -> np.ndarray:
    """The design's own reachable set Omega, RESTRICTED to the target support S_hat
    (project.md Section 6's c_rff_matched ablation, T2.14) -- the frequencies
    `c_rff_matched` should use as its fixed Fourier-feature matrix B. `card.target_omega`
    is already support-restricted (S_hat); intersecting it against `card.omega_set` via
    the same relative-tolerance matching as coverage() means a de-tuned (coverage < 1.0)
    card correctly excludes target frequencies the circuit can't actually reach, rather
    than silently claiming a frequency the model has no way to represent.
    """
    target_omega = np.asarray(card.target_omega, dtype=float)
    omega_set = np.asarray(card.omega_set, dtype=float)
    if target_omega.shape[0] == 0:
        return target_omega
    matched = _match_mask(target_omega, omega_set, rtol)
    return target_omega[matched]


def matched_and_padded_frequencies(
    card: DesignCard, target_param_count: int, tol: float = 0.10, rtol: float = 1e-6
) -> np.ndarray:
    """`matched_target_frequencies(card)` (the REQUIRED subset -- guarantees the
    ablation's frequencies literally include the target support), PADDED with additional
    rows of `card.omega_set` (in increasing |omega| order, i.e. nearest to DC first, a
    simple deterministic policy) until a `FourierFeaturePINN`'s param count
    (`2*m + 1` -- `nn.Linear(2*m, 1)`, `m` = number of frequency rows) lands within `tol`
    of `target_param_count`. Needed because the raw target support alone is typically far
    too small to match q_serial's own parameter count (T2.14's DoD requires BOTH, for P1
    specifically: `c_rff_matched.realised_frequencies()` contains the target frequencies,
    AND its param count is matched to q_serial within 10% -- for P1 the bare 2-frequency
    support gives only 5 params against q_serial's 14, so padding is not optional).

    "Contains the target support" always wins over "matches within tol": if the REQUIRED
    set alone already exceeds target_param_count + tol (verified for Burgers: its
    empirical target support has 16 points, needing 33 params, against a q_serial design
    of only 14 -- richer/broader target spectra than P1's own can genuinely outstrip a
    small quantum design's own capacity), no padding happens and the required set is
    returned as-is, best-effort, WITHOUT raising -- T2.14's DoD only pins the exact 10%
    match for P1; the weaker "runs error-free on all four PDEs" requirement (T1.13's
    deferred criterion) must still hold even when the tighter match can't be achieved.
    Similarly, if `card.omega_set` runs out of distinct rows before reaching the target
    from below, whatever was accumulated is returned rather than raising.
    """
    required = matched_target_frequencies(card, rtol=rtol)
    omega_set = np.asarray(card.omega_set, dtype=float)

    def _n_params(m: int) -> int:
        return 2 * m + 1

    if abs(_n_params(required.shape[0]) - target_param_count) <= tol * target_param_count:
        return required
    if _n_params(required.shape[0]) > target_param_count:
        return required  # already over budget just from the required set; can't shrink further

    order = np.argsort(np.linalg.norm(omega_set, axis=1))
    pool = omega_set[order]
    already = _match_mask(pool, required, rtol) if required.shape[0] > 0 else np.zeros(pool.shape[0], dtype=bool)

    result = required
    for row in pool[~already]:
        if abs(_n_params(result.shape[0]) - target_param_count) <= tol * target_param_count:
            break
        if _n_params(result.shape[0]) > target_param_count:
            break
        result = np.vstack([result, row[None, :]]) if result.shape[0] > 0 else row[None, :]

    return result
