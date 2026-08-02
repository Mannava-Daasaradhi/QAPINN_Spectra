"""SMCD Part 2: Algorithm 1 (T2.9, project.md Section 5.3), with the D5 correction at
step 4 (docs/derivations/07_circuit_fourier_spectrum.md Section 4).

Built AFTER card.py (T2.10) even though the phase doc lists T2.9 first, because this
algorithm's own DoD (weighted coverage = 1.0 for P1/P4) needs DesignCard's coverage()
function to score its own output -- the same pull-forward pattern already used for
T0.14/T0.15.
"""
from __future__ import annotations

import math

import numpy as np

from qapinn.models.circuits import ReuploadCircuit
from qapinn.pdes.base import PDE
from qapinn.smcd.card import DesignCard, coverage, d5_depth, predicted_benefit
from qapinn.smcd.symbol import TargetSpectrum, analytic_spectrum, empirical_spectrum

_ZERO_TOL = 1e-12
_GCD_TOL = 1e-6


def _gcd_like(vals: np.ndarray, tol: float = _GCD_TOL) -> float:
    """Largest real number that (numerically) divides every value in `vals` as an integer
    multiple -- the ternary-scaling base frequency Delta for ONE dimension. NOT the same
    as TargetSpectrum.delta (nearest-neighbor spacing across the WHOLE support, possibly
    multi-dimensional; see symbol.py's docstring on that distinction).

    Near-duplicate values (e.g. P4's four support points all sharing the exact same
    |omega_x|=3*pi) are collapsed via a TOLERANCE comparison, not `np.round` -- an earlier
    version rounded to 8 decimals for dedup and then returned the ROUNDED value itself,
    silently shifting a single-value Delta away from the true (unrounded) K by ~1e-8. That
    is enough for K/Delta to land at 1.0000000000495 instead of exactly 1.0, which
    d5_depth's ceil() then rounds up to the WRONG depth (P4 got L=2 instead of L=1, still
    passing coverage=1.0 by using extra unnecessary depth, but silently wrong). Sorting
    and comparing adjacent full-precision values against `tol` avoids ever discarding
    precision from a value that gets returned.
    """
    vals = np.sort(vals)
    deduped = [vals[0]]
    for v in vals[1:]:
        if v - deduped[-1] > tol:
            deduped.append(v)
    vals = np.array(deduped)
    if vals.size == 1:
        return float(vals[0])

    def _pair_gcd(a: float, b: float) -> float:
        while b > tol:
            a, b = b, math.fmod(a, b)
        return a

    result = float(vals[0])
    for v in vals[1:]:
        result = _pair_gcd(result, float(v))
        if result <= tol:
            return tol
    return max(result, tol)


def _per_dim_band(S: TargetSpectrum, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension (K_i, Delta_i) -- step 3 BAND, computed PER SPATIAL DIMENSION (not
    radially): Algorithm 1's circuit assigns one or more WIRES to each dimension
    independently (circuits.py's wire_to_dim), and D5's depth rule must be applied per
    dimension's own reachable band, not to a single global radial statistic (which would
    be wrong for d>1: P4's four support points all share the SAME radial magnitude
    pi*sqrt(10), which says nothing about the two axes' independent pi and 3*pi scales)."""
    omega_supp = S.omega[S.support]
    K = np.zeros(d)
    delta = np.zeros(d)
    for i in range(d):
        vals = np.abs(omega_supp[:, i])
        vals = vals[vals > _ZERO_TOL]
        if vals.size == 0:
            continue
        K[i] = float(vals.max())
        delta[i] = _gcd_like(vals)
    return K, delta


def _has_cross_terms(S: TargetSpectrum) -> bool:
    """True iff some support frequency has >= 2 nonzero components -- "genuine
    mixed-frequency terms" (step 5's condition), e.g. P4's (+-3*pi, +-1*pi) points. P1
    (d=1) can never have cross terms by construction."""
    omega_supp = S.omega[S.support]
    if omega_supp.shape[1] < 2:
        return False
    nonzero_counts = np.sum(np.abs(omega_supp) > _ZERO_TOL, axis=1)
    return bool(np.any(nonzero_counts >= 2))


def _symbol_and_target(pde: PDE, eps: float) -> tuple[TargetSpectrum, str]:
    """Steps 1 SYMBOL + 2 TARGET. For Burgers (no pde.symbol, nonlinear): use a viscous-
    scale estimate (the Burgers dissipation-cutoff frequency ~ 1/nu, a standard order-of-
    magnitude scale for where the nu*u_xx term suppresses higher modes) unioned with the
    empirical spectrum's own support, recording both in the returned notes string per the
    phase doc ("record both in the card")."""
    notes = ""
    S = analytic_spectrum(pde, eps)
    if S is not None:
        return S, notes

    S = empirical_spectrum(pde, eps)
    nu = pde.params.get("nu")
    if nu is not None and nu > 0:
        omega_visc = 1.0 / nu
        notes = (
            f"no closed-form symbol (nonlinear); target spectrum is the empirical "
            f"(FFT/sine-projection) path, with a viscous-scale estimate omega_visc="
            f"{omega_visc:.3g} (~1/nu) recorded as a cross-check upper bound -- "
            f"empirical K={S.K:.3g}."
        )
    return S, notes


def smcd(
    pde: PDE,
    eps: float = 1e-3,
    n_max: int = 8,
    L_max: int = 6,
    scaling_mode: str = "ternary",
    coverage_target: float | None = None,
) -> DesignCard:
    if scaling_mode != "ternary":
        raise NotImplementedError(
            f"smcd: only scaling_mode='ternary' is implemented (linear-scaling fallback "
            f"for dense low bands is a heuristic noted in the phase doc, not yet built); "
            f"got {scaling_mode!r}"
        )

    # --- steps 1-2: SYMBOL, TARGET ------------------------------------------------
    S, symbol_notes = _symbol_and_target(pde, eps)
    d = S.omega.shape[1]

    # --- step 3: BAND (per dimension) ----------------------------------------------
    K_dim, delta_dim = _per_dim_band(S, d)

    # --- step 4: DEPTH (D5, corrected) ----------------------------------------------
    L_dim = np.array(
        [d5_depth(K_dim[i], delta_dim[i]) if delta_dim[i] > 0 else 1 for i in range(d)]
    )
    L = int(L_dim.max()) if L_dim.size else 1

    # --- step 5: WIDTH ---------------------------------------------------------------
    cross_terms = _has_cross_terms(S)
    n_cross = 1 if cross_terms else 0
    n = d + n_cross

    # --- step 6: CHECK (octave split trigger, T2.15 not yet implemented) ------------
    octave_split = n > n_max or L > L_max
    check_notes = ""
    if octave_split:
        check_notes = (
            f"n={n} > n_max={n_max} or L={L} > L_max={L_max}: octave split needed "
            f"(T2.15, not yet implemented) -- card reports the UNSPLIT design."
        )

    # --- steps 7-8: ENTANGLER, OBSERVABLE --------------------------------------------
    entangler = "ring_cz" if cross_terms else "none"
    observable = "z0"

    # wire assignment: round-robin across dims, so any "extra" (n_cross) wire lands on
    # dim 0 -- gives that dimension a second, independently-trainable wire at the SAME
    # ternary scaling (redundant frequency reach, independent amplitude parameters).
    wire_to_dim = tuple(i % d for i in range(n))

    scalings = np.zeros((L, n))
    for q in range(n):
        dim = wire_to_dim[q]
        base = delta_dim[dim] if delta_dim[dim] > 0 else 1.0
        scalings[:, q] = base * (3.0 ** np.arange(L))

    circuit = ReuploadCircuit(
        n_qubits=n,
        n_layers=L,
        scalings=scalings,
        wire_to_dim=wire_to_dim,
        entangler=entangler,
        observable=observable,
    )
    Omega = circuit.frequencies()
    if Omega.ndim == 1:
        Omega = Omega.reshape(-1, 1)

    # --- steps 9-10: ANSATZ, INIT (recorded as notes; consumed by T2.12's hybrid model
    # construction, not by the card's own fields) -------------------------------------
    ansatz_notes = "ansatz=hard-BC (D6 default); theta~N(0,0.1^2), encoder A=I (D3)."

    # --- coverage / predicted_benefit, on the UNSPLIT design --------------------------
    cov_count, cov_weighted = coverage(S, Omega)
    benefit = predicted_benefit(S, Omega)

    if coverage_target is not None:
        L, scalings, circuit, Omega, cov_count, cov_weighted = _detune_to_target(
            S, L, n, wire_to_dim, entangler, observable, delta_dim, coverage_target
        )
        benefit = predicted_benefit(S, Omega)

    param_count = (L + 1) * n * 2
    K_overall = float(S.K)
    delta_overall = float(delta_dim[delta_dim > 0].min()) if np.any(delta_dim > 0) else 0.0

    notes = " ".join(part for part in (symbol_notes, check_notes, ansatz_notes) if part)

    return DesignCard(
        pde=pde.name,
        eps=eps,
        target_omega=S.omega.tolist(),
        target_weight=S.weight.tolist(),
        omega_set=Omega.tolist(),
        coverage=cov_count,
        coverage_weighted=cov_weighted,
        predicted_benefit=benefit,
        n_qubits=n,
        n_layers=L,
        scalings=scalings.tolist(),
        wire_to_dim=list(wire_to_dim),
        entangler=entangler,
        observable=observable,
        param_count=param_count,
        predicted_ntk_band=(delta_overall, K_overall),
        octave_split=octave_split,
        notes=notes,
    )


def _detune_to_target(
    S: TargetSpectrum,
    L_full: int,
    n: int,
    wire_to_dim: tuple[int, ...],
    entangler: str,
    observable: str,
    delta_dim: np.ndarray,
    coverage_target: float,
):
    """DELIBERATELY de-tune the design to hit approximately coverage_target (Phase-3
    coverage sweep, project.md Section 5.3). Searches over (L, Delta-mismatch-factor)
    pairs -- reducing L shrinks the reachable band; multiplying Delta by a factor shifts
    the reachable LATTICE so target frequencies fall off it -- and returns whichever
    combination's achieved coverage is closest to the target.

    Optimizes against the COUNT metric (|S_hat intersect Omega| / |S_hat|), not the
    weighted one, DELIBERATELY: this project's four PDEs' target supports are small,
    exact point sets (P1/P2: 2 points; P4: 4 points, but symmetric under sign flips in
    each axis independently, so the reachable-set construction can only ever include ALL
    4 or NONE of them, never a proper subset -- verified directly: no (L, Delta-factor)
    combination produces 2-of-4). Weighted coverage for a 2-point target can therefore
    only take the values {0, w_lo/(w_lo+w_hi), w_hi/(w_lo+w_hi), 1} -- for P1's actual
    weights {1, 0.3} that is {0, 0.231, 0.769, 1.0}, which never lands in a generic
    requested band like [0.4, 0.6]. The COUNT metric for the same 2-point target can hit
    exactly 0.5 (cover 1 of 2 points, regardless of which one), which is achievable and is
    what tests/test_smcd_design.py's DoD item 4 actually exercises. Both coverage and
    coverage_weighted in the returned card reflect whatever was ACTUALLY achieved at the
    selected design, not the requested target.
    """
    best = None
    factors = np.concatenate([[1.0], np.linspace(1.01, 3.0, 40)])
    for L_try in range(1, L_full + 1):
        for factor in factors:
            scalings = np.zeros((L_try, n))
            for q in range(n):
                dim = wire_to_dim[q]
                base = (delta_dim[dim] if delta_dim[dim] > 0 else 1.0) * factor
                scalings[:, q] = base * (3.0 ** np.arange(L_try))
            circuit = ReuploadCircuit(
                n_qubits=n,
                n_layers=L_try,
                scalings=scalings,
                wire_to_dim=wire_to_dim,
                entangler=entangler,
                observable=observable,
            )
            Omega = circuit.frequencies()
            if Omega.ndim == 1:
                Omega = Omega.reshape(-1, 1)
            cov_count, cov_weighted = coverage(S, Omega)
            dist = abs(cov_count - coverage_target)
            if best is None or dist < best[0]:
                best = (dist, L_try, scalings, circuit, Omega, cov_count, cov_weighted)

    _, L_best, scalings_best, circuit_best, Omega_best, cov_count_best, cov_weighted_best = best
    return L_best, scalings_best, circuit_best, Omega_best, cov_count_best, cov_weighted_best
