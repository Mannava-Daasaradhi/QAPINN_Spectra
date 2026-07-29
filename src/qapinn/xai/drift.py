"""Encoder spectral drift (T1.8, D3 / project.md instrument 7.7, new instrument).

D3: the quantum encoder is affine, E(x) = A x + b, A identity-initialised, so the model's
realised frequencies in physical coordinates are Omega . A (Omega fixed per-D4, not
trainable). Track how A (and therefore the realised frequency set) moves during training.

No real quantum family exists yet (Phase 2, T2.x). Until then, `encoder_drift` recognises
a model's affine encoder via a small duck-typed protocol -- `model.encoder_A` (the current,
trainable [d,d] matrix) and `model.encoder_omega` (the fixed [m,d] Omega buffer) -- so this
instrument can be implemented and tested now against a minimal stand-in (see
tests/test_drift.py), and will work unmodified once the real quantum model exposes the
same two attributes.
"""
from __future__ import annotations

import numpy as np

from qapinn.models.base import PINNModel


def encoder_drift(model: PINNModel, target_omega: np.ndarray | None = None) -> dict:
    """{'frobenius': ||A - I||_F, 'realised_omega': Omega . A, 'coverage_now': float}.

    Classical families with no frequency structure at all (c_mlp) return {}.

    Fourier-feature families (c_ff, c_rff_matched): the B buffer already IS Omega . A,
    fixed for the model's whole life (A is implicitly, permanently the identity) -- so
    this reports frobenius=0.0 by construction and realised_omega =
    model.realised_frequencies() directly. Reported anyway for symmetry with the quantum
    case, not because it moves (D3).

    Quantum families expose a trainable affine encoder via `model.encoder_A` (current A)
    and `model.encoder_omega` (fixed Omega); frobenius = ||A-I||_F, realised_omega =
    Omega @ A.

    `coverage_now`: fraction of `target_omega` reachable by `realised_omega` (nearest-
    neighbour within a small tolerance). Requires a real target spectrum (SMCD design
    card, Phase 2) -- NaN when `target_omega` is not supplied, which is the common case
    until Phase 2 exists (D3 documents this same "coverage is undefined" caveat for its
    nonlinear-encoder ablation).
    """
    omega = model.realised_frequencies()
    if omega is None:
        return {}

    has_trainable_encoder = hasattr(model, "encoder_A") and hasattr(model, "encoder_omega")
    if not has_trainable_encoder:
        realised_omega = np.asarray(omega)
        frobenius = 0.0
    else:
        A = model.encoder_A.detach().cpu().numpy()
        omega_buf = model.encoder_omega
        Omega = omega_buf.detach().cpu().numpy() if hasattr(omega_buf, "detach") else np.asarray(omega_buf)
        identity = np.eye(A.shape[0])
        frobenius = float(np.linalg.norm(A - identity, ord="fro"))
        realised_omega = Omega @ A

    coverage_now = _coverage(realised_omega, target_omega) if target_omega is not None else float("nan")

    return {"frobenius": frobenius, "realised_omega": realised_omega, "coverage_now": coverage_now}


def _coverage(realised_omega: np.ndarray, target_omega: np.ndarray, tol: float = 1e-6) -> float:
    """Fraction of `target_omega` rows with a matching row in `realised_omega` within
    `tol` (Euclidean distance). Placeholder metric until SMCD's real coverage definition
    (T2.x) supersedes it."""
    target = np.atleast_2d(target_omega)
    realised = np.atleast_2d(realised_omega)
    hits = 0
    for row in target:
        dists = np.linalg.norm(realised - row, axis=-1)
        if dists.min() <= tol:
            hits += 1
    return hits / len(target)
