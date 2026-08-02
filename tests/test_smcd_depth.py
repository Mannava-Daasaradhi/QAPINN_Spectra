"""T2.10 DoD: tests/test_smcd_depth.py pins the D5 formula: for (K/delta) in
{1, 4, 13, 14, 40, 41} assert L in {1, 2, 3, 4, 4, 5} respectively, and assert
max|Omega| >= K in every case. Card round-trips through JSON unchanged.
"""
from __future__ import annotations

import math

import numpy as np

from qapinn.smcd.card import DesignCard, coverage, d5_depth
from qapinn.smcd.symbol import TargetSpectrum

RATIOS_AND_L = [(1, 1), (4, 2), (13, 3), (14, 4), (40, 4), (41, 5)]


def test_d5_depth_pinned_table():
    delta = 1.0
    for ratio, expected_L in RATIOS_AND_L:
        K = ratio * delta
        L = d5_depth(K, delta)
        assert L == expected_L, f"K/delta={ratio}: got L={L}, expected {expected_L}"

        max_omega = delta * (3.0**L - 1.0) / 2.0
        assert max_omega >= K, f"K/delta={ratio}: max|Omega|={max_omega} < K={K}"


def test_d5_depth_rejects_nonpositive_delta():
    import pytest

    with pytest.raises(ValueError):
        d5_depth(K=10.0, delta=0.0)


def _sample_card() -> DesignCard:
    return DesignCard(
        pde="poisson",
        eps=1e-3,
        target_omega=[[math.pi], [15.0 * math.pi]],
        target_weight=[1.0, 0.3],
        omega_set=[[-math.pi], [0.0], [math.pi]],
        coverage=0.5,
        coverage_weighted=0.769,
        predicted_benefit=0.1,
        n_qubits=1,
        n_layers=4,
        scalings=[[math.pi], [3.0 * math.pi], [9.0 * math.pi], [1.0]],
        wire_to_dim=[0],
        entangler="none",
        observable="z0",
        param_count=10,
        predicted_ntk_band=(math.pi, 15.0 * math.pi),
        octave_split=False,
        notes="test card",
    )


def test_design_card_json_roundtrip():
    card = _sample_card()
    restored = DesignCard.from_json(card.to_json())
    assert restored == card


def test_coverage_full_and_partial():
    S = TargetSpectrum(
        omega=np.array([[math.pi], [15.0 * math.pi]]),
        weight=np.array([1.0, 0.3]),
        eps=1e-3,
        support=np.array([True, True]),
        K=15.0 * math.pi,
        delta=14.0 * math.pi,
        source="analytic",
    )
    full_omega = np.array([[-15.0 * math.pi], [0.0], [math.pi], [15.0 * math.pi]])
    count_cov, weighted_cov = coverage(S, full_omega)
    assert count_cov == 1.0
    assert weighted_cov == 1.0

    partial_omega = np.array([[-math.pi], [0.0], [math.pi]])
    count_cov, weighted_cov = coverage(S, partial_omega)
    assert count_cov == 0.5
    np.testing.assert_allclose(weighted_cov, 1.0 / 1.3)


def test_coverage_empty_support_is_vacuous():
    S = TargetSpectrum(
        omega=np.array([[math.pi]]),
        weight=np.array([1.0]),
        eps=1e-3,
        support=np.array([False]),
        K=0.0,
        delta=0.0,
        source="analytic",
    )
    count_cov, weighted_cov = coverage(S, np.array([[0.0]]))
    assert count_cov == 1.0
    assert weighted_cov == 1.0
