"""T2.9 DoD: tests/test_smcd_design.py asserts
1. P1 -> L=4, n=1, entangler="none", weighted coverage = 1.0.
2. P4 k=10 -> n=3 (2 dims + 1 cross), entangler="ring_cz", coverage = 1.0.
3. P2 -> the card's predicted_benefit field is < 0.05 (the a-priori negative prediction).
4. coverage_target=0.5 returns a card with achieved coverage in [0.4, 0.6].
5. Determinism: two calls with the same inputs give byte-identical cards.

Item 4 is checked against `card.coverage` (the COUNT metric), not `coverage_weighted` --
see design.py's `_detune_to_target` docstring: this project's four PDEs have small, exact
point-set target supports (P1/P2: 2 points; P4: 4 points but symmetric under independent
per-axis sign flips, so the ternary/entangler reachable-set construction can only ever
include ALL 4 points or NONE, never a proper subset). Weighted coverage for a 2-point
target can therefore only take 4 discrete values determined by the specific physical
weights (for P1: {0, 0.231, 0.769, 1.0}), which never lands in [0.4, 0.6] for any
selection of included/excluded points -- a mathematical fact about these PDEs' spectra,
not an implementation gap. The COUNT metric can hit exactly 0.5 (cover 1 of 2 points), and
IS achieved.
"""
from __future__ import annotations

import pytest

from qapinn.pdes.burgers import Burgers
from qapinn.pdes.heat import Heat
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson
from qapinn.smcd.design import smcd


def test_p1_design():
    card = smcd(Poisson(alpha=0.3))
    assert card.n_layers == 4
    assert card.n_qubits == 1
    assert card.entangler == "none"
    assert card.coverage_weighted == 1.0


def test_p4_design():
    card = smcd(Helmholtz(k=10.0, a1=3.0, a2=1.0))
    assert card.n_qubits == 3
    assert card.entangler == "ring_cz"
    assert card.coverage == 1.0
    assert card.coverage_weighted == 1.0


def test_p2_predicted_benefit_is_negative_control():
    card = smcd(Heat(alpha=0.3, nu=0.05))
    assert card.predicted_benefit < 0.05


def test_coverage_target_detuning():
    card = smcd(Poisson(alpha=0.3), coverage_target=0.5)
    assert 0.4 <= card.coverage <= 0.6


def test_determinism():
    pde = Poisson(alpha=0.3)
    card_a = smcd(pde)
    card_b = smcd(pde)
    assert card_a.to_json() == card_b.to_json()
    assert card_a == card_b


def test_burgers_runs_via_empirical_fallback():
    card = smcd(Burgers())
    assert card.pde == "burgers"
    assert card.n_qubits >= 1
    assert card.n_layers >= 1
    assert "empirical" in card.notes or "viscous" in card.notes


def test_n_qubits_n_layers_override():
    """T3.3: the depth/qubit sweep needs circuit sizes SMCD would never auto-choose.
    Without this override, models.build()'s q_serial branch always uses smcd()'s own
    auto-computed n_qubits/n_layers -- ModelConfig.n_qubits/n_layers (pre-placed fields)
    would silently have NO effect on the actual circuit, making a depth/qubit sweep over
    those fields build 30 identical circuits."""
    pde = Helmholtz(k=10.0, a1=3.0, a2=1.0)
    card_auto = smcd(pde)
    assert (card_auto.n_qubits, card_auto.n_layers) == (3, 1)

    card_override = smcd(pde, n_qubits=6, n_layers=4)
    assert card_override.n_qubits == 6
    assert card_override.n_layers == 4
    assert len(card_override.scalings) == 4  # L rows
    assert len(card_override.scalings[0]) == 6  # n columns
    assert "OVERRIDDEN" in card_override.notes


def test_n_qubits_override_below_pde_dim_raises():
    pde = Helmholtz(k=10.0, a1=3.0, a2=1.0)  # d=2
    with pytest.raises(ValueError):
        smcd(pde, n_qubits=1)
