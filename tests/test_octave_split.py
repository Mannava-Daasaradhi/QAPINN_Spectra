"""T2.15 DoD (literal): on P4 k=20, smcd(..., L_max=4) triggers the split, produces >= 2
circuits, each with L <= 4, and the union of their Omega covers Shat with weighted
coverage 1.0.

Substituted demo, per an explicit owner decision (this session, T2.15): Helmholtz's exact
solution is always a single sin(a1*pi*x)*sin(a2*pi*y) mode (T0.x), so its target spectrum
always has exactly ONE frequency magnitude per axis -- Delta always exactly equals K
(nothing to take a GCD of beyond the single value itself), so d5_depth always returns
L=1, for ANY (k, a1, a2), including k=20 (a1=6, a2=2). Verified directly:
test_p4_k20_never_needs_a_split below confirms L=1 at k=4/10/20, all three of this
project's matrix configs. There is therefore no way to make Helmholtz ever need L>4 with
the PDE as currently defined -- the literal DoD scenario is mathematically unreachable,
not an implementation gap. The owner chose (recommended option): implement the real,
general octave-split algorithm, and demonstrate it against P1 (whose actual 2-point
target {pi, 15*pi} genuinely spans multiple octaves of pi) with an artificially lowered
L_max=3 (P1's own natural, unsplit depth is exactly L=4) to force the split condition
that P4 cannot produce. See BENCH.md's T2.15 section for the full writeup.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import qapinn.models as models_pkg
from qapinn.config import ModelConfig
from qapinn.models.hybrid import OctaveEnsemble
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson
from qapinn.smcd.design import smcd


def test_p4_k20_never_needs_a_split():
    """Confirms the structural finding this test file's docstring describes, across all
    three (k, a1, a2) matrix configs actually used in this project (helmholtz.py)."""
    for k, a1, a2 in [(4, 1, 1), (10, 3, 1), (20, 6, 2)]:
        pde = Helmholtz(k=float(k), a1=float(a1), a2=float(a2))
        card = smcd(pde, L_max=4)
        assert card.n_layers == 1, f"k={k}: L={card.n_layers}, expected 1"
        assert not card.octave_split


def test_p1_octave_split_triggers_and_covers():
    """The substituted demo: P1's own L=4 exceeds an artificially lowered L_max=3,
    triggering a genuine split. P1's target {pi, 15*pi} spans octave 0 ([pi, 2*pi)) and
    octave 3 ([8*pi, 16*pi)) of the base frequency pi, so it splits into exactly 2
    circuits -- one per point."""
    pde = Poisson(alpha=0.05)
    card = smcd(pde, L_max=3)

    assert card.octave_split
    assert card.octave_configs is not None
    assert len(card.octave_configs) >= 2
    for cfg in card.octave_configs:
        assert cfg["n_layers"] <= 3

    all_omega = []
    for cfg in card.octave_configs:
        from qapinn.smcd.design import _circuit_frequencies

        all_omega.append(_circuit_frequencies(cfg))
    union_omega = np.concatenate(all_omega, axis=0)

    from qapinn.smcd.card import coverage
    from qapinn.smcd.symbol import analytic_spectrum

    S = analytic_spectrum(pde, card.eps)
    _, weighted = coverage(S, union_omega)
    assert weighted == 1.0

    # the card's own recorded coverage (computed the same way inside smcd()) must agree
    assert card.coverage_weighted == 1.0


def test_octave_ensemble_builds_and_trains_from_real_octave_configs():
    """OctaveEnsemble (T2.12) fed the REAL octave_configs from a genuine split (not the
    hand-built fixture T2.12's own test used) forwards correctly and its loss decreases
    over a short training run on P1."""
    pde = Poisson(alpha=0.05)
    card = smcd(pde, L_max=3)
    assert card.octave_configs is not None

    model = OctaveEnsemble(card.octave_configs)
    x = torch.rand(5, 1)
    assert model(x).shape == (5, 1)

    from qapinn.config import TrainConfig
    from qapinn.train.losses import pinn_loss

    gen = torch.Generator().manual_seed(0)
    train_cfg = TrainConfig(bc_mode="hard")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first_loss = last_loss = None
    for _ in range(50):
        x_r = pde.sample_collocation(256, gen)
        x_r.requires_grad_(True)
        loss, _ = pinn_loss(model, pde, {"x_r": x_r}, train_cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()

    assert last_loss < first_loss


def test_build_q_octave_family():
    """models.build(family='q_octave') wiring. At the DEFAULT n_max/L_max, none of this
    project's four PDEs trigger a split (P1 needs L=4<=6; P4 needs L=1<=6), so this
    degenerates to a single-circuit ensemble -- still exercises the real build() path,
    not just OctaveEnsemble's own class mechanics (T2.12's test used a hand-built config).
    """
    pde = Poisson(alpha=0.05)
    model = models_pkg.build(ModelConfig(family="q_octave"), input_dim=pde.dim, pde=pde)
    assert len(model.circuits) == 1
    out = model(torch.rand(4, 1))
    assert out.shape == (4, 1)
