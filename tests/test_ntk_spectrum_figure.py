"""T1.3 DoD: the ntk_spectrum figure renders for c_mlp alone, and the measured
decay_exponent for a tanh MLP on P1 is negative and in a plausible range (roughly -1 to -4)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from make_figures import make_ntk_spectrum_figure

import qapinn.viz.style as style_mod
from qapinn.models.mlp import MLPPINN
from qapinn.pdes.poisson import Poisson
from qapinn.seeding import set_global_seed
from qapinn.xai.ntk import ntk, probe_set, spectrum_stats


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(style_mod, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(style_mod, "MANIFEST_PATH", tmp_path / "manifest.json")


def test_ntk_spectrum_figure_renders_for_c_mlp_alone():
    set_global_seed(0)
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=1, widths=(64, 64, 64))

    decay_exponents = make_ntk_spectrum_figure({"c_mlp": model}, pde)

    assert style_mod.FIGURES_DIR.joinpath("ntk_spectrum.pdf").is_file()
    assert style_mod.FIGURES_DIR.joinpath("ntk_spectrum.png").is_file()
    assert set(decay_exponents.keys()) == {"c_mlp"}


def test_c_mlp_decay_exponent_negative_and_plausible():
    set_global_seed(0)
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=1, widths=(64, 64, 64))

    probe_x = probe_set(pde)
    K = ntk(model, pde, probe_x, output="residual", group="all")
    stats = spectrum_stats(K)

    assert stats["decay_exponent"] < 0.0
    assert -4.0 <= stats["decay_exponent"] <= -1.0
