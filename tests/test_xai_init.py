"""T1.12 DoD: a --smoke run with all instruments enabled completes in < 120 s and writes
one .npz per instrument per checkpoint.
"""
from __future__ import annotations

import time

import pytest
import torch

from qapinn.config import load_config
from qapinn.models.mlp import MLPPINN
from qapinn.pdes.poisson import Poisson
from qapinn.train.loop import train
from qapinn.xai import INSTRUMENTS, run_instruments


def _small_cfg(**overrides):
    all_overrides = {"train.steps_adam": 20, "train.steps_lbfgs": 0, "train.checkpoints": (0, -1)}
    all_overrides.update(overrides)
    return load_config("poisson", "c_mlp", seed=0, overrides=all_overrides)


def test_run_instruments_writes_per_checkpoint_files(tmp_path):
    cfg = _small_cfg()
    pde = Poisson(alpha=cfg.pde.params["alpha"])
    model = MLPPINN(input_dim=1, widths=(4, 4))

    run_instruments(model, pde, cfg, 0, tmp_path, which=("ntk", "attribution", "fisher", "probes"))

    xai_dir = tmp_path / "xai"
    files = {p.name for p in xai_dir.glob("*.npz")}
    assert "ntk_step0.npz" in files
    assert "attribution_step0.npz" in files
    assert "fisher_step0.npz" in files
    assert "probes_step0.npz" in files


def test_run_instruments_final_only_skipped_at_non_final_checkpoint(tmp_path):
    cfg = _small_cfg()
    pde = Poisson(alpha=cfg.pde.params["alpha"])
    model = MLPPINN(input_dim=1, widths=(4, 4))

    run_instruments(model, pde, cfg, 0, tmp_path, which=("gradvar", "landscape"))
    xai_dir = tmp_path / "xai"
    assert list(xai_dir.glob("*.npz")) == []

    total_steps = cfg.train.steps_adam + cfg.train.steps_lbfgs
    run_instruments(model, pde, cfg, total_steps, tmp_path, which=("gradvar", "landscape"))
    files = {p.name for p in xai_dir.glob("*.npz")}
    assert f"gradvar_step{total_steps}.npz" in files
    assert f"landscape_step{total_steps}.npz" in files


def test_run_instruments_specerr_accumulates_single_file(tmp_path):
    import numpy as np

    cfg = _small_cfg()
    pde = Poisson(alpha=cfg.pde.params["alpha"])
    model = MLPPINN(input_dim=1, widths=(4, 4))

    run_instruments(model, pde, cfg, 0, tmp_path, which=("specerr",))
    total_steps = cfg.train.steps_adam + cfg.train.steps_lbfgs
    run_instruments(model, pde, cfg, total_steps, tmp_path, which=("specerr",))

    xai_dir = tmp_path / "xai"
    specerr_files = list(xai_dir.glob("specerr*.npz"))
    assert len(specerr_files) == 1

    data = np.load(specerr_files[0])
    assert data["steps"].tolist() == [0, total_steps]


def test_run_instruments_unknown_instrument_raises(tmp_path):
    cfg = _small_cfg()
    pde = Poisson(alpha=cfg.pde.params["alpha"])
    model = MLPPINN(input_dim=1, widths=(4,))
    with pytest.raises(ValueError):
        run_instruments(model, pde, cfg, 0, tmp_path, which=("not_a_real_instrument",))


def test_run_instruments_which_overrides_config_default(tmp_path):
    cfg = _small_cfg()
    pde = Poisson(alpha=cfg.pde.params["alpha"])
    model = MLPPINN(input_dim=1, widths=(4,))

    run_instruments(model, pde, cfg, 0, tmp_path, which=("ntk",))
    xai_dir = tmp_path / "xai"
    files = {p.name for p in xai_dir.glob("*.npz")}
    assert files == {"ntk_step0.npz"}


def test_instruments_registry_has_all_nine_entries():
    assert set(INSTRUMENTS) == {
        "ntk", "specerr", "attribution", "fisher", "drift", "gradvar", "probes", "landscape",
        "block_mass",
    }


def test_run_block_mass_writes_npz_with_finite_traces(tmp_path):
    import numpy as np

    cfg = _small_cfg(**{"train.bc_mode": "soft"})
    pde = Poisson(alpha=cfg.pde.params["alpha"])
    model = MLPPINN(input_dim=1, widths=(4, 4))

    run_instruments(model, pde, cfg, 0, tmp_path, which=("block_mass",))

    saved = np.load(tmp_path / "xai" / "block_mass_step0.npz")
    assert saved["trace_rr"] > 0
    assert saved["trace_bb"] > 0
    assert np.isfinite(saved["trace_rr"])
    assert np.isfinite(saved["trace_bb"])


def test_run_block_mass_probe_points_are_fixed_across_checkpoints(tmp_path):
    """block_mass is an NTK-family instrument (T1.2's convention: a FIXED probe set
    across checkpoints, not a step-seeded one like _run_landscape) -- two different steps
    on an UNCHANGED model must therefore produce identical traces."""
    import numpy as np

    cfg = _small_cfg(**{"train.bc_mode": "soft"})
    pde = Poisson(alpha=cfg.pde.params["alpha"])
    model = MLPPINN(input_dim=1, widths=(4, 4))

    run_instruments(model, pde, cfg, 0, tmp_path, which=("block_mass",))
    run_instruments(model, pde, cfg, 1, tmp_path, which=("block_mass",))

    step0 = np.load(tmp_path / "xai" / "block_mass_step0.npz")
    step1 = np.load(tmp_path / "xai" / "block_mass_step1.npz")
    assert step0["trace_rr"] == pytest.approx(float(step1["trace_rr"]))
    assert step0["trace_bb"] == pytest.approx(float(step1["trace_bb"]))


def test_full_smoke_run_all_instruments_under_120s_writes_expected_files():
    torch.manual_seed(0)
    cfg = load_config("poisson", "c_mlp", seed=0, overrides={"pde.params.alpha": 0.3})

    t0 = time.time()
    result = train(cfg, smoke=True)
    elapsed = time.time() - t0

    assert elapsed < 120.0

    xai_dir = result.run_dir / "xai"
    files = {p.name for p in xai_dir.glob("*.npz")}

    total_steps = result.cfg.train.steps_adam + result.cfg.train.steps_lbfgs
    for name in ("ntk", "attribution", "fisher", "probes"):
        assert f"{name}_step0.npz" in files
        assert f"{name}_step{total_steps}.npz" in files
    for name in ("gradvar", "landscape"):
        assert f"{name}_step0.npz" not in files
        assert f"{name}_step{total_steps}.npz" in files
    assert "specerr.npz" in files
