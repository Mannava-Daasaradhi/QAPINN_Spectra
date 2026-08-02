"""T0.18 DoD: `python tasks.py run --pde poisson --model c_mlp --smoke` completes in <60s
and writes a complete results/runs/<id>/ directory with all required files."""
from __future__ import annotations

import time

import pandas as pd
import pytest

import qapinn.train.loop as loop_mod
from qapinn.config import load_config
from qapinn.train.loop import train

_EXPECTED_METRIC_KEYS = {
    "rel_l2",
    "l_inf",
    "residual_norm",
    "steps_to_tol_1e-2",
    "steps_to_tol_1e-3",
    "n_params",
    "wall_clock_s",
    "circuit_evals",
    "grad_var_final",
    "ntk_cond",
    "ntk_decay_exponent",
    "smcd_coverage",
    "smcd_coverage_weighted",
    "encoder_drift",
}


@pytest.fixture(autouse=True)
def _isolated_results_root(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "RESULTS_ROOT", tmp_path)


def test_smoke_run_completes_quickly_and_writes_full_artifact_dir():
    cfg = load_config("poisson", "c_mlp", seed=0)

    t0 = time.time()
    result = train(cfg, smoke=True)
    elapsed = time.time() - t0

    assert elapsed < 60.0

    run_dir = result.run_dir
    assert run_dir.is_dir()
    for name in ("config.yaml", "metrics.json", "provenance.json", "design_card.json", "history.parquet"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "xai").is_dir()

    assert set(result.metrics.keys()) == _EXPECTED_METRIC_KEYS

    history = pd.read_parquet(run_dir / "history.parquet")
    assert len(history) == 25  # steps_adam=20 + steps_lbfgs=5 (T2.17: cut from 50+10), the smoke override
    assert list(history.columns) == ["step", "loss", "lr", "grad_norm"]


def test_run_id_matches_the_smoke_adjusted_config():
    # smoke mode overrides steps_adam/steps_lbfgs/n_collocation *inside* train(), so the
    # directory name (hashed from the post-override config) must match result.run_id --
    # not the pre-override cfg the caller built.
    cfg = load_config("poisson", "c_mlp", seed=0)
    result = train(cfg, smoke=True)
    assert result.run_dir.name == result.run_id


def test_smoke_run_soft_bc_mode_on_heat_writes_bc_and_ic_terms():
    cfg = load_config("heat", "c_mlp", seed=0, overrides={"train.bc_mode": "soft"})
    result = train(cfg, smoke=True)
    assert (result.run_dir / "metrics.json").is_file()
    assert set(result.metrics.keys()) == _EXPECTED_METRIC_KEYS


def test_design_card_is_null_for_classical_family():
    import json

    cfg = load_config("poisson", "c_mlp", seed=0)
    result = train(cfg, smoke=True)
    with (result.run_dir / "design_card.json").open() as f:
        assert json.load(f) is None


def test_unsupported_lr_schedule_raises():
    cfg = load_config("poisson", "c_mlp", seed=0, overrides={"train.lr_schedule": "step"})
    with pytest.raises(ValueError):
        train(cfg, smoke=True)
