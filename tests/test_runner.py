"""T3.2 DoD: enumerate_runs expands configs/exp/*.yaml correctly; run_all executes with
resume support and per-run failure isolation."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

import tasks
from qapinn.config import load_config
from qapinn.runner import PROBLEM_INSTANCES, enumerate_runs, run_all
from qapinn.train.loop import RESULTS_ROOT, apply_smoke_overrides


def test_smoke_pde_instances_matches_runner_canonical_list():
    assert tasks.SMOKE_PDE_INSTANCES == PROBLEM_INSTANCES


def test_enumerate_runs_core_matrix_yaml_gives_84_distinct_configs():
    # 210 in the original spec; cut to 84 (seeds 5->2, cut-line #4) under Aug 7 deadline
    # pressure -- see 08_TASK_INDEX.md's "Cuts applied" sections.
    cfgs = enumerate_runs(Path("configs/exp/core_matrix.yaml"))
    assert len(cfgs) == 84
    assert len({c.run_id for c in cfgs}) == 84


def test_enumerate_runs_axes_cartesian_product(tmp_path):
    spec = {
        "problems": ["poisson", "helmholtz_k10"],
        "families": ["q_serial"],
        "seeds": [0, 1, 2],
        "axes": [{"key": "smcd_coverage_target", "values": [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]}],
    }
    exp_path = tmp_path / "coverage_sweep.yaml"
    exp_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    cfgs = enumerate_runs(exp_path)
    assert len(cfgs) == 2 * 1 * 3 * 6  # == 36, matches T3.3's coverage_sweep.yaml spec
    assert {c.smcd_coverage_target for c in cfgs} == {0.2, 0.4, 0.6, 0.8, 0.9, 1.0}


def test_enumerate_runs_two_axes_combine(tmp_path):
    spec = {
        "problems": ["helmholtz_k10"],
        "families": ["q_serial"],
        "seeds": [0, 1, 2],
        "axes": [
            {"key": "model.n_layers", "values": [2, 3, 4, 5, 6]},
            {"key": "model.n_qubits", "values": [4, 6]},
        ],
    }
    exp_path = tmp_path / "depth_sweep.yaml"
    exp_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    cfgs = enumerate_runs(exp_path)
    assert len(cfgs) == 1 * 1 * 3 * 5 * 2  # == 30, the capped depth/qubit sweep (BENCH.md)
    assert {c.model.n_layers for c in cfgs} == {2, 3, 4, 5, 6}
    assert {c.model.n_qubits for c in cfgs} == {4, 6}


def test_enumerate_runs_unknown_problem_raises(tmp_path):
    spec = {"problems": ["not_a_real_problem"], "families": ["c_mlp"], "seeds": [0]}
    exp_path = tmp_path / "bad.yaml"
    exp_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(ValueError):
        enumerate_runs(exp_path)


def test_run_all_resume_skips_existing_run():
    cfg = load_config("poisson", "c_mlp", seed=0)
    # smoke=True mutates the config internally (apply_smoke_overrides, T2.17) before
    # hashing for run_id -- the path run_all actually checks/writes is the POST-override
    # one, not cfg.run_id itself.
    run_dir = RESULTS_ROOT / apply_smoke_overrides(cfg).run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    try:
        result = run_all([cfg], n_workers=1, resume=True, smoke=True)
        assert result.n_total == 1
        assert result.n_skipped == 1
        assert result.n_ok == 0
        assert result.n_failed == 0
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.mark.slow
def test_run_all_isolates_a_failing_run_without_losing_the_good_one():
    good_cfg = load_config("poisson", "c_mlp", seed=1)
    bad_cfg = load_config("poisson", "c_mlp", seed=1, overrides={"train.lr_schedule": "step"})
    good_dir = RESULTS_ROOT / apply_smoke_overrides(good_cfg).run_id
    bad_dir = RESULTS_ROOT / apply_smoke_overrides(bad_cfg).run_id
    try:
        result = run_all([good_cfg, bad_cfg], n_workers=1, resume=False, smoke=True)
        assert result.n_total == 2
        assert result.n_ok == 1
        assert result.n_failed == 1
        assert result.failures[0][0] == apply_smoke_overrides(bad_cfg).run_id
        assert (bad_dir / "error.json").is_file()
        assert (good_dir / "metrics.json").is_file()
    finally:
        shutil.rmtree(good_dir, ignore_errors=True)
        shutil.rmtree(bad_dir, ignore_errors=True)


@pytest.mark.slow
def test_run_all_second_invocation_skips_everything():
    # An unusual seed, deliberately -- this writes into the REAL results/runs/ directory
    # (T2.17's own established pattern for smoke-mode tests), so a common seed risks
    # colliding with some other run's leftover directory from elsewhere in this repo.
    cfg = load_config("poisson", "c_mlp", seed=987654)
    run_dir = RESULTS_ROOT / apply_smoke_overrides(cfg).run_id
    shutil.rmtree(run_dir, ignore_errors=True)  # defensive: guard against stale leftovers
    try:
        first = run_all([cfg], n_workers=1, resume=True, smoke=True)
        assert first.n_ok == 1 and first.n_skipped == 0

        second = run_all([cfg], n_workers=1, resume=True, smoke=True)
        assert second.n_ok == 0 and second.n_skipped == 1
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
