"""T4.6 DoD: results/cost_ledger.json aggregates real per-run cost data (params, wall-clock,
rel_l2) per (family, problem), with the "error at matched wall-clock" disparity operationalised
as wall_clock_ratio_vs_fastest. See scripts/cost_ledger.py's module docstring for why
circuit_evals/flops_per_step are reported as None rather than fabricated, and why a literal
error-vs-wall-clock curve isn't derivable (no per-checkpoint wall-clock timestamp exists,
regardless of checkpoint density).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cost_ledger import (  # noqa: E402
    build_cost_ledger,
    enumerate_core_matrix_run_ids,
    load_run_record,
    render_markdown_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_run(tmp_path, run_id, family, problem, seed, n_params, wall_clock_s, rel_l2,
                steps_adam=20000, steps_lbfgs=2000):
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    config = {
        "model": {"family": family},
        "pde": {"name": problem, "params": {}},
        "seed": seed,
        "train": {"steps_adam": steps_adam, "steps_lbfgs": steps_lbfgs},
    }
    metrics = {
        "n_params": n_params,
        "wall_clock_s": wall_clock_s,
        "rel_l2": rel_l2,
        "circuit_evals": 0,
        "grad_var_final": 0.0,
        "ntk_cond": 0.0,
    }
    (run_dir / "config.yaml").write_text(json.dumps(config), encoding="utf-8")  # valid YAML
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


def test_load_run_record_reads_real_fields_and_derives_per_step_cost(tmp_path):
    run_dir = _write_run(tmp_path, "r1", "c_mlp", "poisson", 0, n_params=100, wall_clock_s=220.0, rel_l2=0.5)
    record = load_run_record(run_dir)
    assert record["family"] == "c_mlp"
    assert record["problem"] == "poisson"
    assert record["n_params"] == 100
    assert record["wall_clock_s"] == pytest.approx(220.0)
    assert record["wall_clock_s_per_step"] == pytest.approx(220.0 / 22000)
    assert record["rel_l2"] == pytest.approx(0.5)


def test_load_run_record_never_fabricates_uninstrumented_fields(tmp_path):
    run_dir = _write_run(tmp_path, "r1", "q_serial", "poisson", 0, n_params=10, wall_clock_s=100.0, rel_l2=0.5)
    record = load_run_record(run_dir)
    assert record["circuit_evals"] is None
    assert record["flops_per_step"] is None


def test_load_run_record_returns_none_for_incomplete_run(tmp_path):
    run_dir = tmp_path / "still_running"
    run_dir.mkdir()
    assert load_run_record(run_dir) is None


def test_build_cost_ledger_aggregates_medians_across_seeds(tmp_path):
    dirs = [
        _write_run(tmp_path, f"r{s}", "c_mlp", "poisson", s, n_params=100, wall_clock_s=w, rel_l2=e)
        for s, (w, e) in enumerate([(100.0, 0.5), (200.0, 0.3), (300.0, 0.4)])
    ]
    ledger = build_cost_ledger(dirs)
    assert len(ledger["entries"]) == 1
    entry = ledger["entries"][0]
    assert entry["family"] == "c_mlp"
    assert entry["problem"] == "poisson"
    assert entry["n_seeds"] == 3
    assert entry["wall_clock_s_median"] == pytest.approx(200.0)
    assert entry["rel_l2_median"] == pytest.approx(0.4)


def test_wall_clock_ratio_vs_fastest_is_one_for_the_fastest_family(tmp_path):
    dirs = [
        _write_run(tmp_path, "fast", "c_rff_matched", "poisson", 0, n_params=10, wall_clock_s=100.0, rel_l2=0.3),
        _write_run(tmp_path, "slow", "q_serial", "poisson", 0, n_params=10, wall_clock_s=650.0, rel_l2=0.4),
    ]
    ledger = build_cost_ledger(dirs)
    by_family = {e["family"]: e for e in ledger["entries"]}
    assert by_family["c_rff_matched"]["wall_clock_ratio_vs_fastest"] == pytest.approx(1.0)
    assert by_family["q_serial"]["wall_clock_ratio_vs_fastest"] == pytest.approx(6.5)


def test_wall_clock_ratio_is_computed_per_problem_independently(tmp_path):
    """A family that's fast on one problem and slow on another must not have its ratio
    on problem A contaminated by problem B's costs."""
    dirs = [
        _write_run(tmp_path, "p1a", "c_mlp", "poisson", 0, n_params=10, wall_clock_s=100.0, rel_l2=0.3),
        _write_run(tmp_path, "p1b", "q_serial", "poisson", 0, n_params=10, wall_clock_s=200.0, rel_l2=0.4),
        _write_run(tmp_path, "p2a", "c_mlp", "heat", 0, n_params=10, wall_clock_s=1000.0, rel_l2=0.3),
        _write_run(tmp_path, "p2b", "q_serial", "heat", 0, n_params=10, wall_clock_s=1000.0, rel_l2=0.4),
    ]
    ledger = build_cost_ledger(dirs)
    by_key = {(e["family"], e["problem"]): e for e in ledger["entries"]}
    assert by_key[("c_mlp", "poisson")]["wall_clock_ratio_vs_fastest"] == pytest.approx(1.0)
    assert by_key[("q_serial", "poisson")]["wall_clock_ratio_vs_fastest"] == pytest.approx(2.0)
    assert by_key[("c_mlp", "heat")]["wall_clock_ratio_vs_fastest"] == pytest.approx(1.0)
    assert by_key[("q_serial", "heat")]["wall_clock_ratio_vs_fastest"] == pytest.approx(1.0)


def test_label_by_run_id_overrides_problem_grouping_and_filters_non_members(tmp_path):
    member = _write_run(tmp_path, "member", "c_mlp", "helmholtz", 0, n_params=10, wall_clock_s=100.0, rel_l2=0.3)
    non_member = _write_run(tmp_path, "not_in_sweep", "c_mlp", "helmholtz", 1, n_params=10, wall_clock_s=999.0, rel_l2=0.9)
    label_by_run_id = {"member": {"problem": "helmholtz_k4", "family": "c_mlp", "seed": 0}}

    ledger = build_cost_ledger([member, non_member], label_by_run_id=label_by_run_id)
    assert ledger["n_production_runs"] == 1
    assert ledger["entries"][0]["problem"] == "helmholtz_k4"


def test_render_markdown_table_never_prints_a_fabricated_number(tmp_path):
    dirs = [_write_run(tmp_path, "r1", "c_mlp", "poisson", 0, n_params=100, wall_clock_s=100.0, rel_l2=0.3)]
    ledger = build_cost_ledger(dirs)
    table = render_markdown_table(ledger)
    assert "n/a (uninstrumented)" in table
    assert "| poisson | c_mlp |" in table


def test_enumerate_core_matrix_run_ids_distinguishes_helmholtz_variants():
    mapping = enumerate_core_matrix_run_ids(REPO_ROOT / "configs" / "exp" / "core_matrix.yaml")
    labels = {v["problem"] for v in mapping.values()}
    assert {"helmholtz_k4", "helmholtz_k10", "helmholtz_k20"} <= labels
    # 6 problems x 7 families x 2 seeds (cut-line #4, seeds 5->2, see 08_TASK_INDEX.md),
    # all distinct run_ids (deterministic hashing, no collisions)
    assert len(mapping) == 6 * 7 * 2


def test_real_core_matrix_production_runs_produce_a_nonfabricated_ledger():
    """The actual real check this task cares about: build the ledger from whatever
    T3.4 core_matrix runs are done so far. Skips gracefully if none exist yet."""
    label_by_run_id = enumerate_core_matrix_run_ids(REPO_ROOT / "configs" / "exp" / "core_matrix.yaml")
    run_dirs = [
        REPO_ROOT / "results" / "runs" / run_id
        for run_id in label_by_run_id
        if (REPO_ROOT / "results" / "runs" / run_id / "metrics.json").is_file()
    ]
    if not run_dirs:
        pytest.skip("no real core_matrix runs completed yet")

    ledger = build_cost_ledger(run_dirs, label_by_run_id=label_by_run_id)
    assert ledger["n_production_runs"] == len(run_dirs)
    for entry in ledger["entries"]:
        assert entry["circuit_evals"] is None
        assert entry["flops_per_step"] is None
        assert entry["wall_clock_s_median"] > 0
        assert entry["wall_clock_ratio_vs_fastest"] >= 1.0 - 1e-9
