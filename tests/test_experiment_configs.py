"""T3.3 DoD: enumerate_runs returns the documented run count for each of the 5
configs/exp/*.yaml files, all with distinct run_ids.

depth_sweep is 30, not the phase doc's literal 45: n_qubits is capped at {4, 6} (dropped
8) per the T2.18 post-gate owner decision (BENCH.md's "Post-T2.18 Owner Decisions") --
n=8 doesn't fit in VRAM at the default batch size and dominated the whole Phase 3
wall-clock budget. This is a deliberate, documented deviation from the literal spec
number, not a bug.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from qapinn.runner import enumerate_runs

EXPECTED_COUNTS = {
    "core_matrix": 210,
    "coverage_sweep": 36,
    "depth_sweep": 30,  # 45 in the phase doc's literal spec; see module docstring
    "alpha_sweep": 54,
    "noise_study": 36,
}


@pytest.mark.parametrize("name,expected", EXPECTED_COUNTS.items())
def test_enumerate_runs_matches_expected_count(name, expected):
    cfgs = enumerate_runs(Path(f"configs/exp/{name}.yaml"))
    assert len(cfgs) == expected
    assert len({c.run_id for c in cfgs}) == expected  # all distinct


def test_depth_sweep_drops_n_qubits_8_and_reduces_steps():
    cfgs = enumerate_runs(Path("configs/exp/depth_sweep.yaml"))
    assert {c.model.n_qubits for c in cfgs} == {4, 6}
    assert {c.model.n_layers for c in cfgs} == {2, 3, 4, 5, 6}
    assert all(c.train.steps_adam == 4500 and c.train.steps_lbfgs == 500 for c in cfgs)


def test_coverage_sweep_never_dropped_and_covers_all_targets():
    # 00_MASTER_PLAN.md §5: "Never cut: the coverage sweep (it is the validation of C1)".
    cfgs = enumerate_runs(Path("configs/exp/coverage_sweep.yaml"))
    assert {c.smcd_coverage_target for c in cfgs} == {0.2, 0.4, 0.6, 0.8, 0.9, 1.0}


def test_alpha_sweep_covers_all_alpha_values_and_families():
    cfgs = enumerate_runs(Path("configs/exp/alpha_sweep.yaml"))
    assert {c.pde.params["alpha"] for c in cfgs} == {0.0, 0.05, 0.1, 0.2, 0.4, 0.8}
    assert {c.model.family for c in cfgs} == {"c_mlp", "c_ff", "q_serial"}


def test_noise_study_covers_both_noise_settings():
    cfgs = enumerate_runs(Path("configs/exp/noise_study.yaml"))
    assert {c.train.noise for c in cfgs} == {"shot_1024", "depol_1e-3"}
