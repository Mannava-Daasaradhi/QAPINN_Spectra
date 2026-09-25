"""T3.3 DoD: enumerate_runs returns the documented run count for each of the 5
configs/exp/*.yaml files, all with distinct run_ids.

depth_sweep, core_matrix, alpha_sweep, and noise_study counts below reflect the Aug 7
deadline cut-line applied 2026-08-05/06 (00_MASTER_PLAN.md §5, recorded in full in
docs/plan/08_TASK_INDEX.md's "Cuts applied" sections), not the phase docs' literal
uncut specs:
- depth_sweep: 45 in the literal spec -> 30 (n_qubits capped {4,6}, T2.18 owner decision,
  VRAM) -> 4 (n_qubits cut to {6} only, cut-line #2; n_layers=6 dropped, genuine CUDA
  OOM at that circuit size -- see depth_sweep.yaml's own header for the full chain).
- core_matrix: 210 -> 84 (seeds 5->2, cut-line #4).
- alpha_sweep: 54 -> 27 (6 alpha values -> 3, cut-line #1).
- noise_study: 36 -> 18 (dropped depol_1e-3, cut-line #3).
- coverage_sweep: unchanged at 36 -- never-cut by rule.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from qapinn.runner import enumerate_runs

EXPECTED_COUNTS = {
    "core_matrix": 84,
    "coverage_sweep": 36,
    "depth_sweep": 4,
    "alpha_sweep": 27,
    "noise_study": 18,
}


@pytest.mark.parametrize("name,expected", EXPECTED_COUNTS.items())
def test_enumerate_runs_matches_expected_count(name, expected):
    cfgs = enumerate_runs(Path(f"configs/exp/{name}.yaml"))
    assert len(cfgs) == expected
    assert len({c.run_id for c in cfgs}) == expected  # all distinct


def test_depth_sweep_drops_n_qubits_8_and_reduces_steps():
    cfgs = enumerate_runs(Path("configs/exp/depth_sweep.yaml"))
    # n_qubits=4 dropped (cut-line #2) and n_layers=6 dropped (genuine CUDA OOM at
    # n_qubits=6, even at this reduced step budget) -- see depth_sweep.yaml's header.
    assert {c.model.n_qubits for c in cfgs} == {6}
    assert {c.model.n_layers for c in cfgs} == {2, 3, 4, 5}
    assert all(c.train.steps_adam == 100 and c.train.steps_lbfgs == 20 for c in cfgs)


def test_coverage_sweep_never_dropped_and_covers_all_targets():
    # 00_MASTER_PLAN.md §5: "Never cut: the coverage sweep (it is the validation of C1)".
    cfgs = enumerate_runs(Path("configs/exp/coverage_sweep.yaml"))
    assert {c.smcd_coverage_target for c in cfgs} == {0.2, 0.4, 0.6, 0.8, 0.9, 1.0}


def test_alpha_sweep_covers_all_alpha_values_and_families():
    # 6 alpha values -> 3 (cut-line #1): both endpoints (0.0, 0.8) plus the mid-transition
    # value (0.1) retained, see alpha_sweep.yaml's header.
    cfgs = enumerate_runs(Path("configs/exp/alpha_sweep.yaml"))
    assert {c.pde.params["alpha"] for c in cfgs} == {0.0, 0.1, 0.8}
    assert {c.model.family for c in cfgs} == {"c_mlp", "c_ff", "q_serial"}


def test_noise_study_covers_shot_noise():
    # depol_1e-3 dropped (cut-line #3) -- shot noise only, see noise_study.yaml's header.
    cfgs = enumerate_runs(Path("configs/exp/noise_study.yaml"))
    assert {c.train.noise for c in cfgs} == {"shot_1024"}


def test_baseline_tuning_confirm_enumerates_exactly_the_committed_f14_runs():
    # The T4.9 full-budget confirmation runs behind FINDINGS.md F14 were launched without
    # a config file; baseline_tuning_confirm.yaml was reconstructed after submission and
    # must hash to exactly the run_ids that are committed, or it is not the same experiment.
    cfgs = enumerate_runs(Path("configs/exp/baseline_tuning_confirm.yaml"))
    assert {c.run_id for c in cfgs} == {"3987924e8b7a", "9b4be2905b57", "cf3a29a94d29", "d2260a58ae0f"}
    assert all(c.train.steps_adam == 20000 and c.train.steps_lbfgs == 2000 for c in cfgs)
    for c in cfgs:
        assert Path("results/runs", c.run_id, "metrics.json").is_file()
