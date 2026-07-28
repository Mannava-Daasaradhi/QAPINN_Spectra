"""T0.20 DoD: two train() calls with identical configs (50 steps) produce bit-identical
metrics.json. Never relax this to a tolerance -- the reproducibility contract (project.md
§11) depends on it.

wall_clock_s is excluded from the bit-identical comparison: it is a measurement of how long
the run took (a timer reading), not a computed result, and by its nature differs between any
two separate runs (e.g. CUDA warm-up on the first call) -- this is not the kind of
non-determinism the DoD is protecting against, unlike loss/rel_l2/n_params etc., which come
straight out of the computation and must match exactly.
"""
from __future__ import annotations

import pytest

import qapinn.train.loop as loop_mod
from qapinn.config import load_config
from qapinn.train.loop import train


@pytest.fixture(autouse=True)
def _isolated_results_root(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "RESULTS_ROOT", tmp_path)


def _without_wall_clock(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "wall_clock_s"}


def test_two_identical_runs_produce_bit_identical_metrics():
    cfg = load_config(
        "poisson",
        "c_mlp",
        seed=0,
        overrides={"train.steps_adam": 40, "train.steps_lbfgs": 10, "pde.n_collocation": 256},
    )

    result1 = train(cfg)
    result2 = train(cfg)

    assert result1.run_id == result2.run_id  # identical config -> identical run_id
    assert "wall_clock_s" in result1.metrics and "wall_clock_s" in result2.metrics
    assert _without_wall_clock(result1.metrics) == _without_wall_clock(result2.metrics)


def test_two_identical_runs_produce_identical_history():
    cfg = load_config(
        "poisson",
        "c_mlp",
        seed=0,
        overrides={"train.steps_adam": 40, "train.steps_lbfgs": 10, "pde.n_collocation": 256},
    )

    result1 = train(cfg)
    history1 = result1.history.copy()

    result2 = train(cfg)
    history2 = result2.history.copy()

    assert history1.equals(history2)


@pytest.mark.parametrize("seed", [0, 1])
def test_different_seeds_need_not_match(seed):
    # sanity check for the harness itself: confirms the test setup can actually detect a
    # difference (i.e. isn't accidentally comparing two constant/trivial outputs)
    cfg_a = load_config(
        "poisson",
        "c_mlp",
        seed=seed,
        overrides={"train.steps_adam": 10, "train.steps_lbfgs": 0, "pde.n_collocation": 128},
    )
    cfg_b = load_config(
        "poisson",
        "c_mlp",
        seed=seed + 100,
        overrides={"train.steps_adam": 10, "train.steps_lbfgs": 0, "pde.n_collocation": 128},
    )
    result_a = train(cfg_a)
    result_b = train(cfg_b)
    assert result_a.metrics != result_b.metrics
