"""T2.17 DoD: run train(..., smoke=True) for every (problem instance x family) pair =
6 x 7 = 42 combinations, with all instruments enabled. All 42 must complete without
error and write complete artifact directories. Total time < 20 min.

Marked @pytest.mark.slow (excluded from the default `pytest`/`tasks.py test` run, matches
this project's convention for expensive tests) -- run explicitly via
`pytest tests/test_pipeline_smoke.py` or `tasks.py test --slow`, or via `tasks.py smoke`
directly (same underlying instance/family list, tasks.SMOKE_PDE_INSTANCES /
tasks.SMOKE_FAMILIES, imported here rather than duplicated so the two can't drift apart).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tasks  # noqa: E402 -- must follow the sys.path fix-up above

TIME_BUDGET_S = 20 * 60


@pytest.mark.slow
def test_full_matrix_smoke():
    import gc

    import torch

    from qapinn.config import load_config
    from qapinn.train.loop import train

    t_start = time.time()
    n_total = len(tasks.SMOKE_PDE_INSTANCES) * len(tasks.SMOKE_FAMILIES)
    failures: list[tuple[str, str]] = []
    n_ok = 0

    for instance_label, pde_yaml, overrides in tasks.SMOKE_PDE_INSTANCES:
        for family in tasks.SMOKE_FAMILIES:
            label = f"{instance_label}/{family}"
            try:
                cfg = load_config(pde_yaml, family, overrides=overrides, seed=0)
                result = train(cfg, smoke=True)
                assert (result.run_dir / "metrics.json").is_file(), f"{label}: missing metrics.json"
                assert (result.run_dir / "config.yaml").is_file(), f"{label}: missing config.yaml"
                assert (result.run_dir / "history.parquet").is_file(), f"{label}: missing history.parquet"
                n_ok += 1
            except Exception as e:  # noqa: BLE001 -- collect every failure, don't stop at the first
                failures.append((label, repr(e)))
            finally:
                # This test runs all 42 combos IN-PROCESS (unlike tasks.py smoke, which
                # isolates each combo in its own subprocess) -- release what's unused
                # after every combo so nothing accumulates across the full matrix.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    if failures:
        detail = "\n".join(f"  {label}: {err}" for label, err in failures)
        pytest.fail(f"{len(failures)}/{n_total} combinations failed:\n{detail}")

    assert n_ok == n_total
    assert elapsed < TIME_BUDGET_S, f"42-combination smoke matrix took {elapsed:.1f}s, budget is {TIME_BUDGET_S}s"
