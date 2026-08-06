"""T3.10 gate check (item 1) DoD: docs/predictions.md's commit must be a git ancestor of
every real run's provenance.json git_sha. See scripts/verify_preregistration.py's module
docstring for why this uses git ancestry rather than a wall-clock timestamp comparison
(provenance.json has no timestamp field at all).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_preregistration import (  # noqa: E402
    preregistration_commit_sha,
    verify_predictions_precede_runs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _head_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _first_commit_sha() -> str:
    log = subprocess.run(
        ["git", "log", "--format=%H", "--reverse"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return log.splitlines()[0]


def test_preregistration_commit_sha_matches_known_real_commit():
    # Pinned to the actual T3.1 commit -- if this ever legitimately changes (predictions.md
    # re-added under a new commit), this test's expectation must be updated deliberately,
    # not silently pass a different SHA.
    assert preregistration_commit_sha() == "92abd7d60c34fc86abc529769dd2d8bde6dfb575"


def test_run_at_or_after_preregistration_passes(tmp_path):
    run_dir = tmp_path / "run_good"
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text(
        json.dumps({"run_id": "run_good", "git_sha": _head_sha()}), encoding="utf-8"
    )

    report = verify_predictions_precede_runs([run_dir])
    assert report["n_checked"] == 1
    assert report["violations"] == []


def test_run_before_preregistration_is_a_violation(tmp_path):
    run_dir = tmp_path / "run_bad"
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text(
        json.dumps({"run_id": "run_bad", "git_sha": _first_commit_sha()}), encoding="utf-8"
    )

    report = verify_predictions_precede_runs([run_dir])
    assert report["n_checked"] == 1
    assert report["violations"] == ["run_bad"]


def test_run_missing_provenance_is_reported_separately(tmp_path):
    run_dir = tmp_path / "run_no_provenance"
    run_dir.mkdir()

    report = verify_predictions_precede_runs([run_dir])
    assert report["n_checked"] == 0
    assert report["missing_provenance"] == ["run_no_provenance"]
    assert report["violations"] == []


def test_run_with_unrecognised_git_sha_is_reported_separately(tmp_path):
    run_dir = tmp_path / "run_bogus_sha"
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text(
        json.dumps({"run_id": "run_bogus_sha", "git_sha": "0" * 40}), encoding="utf-8"
    )

    report = verify_predictions_precede_runs([run_dir])
    assert report["n_checked"] == 0
    assert report["unknown_sha"] == [("run_bogus_sha", "0" * 40)]
    assert report["violations"] == []


def test_real_core_matrix_production_runs_have_zero_violations():
    """The actual, real check this task cares about: every T3.4 core_matrix run
    completed so far must genuinely postdate pre-registration. Skips gracefully if no
    real runs exist yet (e.g. a fresh checkout before T3.4 has ever been launched)."""
    from qapinn.runner import enumerate_runs

    cfgs = enumerate_runs(REPO_ROOT / "configs" / "exp" / "core_matrix.yaml")
    run_dirs = [
        REPO_ROOT / "results" / "runs" / c.run_id
        for c in cfgs
        if (REPO_ROOT / "results" / "runs" / c.run_id / "metrics.json").is_file()
    ]
    if not run_dirs:
        import pytest

        pytest.skip("no real core_matrix runs completed yet")

    report = verify_predictions_precede_runs(run_dirs)
    assert report["violations"] == []
    assert report["unknown_sha"] == []
