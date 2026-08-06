"""Real bug found running T4.7's PR-8 check against real data: a crashed-and-retried run's
specerr.npz gets corrupted with duplicate step entries (save_spectral_error_checkpoint
always appends). See scripts/check_specerr_integrity.py's module docstring for the full
root-cause explanation and why it's flagged rather than fixed at the source mid-sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_specerr_integrity import (  # noqa: E402
    scan_core_matrix_for_contamination,
    specerr_has_duplicate_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_specerr(run_dir, steps):
    xai_dir = run_dir / "xai"
    xai_dir.mkdir(parents=True)
    np.savez(xai_dir / "specerr.npz", omega=np.array([1.0, 2.0]), errors=np.zeros((len(steps), 2)), steps=np.array(steps))


def test_specerr_has_duplicate_steps_true_for_contaminated_run(tmp_path):
    run_dir = tmp_path / "contaminated"
    _write_specerr(run_dir, [0, 0, 100, 500, 1000, 5000, 20000, 22000])
    assert specerr_has_duplicate_steps(run_dir) is True


def test_specerr_has_duplicate_steps_false_for_clean_run(tmp_path):
    run_dir = tmp_path / "clean"
    _write_specerr(run_dir, [0, 100, 500, 1000, 5000, 20000, 22000])
    assert specerr_has_duplicate_steps(run_dir) is False


def test_specerr_has_duplicate_steps_false_when_file_missing(tmp_path):
    run_dir = tmp_path / "no_xai_at_all"
    run_dir.mkdir()
    assert specerr_has_duplicate_steps(run_dir) is False


def test_scan_reports_contaminated_and_clean_runs_separately(tmp_path):
    clean_dir = tmp_path / "clean_run"
    _write_specerr(clean_dir, [0, 100, 22000])
    bad_dir = tmp_path / "bad_run"
    _write_specerr(bad_dir, [0, 0, 100, 22000])
    in_progress_dir = tmp_path / "in_progress"
    in_progress_dir.mkdir()  # no specerr.npz yet -- must be silently skipped, not miscounted

    label_by_run_id = {
        "clean_run": {"family": "c_mlp", "problem": "poisson"},
        "bad_run": {"family": "c_ff", "problem": "heat"},
        "in_progress": {"family": "q_serial", "problem": "poisson"},
    }
    report = scan_core_matrix_for_contamination(label_by_run_id, runs_dir=tmp_path)
    assert report["n_checked"] == 2
    assert report["n_contaminated"] == 1
    assert report["contaminated"][0]["run_id"] == "bad_run"
    assert report["contaminated"][0]["steps"] == [0, 0, 100, 22000]


def test_real_core_matrix_scan_runs_without_crashing_and_reports_a_sane_count():
    """The actual real check: scan whatever T3.4 core_matrix runs exist so far. Does NOT
    assert a specific contaminated count -- the live sweep is still producing new runs,
    so that number is a moving target; this only asserts the scan completes and the
    reported counts are internally consistent."""
    from cost_ledger import enumerate_core_matrix_run_ids

    label_by_run_id = enumerate_core_matrix_run_ids(REPO_ROOT / "configs" / "exp" / "core_matrix.yaml")
    report = scan_core_matrix_for_contamination(label_by_run_id)
    assert report["n_contaminated"] == len(report["contaminated"])
    assert report["n_contaminated"] <= report["n_checked"]
    for entry in report["contaminated"]:
        steps = entry["steps"]
        assert len(steps) != len(set(steps))
