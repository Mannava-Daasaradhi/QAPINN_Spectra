"""Ensure the deterministic cuBLAS workspace config is set before any test can import
torch (01_CONVENTIONS.md §9). tasks.py sets this too for the `tasks.py test` entry point;
this covers bare `pytest` invocations."""
import os

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(autouse=True)
def _isolate_committed_artifacts(tmp_path_factory, monkeypatch):
    """Keep every test away from the committed results/ and paper/figures/ trees.

    Training runs land in RESULTS_ROOT, figures in FIGURES_DIR, and each save_figure call
    rewrites MANIFEST_PATH -- all relative to the repo root by default. Before this fixture
    existed, a full test run overwrote a committed CPU smoke run with a CUDA re-run of the
    same run_id (differences at ~1e-16) and rewrote results/size_matching.json. The env
    var covers sweep workers and `tasks.py _smoke-one` subprocesses, which re-import
    qapinn.train.loop instead of seeing this process's monkeypatch. Tests that read
    committed runs use explicit REPO_ROOT paths and are unaffected; tests that need their
    own isolated root (test_checkpoint.py, test_determinism.py, ...) still override this.
    """
    import qapinn.train.loop as loop_mod
    import qapinn.viz.style as style_mod

    results_root = tmp_path_factory.mktemp("results_runs")
    monkeypatch.setenv("QAPINN_RESULTS_DIR", str(results_root))
    monkeypatch.setattr(loop_mod, "RESULTS_ROOT", results_root)

    artifacts = tmp_path_factory.mktemp("figures")
    monkeypatch.setattr(style_mod, "FIGURES_DIR", artifacts / "figures")
    monkeypatch.setattr(style_mod, "MANIFEST_PATH", artifacts / "manifest.json")
