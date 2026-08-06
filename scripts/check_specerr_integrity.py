"""Diagnostic: flags `results/runs/*/xai/specerr.npz` files corrupted by a real T3.2 gap
found while running T4.7's PR-8 check against real data this session.

`save_spectral_error_checkpoint` (`xai/spectral_error.py`) unconditionally APPENDS to an
existing `specerr.npz` (conventions SS8: one accumulated file per run, not one per
checkpoint -- the correct design when a run completes cleanly in one attempt). If a run
crashes mid-training (`BrokenProcessPool`, T3.4's own documented crash cascade) and the
external retry-wrapper restarts it (`metrics.json`-existence-based resume, T3.2's own
established mechanism), the new attempt's checkpoints get appended ON TOP of the previous
crashed attempt's -- producing duplicate/out-of-order step entries. This silently
corrupts any analysis that assumes `specerr.npz` holds exactly one entry per checkpoint
(T3.9's freq heatmap; this session's own `scripts/adjudicate_predictions.py::check_pr8`
hit this directly on a real run, `14bbe5f763c4`).

Verified scope directly against the live (partial) T3.4 sweep: 9 of 73 real production
runs affected (~12%), one retried at least 3 times (`6261688558ff`, 17 step entries
instead of 7). Every affected run so far is a fault-isolation retry survivor, consistent
with the crash cascade already documented in BENCH.md, not a new/different failure mode.

Deliberately NOT fixed at the source: `xai/spectral_error.py` is live-imported and called
by EVERY currently-running core_matrix task at EVERY checkpoint -- unlike this session's
other mid-sweep fixes (`block_mass`, the noise wiring), an append-vs-overwrite behavior
change here is NOT a provable no-op for in-flight work (every task uses the 'specerr'
instrument). Flagged here for T4.8's adversarial self-review / a future resume-hardening
pass instead of patched under time pressure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results" / "runs"


def specerr_has_duplicate_steps(run_dir) -> bool:
    npz_path = Path(run_dir) / "xai" / "specerr.npz"
    if not npz_path.is_file():
        return False
    steps = np.load(npz_path)["steps"]
    return len(steps) != len(set(steps.tolist()))


def scan_core_matrix_for_contamination(label_by_run_id: dict, runs_dir: Path = RUNS_DIR) -> dict:
    """Returns {'n_checked', 'n_contaminated', 'contaminated': [{'run_id', 'family',
    'problem', 'steps'}, ...]} -- only over runs whose specerr.npz actually exists
    (in-progress/not-yet-instrumented runs are silently skipped, not miscounted)."""
    n_checked = 0
    contaminated = []
    for run_id, info in label_by_run_id.items():
        run_dir = Path(runs_dir) / run_id
        npz_path = run_dir / "xai" / "specerr.npz"
        if not npz_path.is_file():
            continue
        n_checked += 1
        if specerr_has_duplicate_steps(run_dir):
            contaminated.append(
                {
                    "run_id": run_id,
                    "family": info["family"],
                    "problem": info["problem"],
                    "steps": np.load(npz_path)["steps"].tolist(),
                }
            )
    return {"n_checked": n_checked, "n_contaminated": len(contaminated), "contaminated": contaminated}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from cost_ledger import enumerate_core_matrix_run_ids

    label_by_run_id = enumerate_core_matrix_run_ids()
    report = scan_core_matrix_for_contamination(label_by_run_id)
    print(json.dumps(report, indent=2))
