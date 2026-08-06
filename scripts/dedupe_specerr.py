"""One-time cleanup for the crash-retry duplicate-checkpoint corruption documented in
`scripts/check_specerr_integrity.py`. Deliberately deferred there while T3.4-T3.6's
sweeps were still in-flight (touching `xai/spectral_error.py`'s append-only write path
was not a provable no-op for running tasks); now that every sweep is complete, the
post-hoc fix is safe: for each contaminated `xai/specerr.npz`, keep the LAST occurrence
of each real checkpoint step (the most recent, post-crash-recovery value) and drop the
stale duplicate rows written by the crashed attempt before it.

Run once, by hand: `python scripts/dedupe_specerr.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_specerr_integrity import specerr_has_duplicate_steps  # noqa: E402


def dedupe_specerr(run_dir: Path) -> dict:
    """Rewrites `run_dir/xai/specerr.npz` keeping only the last occurrence of each
    distinct step, sorted ascending. Returns {'run_id', 'before', 'after'}."""
    path = Path(run_dir) / "xai" / "specerr.npz"
    data = np.load(path)
    omega, errors, steps = data["omega"], data["errors"], data["steps"]

    last_idx: dict[int, int] = {}
    for i, s in enumerate(steps.tolist()):
        last_idx[s] = i  # later i overwrites earlier -> last occurrence wins
    keep_steps = sorted(last_idx)
    keep_idx = [last_idx[s] for s in keep_steps]

    new_steps = steps[keep_idx]
    new_errors = errors[keep_idx]
    assert len(new_steps) == len(set(new_steps.tolist())), "dedup produced duplicates -- bug"

    np.savez(path, omega=omega, errors=new_errors, steps=new_steps)
    return {"run_id": Path(run_dir).name, "before": len(steps), "after": len(new_steps)}


if __name__ == "__main__":
    from qapinn.runner import enumerate_runs

    runs_dir = REPO_ROOT / "results" / "runs"
    cfgs = enumerate_runs(REPO_ROOT / "configs" / "exp" / "core_matrix.yaml")
    fixed = []
    for c in cfgs:
        run_dir = runs_dir / c.run_id
        if specerr_has_duplicate_steps(run_dir):
            fixed.append(dedupe_specerr(run_dir))

    print(f"Deduped {len(fixed)} specerr.npz files:")
    for f in fixed:
        print(f"  {f['run_id']}: {f['before']} -> {f['after']} checkpoint rows")
