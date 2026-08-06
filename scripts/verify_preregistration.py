"""T3.10 gate check, item 1: mechanically verify that `docs/predictions.md`'s commit
predates every run's `provenance.json`.

`provenance.json` (`train/checkpoint.py::build_provenance`) has no wall-clock timestamp
field at all -- only `git_sha` (whatever commit was HEAD when the run started). A
git-ancestry check is used instead of a timestamp comparison: it is STRONGER (immune to
clock skew/timezone bugs, which a raw datetime comparison would not be) and uses data
that already exists, so no `src/qapinn` change was needed to make this checkable.

A run's `git_sha` passes iff the pre-registration commit is an ancestor of (or equal to)
that run's `git_sha` in the git DAG -- i.e. the pre-registration commit was already part
of history by the time that run's code was checked out.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = REPO_ROOT / "docs" / "predictions.md"


def _run_git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def preregistration_commit_sha() -> str:
    """The commit that FIRST added docs/predictions.md (`--diff-filter=A`, not just the
    latest commit to touch it -- T3.1's own DoD anticipates a later commit recording this
    SHA into FINDINGS.md, which must not be mistaken for a second pre-registration)."""
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(f"{PREDICTIONS_PATH} does not exist -- pre-registration was never committed")
    log = _run_git("log", "--diff-filter=A", "--format=%H", "--", str(PREDICTIONS_PATH.relative_to(REPO_ROOT)))
    commits = log.splitlines()
    if not commits:
        raise RuntimeError(f"{PREDICTIONS_PATH} exists but git has no ADDING commit for it -- uncommitted?")
    return commits[-1]  # oldest first with --diff-filter=A across history; take the original add


def _is_ancestor_or_equal(ancestor_sha: str, descendant_sha: str) -> bool:
    """True/False for a genuine ancestry result (`git merge-base --is-ancestor` exits 0
    for "is an ancestor", 1 for "is not"). Raises ValueError for anything else (e.g. exit
    128, "fatal: Not a valid commit name") -- an UNRESOLVABLE sha is a different failure
    mode than a resolvable-but-not-ancestor one, and callers must not conflate the two."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise ValueError(f"git could not resolve {descendant_sha!r}: {result.stderr.strip()}")


def verify_predictions_precede_runs(run_dirs: list[Path]) -> dict:
    """Returns {'predictions_commit': sha, 'n_checked': int, 'violations': [run_id, ...],
    'missing_provenance': [run_dir_name, ...], 'unknown_sha': [(run_id, sha), ...]}.

    'unknown_sha' covers a `git_sha` that git's local history doesn't recognise at all
    (e.g. a run recorded from a since-rebased/force-pushed branch) -- reported separately
    from 'violations' (a KNOWN commit that is genuinely NOT a descendant of
    pre-registration) since the two failure modes need different remediation.
    """
    predictions_sha = preregistration_commit_sha()
    violations: list[str] = []
    missing_provenance: list[str] = []
    unknown_sha: list[tuple[str, str]] = []
    n_checked = 0

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        prov_path = run_dir / "provenance.json"
        if not prov_path.is_file():
            missing_provenance.append(run_dir.name)
            continue
        provenance = json.loads(prov_path.read_text(encoding="utf-8"))
        run_sha = provenance.get("git_sha")
        run_id = provenance.get("run_id", run_dir.name)
        if not run_sha:
            unknown_sha.append((run_id, "<missing>"))
            continue

        try:
            ok = _is_ancestor_or_equal(predictions_sha, run_sha)
        except ValueError:
            unknown_sha.append((run_id, run_sha))
            continue

        n_checked += 1
        if not ok:
            violations.append(run_id)

    return {
        "predictions_commit": predictions_sha,
        "n_checked": n_checked,
        "violations": violations,
        "missing_provenance": missing_provenance,
        "unknown_sha": unknown_sha,
    }


if __name__ == "__main__":
    run_dirs = sorted((REPO_ROOT / "results" / "runs").iterdir())
    report = verify_predictions_precede_runs(run_dirs)
    print(json.dumps(report, indent=2))
    if report["violations"] or report["unknown_sha"]:
        raise SystemExit(1)
