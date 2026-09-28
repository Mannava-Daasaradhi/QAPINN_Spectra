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


def preregistration_commit_sha(predictions_path: Path = PREDICTIONS_PATH) -> str:
    """The commit that FIRST added docs/predictions.md (`--diff-filter=A`, not just the
    latest commit to touch it -- T3.1's own DoD anticipates a later commit recording this
    SHA into FINDINGS.md, which must not be mistaken for a second pre-registration)."""
    if not predictions_path.is_file():
        raise FileNotFoundError(f"{predictions_path} does not exist -- pre-registration was never committed")
    log = _run_git("log", "--diff-filter=A", "--format=%H", "--", str(predictions_path.relative_to(REPO_ROOT)))
    commits = log.splitlines()
    if not commits:
        raise RuntimeError(f"{predictions_path} exists but git has no ADDING commit for it -- uncommitted?")
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
        check=False,  # exit codes 0/1 are both answers, handled below
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise ValueError(f"git could not resolve {descendant_sha!r}: {result.stderr.strip()}")


def verify_predictions_precede_runs(run_dirs: list[Path], predictions_path: Path = PREDICTIONS_PATH) -> dict:
    """Returns {'predictions_commit': sha, 'n_checked': int, 'violations': [run_id, ...],
    'missing_provenance': [run_dir_name, ...], 'unknown_sha': [(run_id, sha), ...]}.

    'unknown_sha' covers a `git_sha` that git's local history doesn't recognise at all
    (e.g. a run recorded from a since-rebased/force-pushed branch) -- reported separately
    from 'violations' (a KNOWN commit that is genuinely NOT a descendant of
    pre-registration) since the two failure modes need different remediation.
    """
    predictions_sha = preregistration_commit_sha(predictions_path)
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


# The experiments docs/predictions.md pre-registers (T3.1-T3.6), i.e. the T3.10 gate's
# scope. configs/exp/baseline_tuning.yaml (T4.9) is deliberately absent: it is a
# post-adjudication fairness check, not a pre-registered prediction, and its 12 runs
# record a git_sha (d4c6877) that never reached the published history.
PREREGISTERED_EXPERIMENTS = ("core_matrix", "coverage_sweep", "depth_sweep", "alpha_sweep", "noise_study", "soft_bc_ntk")


# v2 (docs/predictions_v2.md): the re-test and groundwater experiments. The lab
# (lab_*.yaml) is deliberately absent: it ran before the v2 predictions, by design.
PREDICTIONS_V2_PATH = REPO_ROOT / "docs" / "predictions_v2.md"
V2_EXPERIMENTS = (
    "v2_matrix_classical",
    "v2_matrix_quantum",
    "v2_coverage_sweep",
    "v2_depth_sweep",
    "v2_groundwater_classical",
    "v2_groundwater_quantum",
)


# The groundwater showcase with v1 models (docs/predictions_groundwater.md).
PREDICTIONS_GROUNDWATER_PATH = REPO_ROOT / "docs" / "predictions_groundwater.md"
GROUNDWATER_EXPERIMENTS = ("groundwater_v1",)


def experiment_run_dirs(experiments: tuple[str, ...] = PREREGISTERED_EXPERIMENTS) -> list[Path]:
    """Every completed run of the pre-registered experiments. A raw glob of results/runs/
    also sweeps in Phase 0-2 development and smoke-test runs whose commits legitimately
    predate pre-registration -- which is what this CLI used to do, so it exited 1 (65
    "violations", 16 unresolvable SHAs, none of them pre-registered runs) on the very
    repository whose gate it documents."""
    import sys

    src_dir = REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from qapinn.runner import enumerate_runs

    runs_dir = REPO_ROOT / "results" / "runs"
    run_dirs = {
        runs_dir / cfg.run_id
        for exp in experiments
        if (REPO_ROOT / "configs" / "exp" / f"{exp}.yaml").is_file()
        for cfg in enumerate_runs(REPO_ROOT / "configs" / "exp" / f"{exp}.yaml")
        if (runs_dir / cfg.run_id / "metrics.json").is_file()
    }
    return sorted(run_dirs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check that pre-registration predates every run.")
    parser.add_argument("--v2", action="store_true", help="check docs/predictions_v2.md against the v2 runs")
    parser.add_argument("--groundwater", action="store_true",
                        help="check docs/predictions_groundwater.md against the groundwater_v1 runs")
    cli = parser.parse_args()
    if cli.groundwater:
        report = verify_predictions_precede_runs(
            experiment_run_dirs(GROUNDWATER_EXPERIMENTS), PREDICTIONS_GROUNDWATER_PATH
        )
    elif cli.v2:
        report = verify_predictions_precede_runs(experiment_run_dirs(V2_EXPERIMENTS), PREDICTIONS_V2_PATH)
    else:
        report = verify_predictions_precede_runs(experiment_run_dirs())
    print(json.dumps(report, indent=2))
    if report["violations"] or report["unknown_sha"]:
        raise SystemExit(1)
