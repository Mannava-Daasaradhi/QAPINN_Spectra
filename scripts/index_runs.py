"""Write results/runs_index.csv: one row per committed run that carries data, saying which
experiment it belongs to, so the 12-character run IDs can be navigated without opening
every config.yaml. Development smoke runs (20+5 or 50+10 training steps) and incomplete
directories (no config.yaml) are counted but not listed.

Run: uv run python scripts/index_runs.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml

from qapinn.runner import enumerate_labeled_runs

RUNS_DIR = REPO_ROOT / "results" / "runs"
INDEX_PATH = REPO_ROOT / "results" / "runs_index.csv"
# T1.5's spectral-bias staircase gate runs (docs/plan/BENCH.md, T1.5 section).
STAIRCASE_RUNS = {"0280f156172b", "a5f1ca54add5", "5669659249e2"}
# Smoke-scale fixtures pinned by tests/test_phase3_figures.py and test_adjudicate_predictions.py.
TEST_FIXTURES = {"13972243cc87", "3cabbf515e05"}
DEV_SMOKE_STEPS = {(20, 5), (50, 10)}
HELMHOLTZ_LABELS = {4.0: "helmholtz_k4", 10.0: "helmholtz_k10", 20.0: "helmholtz_k20"}
FIELDS = ["run_id", "group", "problem", "family", "seed", "steps_adam", "steps_lbfgs", "rel_l2"]


def _problem_label(cfg: dict) -> str:
    pde = cfg["pde"]
    if pde["name"] == "helmholtz":
        return HELMHOLTZ_LABELS.get(float(pde["params"]["k"]), f"helmholtz_k{pde['params']['k']}")
    return pde["name"]


def _group(run_id: str, cfg: dict, experiment_of: dict[str, str]) -> str:
    if run_id in experiment_of:
        return experiment_of[run_id]
    if run_id in STAIRCASE_RUNS:
        return "t1.5_staircase"
    if run_id in TEST_FIXTURES:
        return "test_fixture"
    steps = (cfg["train"]["steps_adam"], cfg["train"]["steps_lbfgs"])
    if steps in DEV_SMOKE_STEPS:
        return "dev_smoke"
    if cfg.get("smcd_coverage_target") is not None:
        return "coverage_sweep_superseded"  # earlier step budgets, see coverage_sweep.yaml
    if steps == (20000, 2000) and cfg["seed"] >= 2:
        return "core_matrix_extra_seed"  # seeds cut from the analysis, core_matrix.yaml
    return "other"


def build_index() -> tuple[list[dict], dict[str, int]]:
    experiment_of = {
        cfg.run_id: exp.stem
        for exp in sorted((REPO_ROOT / "configs" / "exp").glob("*.yaml"))
        for _label, cfg in enumerate_labeled_runs(exp)
    }
    rows: list[dict] = []
    unlisted = {"dev_smoke": 0, "incomplete": 0}
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.is_file():
            unlisted["incomplete"] += 1
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        group = _group(run_dir.name, cfg, experiment_of)
        if group == "dev_smoke":
            unlisted["dev_smoke"] += 1
            continue
        metrics_path = run_dir / "metrics.json"
        rel_l2 = json.loads(metrics_path.read_text(encoding="utf-8"))["rel_l2"] if metrics_path.is_file() else None
        rows.append({
            "run_id": run_dir.name,
            "group": group,
            "problem": _problem_label(cfg),
            "family": cfg["model"]["family"],
            "seed": cfg["seed"],
            "steps_adam": cfg["train"]["steps_adam"],
            "steps_lbfgs": cfg["train"]["steps_lbfgs"],
            "rel_l2": f"{rel_l2:.6g}" if rel_l2 is not None else "",
        })
    rows.sort(key=lambda r: (r["group"], r["problem"], r["family"], r["seed"], r["run_id"]))
    return rows, unlisted


def main() -> None:
    rows, unlisted = build_index()
    with INDEX_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    print(f"wrote {INDEX_PATH.relative_to(REPO_ROOT)}: {len(rows)} runs")
    for group, n in sorted(counts.items()):
        print(f"  {group:28s} {n}")
    print(f"not listed: {unlisted['dev_smoke']} dev smoke runs, {unlisted['incomplete']} incomplete directories")


if __name__ == "__main__":
    main()
