"""Apply the v2 lab selection rule (docs/superpowers/specs/2026-09-28-*-design.md,
"Amendments") and write results/lab_selection.json.

Quantum (lab_quantum*.yaml): among configurations (n_replicas, bc_mode, lr), the smallest
n_replicas whose median rel-L2 over seeds 0-2 is within 10% of the best configuration's;
ties on n_replicas go to the lower median. Classical (lab_classical*.yaml): per family,
the (bc_mode, learning rate) with the lowest median rel-L2.

Run: uv run python scripts/select_v2_config.py [--write]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qapinn.runner import enumerate_runs

RUNS_DIR = REPO_ROOT / "results" / "runs"
OUTPUT = REPO_ROOT / "results" / "lab_selection.json"
PARSIMONY = 1.10


def _median_rel_l2_by_config(exps: tuple[str, ...], key) -> dict:
    groups: dict = {}
    for cfg in (c for exp in exps for c in enumerate_runs(REPO_ROOT / "configs" / "exp" / f"{exp}.yaml")):
        metrics = RUNS_DIR / cfg.run_id / "metrics.json"
        if not metrics.is_file():
            continue
        rel_l2 = json.loads(metrics.read_text(encoding="utf-8"))["rel_l2"]
        groups.setdefault(key(cfg), []).append(rel_l2)
    return {k: {"median_rel_l2": statistics.median(v), "rel_l2": sorted(v), "n": len(v)} for k, v in groups.items()}


def select() -> dict:
    quantum = _median_rel_l2_by_config(
        ("lab_quantum", "lab_quantum_affine"), lambda c: (c.model.n_replicas, c.train.bc_mode, c.train.lr)
    )
    best = min(v["median_rel_l2"] for v in quantum.values())
    eligible = [k for k, v in quantum.items() if v["median_rel_l2"] <= PARSIMONY * best]
    chosen = min(eligible, key=lambda k: (k[0], quantum[k]["median_rel_l2"]))

    classical = {}
    by_family = _median_rel_l2_by_config(
        ("lab_classical", "lab_classical_affine"), lambda c: (c.model.family, c.train.bc_mode, c.train.lr)
    )
    for family in sorted({f for f, _, _ in by_family}):
        rows = {(bc, lr): v for (f, bc, lr), v in by_family.items() if f == family}
        bc, lr = min(rows, key=lambda r: rows[r]["median_rel_l2"])
        classical[family] = {"bc_mode": bc, "lr": lr, "median_rel_l2": rows[(bc, lr)]["median_rel_l2"],
                             "all": {f"bc_mode={k[0]}, lr={k[1]}": v for k, v in sorted(rows.items())}}

    return {
        "rule": "quantum: smallest n_replicas within 10% of the best median rel-L2; "
        "classical: lowest median rel-L2 per family",
        "quantum": {
            "n_replicas": chosen[0],
            "bc_mode": chosen[1],
            "lr": chosen[2],
            "median_rel_l2": quantum[chosen]["median_rel_l2"],
            "best_median_rel_l2": best,
            "all": {f"n_replicas={k[0]}, bc_mode={k[1]}, lr={k[2]}": v for k, v in sorted(quantum.items())},
        },
        "classical": classical,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help=f"write {OUTPUT.relative_to(REPO_ROOT)}")
    cli = parser.parse_args()
    report = select()
    print(json.dumps(report, indent=2))
    if cli.write:
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
