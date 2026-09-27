"""Adjudicate G-1 to G-5 (docs/predictions_v2.md) against the committed groundwater runs.

G-2 and G-3 use v1's PR-1/PR-6 rule (`_beats_by_ratio`: ratio threshold, then the
paired Wilcoxon p-value floor for n = 5), G-4 uses v1's PR-8 overlap computation. Writes
results/adjudication_groundwater.json with --write.

Run: uv run python scripts/adjudicate_groundwater.py [--write]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from adjudicate_predictions import _beats_by_ratio, portable_record

from qapinn.config import load_config
from qapinn.pdes import build as build_pde
from qapinn.pdes.groundwater import waterlogged_length_m

RUNS_DIR = REPO_ROOT / "results" / "runs"
OUTPUT = REPO_ROOT / "results" / "adjudication_groundwater.json"
# docs/predictions_v2.md: one config per family (each carries its lab-selected lr).
EXPERIMENTS = ("v2_groundwater_classical", "v2_groundwater_quantum")
THRESHOLD_M = 7.5  # water table within 1.5 m of a land surface at 9 m


def runs_by_family() -> dict[str, list[Path]]:
    """Completed runs keyed by model config name (c_mlp_matched and c_mlp share a family
    but are different configs)."""
    from cost_ledger import enumerate_core_matrix_run_ids

    out: dict[str, list[Path]] = {}
    for exp in EXPERIMENTS:
        for run_id, info in enumerate_core_matrix_run_ids(REPO_ROOT / "configs" / "exp" / f"{exp}.yaml").items():
            if (RUNS_DIR / run_id / "metrics.json").is_file():
                out.setdefault(info["family"], []).append(RUNS_DIR / run_id)
    return out


def _rel_l2(run_dir: Path) -> float:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["rel_l2"]


def _waterlogged(run_dir: Path, length_m: float) -> float:
    data = np.load(run_dir / "prediction.npz")
    order = np.argsort(data["x"])
    return waterlogged_length_m(data["x"][order] * length_m, data["u"][order], THRESHOLD_M)


def adjudicate() -> dict:
    pde = build_pde(load_config("groundwater", "c_mlp").pde)
    length_m = pde.params["length_m"]
    X = np.linspace(0.0, length_m, 20001)
    exact_length = waterlogged_length_m(X, pde.head_m(X), THRESHOLD_M)

    runs = runs_by_family()
    need = ("c_mlp", "c_mlp_matched", "c_rff_matched", "q_serial_v2", "q_random_v2")
    missing = [f for f in need if len(runs.get(f, [])) < 2]
    if missing:
        return {"verdict": "INSUFFICIENT_DATA", "reason": f"fewer than 2 runs for {missing}"}

    q = runs["q_serial_v2"]
    report: dict = {"families": {
        f: {"median_rel_l2": statistics.median(_rel_l2(r) for r in runs[f]),
            "median_waterlogged_m": statistics.median(_waterlogged(r, length_m) for r in runs[f]),
            "n": len(runs[f])}
        for f in sorted(runs)
    }, "exact_waterlogged_m": exact_length}

    g1 = report["families"]["q_serial_v2"]["median_rel_l2"]
    report["G-1"] = {"verdict": "CONFIRMED" if g1 <= 0.01 else "REFUTED", "median_rel_l2": g1, "threshold": 0.01}
    report["G-2"] = _beats_by_ratio(q, runs["c_mlp_matched"], threshold=1.3)
    report["G-3"] = _beats_by_ratio(q, runs["q_random_v2"], threshold=1.5)

    from make_figures import make_freq_heatmap_figure

    from qapinn.models.circuits import ReuploadCircuit
    from qapinn.smcd.design import smcd

    card = smcd(pde)
    circuit = ReuploadCircuit(card.n_qubits, card.n_layers, card.scalings, tuple(card.wire_to_dim),
                              card.entangler, card.observable)
    omega = circuit.frequencies().reshape(-1, 1)
    overlap = make_freq_heatmap_figure("groundwater", runs["c_mlp"][0], q[0], omega.tolist())
    if overlap["total_improvement"] <= 0:
        report["G-4"] = {"verdict": "REFUTED", "threshold": 0.7, "reason": "no improvement at any frequency", **overlap}
    else:
        report["G-4"] = {"verdict": "CONFIRMED" if overlap["overlap_fraction"] >= 0.7 else "REFUTED",
                         "threshold": 0.7, **overlap}

    g5 = report["families"]["q_serial_v2"]["median_waterlogged_m"]
    rel = abs(g5 - exact_length) / exact_length
    report["G-5"] = {"verdict": "CONFIRMED" if rel <= 0.05 else "REFUTED", "median_waterlogged_m": g5,
                     "exact_m": exact_length, "relative_error": rel, "threshold": 0.05}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help=f"write {OUTPUT.relative_to(REPO_ROOT)}")
    cli = parser.parse_args()
    from qapinn.viz import style

    style.FIGURES_DIR = REPO_ROOT / "results" / "figures" / "v2"
    result = adjudicate()
    print(json.dumps(result, indent=2, default=str))
    if cli.write:
        OUTPUT.write_text(json.dumps(portable_record(result), indent=2) + "\n", encoding="utf-8", newline="\n")
