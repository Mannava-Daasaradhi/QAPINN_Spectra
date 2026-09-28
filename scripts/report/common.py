"""Shared loading and styling for the report figures (scripts/report/*.py).

The report reads only committed run artifacts (config.yaml, metrics.json,
history.parquet, xai/*.npz), like the paper's figures, and writes PNGs to
results/report/figures/. It never touches paper/figures/.

Colour: a report palette validated with the dataviz skill's validate_palette.js (every
adjacent pair passes the colour-vision-deficiency and normal-vision checks; all seven
pairwise cannot, so family identity is never colour alone: families sit on an axis or
carry a direct label and a distinct marker). The paper keeps its own palette
(qapinn.viz.style), which CI pins.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cost_ledger import enumerate_core_matrix_run_ids

from qapinn.viz.style import apply_style

RUNS_DIR = REPO_ROOT / "results" / "runs"
OUT_DIR = REPO_ROOT / "results" / "report" / "figures"
EXP_DIR = REPO_ROOT / "configs" / "exp"

FAMILIES = ["c_mlp", "c_ff", "c_rff_matched", "q_serial", "q_parallel", "q_random", "q_octave"]
PROBLEMS = ["poisson", "heat", "burgers", "helmholtz_k4", "helmholtz_k10", "helmholtz_k20"]
QUANTUM = {"q_serial", "q_parallel", "q_random", "q_octave"}

COLORS = {
    "c_mlp": "#4f7a28",
    "c_ff": "#0072B2",
    "c_rff_matched": "#009E73",
    "q_serial": "#D55E00",
    "q_parallel": "#E69F00",
    "q_random": "#CC79A7",
    "q_octave": "#a0522d",
}
MARKERS = {"c_mlp": "o", "c_ff": "s", "c_rff_matched": "D", "q_serial": "^", "q_parallel": "v",
           "q_random": "P", "q_octave": "X"}
LABELS = {
    "c_mlp": "MLP", "c_ff": "Fourier feat.", "c_rff_matched": "RFF matched",
    "q_serial": "q_serial (SMCD)", "q_parallel": "q_parallel", "q_random": "q_random", "q_octave": "q_octave",
}
PROBLEM_LABELS = {
    "poisson": "Poisson", "heat": "Heat", "burgers": "Burgers",
    "helmholtz_k4": "Helmholtz k=4", "helmholtz_k10": "Helmholtz k=10", "helmholtz_k20": "Helmholtz k=20",
    "groundwater": "Groundwater",
}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


@dataclass(frozen=True)
class Run:
    run_id: str
    problem: str
    family: str
    seed: int
    submitted: bool  # part of the 2-seed matrix judged at WISER 2026

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.run_id

    @property
    def metrics(self) -> dict:
        return json.loads((self.dir / "metrics.json").read_text(encoding="utf-8"))

    @property
    def config(self) -> dict:
        return yaml.safe_load((self.dir / "config.yaml").read_text(encoding="utf-8"))

    def xai(self, name: str) -> dict | None:
        path = self.dir / "xai" / f"{name}.npz"
        if not path.is_file():
            return None
        with np.load(path) as data:
            return {k: data[k] for k in data.files}

    def xai_steps(self, prefix: str) -> list[int]:
        steps = []
        for path in (self.dir / "xai").glob(f"{prefix}_step*.npz"):
            steps.append(int(path.stem.rsplit("step", 1)[1]))
        return sorted(steps)


def core_runs() -> list[Run]:
    """Every completed core-matrix run at the pre-registered 5 seeds."""
    submitted = set(enumerate_core_matrix_run_ids(EXP_DIR / "core_matrix.yaml"))
    runs = []
    for run_id, info in enumerate_core_matrix_run_ids(EXP_DIR / "core_matrix_full.yaml").items():
        if (RUNS_DIR / run_id / "metrics.json").is_file():
            runs.append(Run(run_id, info["problem"], info["family"], info["seed"], run_id in submitted))
    return runs


def select(runs: list[Run], **where) -> list[Run]:
    return [r for r in runs if all(getattr(r, k) == v for k, v in where.items())]


def style() -> None:
    apply_style()
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK, "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        # One typeface across the WISER report: its text is set in DejaVu Sans too.
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
    })


def save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    for attempt in range(5):  # Windows: a scanner can briefly lock a just-written file
        try:
            fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05, metadata={"Software": None})
            break
        except OSError:
            if attempt == 4:
                raise
            import time

            time.sleep(1.0 + attempt)
    plt.close(fig)
    return path


def family_legend(ax, families, **kw) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=COLORS[f], marker=MARKERS[f], linestyle="-", markersize=6, label=LABELS[f])
               for f in families]
    ax.legend(handles=handles, **kw)
