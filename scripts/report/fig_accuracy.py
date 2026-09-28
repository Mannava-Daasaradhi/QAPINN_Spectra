"""Accuracy, cost and training-curve figures from the core matrix (5 seeds)."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from report.common import (
    COLORS,
    FAMILIES,
    GRID,
    LABELS,
    MARKERS,
    MUTED,
    PROBLEM_LABELS,
    PROBLEMS,
    REPO_ROOT,
    core_runs,
    family_legend,
    save,
    select,
    style,
)

STATUS = {  # reserved status colours, always shown with the verdict text
    "CONFIRMED": "#2e7d32", "REFUTED": "#c62828", "INCONCLUSIVE": "#ef6c00", "INSUFFICIENT_DATA": "#9e9e9e",
}


def accuracy_small_multiples(runs) -> None:
    """Every seed of every family on every problem. Filled = submitted seeds (0-1),
    hollow = seeds 2-4 added to complete the protocol; the bar is the 5-seed median."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharey=False)
    for ax, problem in zip(axes.flat, PROBLEMS):
        for i, fam in enumerate(FAMILIES):
            rs = select(runs, problem=problem, family=fam)
            if not rs:
                continue
            vals = np.array([r.metrics["rel_l2"] for r in rs])
            jitter = np.linspace(-0.18, 0.18, len(rs))
            for r, v, j in zip(rs, vals, jitter):
                ax.scatter(i + j, v, marker=MARKERS[fam], s=28, color=COLORS[fam] if r.submitted else "white",
                           edgecolor=COLORS[fam], linewidth=1.2, zorder=3)
            ax.hlines(np.median(vals), i - 0.3, i + 0.3, color=COLORS[fam], linewidth=2.5, zorder=4)
        ax.set_yscale("log")
        ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.set_xticks(range(len(FAMILIES)), [LABELS[f] for f in FAMILIES], rotation=40, ha="right")
        ax.set_title(PROBLEM_LABELS[problem])
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("relative L2 error (log)")
    fig.suptitle("Final error of every run: filled = submitted seeds 0-1, hollow = seeds 2-4, bar = median;"
                 " dotted line = error of predicting zero", fontsize=9, color=MUTED, y=1.0)
    fig.tight_layout()
    save(fig, "accuracy_all_runs")


def verdict_comparison() -> None:
    """Submitted (2 seeds) vs full protocol (5 seeds), every check."""
    paths = {"Submitted, 2 seeds": REPO_ROOT / "results" / "adjudication.json",
             "Full protocol, 5 seeds": REPO_ROOT / "results" / "adjudication_v1_full.json"}
    if not all(p.is_file() for p in paths.values()):
        return
    records = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}
    checks = list(records["Submitted, 2 seeds"])
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(checks) + 0.9))
    for col, (name, rec) in enumerate(records.items()):
        for row, check in enumerate(checks):
            verdict = rec.get(check, {}).get("verdict", "INSUFFICIENT_DATA")
            ax.add_patch(plt.Rectangle((col + 0.04, row + 0.08), 0.92, 0.84, color=STATUS[verdict], alpha=0.9,
                                       linewidth=0))
            ax.text(col + 0.5, row + 0.5, verdict.replace("_", " ").title(), ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(0, 2)
    ax.set_ylim(len(checks), 0)
    ax.set_xticks([0.5, 1.5], list(records), fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(checks))], checks)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    counts = {n: sum(r.get(c, {}).get("verdict") == "CONFIRMED" for c in checks) for n, r in records.items()}
    ax.set_title(" -> ".join(f"{v}/{len(checks)} confirmed" for v in counts.values()), color=MUTED,
                 fontsize=9, pad=24)
    save(fig, "verdicts_submitted_vs_full")


def cost_vs_accuracy(runs) -> None:
    """Parameters and wall-clock time against error, median over seeds, per problem."""
    for metric, xlabel, name in (("n_params", "trainable parameters (log)", "params"),
                                 ("wall_clock_s", "wall-clock minutes per run (log)", "wallclock")):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
        for ax, problem in zip(axes.flat, PROBLEMS):
            for fam in FAMILIES:
                rs = select(runs, problem=problem, family=fam)
                if not rs:
                    continue
                x = np.median([r.metrics[metric] for r in rs]) / (60.0 if metric == "wall_clock_s" else 1.0)
                y = np.median([r.metrics["rel_l2"] for r in rs])
                ax.scatter(x, y, marker=MARKERS[fam], s=60, color=COLORS[fam], edgecolor="white", linewidth=1.5,
                           zorder=3)
                ax.annotate(LABELS[fam], (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7, color=MUTED)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(PROBLEM_LABELS[problem])
        for ax in axes[1]:
            ax.set_xlabel(xlabel)
        for ax in axes[:, 0]:
            ax.set_ylabel("median relative L2 (log)")
        fig.tight_layout()
        save(fig, f"cost_vs_accuracy_{name}")


def training_curves(runs) -> None:
    """Median training loss across seeds, with the seed range shaded."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, problem in zip(axes.flat, PROBLEMS):
        for fam in FAMILIES:
            rs = select(runs, problem=problem, family=fam)
            if not rs:
                continue
            frames = []
            for r in rs:
                h = pd.read_parquet(r.dir / "history.parquet")[["step", "loss"]]
                frames.append(h.set_index("step")["loss"])
            table = pd.concat(frames, axis=1).dropna()
            steps = table.index.values[::50]
            sub = table.iloc[::50]
            ax.fill_between(steps, sub.min(axis=1), sub.max(axis=1), color=COLORS[fam], alpha=0.12, linewidth=0)
            ax.plot(steps, sub.median(axis=1), color=COLORS[fam], linewidth=1.6)
        ax.set_yscale("log")
        ax.axvline(20000, color=GRID, linewidth=1.2)
        ax.set_title(PROBLEM_LABELS[problem])
    for ax in axes[1]:
        ax.set_xlabel("training step (Adam to 20,000, then L-BFGS)")
    for ax in axes[:, 0]:
        ax.set_ylabel("training loss (log)")
    family_legend(axes[0, 2], FAMILIES, loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, "training_curves")


def seed_spread(runs) -> None:
    """How much the error varies across seeds: max/min ratio of rel-L2 over the 5 seeds."""
    grid = np.full((len(FAMILIES), len(PROBLEMS)), np.nan)
    for i, fam in enumerate(FAMILIES):
        for j, problem in enumerate(PROBLEMS):
            vals = [r.metrics["rel_l2"] for r in select(runs, problem=problem, family=fam)]
            if len(vals) >= 2:
                grid[i, j] = max(vals) / min(vals)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(np.log10(grid), cmap="Blues", aspect="auto", vmin=0)
    for i in range(len(FAMILIES)):
        for j in range(len(PROBLEMS)):
            if np.isfinite(grid[i, j]):
                v = grid[i, j]
                ax.text(j, i, f"{v:.1f}x" if v < 100 else f"{v:.0e}x", ha="center", va="center", fontsize=7,
                        color="white" if np.log10(v) > 0.6 * np.nanmax(np.log10(grid)) else "#1f1f1f")
    ax.set_xticks(range(len(PROBLEMS)), [PROBLEM_LABELS[p] for p in PROBLEMS], rotation=30, ha="right")
    ax.set_yticks(range(len(FAMILIES)), [LABELS[f] for f in FAMILIES])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="log10(max / min rel-L2 across seeds)")
    ax.set_title("Seed sensitivity: worst seed / best seed")
    save(fig, "seed_spread")


def main() -> None:
    style()
    runs = core_runs()
    accuracy_small_multiples(runs)
    verdict_comparison()
    cost_vs_accuracy(runs)
    training_curves(runs)
    seed_spread(runs)


if __name__ == "__main__":
    main()
