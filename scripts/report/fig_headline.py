"""Headline charts for the WISER report: the design-vs-random result per seed, and the
v2 lab summary (exploratory: tuning seeds 0-2, best of several configurations)."""
from __future__ import annotations

import json
import statistics

import matplotlib.pyplot as plt
import numpy as np

from qapinn.runner import enumerate_runs
from report.common import (
    COLORS,
    LABELS,
    MUTED,
    PROBLEM_LABELS,
    REPO_ROOT,
    core_runs,
    save,
    select,
    style,
)


def design_vs_random(runs) -> None:
    """q_serial vs q_random, every seed, on the two problems PR-6 tests."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, problem in zip(axes, ("heat", "poisson")):
        for k, fam in enumerate(("q_serial", "q_random")):
            rs = sorted(select(runs, problem=problem, family=fam), key=lambda r: r.seed)
            seeds = [r.seed for r in rs]
            vals = [r.metrics["rel_l2"] for r in rs]
            x = np.arange(len(seeds)) + (k - 0.5) * 0.38
            ax.bar(x, vals, width=0.36, color=COLORS[fam], label="SMCD-designed circuit" if k == 0 else
                   "same circuit, random frequencies")
        ratio = (statistics.median(r.metrics["rel_l2"] for r in select(runs, problem=problem, family="q_random"))
                 / statistics.median(r.metrics["rel_l2"] for r in select(runs, problem=problem, family="q_serial")))
        ax.set_yscale("log")
        ax.set_xticks(range(len(seeds)), [f"seed {s}" for s in seeds])
        ax.set_title(f"{PROBLEM_LABELS[problem]}: designed circuit {ratio:.1f}x lower error (median)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("relative L2 error (log)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "headline_design_vs_random")


def v2_lab() -> None:
    """Median rel-L2 on Poisson (tuning seeds) for each quantum configuration and the best
    learning rate of each classical model."""
    runs_dir = REPO_ROOT / "results" / "runs"

    params: dict = {}

    def med(exp, key):
        groups: dict = {}
        for c in enumerate_runs(REPO_ROOT / "configs" / "exp" / f"{exp}.yaml"):
            m = runs_dir / c.run_id / "metrics.json"
            if m.is_file():
                metrics = json.loads(m.read_text(encoding="utf-8"))
                groups.setdefault(key(c), []).append(metrics["rel_l2"])
                params[key(c)[0]] = metrics["n_params"]
        return {k: statistics.median(v) for k, v in groups.items()}

    q = med("lab_quantum", lambda c: (c.model.n_replicas, c.train.lr))
    cl = med("lab_classical", lambda c: (c.model.family, c.train.lr))
    best_cl = {f: min(v for (ff, _), v in cl.items() if ff == f) for f in {f for f, _ in cl}}
    fig, ax = plt.subplots(figsize=(10, 4))
    labels, vals, colors = [], [], []
    for k in sorted({k for k, _ in q}):
        for lr in sorted({lr for kk, lr in q if kk == k}):
            labels.append(f"{k} cop{'y' if k == 1 else 'ies'}\nlr {lr:g}\n{params[k]} par.")
            vals.append(q[(k, lr)])
            colors.append(COLORS["q_serial"])
    for f in ("c_rff_matched", "c_ff", "c_mlp"):
        labels.append(f"{LABELS[f].replace(' ', chr(10), 1)}\n{params[f]:,} par.")
        vals.append(best_cl[f])
        colors.append(COLORS[f])
    ax.bar(range(len(vals)), vals, color=colors, width=0.7)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3g}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7,
                    color=MUTED)
    ax.set_yscale("log")
    ax.set_xticks(range(len(vals)), labels, fontsize=7)
    ax.set_ylabel("median relative L2 error (log)")
    ax.set_title("v2 lab, Poisson, tuning seeds (exploratory): more circuit copies + higher learning rate")
    ax.grid(axis="x", visible=False)
    save(fig, "headline_v2_lab")


STAIRCASE_RUNS = ("0280f156172b", "a5f1ca54add5", "5669659249e2")  # T1.5, c_mlp on Poisson, seeds 0-2


def staircase_steps() -> dict:
    """Step at which each frequency's error first halves, per staircase seed."""
    out = {"pi": [], "15pi": []}
    for rid in STAIRCASE_RUNS:
        with np.load(REPO_ROOT / "results" / "runs" / rid / "xai" / "specerr.npz") as d:
            for name, w in (("pi", np.pi), ("15pi", 15 * np.pi)):
                i = int(np.argmin(np.abs(d["omega"] - w)))
                e = np.abs(d["errors"][:, i])
                out[name].append(int(d["steps"][np.nonzero(e <= 0.5 * e[0])[0][0]]))
    return out


def staircase() -> None:
    """Spectral bias in a classical PINN: the slow wave is learned long before the fast one."""
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    halves = staircase_steps()
    for w, color, label, key in ((np.pi, COLORS["c_ff"], "slow wave  sin(πx)", "pi"),
                                 (15 * np.pi, COLORS["q_serial"], "fast wave  sin(15πx)", "15pi")):
        curves = []
        for rid in STAIRCASE_RUNS:
            with np.load(REPO_ROOT / "results" / "runs" / rid / "xai" / "specerr.npz") as d:
                steps = np.maximum(d["steps"], 1)
                i = int(np.argmin(np.abs(d["omega"] - w)))
                e = np.abs(d["errors"][:, i])
                curves.append(np.minimum(e / e[0], 1.2))
        curves = np.array(curves)
        ax.fill_between(steps, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.15, linewidth=0)
        ax.plot(steps, np.median(curves, axis=0), color=color, linewidth=2.2, label=label)
        step = float(np.median(halves[key]))
        ax.axvline(step, color=color, linestyle="--", linewidth=1)
        ax.annotate(f"halved by step ~{step:,.0f}", (step, 1.12), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("error / starting error")
    ax.set_title("A classical PINN learns the slow wave first (median of 3 seeds, band = range)")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "staircase")


def main() -> None:
    style()
    design_vs_random(core_runs())
    v2_lab()
    staircase()


if __name__ == "__main__":
    main()
