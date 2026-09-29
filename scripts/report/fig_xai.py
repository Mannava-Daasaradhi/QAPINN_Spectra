"""XAI gallery: every recorded instrument, across families and training, plus three
analyses derived from the saved arrays (no retraining):

- landscape sharpness: curvature of the 25x25 loss slice at the trained point;
- frequency learning order: the step at which each frequency's error first halves;
- NTK vs Fisher: effective rank of the NTK against the Fisher effective dimension.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from report.common import (
    COLORS,
    FAMILIES,
    LABELS,
    MARKERS,
    MUTED,
    PROBLEM_LABELS,
    PROBLEMS,
    QUANTUM,
    core_runs,
    family_legend,
    save,
    select,
    style,
)

CHECKPOINT_PREFIXES = ("ntk", "fisher", "attribution", "drift", "probes")


def _seed0(runs, problem, family):
    rs = sorted(select(runs, problem=problem, family=family), key=lambda r: r.seed)
    return rs[0] if rs else None


def _final_step(run) -> int | None:
    steps = run.xai_steps("ntk")
    return steps[-1] if steps else None


def ntk_spectra(runs) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, problem in zip(axes.flat, PROBLEMS):
        for fam in FAMILIES:
            run = _seed0(runs, problem, fam)
            step = _final_step(run) if run else None
            data = run.xai(f"ntk_step{step}") if step is not None else None
            if data is None:
                continue
            eig = np.sort(data["eigenvalues"])[::-1]
            eig = eig[eig > 0]
            if eig.size:
                ax.loglog(np.arange(1, eig.size + 1), eig / eig[0], color=COLORS[fam], marker=MARKERS[fam],
                          markevery=max(1, eig.size // 6), markersize=4, linewidth=1.4)
        ax.set_title(PROBLEM_LABELS[problem])
    for ax in axes[1]:
        ax.set_xlabel("eigenvalue index")
    for ax in axes[:, 0]:
        ax.set_ylabel("NTK eigenvalue / largest")
    family_legend(axes[0, 2], FAMILIES, loc="lower left", fontsize=7)
    fig.suptitle("Trained NTK spectrum (seed 0): a slower fall-off means less spectral bias", fontsize=9,
                 color=MUTED)
    fig.tight_layout()
    save(fig, "xai_ntk_spectra")


def instrument_evolution(runs, problems=("poisson", "heat", "helmholtz_k4")) -> None:
    """Scalar diagnostics over training, median across seeds."""
    rows = [("ntk", "effective_rank", "NTK effective rank", True),
            ("ntk", "condition_number", "NTK condition number", True),
            ("fisher", "effective_dimension", "Fisher effective dimension", False),
            ("attribution", "correlation", "attribution-residual correlation", False)]
    fig, axes = plt.subplots(len(rows), len(problems), figsize=(10, 10), sharex=True)
    for j, problem in enumerate(problems):
        for i, (prefix, key, label, logy) in enumerate(rows):
            ax = axes[i, j]
            for fam in FAMILIES:
                rs = select(runs, problem=problem, family=fam)
                if not rs:
                    continue
                steps = rs[0].xai_steps(prefix)
                series = []
                for r in rs:
                    vals = [r.xai(f"{prefix}_step{s}") for s in steps]
                    if all(v is not None and key in v for v in vals):
                        series.append([float(v[key]) for v in vals])
                if series:
                    x = np.maximum(steps, 1)
                    ax.plot(x, np.median(series, axis=0), color=COLORS[fam], marker=MARKERS[fam], markersize=4,
                            linewidth=1.4)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(PROBLEM_LABELS[problem])
    for ax in axes[-1]:
        ax.set_xlabel("training step (log; step 0 shown at 1)")
    fig.tight_layout()
    family_legend(axes[0, -1], FAMILIES, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    save(fig, "xai_instrument_evolution")


def landscapes_and_sharpness(runs, problem="poisson") -> None:
    fams = [f for f in FAMILIES if _seed0(runs, problem, f) is not None]
    fig, axes = plt.subplots(1, len(fams), figsize=(2.2 * len(fams), 2.6))
    sharp = {}
    for ax, fam in zip(np.atleast_1d(axes), fams):
        run = _seed0(runs, problem, fam)
        steps = run.xai_steps("landscape")
        data = run.xai(f"landscape_step{steps[-1]}") if steps else None
        if data is None:
            ax.set_visible(False)
            continue
        grid = np.log10(np.maximum(data["grid"], 1e-30))
        ax.contourf(grid, levels=14, cmap="Blues_r")
        c = grid.shape[0] // 2
        ax.plot(c, c, marker="*", color="white", markersize=9, markeredgecolor="#1f1f1f")
        ax.set_title(LABELS[fam], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        raw = data["grid"]
        hxx = raw[c, c + 1] - 2 * raw[c, c] + raw[c, c - 1]
        hyy = raw[c + 1, c] - 2 * raw[c, c] + raw[c - 1, c]
        sharp[fam] = float((hxx + hyy) / 2 / max(raw[c, c], 1e-30))
    fig.suptitle(f"{PROBLEM_LABELS[problem]}: loss landscape around the trained point (log loss, star = trained"
                 " parameters, 2 random directions; darker = lower loss)", fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, f"xai_landscapes_{problem}")

    if sharp:
        fig, ax = plt.subplots(figsize=(6, 3))
        names = list(sharp)
        vals = [sharp[f] for f in names]
        ax.barh(range(len(names)), vals, color=[COLORS[f] for f in names], height=0.6)
        ax.set_yticks(range(len(names)), [LABELS[f] for f in names])
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel("relative curvature at the trained point (higher = sharper minimum)")
        ax.set_title(f"{PROBLEM_LABELS[problem]}: landscape sharpness (new, from the saved slices)")
        ax.grid(axis="y", visible=False)
        save(fig, f"xai_sharpness_{problem}")


def spectral_error_heatmaps(runs, problem="poisson", families=("c_mlp", "c_rff_matched", "q_serial")) -> None:
    fig, axes = plt.subplots(1, len(families), figsize=(3.6 * len(families), 3.2), sharey=True)
    for ax, fam in zip(axes, families):
        run = _seed0(runs, problem, fam)
        data = run.xai("specerr") if run else None
        if data is None:
            continue
        omega, err, steps = data["omega"], data["errors"], data["steps"]
        keep = (omega >= 0) & (omega <= 60 * np.pi)
        im = ax.imshow(np.log10(np.abs(err[:, keep]).T + 1e-12), aspect="auto", origin="lower", cmap="Blues",
                       extent=(-0.5, len(steps) - 0.5, omega[keep].min() / np.pi, omega[keep].max() / np.pi))
        ax.set_xticks(range(len(steps)), [str(s) for s in steps], rotation=45, fontsize=7)
        ax.set_title(LABELS[fam])
        ax.axhline(15, color="#D55E00", linewidth=1, linestyle="--")
        ax.axhline(1, color="#D55E00", linewidth=1, linestyle="--")
        ax.grid(False)
        ax.set_xlabel("checkpoint step")
    axes[0].set_ylabel("frequency / pi (dashed: solution's pi and 15 pi)")
    fig.colorbar(im, ax=axes, label="log10 |error spectrum|")
    fig.suptitle(f"{PROBLEM_LABELS[problem]}: where in frequency the error sits during training", fontsize=9,
                 color=MUTED)
    save(fig, f"xai_spectral_error_{problem}")


def frequency_learning_order(runs, problem="poisson") -> None:
    """New: step at which the error at each target frequency first falls below half its
    initial value (the spectral-bias staircase, per family)."""
    targets = {"pi": np.pi, "15 pi": 15 * np.pi}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.8 / len(targets)
    for k, (name, w) in enumerate(targets.items()):
        for i, fam in enumerate(FAMILIES):
            vals = []
            for r in select(runs, problem=problem, family=fam):
                data = r.xai("specerr")
                if data is None:
                    continue
                idx = int(np.argmin(np.abs(data["omega"] - w)))
                e = np.abs(data["errors"][:, idx])
                hit = np.nonzero(e <= 0.5 * e[0])[0]
                vals.append(data["steps"][hit[0]] if hit.size else np.nan)
            if vals:
                med = np.nanmedian(vals) if np.isfinite(vals).any() else np.nan
                x = i + (k - 0.5) * width
                if np.isfinite(med):
                    ax.bar(x, max(med, 1), width=width * 0.9, color=COLORS[fam], alpha=1.0 if k else 0.45)
                else:
                    ax.text(x, 0.02, "never", rotation=90, ha="center", va="bottom", fontsize=7, color=MUTED,
                            transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xticks(range(len(FAMILIES)), [LABELS[f] for f in FAMILIES], rotation=35, ha="right")
    ax.set_ylabel("step error first halves (log)")
    ax.set_title(f"{PROBLEM_LABELS[problem]}: when each frequency is learned (pale = pi, solid = 15 pi)")
    ax.grid(axis="x", visible=False)
    save(fig, f"xai_frequency_learning_order_{problem}")


def ntk_vs_fisher(runs) -> None:
    """New: two measures of 'how many directions the model uses', compared per run."""
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for fam in FAMILIES:
        xs, ys = [], []
        for r in select(runs, family=fam):
            step = _final_step(r)
            ntk = r.xai(f"ntk_step{step}") if step is not None else None
            fisher = r.xai(f"fisher_step{step}") if step is not None else None
            if ntk is not None and fisher is not None:
                xs.append(float(ntk["effective_rank"]))
                ys.append(float(fisher["effective_dimension"]))
        if xs:
            ax.scatter(xs, ys, color=COLORS[fam], marker=MARKERS[fam], s=26, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("NTK effective rank (final)")
    ax.set_ylabel("Fisher effective dimension (final)")
    ax.set_title("Two views of model capacity, every run")
    family_legend(ax, FAMILIES, loc="best", fontsize=7)
    save(fig, "xai_ntk_vs_fisher")


def quantum_internals(runs, problems=("poisson", "heat")) -> None:
    """Encoder drift, coverage and layer probes for the quantum families."""
    fams = [f for f in FAMILIES if f in QUANTUM]
    fig, axes = plt.subplots(2, len(problems), figsize=(9, 6), sharex=True)
    for j, problem in enumerate(problems):
        for fam in fams:
            rs = select(runs, problem=problem, family=fam)
            if not rs:
                continue
            steps = rs[0].xai_steps("drift")
            for i, key in enumerate(("frobenius", "coverage_now")):
                vals = []
                for r in rs:
                    d = [r.xai(f"drift_step{s}") for s in steps]
                    if all(x is not None for x in d):
                        vals.append([float(x[key]) for x in d])
                if vals:
                    axes[i, j].plot(np.maximum(steps, 1), np.median(vals, axis=0), color=COLORS[fam],
                                    marker=MARKERS[fam], markersize=4, linewidth=1.4)
        axes[0, j].set_title(PROBLEM_LABELS[problem])
        axes[0, j].axhline(0.2, color=MUTED, linestyle=":", linewidth=1)
        for i in range(2):
            axes[i, j].set_xscale("log")
    axes[0, 0].set_ylabel("encoder drift ||A - I||_F (dotted: limit 0.2)")
    axes[1, 0].set_ylabel("SMCD coverage of the target spectrum")
    for ax in axes[1]:
        ax.set_xlabel("training step (log)")
    family_legend(axes[0, -1], fams, loc="best", fontsize=7)
    fig.tight_layout()
    save(fig, "xai_quantum_internals")


def gradient_variance(runs) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.8 / len(PROBLEMS)
    for k, problem in enumerate(PROBLEMS):
        for i, fam in enumerate(FAMILIES):
            vals = []
            for r in select(runs, problem=problem, family=fam):
                steps = r.xai_steps("gradvar")
                d = r.xai(f"gradvar_step{steps[-1]}") if steps else None
                if d is not None:
                    vals.append(float(d["var_mean"]))
            if vals:
                ax.scatter(i + (k - 2.5) * width, np.median(vals), color=COLORS[fam], marker=MARKERS[fam], s=22)
    ax.set_yscale("log")
    ax.set_xticks(range(len(FAMILIES)), [LABELS[f] for f in FAMILIES], rotation=30, ha="right")
    ax.set_ylabel("mean gradient variance (log)")
    ax.set_title("Gradient variance at the end of training; within each family, problems left to right: "
                 + ", ".join(PROBLEM_LABELS[p] for p in PROBLEMS), fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "xai_gradient_variance")


def main() -> None:
    style()
    runs = core_runs()
    ntk_spectra(runs)
    instrument_evolution(runs)
    for problem in ("poisson", "heat"):
        landscapes_and_sharpness(runs, problem)
        frequency_learning_order(runs, problem)
    spectral_error_heatmaps(runs)
    ntk_vs_fisher(runs)
    quantum_internals(runs)
    gradient_variance(runs)


if __name__ == "__main__":
    main()
