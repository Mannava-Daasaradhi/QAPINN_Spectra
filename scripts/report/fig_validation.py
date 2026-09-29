"""Validation figures: how robust the verdicts and the tools behind them are.

1. Bootstrap 95% intervals for every pre-registered error ratio against its threshold.
2. The pre-registered Wilcoxon test next to three alternatives (sign test, paired t-test
   on log error, Mann-Whitney U).
3. Leave-one-seed-out: does any single seed decide whether a ratio clears its threshold?
4. Circuit spectrum: every problem's SMCD circuit at random parameters has no Fourier
   energy outside its designed set Omega.
5. Reference solver: finite-difference error on the groundwater problem shrinks at the
   expected second order.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from report.common import (
    COLORS,
    MUTED,
    PROBLEM_LABELS,
    REPO_ROOT,
    core_runs,
    save,
    select,
    style,
)

# (label, problem, family under test, baseline, threshold, kind). kind "beats": baseline /
# tested >= threshold; "band": tested / baseline within [1/threshold, threshold].
COMPARISONS = [
    ("vs MLP, Poisson", "poisson", "q_serial", "c_mlp", 2.0, "beats"),
    ("vs Fourier features, Poisson", "poisson", "q_serial", "c_ff", 1.3, "beats"),
    ("vs frequency-matched, Poisson", "poisson", "q_serial", "c_rff_matched", 1.3, "band"),
    ("designed vs random, Poisson", "poisson", "q_serial", "q_random", 1.5, "beats"),
    ("designed vs random, heat", "heat", "q_serial", "q_random", 1.5, "beats"),
]


def _paired(runs, problem, fam_a, fam_b):
    a = {r.seed: r.metrics["rel_l2"] for r in select(runs, problem=problem, family=fam_a)}
    b = {r.seed: r.metrics["rel_l2"] for r in select(runs, problem=problem, family=fam_b)}
    seeds = sorted(set(a) & set(b))
    return np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds])


def _ratio(tested, baseline, kind):
    return np.median(baseline) / np.median(tested) if kind == "beats" else np.median(tested) / np.median(baseline)


def bootstrap_forest(runs, n_boot=5000) -> dict:
    rng = np.random.default_rng(0)
    rows = []
    for label, problem, fam, base, thr, kind in COMPARISONS:
        t, b = _paired(runs, problem, fam, base)
        if len(t) < 2:
            continue
        idx = rng.integers(0, len(t), size=(n_boot, len(t)))
        boots = np.array([_ratio(t[i], b[i], kind) for i in idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append((label, problem, _ratio(t, b, kind), lo, hi, thr, kind, len(t)))
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rows) + 1.2))
    for i, (label, problem, point, lo, hi, thr, kind, n) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=MUTED, linewidth=2, solid_capstyle="round")
        ax.scatter(point, i, color=COLORS["q_serial"], s=40, zorder=3)
        if kind == "beats":
            ax.plot([thr, thr], [i - 0.35, i + 0.35], color="#1f1f1f", linewidth=1.5)
        else:
            ax.plot([1 / thr, 1 / thr], [i - 0.35, i + 0.35], color="#1f1f1f", linewidth=1.5)
            ax.plot([thr, thr], [i - 0.35, i + 0.35], color="#1f1f1f", linewidth=1.5)
        ax.annotate(f"{point:.2f} [{lo:.2f}, {hi:.2f}], n={n}", (max(hi, point), i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.axvline(1.0, color=MUTED, linestyle=":", linewidth=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("error ratio (right of 1 = circuit better; frequency-matched: circuit / baseline)")
    ax.set_title("Bootstrap 95% intervals (dot = median ratio, black ticks = pre-registered threshold)", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "validation_bootstrap_ratios")
    return {r[0]: {"ratio": r[2], "ci95": [r[3], r[4]], "threshold": r[5], "n": r[7]} for r in rows}


def alternative_tests(runs) -> dict:
    names = ["Wilcoxon (pre-registered)", "sign test", "paired t (log error)", "Mann-Whitney U"]
    table = {}
    for label, problem, fam, base, _thr, _kind in COMPARISONS:
        t, b = _paired(runs, problem, fam, base)
        if len(t) < 2:
            continue
        d = np.log(b) - np.log(t)
        wins = int((d > 0).sum())
        table[label] = [
            stats.wilcoxon(t, b).pvalue,
            stats.binomtest(wins, len(d)).pvalue,
            stats.ttest_rel(np.log(t), np.log(b)).pvalue,
            stats.mannwhitneyu(t, b).pvalue,
        ]
    labels = list(table)
    grid = np.array([table[k] for k in labels])
    fig, ax = plt.subplots(figsize=(7.4, 0.5 * len(labels) + 1.4))
    ax.imshow(-np.log10(grid), cmap="Blues", aspect="auto", vmin=0, vmax=3)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"p={grid[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if -np.log10(grid[i, j]) > 1.8 else "#1f1f1f")
    ax.set_xticks(range(len(names)), names, rotation=15, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    ax.set_title("Same comparisons, four tests (darker = smaller p). With 5 seeds the smallest two-sided\n"
                 "Wilcoxon p is 0.0625, so it can never go below that however large the effect.", fontsize=8)
    save(fig, "validation_alternative_tests")
    return {k: dict(zip(names, v)) for k, v in table.items()}


def leave_one_seed_out(runs) -> dict:
    out = {}
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(COMPARISONS) + 1.2))
    for i, (label, problem, fam, base, thr, kind) in enumerate(COMPARISONS):
        t, b = _paired(runs, problem, fam, base)
        if len(t) < 3:
            continue
        ratios = [_ratio(np.delete(t, k), np.delete(b, k), kind) for k in range(len(t))]
        passes = [(r >= thr) if kind == "beats" else (1 / thr <= r <= thr) for r in ratios]
        out[label] = {"ratios": ratios, "passes": int(sum(passes)), "of": len(ratios)}
        ax.scatter(ratios, [i] * len(ratios), color=[COLORS["q_serial"] if p else "white" for p in passes],
                   edgecolor=COLORS["q_serial"], s=36, zorder=3)
        ax.annotate(f"clears threshold in {sum(passes)}/{len(passes)}", (max(ratios), i), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=7, color=MUTED)
        for edge in (thr,) if kind == "beats" else (1 / thr, thr):
            ax.plot([edge, edge], [i - 0.35, i + 0.35], color="#1f1f1f", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yticks(range(len(COMPARISONS)), [c[0] for c in COMPARISONS])
    ax.invert_yaxis()
    ax.set_xlabel("median error ratio with one seed left out (filled = clears the threshold)")
    ax.set_title("Leave-one-seed-out: no single seed should decide a verdict", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "validation_leave_one_seed_out")
    return out


def circuit_spectrum_containment(n_draws=40, n=4096) -> dict:
    """Every problem's designed circuit, random angles: energy outside Omega."""
    import torch

    from qapinn.config import load_config
    from qapinn.models.circuits import ReuploadCircuit
    from qapinn.pdes import build as build_pde
    from qapinn.runner import _PROBLEM_BY_LABEL
    from qapinn.smcd.design import smcd

    torch.manual_seed(0)
    results = {}
    for label in ("poisson", "heat", "burgers", "helmholtz_k4", "helmholtz_k10", "helmholtz_k20", "groundwater"):
        pde_yaml, overrides = _PROBLEM_BY_LABEL[label]
        pde = build_pde(load_config(pde_yaml, "c_mlp", overrides=overrides).pde)
        card = smcd(pde)
        circuit = ReuploadCircuit(card.n_qubits, card.n_layers, card.scalings, tuple(card.wire_to_dim),
                                  card.entangler, card.observable).double()
        omega = circuit.frequencies()
        if omega.ndim > 1 and omega.shape[1] > 1:
            continue  # 2-D designs: the tested 1-D FFT argument does not apply directly
        omega = np.abs(omega.reshape(-1))
        base = np.min(omega[omega > 1e-9]) if np.any(omega > 1e-9) else 1.0
        period = 2 * np.pi / base
        z = torch.arange(n, dtype=torch.float64).unsqueeze(-1) / n * period
        if circuit.wire_to_dim and max(circuit.wire_to_dim) > 0:
            continue
        worst = 0.0
        for _ in range(n_draws):
            with torch.no_grad():
                circuit.theta.normal_(0.0, 1.5)
                y = circuit(z).squeeze(-1).numpy()
            spec = np.abs(np.fft.rfft(y)) / n
            bins = {int(np.rint(w / base)) for w in omega}
            outside = [spec[k] for k in range(len(spec)) if k not in bins]
            worst = max(worst, max(outside) / max(spec.max(), 1e-30))
        results[label] = {"omega_size": len(set(np.round(omega, 9))), "worst_outside_relative": worst}
    fig, ax = plt.subplots(figsize=(6.4, 3))
    names = list(results)
    vals = [max(results[k]["worst_outside_relative"], 1e-18) for k in names]
    ax.barh(range(len(names)), vals, color=COLORS["q_serial"], height=0.55)
    ax.set_xscale("log")
    ax.axvline(1e-8, color="#1f1f1f", linewidth=1.2)
    ax.set_yticks(range(len(names)), [PROBLEM_LABELS.get(k, k) for k in names])
    ax.set_xlabel("largest Fourier amplitude outside Omega / largest overall (40 random parameter draws)")
    ax.set_title("Designed circuits emit nothing outside their frequency set (line = 1e-8)", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "validation_circuit_spectrum")
    return results


def reference_convergence() -> dict:
    """Finite-difference error against the closed form, groundwater, as the grid refines."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    from qapinn.config import load_config
    from qapinn.pdes import build as build_pde

    pde = build_pde(load_config("groundwater", "c_mlp").pde)
    L = pde.params["length_m"]
    sizes = [101, 201, 401, 801, 1601, 3201, 6401]
    errs = []
    for n in sizes:
        X = np.linspace(0, L, n)
        dx = X[1] - X[0]
        A = sp.diags([np.ones(n - 3), -2 * np.ones(n - 2), np.ones(n - 3)], [-1, 0, 1]) / dx**2
        rhs = -pde.recharge_m_per_day(X[1:-1]) / pde.params["transmissivity"]
        rhs[0] -= pde.params["head_left"] / dx**2
        rhs[-1] -= pde.params["head_right"] / dx**2
        h = spl.spsolve(A.tocsc(), rhs)
        errs.append(float(np.abs(h - pde.head_m(X[1:-1])).max()))
    slope = float(np.polyfit(np.log(np.array(sizes) - 1), np.log(errs), 1)[0])
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.loglog(np.array(sizes) - 1, errs, marker="o", color=COLORS["c_ff"])
    ax.set_xlabel("grid intervals")
    ax.set_ylabel("max |finite difference - exact| (m)")
    ax.set_title(f"Groundwater reference: error falls at order {-slope:.2f} (expected 2)", fontsize=9)
    save(fig, "validation_reference_convergence")
    return {"sizes": sizes, "max_error_m": errs, "order": -slope}


def main() -> None:
    style()
    runs = core_runs()
    summary = {
        "bootstrap": bootstrap_forest(runs),
        "alternative_tests": alternative_tests(runs),
        "leave_one_seed_out": leave_one_seed_out(runs),
        "circuit_spectrum": circuit_spectrum_containment(),
        "reference_convergence": reference_convergence(),
    }
    out = REPO_ROOT / "results" / "report" / "validation.json"
    out.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
