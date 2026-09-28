"""Groundwater what-if scenarios from the exact closed-form solution (no training).

The practical question: how much of the field is waterlogged (water table within 1.5 m
of a land surface at 9 m, i.e. above 7.5 m), and what would reduce it? Scenarios vary
one scenario value at a time around configs/pde/groundwater.yaml: lining the canals
(less seepage), canal spacing at the same total seepage, soil type (transmissivity),
and lining against soil together.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from qapinn.config import load_config
from qapinn.pdes.groundwater import Groundwater, waterlogged_length_m
from report.common import COLORS, MUTED, REPO_ROOT, save, style

SURFACE_M = 9.0
THRESHOLD_M = SURFACE_M - 1.5
X = np.linspace(0.0, 2000.0, 8001)
WATER, LAND, RISK = "#0072B2", "#8a6d3b", "#c62828"


def _base_params() -> dict:
    return dict(load_config("groundwater", "c_mlp").pde.params)


def _solve(**changes) -> tuple[np.ndarray, float, float]:
    params = {**_base_params(), **changes}
    pde = Groundwater(**params)
    h = pde.head_m(X * params["length_m"] / 2000.0)
    return h, waterlogged_length_m(X, h, THRESHOLD_M), float(h.max())


def profile() -> dict:
    base = _base_params()
    pde = Groundwater(**base)
    h, wet, peak = _solve()
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.fill_between(X, h, 0, color=WATER, alpha=0.18, linewidth=0)
    ax.plot(X, h, color=WATER, label="water table (exact)")
    ax.axhline(SURFACE_M, color=LAND, linewidth=2, label="land surface")
    ax.axhline(THRESHOLD_M, color=RISK, linewidth=1.2, linestyle="--", label="waterlogging threshold (1.5 m below surface)")
    ax.fill_between(X, THRESHOLD_M, np.maximum(h, THRESHOLD_M), where=h > THRESHOLD_M, color=RISK, alpha=0.35,
                    linewidth=0)
    for c in pde.canal_centres_m():
        ax.axvline(c, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylabel("height above river datum (m)")
    ax.set_ylim(0, SURFACE_M + 0.8)
    ax.legend(loc="lower center", fontsize=7, ncol=2)
    ax.set_title(f"Water table between two rivers: {wet:.0f} m of the 2 km field is waterlogged, peak {peak:.2f} m",
                 fontsize=10)
    axr.plot(X, pde.recharge_m_per_day(X) * 1000, color=COLORS["c_ff"])
    axr.set_ylabel("recharge\n(mm/day)")
    axr.set_xlabel("distance from the upstream river (m); dotted lines = canals")
    fig.tight_layout()
    save(fig, "groundwater_profile")
    return {"waterlogged_m": wet, "peak_m": peak}


def canal_lining() -> dict:
    base = _base_params()["canal_seepage"]
    cuts = np.linspace(0, 0.9, 19)
    wet, peak = [], []
    for cut in cuts:
        _, w, p = _solve(canal_seepage=base * (1 - cut))
        wet.append(w)
        peak.append(p)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(cuts * 100, wet, color=RISK, marker="o", markersize=4)
    axes[0].set_ylabel("waterlogged length (m)")
    axes[1].plot(cuts * 100, peak, color=WATER, marker="o", markersize=4)
    axes[1].axhline(THRESHOLD_M, color=RISK, linestyle="--", linewidth=1)
    axes[1].set_ylabel("peak water table (m)")
    for ax in axes:
        ax.set_xlabel("seepage cut by lining the canals (%)")
    first_dry = next((c for c, w in zip(cuts, wet) if w == 0), None)
    note = f"field fully dry from a {first_dry * 100:.0f}% cut" if first_dry is not None else "never fully dry"
    fig.suptitle(f"What lining the canals buys: {note}", fontsize=10)
    fig.tight_layout()
    save(fig, "groundwater_canal_lining")
    return {"seepage_cut": cuts.tolist(), "waterlogged_m": wet, "peak_m": peak, "first_dry_cut": first_dry}


def canal_spacing() -> dict:
    base = _base_params()
    total = base["canal_seepage"] * base["n_canals"]
    counts = list(range(2, 13))
    wet, ripple = [], []
    for n in counts:
        h, w, _ = _solve(n_canals=float(n), canal_seepage=total / n)
        wet.append(w)
        smooth = np.polyval(np.polyfit(X, h, 2), X)
        ripple.append(float(np.abs(h - smooth).max()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(counts, wet, color=RISK, marker="o", markersize=5)
    axes[0].set_ylabel("waterlogged length (m)")
    axes[1].plot(counts, np.array(ripple) * 100, color=WATER, marker="o", markersize=5)
    axes[1].set_ylabel("ripple over the canals (cm)")
    for ax in axes:
        ax.set_xlabel("number of canals (same total seepage)")
    fig.suptitle("Fewer, larger canals make taller mounds under each canal; the total load sets the rest",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "groundwater_canal_spacing")
    return {"n_canals": counts, "waterlogged_m": wet, "ripple_m": ripple}


def soil_type() -> dict:
    soils = {"clay loam": 25.0, "silt": 50.0, "silty sand (base)": 100.0, "fine sand": 200.0, "medium sand": 400.0}
    rows = {name: _solve(transmissivity=t)[1:] for name, t in soils.items()}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    names = list(rows)
    ax.barh(range(len(names)), [rows[n][0] for n in names], color=RISK, height=0.55)
    for i, n in enumerate(names):
        ax.annotate(f"peak {rows[n][1]:.1f} m", (rows[n][0], i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(names)), [f"{n} (T={soils[n]:.0f} m²/day)" for n in names])
    ax.set_xlabel("waterlogged length (m)")
    ax.set_title("Soil decides how far water spreads: tighter soils waterlog more of the field", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "groundwater_soil_type")
    return {n: {"transmissivity": soils[n], "waterlogged_m": rows[n][0], "peak_m": rows[n][1]} for n in names}


def lining_vs_soil() -> dict:
    base = _base_params()["canal_seepage"]
    cuts = np.linspace(0, 0.9, 10)
    ts = np.array([25, 50, 75, 100, 150, 200, 300, 400], dtype=float)
    grid = np.array([[_solve(canal_seepage=base * (1 - c), transmissivity=t)[1] for c in cuts] for t in ts])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(grid, cmap="Reds", aspect="auto", origin="lower", vmin=0)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] > 0.6 * grid.max() else "#1f1f1f")
    ax.set_xticks(range(len(cuts)), [f"{c * 100:.0f}%" for c in cuts])
    ax.set_yticks(range(len(ts)), [f"{t:.0f}" for t in ts])
    ax.set_xlabel("seepage cut by lining")
    ax.set_ylabel("transmissivity (m²/day)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="waterlogged length (m)")
    ax.set_title("Planning map: waterlogged metres for each soil and lining level", fontsize=9)
    save(fig, "groundwater_planning_map")
    return {"seepage_cut": cuts.tolist(), "transmissivity": ts.tolist(), "waterlogged_m": grid.tolist()}


def model_profiles(model_set: str = "v1") -> None:
    """Each trained model's predicted water table (median over seeds 10-14) against the
    exact one, for the pre-registered groundwater runs."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from adjudicate_groundwater import MODEL_SETS, runs_by_family

    from report.common import LABELS, MARKERS

    spec = MODEL_SETS[model_set]
    runs = runs_by_family(spec["experiments"])
    if not runs:
        return
    pde = Groundwater(**_base_params())
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(X, pde.head_m(X), color="#1f1f1f", linewidth=2.6, label="exact")
    order = ["c_mlp", spec["matched_mlp"], spec["rff"], spec["q_random"], spec["q"]]
    for name in order:
        dirs = runs.get(name, [])
        if not dirs:
            continue
        preds = []
        for d in dirs:
            with np.load(d / "prediction.npz") as p:
                idx = np.argsort(p["x"])
                preds.append(np.interp(X, p["x"][idx] * 2000.0, p["u"][idx]))
        base = name.removesuffix("_v2").replace("_matched_v2", "_matched")
        fam = base if base in COLORS else ("c_mlp" if base.startswith("c_mlp") else base)
        label = {"c_mlp_matched": "MLP, same size as circuit"}.get(base, LABELS.get(base, name))
        ax.plot(X, np.median(preds, axis=0), color=COLORS[fam], linewidth=1.8, marker=MARKERS[fam],
                markevery=800, markersize=5, linestyle="--" if base == "c_mlp_matched" else "-", label=label)
    ax.axhline(THRESHOLD_M, color=RISK, linewidth=1.2, linestyle=":", label="waterlogging threshold")
    ax.set_xlabel("distance from the upstream river (m)")
    ax.set_ylabel("water table (m)")
    ax.set_title(f"Trained models vs the exact water table ({model_set} models, median of 5 fresh seeds)")
    ax.legend(loc="lower center", fontsize=7, ncol=3)
    save(fig, f"groundwater_models_{model_set}")


def main() -> None:
    style()
    for model_set in ("v1", "v2"):
        model_profiles(model_set)
    summary = {
        "threshold_m": THRESHOLD_M,
        "base": profile(),
        "canal_lining": canal_lining(),
        "canal_spacing": canal_spacing(),
        "soil_type": soil_type(),
        "lining_vs_soil": lining_vs_soil(),
    }
    out = REPO_ROOT / "results" / "report" / "groundwater_scenarios.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
