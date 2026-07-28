"""Figure generation entry points. T1.3 starts this file with `ntk_spectrum`;
T5.1 fills in the remaining ~11 figures and a `--regenerate-all` driver.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np

from qapinn.models.base import PINNModel
from qapinn.pdes.base import PDE
from qapinn.viz.style import family_color, save_figure
from qapinn.xai.ntk import ntk, probe_set, spectrum_stats


def make_ntk_spectrum_figure(
    models: dict[str, PINNModel],
    pde: PDE,
    output: str = "residual",
    run_ids: list[str] | None = None,
    omega_band: tuple[float, float] | None = None,
) -> dict[str, float]:
    """models: {family_name: model}. Renders log-log eigenvalue-vs-index for each family
    on the same axes. omega_band: (lo, hi) index range to shade as the encoded band Omega
    -- Phase 2 supplies the real value; this is the shading hook called for in T1.3.

    Returns {family_name: decay_exponent}.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    decay_exponents: dict[str, float] = {}
    for family, model in models.items():
        probe_x = probe_set(pde)
        K = ntk(model, pde, probe_x, output=output, group="all")
        stats = spectrum_stats(K)
        eigs = np.array(stats["eigenvalues"])
        eigs_positive = eigs[eigs > 0]
        idx = np.arange(1, len(eigs_positive) + 1)
        ax.loglog(idx, eigs_positive, label=family, color=family_color(family))
        decay_exponents[family] = stats["decay_exponent"]

    if omega_band is not None:
        ax.axvspan(omega_band[0], omega_band[1], color="grey", alpha=0.2, label=r"encoded band $\Omega$")

    ax.set_xlabel("index $i$")
    ax.set_ylabel(r"eigenvalue $\lambda_i$")
    ax.legend()
    fig.tight_layout()

    save_figure(fig, "ntk_spectrum", run_ids=run_ids)
    plt.close(fig)
    return decay_exponents


if __name__ == "__main__":
    print("scripts/make_figures.py: no CLI driver yet (arrives in T5.1); import and call directly.")
