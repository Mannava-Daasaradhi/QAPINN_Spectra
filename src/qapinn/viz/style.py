"""Single matplotlib style used by every figure in the project -- paper and slides share
figures, so they must share style (T1.1).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Fixed family -> colour map (colourblind-safe, Wong 2011 palette). Never let matplotlib
# pick: every figure comparing families must use these exact colours for comparability.
FAMILY_COLORS = {
    "c_mlp": "#7f7f7f",  # grey
    "c_ff": "#0072B2",  # blue
    "c_rff_matched": "#009E73",  # green
    "q_serial": "#D55E00",  # red/vermillion
    "q_parallel": "#E69F00",  # orange
    "q_random": "#CC79A7",  # purple/pink
    "q_octave": "#8c564b",  # brown
}

FIGSIZE = {
    "single": (3.5, 2.8),  # one-column paper figure, inches
    "double": (7.0, 2.8),  # two-column paper figure
    "slide": (10.0, 6.0),  # 16:9-ish slide figure
}

FIGURES_DIR = Path("paper/figures")
MANIFEST_PATH = Path("results/manifest.json")


def apply_style() -> None:
    """Applies the project-wide matplotlib style. Also applied once at import time."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.prop_cycle": matplotlib.cycler(color=list(FAMILY_COLORS.values())),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "legend.frameon": False,
            "axes.grid": False,
        }
    )


def family_color(family: str) -> str:
    try:
        return FAMILY_COLORS[family]
    except KeyError as e:
        raise ValueError(f"unknown family {family!r}; known families: {sorted(FAMILY_COLORS)}") from e


def save_figure(fig, name: str, run_ids: list[str] | None = None) -> dict[str, Path]:
    """Writes <name>.pdf (paper) and <name>.png (slides, 200 dpi) into paper/figures/, and
    appends an entry to results/manifest.json mapping figure name -> the run_ids used.

    Output is byte-reproducible: the PDF carries no CreationDate (matplotlib otherwise
    stamps the wall-clock time, so every regeneration rewrote every committed PDF), and
    the tight bounding box keeps axis labels that sit outside the axes from being clipped.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.02)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    if MANIFEST_PATH.is_file():
        with MANIFEST_PATH.open(encoding="utf-8") as f:
            manifest = json.load(f)
    manifest[name] = {"run_ids": list(run_ids or [])}
    # newline="\n": the same bytes on every OS (text mode would write CRLF on Windows).
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return {"pdf": pdf_path, "png": png_path}


apply_style()
