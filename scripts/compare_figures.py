"""Compare the figures in paper/figures/ against the committed versions (git HEAD).

Run after `tasks.py figures`: uv run python scripts/compare_figures.py [--report]

On one machine regeneration is byte-for-byte identical. Across operating systems and CPUs
it is not: anti-aliasing and float rounding in the rasterizer differ, so every PNG (and
every PDF with rasterized content) comes out slightly different on, say, Linux CI than on
the Windows machine that produced the committed files. So each changed PNG is decoded and
compared with a tolerance instead: the size may differ by up to MAX_SHIFT pixels (tight
bounding boxes round differently), and after the best alignment at most
MAX_CHANGED_FRACTION of pixels may change by more than PIXEL_THRESHOLD. A figure that
regenerates with different content, such as a moved band or other data, changes far more
than that. PDFs are only reported; every PDF is drawn from the same figure as its PNG.
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib import image as mpl_image

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = "paper/figures"
MAX_SHIFT = 2
PIXEL_THRESHOLD = 0.25  # per-channel difference, on a 0-1 scale
MAX_CHANGED_FRACTION = 0.02


def _committed_bytes(path: str) -> bytes:
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, capture_output=True, check=True).stdout


def _rgb(data: bytes) -> np.ndarray:
    img = mpl_image.imread(io.BytesIO(data), format="png").astype(np.float64)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:  # composite onto white
        img = img[..., :3] * img[..., 3:] + (1.0 - img[..., 3:])
    return img[..., :3]


def changed_fraction(a: np.ndarray, b: np.ndarray, max_shift: int = MAX_SHIFT) -> float:
    """Smallest fraction of changed pixels over every alignment within max_shift pixels."""
    best = 1.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            a_part = a[max(dy, 0):, max(dx, 0):]
            b_part = b[max(-dy, 0):, max(-dx, 0):]
            h = min(a_part.shape[0], b_part.shape[0])
            w = min(a_part.shape[1], b_part.shape[1])
            if h == 0 or w == 0:
                continue
            diff = np.abs(a_part[:h, :w] - b_part[:h, :w]).max(axis=-1)
            best = min(best, float((diff > PIXEL_THRESHOLD).mean()))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="print the comparison but never fail")
    args = parser.parse_args()

    tracked = subprocess.run(
        ["git", "ls-files", FIGURES], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    failures = []
    for path in sorted(tracked):
        current = (REPO_ROOT / path).read_bytes()
        committed = _committed_bytes(path)
        if current == committed:
            continue
        if path.endswith(".pdf"):
            print(f"  differs    {path} (not decoded; its PNG is checked)")
            continue
        a, b = _rgb(current), _rgb(committed)
        dh, dw = a.shape[0] - b.shape[0], a.shape[1] - b.shape[1]
        if abs(dh) > MAX_SHIFT or abs(dw) > MAX_SHIFT:
            failures.append(path)
            print(f"  FAIL       {path}: size {a.shape[1]}x{a.shape[0]} vs committed {b.shape[1]}x{b.shape[0]}")
            continue
        frac = changed_fraction(a, b)
        ok = frac <= MAX_CHANGED_FRACTION
        if not ok:
            failures.append(path)
        print(f"  {'ok' if ok else 'FAIL':10s} {path}: {frac:.2%} of pixels changed (limit {MAX_CHANGED_FRACTION:.0%})")

    n_png = sum(p.endswith(".png") for p in tracked)
    print(f"{n_png} PNGs checked, {len(failures)} outside tolerance")
    return 0 if args.report or not failures else 1


if __name__ == "__main__":
    sys.exit(main())
