# QAPINN-Spectra

**Spectrum-Matched Circuit Design (SMCD):** given a PDE, compute its target spectral
support analytically from the operator's Fourier symbol, then construct a
data-re-uploading quantum circuit — encoding-generator eigenvalues, re-uploading depth,
qubit count, entangler pattern — so its realized frequency set covers that support,
deterministically, with no architecture search. Six explainable-AI instruments, run
identically for matched classical and hybrid models, attribute any accuracy delta to the
quantum layer specifically rather than reporting it as an aggregate number. Full
methodology: `paper/main.pdf`.

## Headline result

At the sample size this run achieved (**n=2 seeds**, cut from the pre-registered n=5
under a hard submission deadline — see `FINDINGS.md` and `docs/plan/08_TASK_INDEX.md`'s
cut-line), **the quantum-advantage hypothesis is not supported**: on Poisson, the one
problem with a full four-family ablation, the hybrid model underperforms every classical
comparator tested. What does hold up: the smoothing-operator negative control mostly
matches its a-priori prediction, one genuine per-frequency win exists (Poisson only),
and — most solidly — the pre-registered, mechanically-adjudicated measurement protocol
itself worked end-to-end on all 12 predictions with no result silently dropped or
reinterpreted. See `FINDINGS.md` for the full, numbered breakdown, including three real
bugs the adjudication pipeline caught in its own supporting code before they could
silently produce a wrong verdict.

![Coverage vs. error, per problem, with Spearman rho annotated](paper/figures/coverage_vs_error.png)

Poisson confirms the predicted negative correlation (rho=-0.82); Helmholtz_k10 does not
(rho=+0.28, wrong sign) — the headline figure ships exactly as measured either way.

## Repository layout

```
src/qapinn/          PDEs, models (classical + hybrid quantum), training loop, XAI instruments
configs/exp/          Experiment sweep definitions (configs/exp/*.yaml)
scripts/              Figure generation, prediction adjudication, cost ledger
docs/predictions.md   Pre-registered falsifiable predictions (T3.1), committed before any run
docs/plan/            Full phase-by-phase execution plan and task index
FINDINGS.md           Numbered findings, C1-C5 adjudication table, decision table
docs/self_review.md   Adversarial self-review (T4.8)
paper/                LaTeX paper (paper/main.tex -> paper/main.pdf)
slides/               Marp slides (slides/main.md)
results/              Committed run artifacts (metrics.json, xai/*.npz) figures regenerate from
tests/                pytest suite
```

## Reproducing

```
uv sync                        # pin the environment (Python 3.12, torch cu128, PennyLane)
python tasks.py test            # run the test suite
python tasks.py figures         # regenerate every figure from committed results/runs/ data
python tasks.py paper           # build paper/main.pdf via tectonic
python tasks.py repro-quick     # ~15 min, CPU-only, reduced-fidelity reproduction
```

No figure in `paper/figures/` is generated from a live model reload — every one comes
from `results/runs/*/metrics.json` and `xai/*.npz` only (`scripts/make_figures.py`).

## Honesty notes

This project runs on a strict pre-registration discipline: `docs/predictions.md` was
committed before any Phase 3 run, and every claim in the paper is adjudicated against
those pre-registered falsifiers (`scripts/adjudicate_predictions.py`), not fitted to the
data after the fact. Every scope cut made under deadline pressure is documented, with
its reasoning, in `docs/plan/08_TASK_INDEX.md` and disclosed again in `FINDINGS.md` —
including the seed-count reduction (n=5 planned, n=2 shipped) that structurally limits
which verdicts this run's statistics can support (`FINDINGS.md`, finding F10).
