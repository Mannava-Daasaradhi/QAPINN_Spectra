# Changelog

## Unreleased

Fixes found by the first CI runs on GitHub's Linux runners. No result changed.

- `results/adjudication.json`: all 15 verdicts and their numbers as a committed,
  machine-readable record (`python scripts/adjudicate_predictions.py --write`). Floats
  are rounded to 10 significant digits so the record compares exactly across machines.
  CI regenerates it and requires an exact match.
- Figures are byte-identical only on the machine that drew them. Linux rasterizes all 20
  PNGs, and the 7 PDFs with raster content (heatmaps, loss landscape), slightly
  differently from Windows (anti-aliasing and float rounding). The other PDFs and all
  JSON and CSV outputs match exactly. `scripts/compare_figures.py` checks each PNG
  against the committed one with a pixel tolerance: size within 2 px, and at most 2% of
  pixels changed by more than a quarter of the colour range after the best alignment. On
  the same machine it reports every figure identical. The submitted, wrong-band Poisson
  heatmap fails it: different size, and 61.5% of pixels changed even when cropped to
  match.
- CI: `astral-sh/setup-uv` is pinned to an exact version (no floating `@v10` tag
  exists), and the test job checks out full history (the pre-registration check verifies
  commit ancestry, which a depth-1 clone lacks). Regenerated figures are uploaded as a
  build artifact.
- `tasks.py repro-all` refreshes `results/adjudication.json`.
- The paper's abstract now states the results. It described only the method, and said
  the XAI instruments "prove the mechanism", which the adjudicated results do not
  support.

## 1.1.0 — 2026-09-25 (post-submission)

Changes after the WISER 2026 BQP Challenge. The judged version is tag `v1.0-submission`
(`d20e961`). None of the 15 pre-registered verdicts changed, and no experiment was re-run.

### Corrected (see FINDINGS.md, "Post-submission errata")

- **PR-8, Poisson: 99.9% of the improvement lies inside Ω, not 71.3%.** The overlap code
  folded the symmetric frequency set Ω onto |ω| while Poisson's error spectrum is
  two-sided, so the mirrored half of Ω counted as outside. Verdict unchanged (CONFIRMED).
- **PR-7, Helmholtz k=10: the decay-exponent gap is −8.37, not −9.67.** The old number
  predates PR-7's switch to committed data (`16ff015`). Verdict unchanged (REFUTED).
- **Paper figures marked the wrong band.** `tasks.py figures` shaded the design card's four
  encoding scalings as Ω instead of the realised frequency set in every `freq_heatmap_*`
  and in `ntk_spectrum_p1/p4`. Adjudication always used the correct Ω.
- Caveats added to F1 (coverage sweep: 9 distinct measurements, reduced budget, ordering
  reverses at full budget), F5/F13 (Helmholtz k=10/20 "parity" is between models that all
  score rel-L2 ≈ 1) and F2 (classical baselines were not size-matched).

### Fixed

- Figure regeneration is deterministic. PDFs no longer embed a timestamp, and the
  adjudicator and `tasks.py figures` draw identical figures from one Ω source, so on a
  given machine running either leaves the working tree clean. (Across operating systems
  raster pixels still differ slightly; see Unreleased.)
- The test suite no longer writes into the committed `results/`. `test_runner.py` and
  one XAI test trained into `results/runs/`, and one of them deleted a committed smoke run
  that a later test recreated on CUDA. That is the "determinism violation" noted during
  T4.9. It was CPU-vs-CUDA rounding (1e-15 in rel-L2 on the run re-checked), not
  same-device nondeterminism.
  `test_size_matching.py` rewrote `results/size_matching.json`. A new
  `QAPINN_RESULTS_DIR` variable and an autouse fixture now keep every test in a temp
  directory.
- Re-running a config into its existing run directory (a retry after a crash, or
  `sweep --no-resume`) appended to `xai/specerr.npz` instead of replacing it. This is the
  root cause of the duplicate-row corruption `scripts/dedupe_specerr.py` repaired in 10
  core-matrix runs. `train()` now clears a run's old instrument output first.
- `scripts/verify_preregistration.py` run directly exited 1: it scanned development smoke
  runs that predate pre-registration. It now checks the six pre-registered experiments:
  181 runs, 0 violations.
- `tasks.py figures` returned 0 even when a figure errored, and `--strict` could never
  pass because of the deliberately excluded staircase figure.
- `tasks.py slides` passed an argument list with `shell=True`, which ran a bare `npx` on
  Linux.
- Figure legibility: the headline figure's ρ labels overlapped the data and printed
  p = 3.2e-5 as "p=0.000"; the ablation figure was a six-panel strip whose text was ~3 pt
  at page width; the decision map collapsed five of six problems onto one point; heatmap
  axis labels were clipped and the Ω band was lost in a ±3,200 axis.
- A dropped `≈` glyph and straight double quotes in the paper source.
- Six cited references had no authors, and some had no year or venue either (the arXiv
  IDs were in `refs.bib` but never printed). They were completed from arXiv and Crossref
  metadata, and each cited arXiv ID was checked to exist and match its title.

### Added

- CI (GitHub Actions): lint, the test suite, and checks that regenerating from committed
  results reproduces the committed files (see Unreleased for the final form).
- `tasks.py repro-all`, previously a stub: every experiment sweep (resumable), figures,
  adjudication and the pre-registration check.
- `configs/exp/baseline_tuning_confirm.yaml`: the config behind F14's four full-budget
  runs, which were launched without one. A test pins it to their exact run IDs.
- Figures embedded in the paper (21 pages; the submitted version, 15 pages, had none) and
  an errata section.
- `results/README.md`, and `results/runs_index.csv` (generated by `scripts/index_runs.py`)
  mapping every run ID to its experiment.
- `CITATION.cff`, this changelog, and the Demo Day deck (`slides/demoday/`).
- The `v1.0-submission` tag, planned in T5.15 but never created.

### Changed

- `pytest` moved from runtime to dev dependencies; package metadata added to
  `pyproject.toml`; version 1.1.0.
- Lint-clean under ruff 0.16's default rules.
- Removed empty `.gitkeep` files from directories that have content.

## 1.0.0 — 2026-08-07 (`v1.0-submission`)

The version submitted to and judged at the WISER Summer Program 2026 Industry Challenge.
