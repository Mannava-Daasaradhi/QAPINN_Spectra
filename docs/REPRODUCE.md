# Reproduction instructions (T5.13)

Every command below was actually run; output is real, pasted or paraphrased from the
actual run, not estimated. Sections 2, 3, 4 and 9 were re-run for the post-submission
revision (2026-09-25); the rest date from the submission (2026-08-06). Commands assume a
Windows/PowerShell or Git-Bash shell in the repository root, with `uv` installed; prefix
them with `uv run` if the environment is not activated. CI (`.github/workflows/ci.yml`)
runs sections 2–4 on every push, on a CPU-only Linux runner.

## 1. Environment

```
uv sync
```

Pins Python 3.12, torch (cu128 build), PennyLane, and every other dependency exactly as
recorded in `pyproject.toml`/`uv.lock`. `torch.cuda.is_available()` should return `True`
on a CUDA-capable machine; the reduced CPU-only paths below (`repro-quick`) do not
require this.

## 2. Test suite

```
python tasks.py test
```

Actual output (post-submission revision, CUDA hidden so it runs CPU-only like CI; with
the GPU visible the same suite took 312 s):

```
...
tests\test_train_loop.py ......                                          [ 94%]
tests\test_verify_preregistration.py ......                              [ 96%]
tests\test_viz_style.py .....                                            [ 97%]
tests\test_xai_init.py .........                                         [100%]

================ 385 passed, 5 deselected in 244.37s (0:04:04) ================
```

`--slow` adds the 5 deselected tests (including the 42-combination smoke matrix). No
test writes into the committed `results/` or `paper/figures/`: `tests/conftest.py`
points `QAPINN_RESULTS_DIR` and the figure paths at a temp directory for every test.

## 3. Regenerate every figure from committed results

```
python tasks.py figures
```

Actual output (post-submission revision):

```
Regenerating figures from results/runs/ (core_matrix: 84/84 runs available)...
  EXCL  staircase_cmlp_p1: build_staircase_data() retrains from scratch -- needs an npz-based rewrite first
  OK    coverage_vs_error
  OK    ntk_spectrum_p1
  OK    ntk_spectrum_p4
  OK    ntk_spectrum_poisson_pr7
  OK    ntk_spectrum_helmholtz_k10_pr7
  OK    freq_heatmap_poisson
  OK    freq_heatmap_heat
  OK    freq_heatmap_burgers
  OK    freq_heatmap_helmholtz_k4
  OK    freq_heatmap_helmholtz_k10
  OK    freq_heatmap_helmholtz_k20
  OK    ablation_matched
  OK    decision_map
  OK    barren_frontier
  OK    cost_ledger_table
  OK    attribution_p3
  OK    attribution_p4
  OK    fisher_effdim
  OK    probes_cka
  OK    landscape_grid

20/21 produced, 0 skipped, 1 excluded by design, 0 errored
```

`staircase_cmlp_p1` is the one figure this driver deliberately does not attempt (see its
`EXCL` reason above): it has its own already-verified T1.5 gate-time artifact instead
of a committed-npz regeneration path. `--strict` fails on any skipped sweep or error,
not on this exclusion.

On the machine that drew the committed figures the output is byte-for-byte identical, so
running this command (or the adjudicator in §4, which draws some of the same figures)
leaves `git status` clean. Other platforms rasterize slightly differently: every PNG, and
the PDFs with raster content, change by a few pixels, while everything else is
identical. To check regenerated figures against the committed ones with a pixel
tolerance (CI runs this on Linux):

```
python scripts/compare_figures.py
```

Actual output, this machine: `20 PNGs checked, 0 outside tolerance`.

Every figure comes from `results/runs/*/metrics.json` and `xai/*.npz` only — never a
live model reload (`scripts/make_figures.py`).

## 4. Re-run the mechanical prediction adjudication

```
python scripts/adjudicate_predictions.py
```

Produces a JSON report with all 12 pre-registered predictions (15 checks, counting
per-problem splits) adjudicated against `docs/predictions.md`'s exact thresholds. The
verdicts in this JSON are the source of every claim in `FINDINGS.md` and
`paper/sections/results.tex`/`negative_results.tex`/`conclusion.tex` — nothing in the
paper states a stronger verdict than what this script reports.

`--write` also saves the verdicts to `results/adjudication.json`, with numbers rounded to
10 significant digits so the record is identical on every machine; CI regenerates it
and requires an exact match with the committed file.

The pre-registration ordering check (T3.10) re-verifies that `docs/predictions.md` was
committed before every run of the six pre-registered experiments:

```
python scripts/verify_preregistration.py
```

Actual result: `"n_checked": 181`, `"violations": []`, exit 0.

## 5. Build the paper

```
python tasks.py paper
```

Wraps `tectonic main.tex` inside `paper/`. Actual result, this session: 15-page PDF,
clean build (only cosmetic `Overfull \hbox`/`Underfull \hbox` LaTeX line-breaking
warnings, no errors).

## 6. Build the slides

```
python tasks.py slides
```

Wraps `npx @marp-team/marp-cli slides/main.md -o slides/main.pdf --allow-local-files`.
Actual result, this session: 18-page PDF (within the required 15–20 slide range),
reusing the same `paper/figures/*.png` assets as the paper.

## 7. Quick, CPU-only reproduction (~15 min target)

```
python tasks.py repro-quick
```

Runs `train(..., smoke=True)` across all 6 problem instances × 7 model families (42
combinations), `--cpu`, one process per combination for fault isolation. This is a fast
correctness signal ("does the whole pipeline still run end-to-end with no CUDA"), not a
reproduction of any specific paper number — see §8 for that.

Actual result, this session (`cmd_repro_quick` was an unimplemented stub before this
session — implemented by wrapping `cmd_smoke`'s existing `--cpu` path):

```
42/42 combinations completed in 513.7s (device=cpu)

repro-quick: 513.8s elapsed (CPU-only, target ~15 min / 900s)
```

8.6 minutes, well under the 15-minute target.

## 8. Reproducing a specific paper number

No figure or number in this paper is generated from a live model reload — every one
reads only `results/runs/*/metrics.json` and `xai/*.npz`, which are committed to the
repository. To recompute a specific number from scratch instead of trusting the
committed artifact, re-run the specific experiment config that produced it, e.g.:

```
python tasks.py sweep --exp core_matrix --workers 2 --mem-fraction 0.35
```

`--workers 2` was the maximum verified safe on the development machine this project was
built on (`docs/plan/08_TASK_INDEX.md`, `docs/plan/BENCH.md`'s T2.17 finding: higher
worker counts caused OS/driver-level CUDA resource contention, not a VRAM capacity
issue, on that specific hardware) — reduce to `--workers 1` on a machine with less
system RAM, or increase cautiously on one with more, watching free system memory as you
go. `sweep` is resumable: re-running the same command skips any run whose
`metrics.json` already exists (`qapinn.runner.enumerate_runs`/`run_all`).

Every `configs/exp/*.yaml` file's own header comment documents any deadline-driven cuts
applied to it (seed count, step budget, axis values) — read that comment before
comparing a fresh run's `run_id` against a committed one, since a config change changes
the hash. `results/runs_index.csv` maps every committed run ID to its experiment.

To retrain into a separate directory rather than alongside the committed runs, set
`QAPINN_RESULTS_DIR` (e.g. `QAPINN_RESULTS_DIR=/tmp/runs python tasks.py sweep ...`).
Figures and the adjudicator always read the committed `results/runs/`.

## 9. Everything, end to end

```
python tasks.py repro-all
```

Runs every experiment sweep in `tasks.REPRO_EXPERIMENTS` (resumable, so committed runs
are skipped), then `figures --strict`, the adjudicator and the pre-registration check,
stopping at the first failure. Actual result against the committed runs: 197/197 runs
skipped, 20 figures regenerated, all checks passed, 56 s. With `--no-resume` it retrains
all 197 runs in place (overwriting `results/runs/`, so the figures and verdicts that
follow come from the fresh runs): 67.2 h of training wall-clock on the development
machine (the sum of the runs' recorded `wall_clock_s`; `core_matrix` alone is 58.5 h).
