# Phase 3 — Main Experiments (Days 8–9)

**Goal:** run the matrix and the three sweeps, and produce the coverage-vs-error plot.

**Exit gate (`project.md` §12 Phase 3):**
> Predictions pre-registered **before** running. The coverage-vs-error plot exists, whatever it says.

**The discipline that makes this phase worth anything** (`project.md` §7): every XAI claim needs
(a) a prediction stated *before* the run, committed to git with a timestamp, and (b) a matched control.
Post-hoc storytelling over a plot is the exact failure mode the XAI-for-PINN literature is criticised
for. T3.1 is therefore a **blocking** task: no run may start before its commit exists.

---

## T3.1 — ★★ Pre-registration (BLOCKING — no runs before this is committed)

**Depends on:** T2.18
**Files:** `docs/predictions.md`

Write and commit, with a git timestamp, a numbered list of falsifiable predictions. Each entry:
`ID · prediction · the metric that decides it · the threshold · the figure that will show it`.

Minimum required entries:

| ID | Prediction | Deciding metric | Threshold |
|---|---|---|---|
| PR-1 | `q_serial` beats `c_mlp` on P1 (α=0.3) | median rel-L2 over 5 seeds | ≥ 2× lower |
| PR-2 | `q_serial` beats `c_ff` on P1 | median rel-L2 | ≥ 1.3× lower |
| PR-3 | `q_serial` ≈ `c_rff_matched` on P1 | median rel-L2 ratio | within 1.3× either way |
| PR-4 | **P2 (heat): no family beats `c_mlp` by more than 5 %** | median rel-L2 | the a-priori SMCD number from T2.8 is 2.5 % |
| PR-5 | P4 advantage grows with `k` | rel-L2 ratio `c_mlp / q_serial` at k=4,10,20 | monotonically increasing |
| PR-6 | `q_serial` beats `q_random` at matched size | median rel-L2 | ≥ 1.5× lower (**C1's falsifier**) |
| PR-7 | NTK spectrum of `q_serial` shows a plateau at the encoded band | `decay_exponent` measured inside `Ω` vs outside | flatter inside by ≥ 0.5 |
| PR-8 | Per-frequency error improvement **coincides with `Ω`** | overlap of the improved band with `Ω` | ≥ 70 % of improvement inside `Ω` (**C3's falsifier**) |
| PR-9 | Final L2 error decreases monotonically with SMCD coverage | Spearman ρ (coverage vs error) | ρ ≤ −0.7 |
| PR-10 | Gradient variance decays with `n` but stays above the `2^{−n}` line | `barren_plateau_fit.slope_b` | `b < log 2` |
| PR-11 | Hybrid leaves the lazy regime earlier than classical | NTK drift `‖Θ_t−Θ_0‖/‖Θ_0‖` at 5k steps | hybrid larger |
| PR-12 | Encoder drift stays small enough that coverage remains valid | `‖A−I‖_F` at final step | < 0.2 |

Also record, for each: **what result would refute our explanation** (`project.md` §2, C3's falsifier
is explicit — "improvement spread uniformly across `ω`, or landing outside the encoded band, means our
explanation is wrong and we say so").

**DoD:** committed to git with the message `T3.1: pre-register predictions`. Record the commit SHA in
`FINDINGS.md` later so the timestamp is auditable.

---

## T3.2 — Run orchestration

**Depends on:** T3.1
**Files:** `src/qapinn/runner.py`, `scripts/run_experiment.py`, `scripts/sweep.py`

```python
def enumerate_runs(exp_cfg_path: Path) -> list[ExpConfig]
def run_all(cfgs: list[ExpConfig], n_workers: int = 6, resume: bool = True) -> None
```

- **Parallelism:** `concurrent.futures.ProcessPoolExecutor`, default 6 workers (measured against
  16 GB VRAM in T2.18; lower it if OOM). Each worker gets its own CUDA context; set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Resume:** skip any run whose `results/runs/<run_id>/metrics.json` already exists. This makes the
  matrix restartable — essential, because it will be interrupted.
- **Failure isolation:** a crashed run writes `error.json` with the traceback and does not kill the
  pool. Report a summary of failures at the end.
- **Progress:** `tqdm` over completed runs, with running ETA.

**DoD:** `python tasks.py sweep --exp core_matrix --smoke` enumerates the full run list, executes it
in smoke mode, and a second invocation skips everything (resume works).

---

## T3.3 — Experiment config files

**Depends on:** T3.2
**Files:** `configs/exp/{core_matrix,coverage_sweep,depth_sweep,alpha_sweep,noise_study}.yaml`

Encode the §5 budget of the master plan exactly:

```yaml
# core_matrix.yaml
problems: [poisson, heat, burgers, helmholtz_k4, helmholtz_k10, helmholtz_k20]
families:  [c_mlp, c_ff, c_rff_matched, q_serial, q_parallel, q_random, q_octave]
seeds:     [0, 1, 2, 3, 4]
train:     {bc_mode: hard, noise: none}
# 6 x 7 x 5 = 210 runs
```
```yaml
# coverage_sweep.yaml   (THE validation of C1 — never cut)
problems: [poisson, helmholtz_k10]
families:  [q_serial]
coverage_targets: [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
seeds: [0, 1, 2]                     # 2 x 6 x 3 = 36 runs
```
```yaml
# depth_sweep.yaml      (barren-plateau frontier)
problems: [helmholtz_k10]
families: [q_serial]
n_layers: [2, 3, 4, 5, 6]
n_qubits: [4, 6, 8]
seeds: [0, 1, 2]                     # 45 runs
```
```yaml
# alpha_sweep.yaml      (where does quantum start winning?)
problems: [poisson]
families: [c_mlp, c_ff, q_serial]
alpha: [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
seeds: [0, 1, 2]                     # 54 runs
```
```yaml
# noise_study.yaml
problems: [poisson, helmholtz_k10]
families: [c_ff, q_serial, q_random]
noise: [shot_1024, depol_1e-3]
seeds: [0, 1, 2]                     # 36 runs
```

**DoD:** `enumerate_runs` returns 210 / 36 / 45 / 54 / 36 configs respectively, all with distinct
`run_id`s.

---

## T3.4 — Launch the core matrix

**Depends on:** T3.3
**Files:** — (produces `results/runs/**`)

`python tasks.py sweep --exp core_matrix`. Run in the background; monitor.

**Watch for and stop on:**
- Any family failing on all seeds of a problem (an integration bug, not a scientific result).
- `q_serial` gradient variance < 1e-10 (barren plateau — switch to `q_octave` and record why).
- Wall-clock diverging > 2× from the T2.18 estimate (re-apply master-plan §5 cut lines).

**DoD:** 210 `metrics.json` files exist (or documented failures with reasons). Append the real total
wall-clock to `BENCH.md`.

---

## T3.5 — Launch the three sweeps + noise study

**Depends on:** T3.4
**Files:** — (produces `results/runs/**`)

Order matters if time is short: **coverage sweep first** (it validates C1), then depth/qubit, then α,
then noise.

**DoD:** all four sweeps complete or explicitly cut with the cut recorded in `FINDINGS.md`.

---

## T3.6 — Soft-BC NTK block-imbalance study (D6)

**Depends on:** T3.4
**Files:** `configs/exp/soft_bc_ntk.yaml`

`bc_mode: soft` on P1 and P4(k=10), families `{c_mlp, q_serial}`, 3 seeds. This is the only place the
per-loss-block NTK eigenvalue mass (`project.md` §7.1: residual vs BC vs IC) is measurable, since hard
constraints remove the BC loss entirely.

**DoD:** `block_mass` recorded at every checkpoint; the classical loss-imbalance story (residual block
eigenvalues ≫ BC block, per Wang et al.) is either reproduced or reported as not reproduced.

---

## T3.7 — ★★ THE COVERAGE-VS-ERROR PLOT

**Depends on:** T3.5
**Files:** `scripts/make_figures.py` (fig: `coverage_vs_error`)

**The single most important figure in the paper** (`project.md` §5.3 step 11).

x-axis: `coverage_weighted` from the design card. y-axis: final median rel-L2 (log scale), IQR bands
over seeds. One series per problem (P1, P4 k=10). Annotate each point with `(n, L)`. Overlay the
Spearman ρ and its p-value.

Because coverage is read from the **achieved** value in the card (T2.9), the x-axis is a measured
quantity, not the requested one — the points will not be evenly spaced, and that is correct.

**DoD:** `paper/figures/coverage_vs_error.pdf` exists with ρ reported. **Whatever the plot says, it
ships.** A flat or non-monotone curve refutes C1 and is written up as such.

---

## T3.8 — NTK spectra figure (two families, same axes)

**Depends on:** T3.4
**Files:** `scripts/make_figures.py` (fig: `ntk_spectrum`)

Complete the T1.3 figure now that hybrids exist: `λ_i` vs `i`, log-log, `c_mlp` and `q_serial`
overlaid at matched steps, with the encoded band `Ω` shaded (the hook left in T1.3).
Report the decay exponent measured **inside** vs **outside** the band — that difference is PR-7.

**DoD:** figure exists for P1 and P4(k=10); the two decay exponents are recorded in `metrics.json`.

---

## T3.9 — ★ Per-frequency heatmaps with `Ω` overlay (C3's decisive figure)

**Depends on:** T3.4
**Files:** `scripts/make_figures.py` (fig: `freq_heatmap_{problem}`)

Side-by-side `|ê(ω, t)|` heatmaps for `c_mlp` and `q_serial`, **with the design card's `Ω` drawn as
horizontal lines over both panels** (`project.md` §7.2).

> If the improved band and `Ω` coincide, C3 is proven visually in one figure.

Compute PR-8 quantitatively as well as visually: the fraction of total error improvement
(`|ê_cmlp| − |ê_qserial|`, summed over checkpoints) that falls at frequencies inside `Ω`.

**DoD:** heatmaps for all six problem instances; the PR-8 overlap fraction recorded per problem.

---

## T3.10 — Phase 3 gate check

**Depends on:** T3.7, T3.8, T3.9

Confirm:
1. `docs/predictions.md` commit predates every run's `provenance.json` timestamp — **verify this
   mechanically**, it is the credibility of the whole pre-registration claim.
2. Coverage-vs-error plot exists.
3. NTK spectra and frequency heatmaps exist for all six problem instances.
4. `results/manifest.json` maps every figure to its runs.

**DoD:** commit tagged `phase3-complete`.
