# v1 completion, v2 model and the groundwater showcase: design

Date: 2026-09-28. Branch: `v2`. Status: approved section by section in chat (sections 1
and 2 explicitly; the owner then delegated section 3 and all remaining work overnight).

## Goal

1. Say, truthfully: "we ran all 12 pre-registered predictions to the protocol as written."
2. Show a genuinely improved quantum model re-tested against the same 12 claims, locked
   in advance, labelled "the improved model meets N of 12".
3. Solve one real-world problem with it: the water table under canal-irrigated farmland.

## Non-negotiable rules

- v1's predictions, thresholds and checks (`docs/predictions.md`,
  `scripts/adjudicate_predictions.py`'s `check_pr*`) are not changed. `v1.0-submission`
  and `results/adjudication.json` stay as the dated record.
- v1 re-adjudication reports whatever the full protocol gives, REFUTED included.
- v2 predictions are committed, and the `v2` branch pushed so GitHub records the time,
  before any v2 re-test or groundwater run. A script checks every such run's provenance
  timestamp is after that commit.
- Model tuning (Phase B) happens only on v1's problems and only with seeds 0-4. The
  re-test uses seeds 10-14. The groundwater problem is never trained on before the lock
  except by `--smoke` runs (a few dozen steps, used only to check the code runs).
- Every classical family gets the same number of tuning configurations as the quantum
  model.
- Every deviation is dated and written down.

## Phase A: finish v1 to its pre-registered protocol

| Config | What | Runs to train |
|---|---|---|
| `configs/exp/core_matrix_full.yaml` | core matrix at the pre-registered 5 seeds | 34 (of 210) |
| `configs/exp/depth_sweep_full.yaml` | PR-10 at n_qubits {4, 6} (8 does not fit in memory), 3 seeds | 20 (of 24) |
| `configs/exp/coverage_sweep_full.yaml` | PR-9 at the original 20000+2000 budget | 10 (of 36) |

Existing runs are reused by run_id. `adjudicate_predictions.py --protocol full` points
the unchanged checks at these configs and writes `results/adjudication_v1_full.json`.

Known outcome, stated now: **PR-7 (Poisson) cannot be measured by any run.** The check
splits the q_serial NTK spectrum at eigen-index |Ω| = 81, but q_serial has about 20
trainable parameters, so its NTK has about 20 non-zero eigenvalues and none past index
81. It stays INSUFFICIENT_DATA with that reason written out. No grid change fixes it.

## Phase B: the v2 quantum model (lab)

Cause of v1's loss, from v1 data: q_serial's output is one expectation value through
`nn.Linear(1, 1)`, so all Fourier amplitudes are set indirectly through ~10 angles; its
Poisson loss stalls at ~3.3e4 while `c_rff_matched` (a linear head on the same waves)
goes further.

Candidates, tried on v1's Poisson and heat problems:

1. **Multi-observable readout**: <X>, <Y>, <Z> on every qubit (all single-qubit, so
   Alg. 1's ban on global observables holds), head `Linear(3n, 1)`.
2. **Residual normalisation** for every family: divide the PDE residual by the RMS of the
   forcing on the collocation set, so the (15π)² ripple term does not swamp the loss.
3. **Width vs depth** at the same frequency set, where SMCD offers a choice.

Selection rule: keep a candidate if it lowers the median rel-L2 of q_serial on Poisson
over seeds 0-2 at the full budget. The chosen model is frozen as `q_serial_v2` (and
`q_random_v2`, `q_octave_v2`, `q_parallel_v2` with the same readout, so v1's
comparisons stay like for like). Classical families get the same normalisation and the
same count of configurations tried. Every lab run is recorded under `configs/exp/lab_*.yaml`.

## Phase C: groundwater, locked predictions, re-test

### The problem

Steady groundwater flow in a confined aquifer between two rivers (Dupuit, 1-D):

    T h''(X) = -R(X),   0 <= X <= L,   h(0) = h_A,   h(L) = h_B

- h: hydraulic head (water-table height above datum, m); T: transmissivity (m²/day);
  R: recharge (m/day) from irrigation return flow plus seepage from lined-but-leaking
  canals running parallel to the rivers.
- Scenario values, within textbook ranges (Freeze & Cherry 1979; Todd & Mays 2005),
  illustrative rather than calibrated to one site: L = 2,000 m; T = 100 m²/day (silty
  sand); background recharge R0 = 1 mm/day; 6 canals at equal spacing, each a smooth
  strip of seepage; land surface 6 m above the river level.
- Non-dimensionalised to x = X/L on [0, 1], so the solver sees `-u'' = f(x)` with the
  lift B(x) = h_A + (h_B - h_A) x and mask D(x) = x(1 - x). This is v1's Poisson
  interface with non-zero boundary values and a different forcing, so every model,
  SMCD and instrument works unchanged.
- The canal spacing sets the frequencies: harmonics of 2π·6 in x. SMCD reads them from
  the forcing exactly as it does for v1's problems.

### Reference solution

A second-order finite-difference solver on 20,001 points (`src/qapinn/reference/`),
checked in tests against the closed-form solution for a cosine recharge pattern. The
PINN's `exact()` is this reference, interpolated.

### What "solved" means

- Accuracy: rel-L2 against the reference.
- The practical answer: the length of field where the water table is within 1.5 m of
  the land surface (the usual waterlogging threshold for crop root zones), and the peak
  water-table height. Reported for the reference and every model.

### v2 predictions (locked in `docs/predictions_v2.md` before the re-test)

- **R-1 to R-12**: v1's PR-1 to PR-12, word for word, with `q_serial` read as
  `q_serial_v2` (and the other quantum families as their v2 versions), on seeds 10-14,
  adjudicated by the unchanged checks. Headline: "meets N of 12".
- **G-1 to G-5** (groundwater, seeds 10-14):
  - G-1 `q_serial_v2` median rel-L2 <= 1% against the reference.
  - G-2 `q_serial_v2` beats a size-matched (±10% parameters) `c_mlp` by >= 1.3×.
  - G-3 `q_serial_v2` beats `q_random_v2` by >= 1.5× (v1's PR-6 rule).
  - G-4 >= 70% of `q_serial_v2`'s per-frequency improvement over `c_mlp` lies inside Ω
    (v1's PR-8 rule).
  - G-5 `q_serial_v2`'s waterlogged length is within 5% of the reference's.

The G thresholds may be adjusted only before the lock, using Phase B data (never
groundwater data), and the final numbers are the ones in the committed file.

### Expected results, stated in advance

Likely to improve: R-1, R-2, R-6, R-8. Likely to fail however good the model:
R-4 (predicts nobody beats `c_mlp` on heat), R-5 and R-9 on Helmholtz k=10/20 (every
model scores rel-L2 ≈ 1 there), R-7 on Poisson (unmeasurable, as above).

## Amendments (2026-09-28 02:50 IST, before any lab result existed)

1. **Candidate 1 replaced.** A readout over <X>, <Y>, <Z> adds nothing on Poisson: SMCD
   gives it a single qubit, and a linear head over one qubit's X, Y, Z is equivalent to
   v1's final trainable rotation plus head scale. The v2 candidate is instead
   `n_replicas`: K copies of the designed circuit (same scalings, so the same Ω; own
   angles) reading the one encoded input, mixed by the head. **Candidate 2 dropped:**
   dividing the residual by a constant does not change Adam's or L-BFGS's steps, so it
   cannot help. It is replaced by a learning-rate axis {1e-3 (v1), 1e-2}. Lab config:
   `configs/exp/lab_quantum.yaml` (K ∈ {1, 2, 4, 8} × 2 learning rates × seeds 0-2).
2. **Selection rule, with parsimony:** the chosen configuration is the one with the
   smallest K whose median rel-L2 is within 10% of the best configuration's. (Larger K
   costs K times the compute, and the Helmholtz runs already take ~100 min at K = 1.)
3. **v2 re-test scope (seeds 10-14):** each problem runs the families its claims read.
   Poisson: `c_mlp`, `c_ff`, `c_rff_matched`, `q_serial_v2`, `q_random_v2`. Heat: all seven
   families (R-4 reads every family), with `q_serial_v2` and `q_random_v2` in place of their v1
   versions. Helmholtz k4/k10/k20: `c_mlp`, `q_serial_v2`. Burgers: none (no claim reads
   it). Instruments run only where a claim reads them: NTK, spectral error and drift for
   Poisson and Helmholtz k10 (`c_mlp`, `q_serial_v2`); none elsewhere, since rel-L2 does
   not depend on them. R-9 uses `coverage_sweep_full`'s settings and R-10 uses
   `depth_sweep_full`'s, with the v2 model.

4. **(04:30 IST, after the first 6 lab runs, before any run of the new arm.)** Those runs
   showed full-budget Poisson rel-L2 of 1.6 for one copy, and 0.69 for a 500-step pilot:
   more training made it worse. Diagnosis: v1's hard ansatz u = B + x(1-x)·N makes N
   learn u/(x(1-x)), which is not band-limited even though u = sin πx + 0.3 sin 15πx lies
   inside Ω. So v1's boundary treatment defeated SMCD's premise for every family.
   New ansatz `bc_mode: hard_affine` (`models.base.AffineBCWrapper`): subtract N's own
   boundary values, interpolated linearly per spatial axis, and N at t = 0; exact
   whenever N equals the solution minus the lift. The boundary treatment becomes a
   tuned choice **for every family**: `lab_quantum_affine.yaml` and
   `lab_classical_affine.yaml` repeat the two labs under `hard_affine`, so quantum and
   classical each get 16 configurations. Selection per family: the best (ansatz, lr) —
   for the quantum model also K — by median rel-L2 over seeds 0-2, with the parsimony
   rule on K from amendment 2 applied across all 16 quantum configurations.

## Compute

One GPU process at a time (concurrent CUDA processes have crashed this laptop). Phase A
≈ 62 GPU-hours; the v2 re-test repeats the 210-run matrix (≈ 70 GPU-hours, dominated by
Helmholtz quantum runs) plus ≈ 35 groundwater runs. Lab runs for 1-qubit Poisson models
can use the CPU in parallel.

## Deliverables

`results/adjudication_v1_full.json`, `docs/predictions_v2.md`,
`results/adjudication_v2.json`, a groundwater figure and results section in FINDINGS.md
and the paper, and a CHANGELOG entry. Nothing is merged to `main` or tagged until the
owner reviews.
