# Pre-registered predictions (T3.1)

Committed **before** any Phase 3 experiment run executes (`08_TASK_INDEX.md`: T3.1 is
★★ and blocking — "no runs before this is committed"). This file exists so that every
claim in the paper can be checked against a prediction made in advance, not one fitted
to the data after the fact. T3.10's gate mechanically verifies this commit predates
every run's `provenance.json` timestamp.

Format per `05_PHASE3_experiments.md`: `ID · prediction · deciding metric · threshold ·
artifact that shows it`, plus what result would refute the explanation for each.

| ID | Prediction | Deciding metric | Threshold | Artifact |
|---|---|---|---|---|
| PR-1 | `q_serial` beats `c_mlp` on P1 (α=0.3) | median rel-L2 over 5 seeds | ≥ 2× lower | core matrix results table |
| PR-2 | `q_serial` beats `c_ff` on P1 | median rel-L2 | ≥ 1.3× lower | core matrix results table |
| PR-3 | `q_serial` ≈ `c_rff_matched` on P1 | median rel-L2 ratio | within 1.3× either way | core matrix results table |
| PR-4 | P2 (heat): no family beats `c_mlp` by more than 5% | median rel-L2 | the a-priori SMCD number from T2.8 is 2.5% | core matrix results table |
| PR-5 | P4 advantage grows with `k` | rel-L2 ratio `c_mlp / q_serial` at k=4,10,20 | monotonically increasing | core matrix results table (P4 rows) |
| PR-6 | `q_serial` beats `q_random` at matched size | median rel-L2 | ≥ 1.5× lower | core matrix results table |
| PR-7 | NTK spectrum of `q_serial` shows a plateau at the encoded band | `decay_exponent` measured inside `Ω` vs outside | flatter inside by ≥ 0.5 | `paper/figures/ntk_spectrum_{p1,p4k10}.pdf` (T3.8) |
| PR-8 | Per-frequency error improvement coincides with `Ω` | overlap of the improved band with `Ω` | ≥ 70% of improvement inside `Ω` | `paper/figures/freq_heatmap_{problem}.pdf` (T3.9) |
| PR-9 | Final L2 error decreases monotonically with SMCD coverage | Spearman ρ (coverage vs error) | ρ ≤ −0.7 | `paper/figures/coverage_vs_error.pdf` (T3.7) |
| PR-10 | Gradient variance decays with `n` but stays above the `2^{-n}` line | `barren_plateau_fit.slope_b` | `b < log 2` | barren-plateau frontier (T4.5, from `depth_sweep` runs) |
| PR-11 | Hybrid leaves the lazy regime earlier than classical | NTK drift `‖Θ_t − Θ_0‖ / ‖Θ_0‖` at 5k steps | hybrid larger | NTK drift table (`FINDINGS.md`) |
| PR-12 | Encoder drift stays small enough that coverage remains valid | `‖A − I‖_F` at final step | < 0.2 | encoder drift table (`FINDINGS.md`) |

## Falsifiers

What result, for each prediction, means the explanation is wrong and gets reported as
such — not silently reinterpreted or dropped:

- **PR-1**: `q_serial`'s median rel-L2 is not at least 2× lower than `c_mlp`'s (including
  if `q_serial` is worse).
- **PR-2**: `q_serial`'s median rel-L2 is not at least 1.3× lower than `c_ff`'s.
- **PR-3**: either family is decisively better than the other (ratio outside `[1/1.3,
  1.3]`) — the "matched capacity, matched result" claim would not hold.
- **PR-4** (**C4's core claim**): any family beats `c_mlp` by more than 5% on P2 — would
  mean hybridization is NOT harmless-at-best on a smoothing operator, contradicting the
  negative-result map's stated mathematical reason (smoothing kills high-frequency
  content, so a high-frequency-rich circuit should add nothing but risk).
- **PR-5**: the `c_mlp`/`q_serial` rel-L2 ratio does not strictly increase across
  `k=4→10→20` (decreases or plateaus at any step).
- **PR-6** (**C1's falsifier**, `project.md` §2, verbatim): "a spectrum-matched circuit
  does not beat a size-matched random-frequency circuit on the same PDE."
- **PR-7**: the inside-`Ω` vs outside-`Ω` decay-exponent gap is smaller than 0.5, or the
  inside band is not flatter at all — the NTK would not show the predicted
  eigen-direction concentration at the encoded frequencies (C2).
- **PR-8** (**C3's falsifier**, `project.md` §2, verbatim): "accuracy improves but the
  improvement is spread uniformly across `k`, or lands outside the encoded band."
- **PR-9**: `ρ > −0.7` (weak, absent, or positive correlation) — coverage would not
  predict accuracy, undermining C1's central constructive claim.
- **PR-10**: `slope_b ≥ log 2` — gradient variance would decay at (or faster than) the
  generic unstructured-circuit barren-plateau rate, meaning SMCD's structured design does
  not actually buy any barren-plateau resistance.
- **PR-11**: hybrid's NTK drift at 5k steps is not larger than classical's — the "hybrid
  leaves the lazy/NTK-linear regime earlier" claim would not hold.
- **PR-12**: `‖A − I‖_F ≥ 0.2` at the final step — the encoder would have moved enough
  during training that the pre-registered `Ω`/coverage claims (computed at init, D3) may
  no longer describe the trained model, and any PR-7/PR-8/PR-9 result would need that
  caveat attached.
