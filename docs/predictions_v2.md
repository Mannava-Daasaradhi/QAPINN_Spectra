# Pre-registered predictions, v2

Committed, and pushed to GitHub, **before** any run below executes.
`scripts/verify_preregistration.py --v2` checks that the commit adding this file is an
ancestor of every v2 run's recorded `git_sha`, as v1's T3.10 gate does for
`docs/predictions.md`.

## What v2 is

v1 (`docs/predictions.md`, tag `v1.0-submission`) is left exactly as it was. After v1 we
changed the quantum model, tuned it on v1's Poisson problem with seeds 0-2, and now
re-test it on seeds never used before (10-14) against v1's claims, word for word, and on
the groundwater problem. The headline this file allows is "the improved model meets N of
the claims re-tested", with N whatever the unchanged adjudication scripts report.

**The v2 quantum model** (`configs/model/q_serial_v2.yaml`): v1's SMCD-designed circuit
with 8 copies of it (same encoding scalings, so the same frequency set Ω; each copy has
its own trainable angles) reading the one affine-encoded input, mixed by a linear head:
91 parameters on Poisson and groundwater. `q_random_v2` is the same with v1's random
scalings. Learning rate 1e-2. Chosen in the lab by the rule committed before the lab
ran (`scripts/select_v2_config.py`, `results/lab_selection.json`; design spec
`docs/superpowers/specs/2026-09-28-v1-completion-v2-groundwater-design.md`, amendments
2 and 5). **The lab result was seen before this file was written**: on seeds 0-2 this
configuration reached a median rel-L2 of about 0.003 on Poisson, the best of 8 quantum
configurations. That number is optimistic by construction and is not a finding; the
re-test below is.

**Baselines.** Each classical family uses the learning rate the same lab chose for it
from 8 candidates (equal tuning effort): `c_mlp` 1e-2, `c_ff` 1e-2, `c_rff_matched`
5e-4. `c_rff_matched_v2` is size-matched to `q_serial_v2` (83 parameters; v1's was
matched to the single circuit). `c_mlp` and `c_ff` keep v1's sizes (8,513 and 129
parameters), so R-1 and R-2 compare against the same models v1 did.

**Protocol.** Seeds 10-14, the full 20000 Adam + 2000 L-BFGS budget, v1's hard boundary
ansatz, no noise. Configs: `configs/exp/v2_poisson_*.yaml` and
`configs/exp/v2_groundwater_*.yaml`. Adjudicated by
`scripts/adjudicate_predictions.py --protocol v2` (v1's checks, unchanged) and
`scripts/adjudicate_groundwater.py --models v2`.

## R-claims: v1's claims, re-tested on the v2 model (Poisson)

Word for word from `docs/predictions.md`, with `q_serial` read as `q_serial_v2`,
`q_random` as `q_random_v2` and `c_rff_matched` as `c_rff_matched_v2`. Thresholds,
falsifiers and statistics are v1's.

| ID | Prediction | Deciding metric | Threshold |
|---|---|---|---|
| R-1 | `q_serial_v2` beats `c_mlp` on P1 (α=0.3) | median rel-L2 over 5 seeds | ≥ 2× lower |
| R-2 | `q_serial_v2` beats `c_ff` on P1 | median rel-L2 | ≥ 1.3× lower |
| R-3 | `q_serial_v2` ≈ `c_rff_matched_v2` on P1 | median rel-L2 ratio | within 1.3× either way |
| R-6 | `q_serial_v2` beats `q_random_v2` at matched size on P1 | median rel-L2 | ≥ 1.5× lower |
| R-7 | NTK spectrum of `q_serial_v2` is flatter inside Ω than outside on P1 | decay-exponent gap | ≥ 0.5 |
| R-8 | Per-frequency improvement over `c_mlp` coincides with Ω on P1 | overlap with Ω | ≥ 70% inside |
| R-11 | Hybrid leaves the lazy regime earlier than classical on P1 | NTK drift at 5k steps | hybrid larger |
| R-12 | Encoder drift stays small enough that coverage remains valid on P1 | `‖A − I‖_F` at final step | < 0.2 |

**Not re-tested in this round** (design spec amendment 5, decided before any v2 run):
R-4 and R-6 on heat, R-5, R-7 and R-8 on Helmholtz k=10, R-9 and R-10. They are
reported as "not run", not as failed.

**Stated in advance.** R-1 is expected to fail: in the lab the tuned 8,513-parameter
`c_mlp` reached rel-L2 of about 4e-4 on two of three seeds. R-3 predicts parity, so it
fails if `q_serial_v2` is clearly better than `c_rff_matched_v2`. R-7 on P1 cannot be
measured if fewer than two NTK eigenvalues lie beyond index |Ω| = 81.

## G-claims: the groundwater showcase with the v2 model

Same problem, thresholds and rules as `docs/predictions_groundwater.md` (the v1-model
version), with the v2 models: `q_serial_v2`, `q_random_v2`, `c_mlp` (lr 1e-2),
`c_mlp_matched_v2` (one hidden layer of 30, 91 parameters, lr 1e-2) and
`c_rff_matched_v2` (lr 5e-4). Exact waterlogged length: 713 m.

| ID | Prediction | Deciding metric | Threshold |
|---|---|---|---|
| G-1 | `q_serial_v2` solves the problem accurately | median rel-L2 against the exact solution | ≤ 1% |
| G-2 | `q_serial_v2` beats a same-size classical network | median rel-L2 ratio `c_mlp_matched_v2` / `q_serial_v2` | ≥ 1.3×, v1's PR-1 rule |
| G-3 | The SMCD design beats a random design of the same size | median rel-L2 ratio `q_random_v2` / `q_serial_v2` | ≥ 1.5×, v1's PR-6 rule |
| G-4 | Its improvement over `c_mlp` lies on the designed frequencies | v1's PR-8 overlap | ≥ 70% inside Ω |
| G-5 | It gets the practical answer right | median waterlogged length vs 713 m | within 5% |

Falsifiers: each claim is REFUTED when its metric misses its threshold, and
INCONCLUSIVE where v1's statistical rule says five seeds cannot separate the two models.
No claim is dropped or reworded after the runs.
