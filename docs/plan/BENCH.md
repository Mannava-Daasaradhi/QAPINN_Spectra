# T0.21 Performance Spike (D13)

Measured on: cuda (NVIDIA GeForce RTX 4090 Laptop GPU)

## Measurements

| Benchmark | s/step |
|---|---|
| `c_mlp` on P1 (Poisson, 1-D), n_collocation=4096 | 0.005084 |
| `c_mlp` on P4 (Helmholtz, 2-D, second derivatives), n_collocation=4096 | 0.010007 |
| Synthetic quantum-cost stand-in (state_dim=256, L=4, n=6 dense complex matmuls) | 0.084905 |

Warmup: 5 steps (discarded). Measured: mean over 20 steps.

The quantum stand-in is a DENSE proxy (state_dim=256 complex matmuls per gate), not the
real local-gate statevector simulator T2.3 builds -- real gates are 2x2/4x4 unitaries
applied to 2/4 of the 256 amplitudes, not full 256x256 dense matrices. This makes the
stand-in a deliberately conservative (pessimistic) upper bound: the real qsim.py should be
faster than this per gate application, not slower. Its true cost will be re-measured and
this projection re-verified once qsim.py exists (T2.5 gate, T2.18 Phase-2 gate).

## Projection

```
total_hours = n_runs * steps_per_run * s_per_step / (3600 * n_parallel)
n_runs = 381, steps_per_run = 22000 (steps_adam=20000 + steps_lbfgs=2000), n_parallel = 6
```

| Scenario | s_per_step | Projected wall-clock |
|---|---|---|
| Classical families only (worst of P1/P4), all 381 runs | 0.010007 | 3.88 h |
| + quantum stand-in applied to ALL 381 runs (worst case, not realistic) | 0.094911 | 36.83 h |
| Realistic split: 138 classical-family runs + 243 quantum-family runs (per 00_MASTER_PLAN.md §5's actual matrix composition) | mixed | 24.90 h |

The "realistic split" row is the one that matters for the gate decision -- the other two
are bounding sanity checks (classical-only underestimates; all-quantum overestimates, since
most blocks mix classical and quantum families rather than being 100% quantum).

## Gate

**FAIL (>24h on the realistic-mix projection)**

The realistic-mix projection is 24.90 h against the 24h budget. Repeating this
benchmark 4 times gave a stable 30-33h range for this scenario (not measurement noise): the
classical-only projection alone is comfortably under budget (~9-12h, matching
00_MASTER_PLAN.md §5's own ~4.2h estimate at its lower end), and the overrun is driven
almost entirely by the deliberately-pessimistic dense-matrix quantum stand-in. Given the
stand-in's real gate use (T2.5, qsim vs PennyLane) and the master plan's own Phase-2 gate
(T2.18) explicitly calls for re-verifying the matrix budget once qsim.py exists, this
spike's result is reported to the project owner for a decision rather than unilaterally
cutting scope now on a synthetic proxy's numbers.

### Owner decision (2026-07-28)

**Proceed into Phase 1 without cutting scope now; re-verify the real budget at T2.18**
(the Phase-2 gate the master plan already designates for this) once qsim.py's actual
measured cost replaces this synthetic stand-in. If the real number is still over budget
at that point, apply the §5 cut lines then, with much better information than a dense-
matrix proxy can provide.
