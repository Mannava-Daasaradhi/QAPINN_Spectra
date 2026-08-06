# Adversarial self-review (T4.8)

Six-item minimum checklist, `docs/plan/06_PHASE4_honesty.md`. Answered against the real
adjudicated data in `FINDINGS.md` and `results/`, not asserted.

## 1. Is any improvement explained by parameter count rather than spectrum?

**Not applicable in the direction the checklist worried about, and the data rules out
the concern for the one comparison that matters most.** There is no measured
`q_serial` improvement over a matched classical family in this run to explain away —
PR-3 (Poisson, `q_serial` vs `c_rff_matched`) is REFUTED with `c_rff_matched`
*decisively beating* `q_serial` (ratio 4.25×, `FINDINGS.md` F2). Checking against T2.16
(`results/size_matching.json`): on Poisson, `q_serial` has 14 parameters,
`c_rff_matched` has 13 (7.1% difference, inside the ±10% matching tolerance). So the
result is not "q\_serial wins because it has more parameters" — the two are
near-identical in size and the classical, non-quantum family still wins substantially.
If anything this strengthens the negative finding: matched capacity did not help
`q_serial` close the gap.

## 2. Is any improvement explained by the linear head alone?

**Real gap — no control run exists.** No linear-head-on-frozen-random-features control
was built or run this session. Given item 1's finding (no positive `q_serial`
improvement exists in this run to attribute to anything, quantum or otherwise), this
control is less urgent than it would be if a win needed explaining, but it remains
unanswered as a check on its own terms. **Not done; flagged, not faked.**

## 3. Did encoder drift invalidate the coverage numbers?

**No.** PR-12: median `‖A−I‖_F` = 0.132 at the final checkpoint, under the 0.2 bar,
`n=2` runs (`FINDINGS.md` F8). The encoder does not move far enough during training to
invalidate the Ω/coverage claims computed once at initialization (D3). This is the one
item on this checklist with a clean, already-adjudicated CONFIRMED answer.

## 4. Are the classical baselines properly tuned?

**Checked (T4.9, resolved). Neither verdict changes; the story is more interesting than
"was under-tuned or wasn't."** Ran a 12-config exploratory grid (3 learning rates ×
2 widths, `c_mlp`/`c_ff`, Poisson, reduced budget) to find each family's best
configuration, then re-ran the winner at `core_matrix`'s actual full budget
(20000+2000 steps) and seeds `{0,1}` — the only comparison that actually matters, since
that is what PR-1/PR-2 are adjudicated against.

- **`c_ff` was mildly under-tuned**: `lr=0.003` (vs.\ default `1e-3`) improves median
  rel-L2 from 0.115 to 0.094 at full budget. This makes PR-2 *more* decisively REFUTED,
  not less — tuned `c_ff` beats `q_serial` by ≈18× (vs.\ the original 14.8×), not the
  other direction. Seed variance at the tuned LR is large (0.013 vs.\ 0.174, nearly 14×
  apart) — one seed converges cleanly, the other doesn't; the median still holds.
- **`c_mlp`'s reduced-budget "winner" did NOT transfer to full budget — it made
  things worse.** `lr=0.003` looked better than the default at 1500 steps (1.54 vs.\
  1.97), but at the real 20000-step budget it is dramatically worse (median 2.36 vs.\
  the default's 0.877) — the higher rate that helped short-horizon convergence
  destabilized the long-horizon schedule. This is itself a real methodological lesson,
  not just a null result: tuning at one step budget does not reliably predict the
  optimum at another, so any future tuning pass must search at the actual comparison
  budget, not a cheaper proxy.
- **Net effect on PR-1/PR-2**: PR-2 (`c_ff`) is now REFUTED more strongly. PR-1
  (`c_mlp`) is unaffected in verdict — `q_serial` (1.702) still doesn't beat `c_mlp`
  by the required 2× margin either way — but the *default* `c_mlp` (0.877) turns out
  to be a better full-budget baseline than the naive "improvement" this sweep initially
  found, which is reassuring evidence against under-tuning in `c_mlp`'s specific case,
  for the opposite reason originally worried about.

Config: `configs/exp/baseline_tuning.yaml` (exploratory) plus a direct `run_all` call
at full budget for the winning configs. See `FINDINGS.md` F14.

## 5. Does the result survive dropping the best and worst seed?

**Cannot be meaningfully checked at this run's sample size.** Every core-matrix cell
has `n=2` seeds (cut from the pre-registered `n=5` under the Aug 7 deadline,
`08_TASK_INDEX.md`). Dropping the best and worst of 2 seeds leaves `n=0` (or, read
charitably as "drop one outlier," `n=1`) — not a meaningful robustness check, just a
different single-point estimate. This is a direct, structural consequence of the seed
cut (see `FINDINGS.md` F10 for the related p-floor finding) rather than a check that
was skipped by oversight. **Answer: the check is not well-defined at this sample size,
stated honestly rather than performed on a degenerate n=1 and reported as if
meaningful.**

## 6. Are the reference solutions right?

**Yes, already verified at the Phase 0 gate.** T0.12's Cole-Hopf Burgers cross-check
(two independent solution methods agreeing to `<1e-6`) and T0.10's analytic references
(agreeing with `pde.exact` to `1e-12`) were both part of the Phase 0 gate
(`phase0-complete` tag), which every subsequent result in this project depends on. Not
re-verified here since nothing downstream of Phase 0 touches how the reference
solutions themselves are computed.

## Does any of this change a T4.7 verdict?

**No PR verdicts change, and T4.9 (item 4, baseline tuning) is now resolved rather than
an open gap.** Item 1 (parameter count) rules out one specific alternative explanation
for PR-3's result without changing the verdict itself (already REFUTED). Item 3 and
item 6 are clean confirmations of existing verdicts. Item 4 was run to completion
(T4.9): `c_ff` was mildly under-tuned, but fixing it makes PR-2's REFUTED verdict
*stronger*, not weaker; `c_mlp`'s apparent improvement didn't survive being re-tested
at the actual comparison budget, leaving PR-1 unaffected. Items 2 and 5 remain real,
stated gaps — neither currently contradicts a reported verdict.
