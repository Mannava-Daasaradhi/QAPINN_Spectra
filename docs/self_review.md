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

**Real gap — no `c_ff` learning-rate/width sweep was run.** Every family in this run
used its default configuration. This is a genuine unanswered risk: an under-tuned
`c_ff` would understate the classical baseline and could make a real `q_serial`
disadvantage look smaller than it should, or (more relevant to what was actually
measured) could make `c_ff`'s already-substantial win over `q_serial` (PR-2:
`c_ff` beats `q_serial` by 14.8× on Poisson) look artificially large if `c_ff` happens
to be over-tuned by luck of its default hyperparameters — though there is no specific
evidence of that here, only the absence of a check that would rule it out. **Not done;
this is the item most likely to change a verdict if run** (T4.9's own trigger
condition), specifically PR-1/PR-2's REFUTED verdicts, which rest on `c_mlp`/`c_ff`
beating `q_serial`, not the reverse.

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

**No verdict changes as a result of this review.** Item 1 (parameter count) rules out
one specific alternative explanation for PR-3's result without changing the verdict
itself (already REFUTED). Item 3 and item 6 are clean confirmations of existing
verdicts. Items 2, 4, and 5 are real, stated gaps — none of them currently contradicts
a reported verdict, but item 4 (baseline tuning) is the one most likely to matter if
pursued: T4.9 is triggered by exactly this finding, and the recommended next step
(`FINDINGS.md`'s Recommendations) is to run it before treating PR-1/PR-2's REFUTED
verdicts as final.
