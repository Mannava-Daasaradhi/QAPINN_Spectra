# Findings

**Pre-registration:** `docs/predictions.md`, commit `92abd7d60c34fc86abc529769dd2d8bde6dfb575`
(2026-08-02T21:00:03+05:30). Mechanically verified (T3.10) to predate every completed
run's `provenance.json` `git_sha` in the real T3.1-T3.6 experiment matrix.

> **Post-submission errata (2026-09-25).** The version judged at the WISER 2026 BQP
> Challenge is tag `v1.0-submission`. After the competition we found two stale or
> mis-computed numbers (F3, F4), wrong frequency bands in the paper figures, and four
> caveats that the findings below understate (F1, F2, F5, F13). **All 15 pre-registered
> verdicts are unchanged.** Details and reproduction commands:
> [Post-submission errata](#post-submission-errata-2026-09-25).

## Headline finding

At the sample size this run achieved (**n=2 seeds, not the pre-registered n=5** — cut
under the Aug 7 deadline, `docs/plan/08_TASK_INDEX.md`), the quantum-advantage hypothesis
SMCD was built to test is **not supported**: on Poisson, the one problem with a full
ablation, `q_serial` underperforms every classical comparator tested, and the two
mechanistic explanations proposed for *why* it should help (NTK-block flattening, C2;
spectral-bias relief, C3) point the wrong direction or hold on only one of two checked
instances. What does hold up: the negative-result mechanism on the smoothing control
(C4, mostly), one genuine per-frequency win (C3, Poisson only), and — most solidly — the
measurement protocol itself (C5), which produced a falsifiable, mechanically-checked
verdict for all 12 pre-registered predictions with no `?` left, including catching real
bugs in its own supporting tooling along the way.

## Findings

**F1 — Coverage predicts accuracy on Poisson, not on Helmholtz\_k10 (PR-9).**
Figure: `paper/figures/coverage_vs_error.pdf`. Poisson: Spearman ρ=−0.82 (p=3.2×10⁻⁵,
n=18 points, 6 coverage targets × 3 seeds — the one sweep never seed-cut). Helmholtz\_k10:
ρ=+0.28 (p=0.27, wrong sign). **Confidence: medium** on Poisson alone (clean, significant,
large-n signal); **verdict REFUTED overall** since the adjudication rule takes the
worst case across problems. Caveat: two problem instances is a thin basis for "coverage
predicts accuracy" as a general claim; the mechanism may be problem-dependent in a way
this run cannot characterize further. *Post-submission: the 18 Poisson points are 9
distinct measurements from a reduced training budget, and the ordering reverses at full
budget — see erratum E4 before citing ρ or p.*

**F2 — `q_serial` underperforms every classical family checked, on Poisson, at n=2
(PR-1, PR-2, PR-3).** Figure: `paper/figures/ablation_matched.pdf`. Median rel-L2:
`q_serial` 1.702, `c_mlp` 0.877, `c_ff` 0.115, `c_rff_matched` 0.400. **Confidence: low**
— n=2 seeds, no paired test can reach significance at this sample size (see F10). Caveat:
direction is consistent (q\_serial loses to all three), which is at least not
random-looking, but consistency across 2 points is weak evidence on its own.
*Post-submission: `c_mlp` and `c_ff` were not size-matched to `q_serial` (8,513 and 129
parameters vs 14) — see erratum E6.*

**F3 — NTK spectrum is steeper, not flatter, inside the encoded band, where
measurable (PR-7).** Figure: `paper/figures/ntk_spectrum_{poisson,helmholtz_k10}_pr7.pdf`,
read from each run's committed `xai/ntk_step*.npz` (an earlier version of this check
reconstructed models from `results/runs/*/checkpoints/*.pt`, which is deliberately
gitignored — that made the check silently unreproducible from a clean clone; caught by
actually running the T5.14 clean-clone verification, not by inspection). Helmholtz\_k10:
gap (outside-exponent minus inside-exponent, predicted ≥+0.5) = −8.37 (stated as −9.67
in v1.0; erratum E1), REFUTED. Poisson:
**INSUFFICIENT\_DATA**, not REFUTED — fewer than 2 positive eigenvalues fall in the
outside-band index range of this run's committed NTK spectrum, so the outside-band decay
exponent is undefined there, not just noisy; reported honestly rather than silently
letting `NaN >= 0.5` evaluate to `False` and report REFUTED with no stated reason (the
same failure shape F4/PR-8 already had, before its own fix). **Confidence: medium** on
Helmholtz\_k10 — direct spectral measurement, not seed-noise-limited, but one
representative run per family, not seed-averaged; **n/a** on Poisson.

**F4 — Per-frequency improvement concentrates in Ω on Poisson; there is no improvement
to attribute on Helmholtz\_k10 (PR-8).** Figure:
`paper/figures/freq_heatmap_{poisson,helmholtz_k10}.pdf`. Poisson: 99.9% (stated as
71.3% in v1.0; erratum E2) of a real, positive total improvement (13,751 accumulated
units) lands inside Ω — **CONFIRMED**, well past the 70% bar. Helmholtz\_k10: `q_serial` beats `c_mlp` at zero frequencies
(total\_improvement=0) — REFUTED, and not narrowly: there is nothing for the mechanism
to explain. **Confidence: medium-high on Poisson** (clean measurement, large effect);
n/a on Helmholtz\_k10.

**F5 — P4's k-scaling trend is directionally confirmed, but driven by catastrophic
`q_serial` failure at k=4, not a growing advantage (PR-5).** Figure:
`paper/figures/decision_map.pdf`; ratios (`c_mlp`/`q_serial`): k=4: 8.79×10⁻⁶, k=10:
0.918, k=20: 1.024. **Confidence: medium** on the direction (strictly monotone, clean);
**low** on the interpretation as "quantum advantage grows with k" — at k=20 the ratio is
still ≈1 (parity, not advantage), and the trend is dominated by how badly `q_serial`
fails at k=4 (error ~5 orders of magnitude larger than `c_mlp`'s), not by a genuine
crossover into quantum outperforming classical. *Post-submission: at k=10 and k=20 every
family's median rel-L2 is 0.977–1.089, i.e. nothing learned either solution — see
erratum E5.*

**F6 — P2's negative-result mechanism holds for 3 of 4 quantum families; `q_serial`
itself is the one quantum exception (PR-4).** Measured benefit over `c_mlp`: `q_serial`
+7.7%, `q_parallel` −67.5%, `q_random` −1023%, `q_octave` −51.2%; two classical families
also beat `c_mlp` (`c_ff` +24.4%, `c_rff_matched` +34.9%, unrelated to the quantum-specific
claim). **Confidence: medium** — the literal 5% uniform-threshold check is REFUTED (three
families cross it), but the per-family breakdown supports the underlying mechanistic
argument (smoothing operators give an unmatched, high-frequency-rich circuit nothing to
earn its keep on) better than the single pre-registered threshold could show on its own.

**F7 — Hybrid models show far smaller NTK drift than classical ones at 5k steps — the
opposite of the predicted direction (PR-11).** Median drift: hybrid 1.03, classical
6,878,097 (n=2 each; raw per-seed values checked directly — both classical seeds are in
the millions, both hybrid seeds are ≈1.0, so this is not a single-run outlier artifact).
**Confidence: low-medium** — consistent across both available seeds, but the sheer
magnitude (millions, not just "larger") raises a real alternative explanation this run
did not chase down: `‖Θ_t−Θ_0‖/‖Θ_0‖` blows up if `‖Θ_0‖` happens to be very small at
initialization for `c_mlp` specifically, independent of how much the kernel actually
moved in absolute terms. Reported as REFUTED (correctly, by the pre-registered metric)
with this normalization caveat attached rather than read as an uncomplicated "hybrid is
6.7-million-times lazier" claim.

**F8 — Encoder drift stays well under the validity threshold (PR-12).** Median
`‖A−I‖_F`=0.132 < 0.2, n=2. **Confidence: high** — clean threshold check, and this is
the one result every frequency-domain claim above (F1, F3, F4) structurally depends on:
if this had failed, the Ω/coverage claims computed at initialization would not describe
the trained model, and F1/F3/F4 would need that caveat attached. It did not fail.

**F9 — The barren-plateau decay rate is unmeasurable on this run's data, and is reported
as such rather than computed anyway (PR-10).** Figure: `paper/figures/barren_frontier.pdf`
(the practical frontier itself — largest n before grad-var collapses below the 10⁻¹⁰
floor — is unaffected and still shown). `depth_sweep`'s grid, cut to `n_qubits=6` only
under the same deadline pressure, has zero variation in the fit's own independent
variable; `barren_plateau_fit` now detects this and refuses to report a slope rather
than return `numpy.polyfit`'s ill-conditioned solver artifact. **Confidence: n/a
(explicitly not measured).**

**F10 — At n=2 seeds, CONFIRMED is structurally unreachable for every ratio-and-
significance prediction (PR-1, PR-2, PR-6), independent of true effect size.** The
smallest two-sided Wilcoxon p achievable at n=2 paired differences is 0.5 (verified
directly against `scipy.stats.wilcoxon`: both seeds agreeing in sign — the most extreme
possible n=2 outcome — still returns p=0.5, not the n=5 plan's 0.0625).
`docs/predictions.md`'s CONFIRMED gate was left at its original, pre-registered 0.0625
value rather than loosened to compensate. **Confidence: high** (mathematical property of
the test, not a data-dependent claim). This is the single most important lens for reading
every REFUTED/INCONCLUSIVE verdict above: some are REFUTED because the ratio itself points
the wrong way (F2, F3, F4/Helmholtz), and some are INCONCLUSIVE-by-construction because no
amount of true effect size could have cleared this sample size's floor (PR-6 on both
problems).

**F11 — Under soft BCs, the residual loss's NTK block dominates the boundary block by
3-6 orders of magnitude, in every family/problem combination checked (T3.6, D6).**
Figure: none dedicated (`results/runs/*/xai/block_mass_step*.npz`, 12/12 runs, `n=3`
seeds — the one part of this sweep not cut on seed count). `trace_rr` (residual) vs.
`trace_bb` (boundary) at the final checkpoint: Poisson/`c_mlp` ≈5,000×, Poisson/`q_serial`
≈1,000,000×, Helmholtz\_k10/`c_mlp` ≈1,500×, Helmholtz\_k10/`q_serial` ≈430×.
**Confidence: high** — consistent across every one of the 4 (problem, family)
combinations and all 3 seeds each, with no sign flips or near-parity cases. This is
direct empirical evidence for the standard PINN soft-BC pathology motivating this
project's own choice of hard-constrained BCs everywhere else (`u = B(x) + D(x)\cdot[...]`,
Section~4 of the paper): under a soft penalty, gradient descent's NTK-driven training
dynamics pay overwhelmingly more attention to the residual term than to the boundary
term, so nothing here should be read as "soft BCs converge, just slower" — the block
imbalance suggests the boundary term is close to invisible to training at these
magnitudes, independent of how long training runs.

**F12 — The depolarizing-noise surrogate matches a real density-matrix simulation to a
3.4% median relative error, well under the 10% "downgrade to qualitative" bar — with a
large-magnitude tail worth flagging (T4.4).** `GlobalDepolarizing`'s single global
`(1-p)^L` factor vs. PennyLane's `default.mixed` device applying local
`DepolarizingChannel`s after every layer (`n=4`, `L=3`, `p=0.01`, `N=200` random draws,
`tests/test_noise_vs_density_matrix.py`): median relative error 3.41%, max 1058%.
**Confidence: medium** — the median is a clean pass, but the max is not a rounding
artifact to dismiss outright; it is most likely a **relative-error-near-zero** effect
(a handful of draws where the true noisy expectation value crosses near 0, inflating any
relative-error ratio regardless of how small the absolute discrepancy is), not evidence
the surrogate is failing on a meaningful fraction of realistic inputs — but this was not
separated out from a genuine large-discrepancy case this run, so it is reported as an
open question rather than resolved. The shot-noise surrogate (the one actually used in
`noise_study.yaml`, since the depolarizing sweep arm was cut — cut-line item #3) was not
separately validated against a density-matrix reference this run.

**F13 — PR-3's matched-capacity equivalence claim generalizes: REFUTED on 4 of 6
problems, but genuinely CONFIRMED on the two hardest instances (T4.2).** Figure:
`paper/figures/ablation_matched.pdf`. `q_serial`/`c_rff_matched` median rel-L2 ratio
per problem: Poisson 4.25× (REFUTED), heat 1.42× (REFUTED), burgers 3.58× (REFUTED),
Helmholtz\_k4 8.97× (REFUTED — same catastrophic k=4 failure as F5), Helmholtz\_k10
1.09× (**CONFIRMED**, inside the [0.77,1.3] band), Helmholtz\_k20 0.98×
(**CONFIRMED**, `q_serial` marginally ahead). **Confidence: medium** — the direction
(low-k: classical wins decisively; high-k: parity) is consistent with F5's k-scaling
trend and not a single-point artifact, but every individual comparison is still an
n=2 read. This is the one place in this run's data where "no longer a clear quantum
disadvantage" (not the same as an advantage) shows up, and only the single-problem
(Poisson) view in the original T4.2 scope would have missed it entirely.
*Post-submission: the two CONFIRMED instances are parity between models that all
failed (rel-L2 ≈ 1) — see erratum E5.*

**F14 — Baseline-tuning check (T4.9) reinforces PR-2's REFUTED verdict and leaves PR-1
unaffected, for two different reasons.** A 12-config exploratory grid
(`configs/exp/baseline_tuning.yaml`: 3 learning rates × 2 widths, `c_mlp`/`c_ff`,
Poisson, reduced budget) found each family's best configuration; the winner was then
re-run at `core_matrix`'s real full budget (20000+2000 steps) and seeds {0,1} — the
actual comparison PR-1/PR-2 are adjudicated against. **`c_ff`** was mildly under-tuned:
`lr=0.003` improves median rel-L2 from 0.115 to 0.094, making its win over `q_serial`
(1.702) *larger* (≈18×, up from 14.8×) — PR-2 REFUTED more decisively, not less.
**`c_mlp`**'s reduced-budget "winner" (`lr=0.003`, which looked better at 1500 steps)
was dramatically *worse* at the real 20000-step budget (median 2.36 vs.\ the default's
0.877) — a higher rate that helps short-horizon convergence destabilizes the long
schedule. PR-1's verdict is unaffected (`q_serial` still doesn't clear the 2× bar
either baseline), but the finding rules out under-tuning as the reason, for the
opposite of the originally-suspected direction. **Confidence: medium-high** — this is a
direct, full-budget, apples-to-apples re-measurement (not an estimate or a proxy-budget
extrapolation), though still `n=2` seeds and one problem (Poisson) only. **Methodological
lesson, stated explicitly**: hyperparameters tuned at a reduced step budget do not
reliably transfer to a longer one — any future tuning pass must search at the actual
comparison budget.

## C1–C5 adjudication table

| Claim | Status | Deciding evidence |
|---|---|---|
| C1 — SMCD constructive map | **Not supported at this sample size, one qualified exception** | Falsifier PR-6 INCONCLUSIVE (both problems, n=2 floor — F10); direct ablation PR-1/2/3 REFUTED on Poisson (F2), and confirmed not an under-tuned-baseline artifact (T4.9, F14); generalized to all 6 problems, REFUTED on 4, but genuinely CONFIRMED (statistical parity, not advantage) on Helmholtz_k10/k20 — the two highest-wavenumber instances (F13) |
| C2 — Quantum-NTK block flattening | **Reversed where measurable** | PR-7 REFUTED on Helmholtz_k10 — spectrum steeper, not flatter, inside Ω; Poisson INSUFFICIENT_DATA (undefined outside-band exponent, not computed) (F3) |
| C3 — Spectral-bias relief mechanism | **Holds on Poisson only** | PR-8 CONFIRMED (Poisson, 99.9% — 71.3% in v1.0, erratum E2), REFUTED (Helmholtz_k10, 0% — no improvement exists) (F4) |
| C4 — Negative-result map | **Mechanism partially right, uniform threshold REFUTED** | PR-4 REFUTED (3 families cross 5%), but 3 of 4 *quantum* families underperform as predicted; only `q_serial` is a quantum exception (F6) |
| C5 — XAI protocol | **Demonstrated** | All 12 predictions mechanically adjudicated from committed artifacts, no `?` left; caught real bugs in its own tooling (PR-8 NaN mishandling, PR-10 degenerate fit, `specerr.npz` corruption, PR-7 silently unreproducible from a clean clone + a second masked NaN) rather than silently propagating them |

## Decision table: when to (not) go quantum

Based on this run's actual, adjudicated evidence — not the pre-registered predictions'
intent:

| PDE property | Recommendation | Basis |
|---|---|---|
| Purely diffusive / smoothing (e.g. heat, P2) | **Don't.** 3 of 4 quantum families measurably underperform `c_mlp`, one by an order of magnitude. | F6 |
| High-wavenumber, `q_serial` specifically | **Not yet demonstrated to help** at the sample size available; underperformed every classical comparator on the one fully-ablated instance (Poisson). Re-test at n=5 before drawing a conclusion either way. | F2, F10 |
| Any problem, if only 1-2 seeds are affordable | **Don't rely on a CONFIRMED verdict either way** — the statistical test cannot distinguish real effects from noise at n=2 (F10). Report ratios and directions, not significance. | F10 |
| Circuit-architecture scaling questions (barren plateau, depth) | **Ensure the sweep grid retains variation in every axis the fit needs** — this run's own `n_qubits` cut made PR-10 unmeasurable after the fact. | F9 |

## Recommendations

1. Re-run the core matrix at the originally planned n=5 seeds before treating any REFUTED
   or INCONCLUSIVE verdict above as final — F10 shows n=2 cannot support a CONFIRMED
   verdict regardless of true effect size, so several REFUTED results may simply be
   underpowered rather than genuinely negative.
2. Restore `n_qubits` variation in `depth_sweep` (at minimum, one cheap `n_qubits=4`
   point) so PR-10 can be measured, not just correctly reported as unmeasurable (F9).
3. Investigate F7's NTK-drift ratio for a possible small-`‖Θ_0‖` normalization artifact
   before citing the raw 6.7-million-times figure in any follow-on claim.
4. ~~Run T4.9's baseline-tuning sweep~~ **Done (F14).** `c_ff` was mildly under-tuned;
   fixing it strengthens PR-2's REFUTED verdict. `c_mlp`'s reduced-budget "improvement"
   did not transfer to full budget and made things worse there, leaving PR-1 unaffected.
   Extend to `helmholtz_k10`/`k20` (F13's two CONFIRMED-parity instances) next, where a
   genuine baseline improvement could plausibly flip PR-3's per-problem verdict, unlike
   Poisson where both PR-1/PR-2 were REFUTED by a wide enough margin that this check
   couldn't realistically flip them.

## What we would do next with more time

Re-run the full core matrix and `depth_sweep` at the pre-registered n=5 seeds and
restored `n_qubits` grid, so F13's already-generalized 6-problem ablation and every
other n=2 finding above can be re-checked with real statistical power; chase down F7's
drift-normalization question directly from the raw `‖Θ_0‖`/`‖Θ_t‖` values rather than
the ratio alone; separate F12's
max-error tail (1058%) from its typical case to confirm it is a relative-error-near-zero
artifact and not a genuine surrogate failure mode; and validate the shot-noise surrogate
(the one arm actually used in `noise_study.yaml`) against a density-matrix reference —
only the depolarizing surrogate was checked this run (F12).

## Post-submission errata (2026-09-25)

Added after the WISER 2026 BQP Challenge; the judged version is tag `v1.0-submission`
(`d20e961`). The pre-registered thresholds and all 15 verdicts are unchanged
(`python scripts/adjudicate_predictions.py`). Every number below was re-derived from the
committed `results/runs/` data, not copied from earlier notes. E1–E3 are corrections;
E4–E6 are caveats the findings above understate.

**E1 — PR-7 (F3): the Helmholtz\_k10 gap is −8.37, not −9.67.** The −9.67 predates
`16ff015`, which moved PR-7 from reloading gitignored checkpoints to the committed
`xai/ntk_step*.npz`; the committed adjudicator has reported −8.37 since then. Still
REFUTED (bar ≥ +0.5). Corrected in F3 and the paper.

**E2 — PR-8 (F4): 99.9% of Poisson's improvement lies inside Ω, not 71.3%.** Poisson's
error spectrum sits on a two-sided FFT axis, and Ω holds both +ω and −ω, but the
overlap code folded Ω onto |ω|. So the mirrored negative half of Ω counted as "outside".
The improvement mass is symmetric: 3,943.6 at ω < 0, 3,943.6 at ω > 0 and 5,863.6 at
ω = 0. Counting both halves, 13,742 of 13,751 units (99.94%) fall inside Ω. The verdict
is the same (CONFIRMED, bar ≥ 70%), but it is not the marginal pass 71.3% suggested.
Fixed in `scripts/make_figures.py::_omega_lines_from_design_card`, with a regression test
in `tests/test_phase3_figures.py`. The other five problems use a radially binned axis
(64 bins, about 35 rad/unit wide), too coarse to resolve Ω. For Helmholtz all of Ω falls
in the first bin, so their heatmaps show where the error sits, not Ω's shape. No verdict
depends on this: Helmholtz\_k10's PR-8 is REFUTED because no improvement exists at any
frequency.

**E3 — The paper figures marked the wrong band.** `tasks.py figures` passed the design
card's four encoding scalings (π, 3π, 9π, 27π) as Ω instead of the realised frequency set
(81 frequencies for Poisson). As a result, every `freq_heatmap_*` overlay and the shaded
band in `ntk_spectrum_p1`/`p4` were wrong in v1.0. Adjudication always used the correct
Ω, so no number or verdict was affected. The adjudicator and `tasks.py figures` also
wrote the same file names with different content, and the PDFs embedded a timestamp.
Figures are now regenerated from one Ω source, byte-for-byte reproducibly.

**E4 — PR-9 (F1): the coverage evidence is thinner than "n=18, p=3.2×10⁻⁵" and depends
on training budget.**
- `coverage_sweep` trained at 1500+150 steps, not the core matrix's 20000+2000
  (`configs/exp/coverage_sweep.yaml`); F1 did not say so.
- On Poisson, SMCD's coverage de-tuning builds only three circuits. Targets 0.4 and 0.6
  give the same 8-parameter circuit (1 qubit, 1 layer), and 0.8, 0.9 and 1.0 give the
  same 14-parameter circuit (1 qubit, 4 layers), with bit-identical rel-L2. Target 0.2
  gives a third 8-parameter circuit. The 18 points are therefore 9 distinct measurements
  (3 circuits × 3 seeds), and the p-value treats the duplicates as independent. Coverage
  is also confounded with parameter count (8 vs 14).
- What holds at this budget: the full-coverage circuit beats every under-covering one on
  every seed (rel-L2 0.718–0.729 vs 0.909–1.084).
- At the full 20000+2000 budget the order reverses on 2 of 3 seeds. The 18 superseded
  full-budget runs are still in `results/runs/`. There the 1-layer circuits reach
  0.66–0.70 on seeds 0–1 (and diverge to 11–21 on seed 2), while the 4-layer circuit
  stays at 1.54–1.58.
- On Helmholtz\_k10 all six targets build the same 20-parameter circuit (3 qubits, 1
  layer), and every run scores 0.991–1.002. Its ρ = +0.28 is rank noise among
  near-identical failures.

**E5 — PR-5 and PR-3 (F5, F13): "parity" at Helmholtz k=10 and k=20 is parity among
failures.** On these two instances every family's median rel-L2 is 0.977–1.089. A model
that outputs u ≡ 0 scores exactly 1.0, so no family learned either solution. F13's two
CONFIRMED verdicts (`q_serial` ≈ `c_rff_matched`) and F5's ratios near 1 reflect shared
failure, not matched capability. They should not be read as "the quantum model catches up
at high k". SMCD's own a-priori `predicted_benefit` is 0.0 at k=4 and k=10 (and 1.0 at
k=20; `results/design_cards.json`).

**E6 — F2: the classical baselines were not size-matched.** On Poisson, `q_serial` has
14 parameters, `c_mlp` 8,513, `c_ff` 129 and `q_parallel` 1,166. Only `c_rff_matched`
(13), `q_random` (14) and `q_octave` (13) are within ±10% of `q_serial`; other problems
show the same pattern. `results/size_matching.json` shows that matched configurations
exist for every family, but the core matrix trained each family at its default size
(`configs/model/*.yaml`). The capacity-matched comparison on Poisson is `q_serial` vs
`c_rff_matched` (PR-3): 1.702 vs 0.400.

**Reproducibility notes (no effect on any number).** F14's four full-budget
confirmation runs were launched without a config file. `configs/exp/baseline_tuning_confirm.yaml`
now reproduces their exact run IDs, and a test pins them. The twelve
`baseline_tuning` runs record a commit (`d4c6877`) that never reached the published
history. Pre-registration ordering is therefore verified for the 181 runs of the six
pre-registered experiments (`python scripts/verify_preregistration.py`: 0 violations);
T4.9 was a post-adjudication fairness check, not a prediction.
