# Phase 4 — Honesty Pass (Days 10–11)

**Goal:** attack our own results before a reviewer does, and adjudicate every claim.

**Exit gate (`project.md` §12 Phase 4):**
> Every claim in C1–C5 marked **confirmed / refuted / inconclusive**.

This phase is where the project's actual contribution is decided. `project.md` §0 states the expected
headline: *the quantum layer helps on high-wavenumber and multiscale problems, does nothing on smooth
diffusive problems, and hurts when the circuit spectrum is mismatched.* A design rule plus a
"when not to go quantum" recommendation is a stronger contribution than a cherry-picked win — so a
refuted claim here is a **deliverable**, not a failure.

---

## T4.1 — Statistics layer

**Depends on:** T3.10
**Files:** `src/qapinn/stats.py`, `tests/test_stats.py`

`project.md` §8: 5 seeds, report **median + IQR**, paired tests across families **on the same seeds**.
No single-run bar charts anywhere in the paper.

```python
def paired_comparison(runs_a: list[dict], runs_b: list[dict], metric: str) -> dict:
    """Pairs runs by seed. Returns
    {'median_a','median_b','iqr_a','iqr_b','ratio_median',
     'wilcoxon_stat','p_value','n_pairs','cliffs_delta'}
    Wilcoxon signed-rank (not a t-test — n=5 and no normality assumption).
    Cliff's delta as the effect size, because with n=5 a p-value alone is thin."""

def bootstrap_ci(values, statistic=np.median, n_boot=10000, alpha=0.05) -> tuple[float,float]
def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]
```

**Multiple-comparison discipline:** we test 12 pre-registered predictions. Apply Holm–Bonferroni
across the PR-set and report both raw and corrected p-values. State `n=5` honestly — with five seeds
the smallest achievable two-sided Wilcoxon p is 0.0625, so **no single comparison can reach p<0.05**.
Say this explicitly in the paper and lean on effect sizes and the *consistency of direction* across
problems rather than on significance stars. Getting this right is what separates this report from the
QPINN papers we are critiquing.

**DoD:** `tests/test_stats.py` — Wilcoxon matches `scipy.stats.wilcoxon` on a known input; bootstrap
CI covers the true median at the nominal rate on synthetic data.

---

## T4.2 — ★ The `c_rff_matched` adjudication (C1's real test)

**Depends on:** T4.1
**Files:** `scripts/make_figures.py` (fig: `ablation_matched`), section in `FINDINGS.md`

The sharpest ablation in the project (`project.md` §6, §13). Three-way paired comparison on every
problem: `q_serial` vs `c_rff_matched` vs `q_random`.

Interpretation table — write the conclusion that the data supports, not the one we hoped for:

| Outcome | Meaning |
|---|---|
| `q_serial` ≈ `c_rff_matched` ≫ `q_random` | **The benefit is spectral matching, not quantum.** SMCD survives intact as a *classical* design rule with a quantum instantiation. `project.md` §13 pre-frames this as the paper's most useful finding — report it as the headline. |
| `q_serial` ≫ `c_rff_matched` | The quantum layer contributes beyond matched frequencies. Investigate *why* (parameter efficiency? the `c_ω(θ)` coupling across frequencies?) and say what evidence supports the mechanism. |
| `q_serial` ≈ `q_random` | **C1 refuted.** The design rule buys nothing. Report it. |
| All ≈ `c_ff` | The whole spectral story is unsupported on this problem. Report it. |

**DoD:** figure + a written adjudication paragraph per problem, each naming the outcome row above.

---

## T4.3 — Negative-result map (C4)

**Depends on:** T4.1
**Files:** `scripts/make_figures.py` (fig: `decision_map`), `FINDINGS.md` decision table

Assemble across problems: for each, the SMCD a-priori prediction (`predicted_benefit` from the design
card, computed **before** training) against the measured benefit. The plot is
predicted-vs-measured — if SMCD's a-priori predictions track reality, the methodology has predictive
power, which is a much stronger claim than post-hoc correlation.

P2 (heat) is the anchor: SMCD predicted ≈ 2.5 % max benefit before any training (T2.8). Report
whether the measured benefit landed there.

Produce the **decision table** (`project.md` §10):

| PDE property | Recommendation | Settings | Evidence |
|---|---|---|---|
| Smoothing operator (parabolic, high ν) | **Do not** hybridise | — | P2 |
| Known discrete high-frequency support | Hybridise, ternary scalings | `L = ⌈log₃(2K/Δ+1)⌉` | P1, P4 |
| Near-resonant (Helmholtz, large k) | Hybridise, entangle coordinate wires | `n = d + 1`, `ring_cz` | P4 |
| Broadband growing spectrum (shocks) | Octave-split | `q_octave` | P3 |
| Spectrum unknown / dense low band | Use `c_ff`; quantum adds cost, not reach | linear scalings | α sweep |

**DoD:** figure + completed decision table with an evidence column citing specific figures.

---

## T4.4 — Noise-surrogate validation (D8's honesty debt)

**Depends on:** T3.5
**Files:** `tests/test_noise_vs_density_matrix.py`, section in `FINDINGS.md`

Our depolarizing model is an analytic surrogate (D8). Validate it: for a small case
(`n = 4`, `L = 3`, 200 random points), compare against PennyLane `default.mixed` with **local**
depolarizing channels on every wire after every layer.

**DoD:** the relative discrepancy is measured and **reported in the paper**, not hidden. If it exceeds
10 %, downgrade every noise conclusion to qualitative and say so explicitly.

---

## T4.5 — Barren-plateau frontier

**Depends on:** T3.5
**Files:** `scripts/make_figures.py` (fig: `barren_frontier`)

From the depth/qubit sweep: `log Var[∂_θ L]` vs `n` for each `L`, with the theoretical `2^{−n}` line
overlaid (`project.md` §7.5). This is the **cost side of the ledger** and belongs in the
recommendations.

Report the practical frontier: the largest `(n, L)` at which training still converges, and the
gradient variance at that point.

**DoD:** figure + a stated frontier, e.g. "training remains reliable to `n ≤ 8`, `L ≤ 5` with local
observables and small-angle init; beyond that, use the octave split."

---

## T4.6 — Cost ledger

**Depends on:** T3.5
**Files:** `results/cost_ledger.json`, table in the paper

`project.md` §6: "fewer parameters" is a claim QPINN papers make loosely and we should make it
precisely. Per family per problem: parameter count, FLOPs/step, wall-clock/step, circuit evaluations,
total training time, and rel-L2 achieved.

Also report **error at matched wall-clock**, not just at matched parameters. A model that needs 40×
the wall-clock per step is not "more efficient" because it has fewer parameters, and saying so is part
of the honesty pass.

**DoD:** `results/cost_ledger.json` + a paper-ready table.

---

## T4.7 — ★★ Claim adjudication

**Depends on:** T4.2, T4.3, T4.4, T4.5, T4.6
**Files:** `FINDINGS.md` (adjudication table)

For each of C1–C5 and each of PR-1…PR-12: **confirmed / refuted / inconclusive**, with the deciding
figure, the effect size, and the corrected p-value.

| Claim | Falsifier (`project.md` §2) | Verdict | Evidence |
|---|---|---|---|
| C1 SMCD design rule | matched circuit does not beat size-matched random-frequency circuit | ? | PR-6, PR-9, T4.2 |
| C2 Quantum-NTK block decomposition | eigenspectra statistically indistinguishable after size-matching | ? | PR-7, T3.8 |
| C3 Spectral-bias relief is the mechanism | improvement spread uniformly across ω, or outside `Ω` | ? | PR-8, T3.9 |
| C4 Negative-result map | — (constructive) | ? | T4.3 |
| C5 XAI protocol | — (constructive) | ? | whole instrument suite |

**Rule:** a verdict of *inconclusive* is acceptable and must be used where `n=5` cannot separate the
alternatives. Do not upgrade inconclusive to confirmed on the strength of a favourable median.

**DoD:** the table is complete with no `?` remaining.

---

## T4.8 — Adversarial self-review

**Depends on:** T4.7
**Files:** `docs/self_review.md`

Deliberately try to break our own conclusions. Minimum checklist:
- Is any improvement explained by parameter count rather than spectrum? (Check against T2.16.)
- Is any improvement explained by the *linear head* alone? Fit a linear head on frozen random
  features of the same dimension as a control.
- Did the encoder drift (T1.8) invalidate the coverage numbers? (PR-12.)
- Are the classical baselines properly tuned? An under-tuned `c_ff` would fake a quantum win.
  **Run a small learning-rate/width sweep on `c_ff` and use its best configuration**, not its default.
- Does the result survive dropping the best and worst seed?
- Are the reference solutions right? (T0.12 cross-check.)

**DoD:** each item answered in writing with the evidence. Any that changes a verdict sends T4.7 back
for revision.

---

## T4.9 — Baseline fairness re-run (if T4.8 finds under-tuning)

**Depends on:** T4.8
**Files:** `configs/exp/baseline_tuning.yaml`

If the `c_ff` tuning sweep finds a materially better configuration, **re-run the core matrix rows for
the classical families with it** and re-adjudicate. Budget: ~40 runs, half a day.

**DoD:** either "no change needed, tuning sweep found nothing better" recorded in `docs/self_review.md`,
or the re-run completed and T4.7 revised.

---

## T4.10 — Phase 4 gate check

**Depends on:** T4.7, T4.8, T4.9

Confirm: every C1–C5 and PR-1…PR-12 adjudicated; noise surrogate discrepancy reported; cost ledger
complete; self-review complete with no unanswered item.

**DoD:** commit tagged `phase4-complete`.
