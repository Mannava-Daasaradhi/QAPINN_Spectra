# Task Index — Chronological Execution Order

**88 tasks.** This ordering is a topological sort of the dependency DAG. **Execute top to bottom.**
Full task descriptions live in the phase documents; this is the checklist and dependency reference.

**Legend:** ★ = load-bearing · ★★ = hard gate, do not proceed on failure · ⚡ = parallelisable
side-branch (safe to work on while a long run executes).

---

## How the implementing agent should work

1. Read `00_MASTER_PLAN.md` §3 (decisions D1–D13) and all of `01_CONVENTIONS.md` before task 1.
2. Take the next unchecked task. Open its phase document and read the full description.
3. Implement. Run the DoD check. **The DoD check must actually execute and pass.**
4. Commit as `T<id>: <subject>`. Tick the box here.
5. At a ★★ gate: stop, verify, and report status before continuing.
6. If blocked, report which task and why — do not skip ahead past a dependency, and do not weaken
   a DoD to make it pass.

**Resuming after a context reset:** the ticked boxes in this file plus `git log` are the source of
truth for progress. Re-read `00_MASTER_PLAN.md` §3 and `01_CONVENTIONS.md` before continuing.

---

## Phase 0 — Foundations (Day 1–2) · `02_PHASE0_foundations.md`

| # | Task | Depends on | DoD in one line |
|---|---|---|---|
| ☑ T0.1 | Init repo + directory skeleton | — | `git status` clean; tree exists |
| ☑ T0.2 | Pin environment with `uv` (Python 3.12, torch cu128, PennyLane) | T0.1 | `torch.cuda.is_available()` is True; `ENV_RESOLVED.md` written |
| ☑ T0.3 | Task runner `tasks.py` + `Makefile` mirror | T0.2 | `python tasks.py test` exits 0 |
| ☑ T0.4 | Config schema, composition, `run_id` hashing | T0.2 | key-order-invariant hash; dotted overrides work |
| ☑ T0.5 | Determinism utils + device/dtype resolution | T0.2 | same seed ⇒ bit-identical tensors |
| ☑ T0.6 | `PDE` ABC + `Domain` | T0.4, T0.5 | dummy subclass instantiates; `eval_grid` shape correct |
| ☑ T0.7 | Autograd differential operators | T0.6 | matches closed forms to 1e-10; `create_graph` verified |
| ☑ T0.8 | ★ The four PDEs (P1–P4) with all constants | T0.7 | **exact solution satisfies its own residual to 1e-8** |
| ☐ T0.9 | FFT convention test (D12 guard) | T0.8 | peak of `sin(15πx)` at `ω=15π`; factor-of-2 pinned |
| ☐ T0.10 | Analytic reference solutions | T0.8 | agrees with `pde.exact` to 1e-12 |
| ☐ T0.11 | Pseudospectral solver + MMS convergence | T0.10 | spectral in space, order ≥ 3.8 in time |
| ☐ T0.12 | ★ Cole–Hopf Burgers + cross-check | T0.11 | two independent methods agree < 1e-6 |
| ☐ T0.13 | Reference solution cache + registry | T0.12 | second call hits cache |
| ☐ T0.14 | `PINNModel` ABC + `match_param_count` | T0.5 | matches a 3000-param target within 10 % |
| ☐ T0.15 | `c_mlp` | T0.14 | forwards `[B,d]→[B,1]`; empty quantum group |
| ☐ T0.16 | `c_ff` + `c_rff_matched` | T0.15 | matched basis fits `sin(πx)+0.3sin(15πx)` to 1e-8 |
| ☐ T0.17 | Loss assembly (hard + soft BC) | T0.8, T0.15 | exact solution ⇒ loss < 1e-10 |
| ☐ T0.18 | ★ Training loop (Adam → LBFGS, checkpoints) | T0.17 | `run --smoke` < 60 s, full artifact dir |
| ☐ T0.19 | Checkpointing + provenance | T0.18 | reload reproduces `rel_l2` to 1e-12 |
| ☐ T0.20 | Determinism test | T0.19 | two runs ⇒ bit-identical `metrics.json` |
| ☐ T0.21 | ★★ **Performance spike (budget gate, D13)** | T0.18 | projection recorded; **cut matrix now if > 24 h** |
| ☐ T0.22 | ★★ **Phase 0 gate** | T0.9, T0.11, T0.12, T0.20, T0.21 | 5 exit criteria; tag `phase0-complete` |

---

## Phase 1 — Instruments (Day 3–4) · `03_PHASE1_instruments.md`

| # | Task | Depends on | DoD in one line |
|---|---|---|---|
| ☐ T1.1 | Plot style + `save_figure` + manifest | T0.22 | emits `.pdf`+`.png`, appends manifest entry |
| ☐ T1.2 | ★ NTK core (Jacobians, blocks, spectrum stats) | T0.19 | linear-model check; **Prop. 4 identity to 1e-10**; PSD |
| ☐ T1.3 | NTK reporting + `ntk_spectrum` figure | T1.2, T1.1 | classical decay exponent recorded as baseline |
| ☐ T1.4 | ★ Per-frequency error trajectory | T0.9, T0.19 | two synthetic peaks recovered; Parseval holds |
| ☐ T1.5 | ★★ **Spectral-bias staircase (Phase-1 gate figure)** | T1.4, T1.1 | staircase visible; **steps-to-tol ratio ≥ 10** |
| ☐ T1.6 | ⚡ Integrated-gradients attribution | T0.19 | linear-model exactness; completeness < 1e-6 |
| ☐ T1.7 | ⚡ Fisher + effective dimension | T0.19 | recovers `k` for a `k`-feature linear model |
| ☐ T1.8 | ⚡ Encoder spectral drift (new, from D3) | T0.19 | identity ⇒ drift 0; scaling `s` ⇒ `s·Ω` |
| ☐ T1.9 | ⚡ Gradient variance / barren-plateau tracking | T0.19 | classical MLP shows slope ≈ 0 |
| ☐ T1.10 | ⚡ Layerwise probes + CKA | T0.19 | input-layer `R²` low, final-layer high |
| ☐ T1.11 | ⚡ Loss-landscape slices | T0.19 | minimum at centre for a converged model |
| ☐ T1.12 | XAI hook registry wired into training loop | T1.2–T1.11 | smoke run with all instruments < 120 s |
| ☐ T1.13 | ★★ **Phase 1 gate** | T1.5, T1.12 | staircase + all instruments on 3 classical families; tag `phase1-complete` |

---

## Phase 2 — Theory + SMCD (Day 5–7) · `04_PHASE2_theory_smcd.md`

| # | Task | Depends on | DoD in one line |
|---|---|---|---|
| ☐ T2.1 | ★ **Prop. 1 derivation, by hand, FIRST** | T1.13 | 5 required items incl. corrected depth rule (D5) |
| ☐ T2.2 | ⚡ Remaining §3 derivations (10 files) | T2.1 | ten files, each a derivation not a summary |
| ☐ T2.3 | ★ Batched statevector simulator `qsim.py` | T1.13 | norm preserved; **second derivatives flow** |
| ☐ T2.4 | ★ Re-uploading circuit module | T2.3 | `frequencies()` returns exactly 27 values for `L=3` ternary |
| ☐ T2.5 | ★★ **`qsim` vs PennyLane oracle** | T2.4 | **max diff < 1e-10 — never loosen** |
| ☐ T2.6 | ★★ **`test_circuit_spectrum.py` (Prop. 1 verified)** | T2.5 | FFT peaks land on `Ω` to 1e-8; tag `prop1-verified` |
| ☐ T2.7 | Parameter-shift derivatives (`∂_θ`, `∂_x`, `∂²_x`) | T2.4 | agrees with autodiff to 1e-9 |
| ☐ T2.8 | ★ SMCD target spectrum (Prop. 2) | T2.6 | P1/P4 supports exact; **P2 high-mode weight ≈ 0.025** |
| ☐ T2.9 | ★ SMCD Algorithm 1 (with D5 correction) | T2.8 | P1⇒`L=4,n=1`; P4⇒`n=3`,`ring_cz`; deterministic |
| ☐ T2.10 | Design card + coverage metric | T2.9 | depth formula pinned for 6 cases; JSON round-trips |
| ☐ T2.11 | Noise models (shot, depolarizing surrogates) | T2.4 | correct empirical variance; gradients flow |
| ☐ T2.12 | ★ Hybrid models (serial / parallel; 4 q-families) | T2.4, T2.10 | `realised_frequencies() == Ω` at init |
| ☐ T2.13 | ⚡ Reference [4] re-positioning | owner input | one differentiation paragraph written |
| ☐ T2.14 | ★ Wire `c_rff_matched` to the design card | T2.10, T0.16 | contains `{π,15π}` for P1; size-matched |
| ☐ T2.15 | Octave-split ensemble `q_octave` | T2.12 | P4 k=20 splits, each `L ≤ 4`, coverage 1.0 |
| ☐ T2.16 | Cross-family size matching (±10 %) | T2.12, T2.14 | table emitted to `results/size_matching.json` |
| ☐ T2.17 | ★ Full-pipeline smoke: 6 problems × 7 families | T2.16, T1.12 | all 42 pass in < 20 min |
| ☐ T2.18 | ★★ **Phase 2 gate** | T2.6, T2.9, T2.17 | 6 checks; **re-verify matrix budget**; tag `phase2-complete` |

---

## Phase 3 — Main Experiments (Day 8–9) · `05_PHASE3_experiments.md`

| # | Task | Depends on | DoD in one line |
|---|---|---|---|
| ☐ T3.1 | ★★ **Pre-registration — BLOCKING, no runs before this commit** | T2.18 | 12 predictions with thresholds + falsifiers, committed |
| ☐ T3.2 | Run orchestration (parallel, resumable, fault-isolated) | T3.1 | smoke sweep runs, second invocation skips all |
| ☐ T3.3 | Experiment config files (5 files) | T3.2 | enumerates 210/36/45/54/36 unique runs |
| ☐ T3.4 | Launch core matrix (210 runs) | T3.3 | 210 `metrics.json`; wall-clock recorded |
| ☐ T3.5 | Sweeps: coverage → depth → α → noise | T3.4 | all complete, or cuts recorded |
| ☐ T3.6 | Soft-BC NTK block-imbalance study (D6) | T3.4 | per-block eigenvalue mass recorded |
| ☐ T3.7 | ★★ **Coverage-vs-error plot (the headline figure)** | T3.5 | figure + Spearman ρ; **ships whatever it says** |
| ☐ T3.8 | NTK spectra figure, both families, band shaded | T3.4 | decay exponent inside vs outside `Ω` recorded |
| ☐ T3.9 | ★ Per-frequency heatmaps with `Ω` overlay (C3) | T3.4 | PR-8 overlap fraction recorded per problem |
| ☐ T3.10 | ★★ **Phase 3 gate** | T3.7–T3.9 | **pre-registration timestamp verified mechanically**; tag `phase3-complete` |

---

## Phase 4 — Honesty Pass (Day 10–11) · `06_PHASE4_honesty.md`

| # | Task | Depends on | DoD in one line |
|---|---|---|---|
| ☐ T4.1 | Statistics layer (Wilcoxon, bootstrap, Holm–Bonferroni) | T3.10 | matches scipy; `n=5` p-floor stated honestly |
| ☐ T4.2 | ★ `c_rff_matched` adjudication (C1's real test) | T4.1 | written verdict per problem against the 4-row table |
| ☐ T4.3 | ★ Negative-result map + decision table (C4) | T4.1 | predicted-vs-measured benefit plot; table complete |
| ☐ T4.4 | Noise-surrogate validation vs density matrix | T3.5 | discrepancy measured and **reported**, not hidden |
| ☐ T4.5 | Barren-plateau frontier | T3.5 | stated practical `(n, L)` frontier |
| ☐ T4.6 | Cost ledger (params, FLOPs, wall-clock, evals) | T3.5 | includes **error at matched wall-clock** |
| ☐ T4.7 | ★★ **Claim adjudication C1–C5 + PR-1…PR-12** | T4.2–T4.6 | no `?` left; inconclusive used where honest |
| ☐ T4.8 | ★ Adversarial self-review (6-item checklist) | T4.7 | each item answered with evidence |
| ☐ T4.9 | Baseline fairness re-run (if T4.8 finds under-tuning) | T4.8 | either "nothing better found" or re-run + re-adjudicate |
| ☐ T4.10 | ★★ **Phase 4 gate** | T4.7–T4.9 | tag `phase4-complete` |

---

## Phase 5 — Package (Day 12–14) · `07_PHASE5_package.md`

| # | Task | Depends on | DoD in one line |
|---|---|---|---|
| ☐ T5.1 | Figure regeneration pipeline (12 figures) | T4.10 | **delete `paper/figures/`, regenerate, all return** |
| ☐ T5.2 | `FINDINGS.md` (findings + confidence + tables) | T5.1 | every finding cites an existing figure |
| ☐ T5.3 | Limitations section | T5.2 | all 7 known limitations present |
| ☐ T5.4 | `results/manifest.json` completeness check | T5.1 | no orphans in either direction |
| ☐ T5.5 | `repro-quick` target (~15 min, **CPU-only**) | T5.1 | timed with CUDA disabled; time recorded |
| ☐ T5.6 | ⚡ Paper skeleton + **math section (the centrepiece)** | T5.2 | builds; 5–15 pp |
| ☐ T5.7 | Results + negative-results sections | T5.6, T4.10 | no claim exceeds its T4.7 verdict |
| ☐ T5.8 | Reproducibility appendix | T5.5 | every number measured, not estimated |
| ☐ T5.9 | Paper review vs `project.md` §1 mapping table | T5.7, T5.8 | mapping table reproduced with real section numbers |
| ☐ T5.10 | ⚡ Slides (~18, Marp, same figures) | T5.9 | 15 ≤ count ≤ 20 |
| ☐ T5.11 | ⚡ Notebooks (narrative only) | T5.1 | execute top-to-bottom, no GPU |
| ☐ T5.12 | `README.md` + headline figure | T5.2 | contribution clear in 60 seconds |
| ☐ T5.13 | `docs/REPRODUCE.md` | T5.5 | every command actually run, output pasted |
| ☐ T5.14 | ★★ **Clean-clone verification (Phase 5 gate)** | T5.13 | clone → `uv sync` → test → repro-quick → figures |
| ☐ T5.15 | Final delivery checklist | T5.14 | 10 boxes; tag `v1.0-submission` |

---

## The eleven hard gates (★★)

Stop and verify at each. These are the points where continuing on a failure wastes the most work.

| Gate | What it protects |
|---|---|
| T0.21 | The 2-week budget — catches an unaffordable matrix before any science is built on it |
| T0.22 | Trustworthy ground truth and a deterministic harness |
| T1.5 | **Spectral bias is visible.** Without it, no downstream result means anything |
| T1.13 | Instruments are trustworthy before quantum models exist |
| T2.5 | Our fast simulator is not lying |
| T2.6 | **Prop. 1 holds for the actual shipped circuits** — the project's scientific foundation |
| T2.18 | Everything integrates, and the budget still holds with real numbers |
| T3.1 | **Pre-registration** — the credibility of every XAI claim |
| T3.7 | The headline result exists, whatever it says |
| T3.10 | Pre-registration timestamp is mechanically verified |
| T4.7 | Every claim honestly adjudicated |
| T5.14 | The repository is actually usable by a reviewer |

---

## Cut-line quick reference (from `00_MASTER_PLAN.md` §5)

If behind schedule, cut in this order:
**1.** α sweep → 3 values · **2.** drop `n=4` from depth sweep · **3.** shot-noise only ·
**4.** seeds 5→3 (**state it in FINDINGS**) · **5.** drop `q_parallel`/`q_octave` on P3 · **6.** drop P4 k=4.

**Never cut:** coverage sweep (validates C1) · `c_rff_matched` (C1's honest baseline) ·
P2 (negative control for C4) · T3.1 pre-registration · T2.6.
