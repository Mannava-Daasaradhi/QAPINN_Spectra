# Phase 5 — Package & Deliverables (Days 12–14)

**Goal:** ship the assignment. Paper, slides, findings, reproducibility.

**Exit gate (`project.md` §12 Phase 5):**
> Clean-clone `python tasks.py repro-quick` succeeds on a different directory/machine.

**Non-negotiable from `project.md` §9:** every figure regenerable by one task-runner target from
committed metrics. If a figure cannot be regenerated, it does not go in the paper.

---

## T5.1 — Figure regeneration pipeline

**Depends on:** T4.10
**Files:** `scripts/make_figures.py`, `src/qapinn/viz/figures.py`

One entry point: `python tasks.py figures` regenerates **every** figure from committed
`results/runs/*/metrics.json` and `xai/*.npz` — never from a live model, never from a checkpoint.

Figure inventory (each a named function, each registered in `results/manifest.json`):

| Name | Source | Used in |
|---|---|---|
| `staircase_cmlp_p1` | T1.5 | paper §Results, slides |
| `coverage_vs_error` | T3.7 | **headline figure** — paper, README, slides |
| `ntk_spectrum_{p1,p4}` | T3.8 | paper §7.1 |
| `freq_heatmap_{6 instances}` | T3.9 | paper §7.2, slides |
| `ablation_matched` | T4.2 | paper §Results |
| `decision_map` | T4.3 | paper §Negative results |
| `barren_frontier` | T4.5 | paper §Limitations, slides |
| `attribution_{p3,p4}` | T1.6 | paper §7.3 |
| `fisher_effdim` | T1.7 | paper §7.4 |
| `probes_cka` | T1.10 | paper §7.6 |
| `landscape_grid` | T1.11 | slides |
| `cost_ledger_table` | T4.6 | paper appendix |

**DoD:** delete `paper/figures/` entirely, run `python tasks.py figures`, and every figure comes back
identical. This is the real test of the regeneration claim — do it.

---

## T5.2 — `FINDINGS.md`

**Depends on:** T5.1
**Files:** `FINDINGS.md`

Per `project.md` §10: numbered findings, each with the figure that supports it and a **confidence
level**, plus the decision table from T4.3.

Structure:
1. Headline finding (one sentence).
2. Findings F1…Fn — each: statement · supporting figure · effect size · confidence
   (high/medium/low) · the caveat that most threatens it.
3. The C1–C5 adjudication table (T4.7).
4. The decision table ("PDE has property X ⇒ use/don't use a quantum layer, with these settings").
5. Recommendations, including explicitly **when not to go quantum**.
6. Pre-registration commit SHA (T3.1) so the timestamp is auditable.
7. What we would do next with more time.

**DoD:** every finding cites a figure that exists; every confidence level is justified in one clause.

---

## T5.3 — Limitations section

**Depends on:** T5.2
**Files:** section in `FINDINGS.md` and in the paper

State plainly: `n = 5` seeds cannot reach p < 0.05 (T4.1); simulation only, no hardware; ≤ 8 qubits;
noise models are surrogates with a measured discrepancy (T4.4); P5 was cut; the affine-encoder
constraint (D3) is a real restriction on the hybrid's expressivity that we imposed to keep coverage
well-defined; `predicted_benefit` is a heuristic.

**DoD:** every item from this list appears, plus anything T4.8 surfaced.

---

## T5.4 — `results/manifest.json` completeness

**Depends on:** T5.1
**Files:** `results/manifest.json`

Every figure → the `run_id`s that produced it (`project.md` §11).

**DoD:** a script verifies every referenced `run_id` exists in `results/runs/` and every figure in
`paper/figures/` has a manifest entry. No orphans in either direction.

---

## T5.5 — `repro-quick` target

**Depends on:** T5.1
**Files:** `tasks.py`

Per `project.md` §11: P1 + P4(k=4), 1 seed, ~15 min CPU, reproducing the two headline figures at
reduced fidelity. Must work **CPU-only** — a reviewer may not have a GPU.

**DoD:** timed on this machine with CUDA disabled (`CUDA_VISIBLE_DEVICES=""`); actual wall-clock
recorded in `docs/REPRODUCE.md`. If it exceeds 20 min, reduce steps until it fits.

---

## T5.6 — Paper skeleton and math section

**Depends on:** T5.2 (can start in parallel with Phase 4)
**Files:** `paper/main.tex`, `paper/refs.bib`, `paper/sections/*.tex`

12 pp per `project.md` §10:
`Intro · Background (§3 condensed) · **Methodology/SMCD: Prop 1–4 + Alg 1** · XAI protocol · Results ·
Negative results · Limitations · Conclusion`.

> **The math section is the centrepiece — the brief explicitly asks for the mathematical basis.**
> Budget the most space and the most care here. It is built from `docs/derivations/` (T2.1, T2.2),
> and Prop. 1 must include the corrected depth rule (D5) with the balanced-ternary argument.

Cite everything in `project.md` §15 plus reference [4] when it arrives (T2.13).

**DoD:** `python tasks.py paper` builds `paper/main.pdf` without errors; page count in 5–15.

---

## T5.7 — Results and negative-results sections

**Depends on:** T5.6, T4.10
**Files:** `paper/sections/results.tex`, `paper/sections/negative.tex`

Lead Results with `coverage_vs_error`. Lead Negative Results with P2 and the a-priori 2.5 % prediction
— an a-priori quantitative negative prediction that held is a stronger result than a positive one that
was found post hoc, and the writing should make that argument explicitly.

**DoD:** every claim in the text has a figure or table reference; no claim exceeds its T4.7 verdict.

---

## T5.8 — Reproducibility appendix

**Depends on:** T5.5
**Files:** `paper/sections/repro.tex`

Hardware actually used (RTX 4090 Laptop, 32 cores), measured wall-clock for `repro-all`, resolved
package versions from `docs/ENV_RESOLVED.md`, determinism settings, and the seed list.

**DoD:** every number is measured, not estimated.

---

## T5.9 — Paper review pass

**Depends on:** T5.7, T5.8
**Files:** `paper/main.pdf`

Self-review against the assignment mapping table in `project.md` §1: walk each requirement row and
point at the section that satisfies it. Check every figure is referenced in the text, every claim is
adjudicated, and no verdict is overstated relative to T4.7.

**DoD:** the §1 mapping table is reproduced in the paper's introduction with real section numbers.

---

## T5.10 — Slides

**Depends on:** T5.9
**Files:** `slides/` (Marp markdown — no LaTeX toolchain needed, renders from the same PNGs)

~18 slides per `project.md` §10:
`problem → the one theorem → the design rule → the two XAI figures → the honest negative result →
recommendations`.

Built from the **same figures** as the paper (T5.1 emits both `.pdf` and `.png` for exactly this).
Include one derivation slide per `docs/derivations/` item that made it into the talk.

**DoD:** `python tasks.py slides` produces `slides/main.pdf` (or HTML); 15 ≤ slide count ≤ 20.

---

## T5.11 — Notebooks

**Depends on:** T5.1
**Files:** `notebooks/01_theory_walkthrough.ipynb` … `05_results.ipynb`

Narrative only — **not the source of truth** (`project.md` §9). Each notebook imports from
`src/qapinn` and reads committed results; none re-derives anything.

**DoD:** all notebooks execute top-to-bottom against committed results without a GPU.

---

## T5.12 — `README.md`

**Depends on:** T5.2
**Files:** `README.md`

60-second version + the headline figure (`coverage_vs_error`) + install + `repro-quick` + repo map +
MIT licence.

**DoD:** a reader who knows PINNs can understand the contribution in 60 seconds.

---

## T5.13 — `docs/REPRODUCE.md`

**Depends on:** T5.5
**Files:** `docs/REPRODUCE.md`

Per `project.md` §11: `uv sync`, `repro-quick` (measured time), `repro-all` (measured time on stated
hardware), determinism notes, the `--smoke` flag on every script, and the manifest explanation.
State plainly that no quantum hardware is required.

**DoD:** every command in the document has actually been run and its output pasted.

---

## T5.14 — ★★ Clean-clone verification (Phase 5 gate)

**Depends on:** T5.13
**Files:** —

```powershell
git clone <repo> C:\Users\daasa\AppData\Local\Temp\qapinn_clean
cd C:\Users\daasa\AppData\Local\Temp\qapinn_clean
uv sync
python tasks.py test
python tasks.py repro-quick
python tasks.py figures
```

**DoD:** all four succeed from a clean clone with no manual intervention. Any missing file, unpinned
dependency, or hard-coded absolute path is a bug to fix here — this is the last gate and it catches
the things that make a repository useless to a reviewer.

---

## T5.15 — Final delivery checklist

**Depends on:** T5.14

- [ ] `paper/main.pdf` — 5–15 pp, math section centred on Prop. 1–4 + Alg. 1
- [ ] `slides/` — ~18 slides from the same figures
- [ ] `FINDINGS.md` — numbered findings + confidence + decision table
- [ ] `README.md` — headline figure
- [ ] `docs/REPRODUCE.md` — verified
- [ ] `pytest` green, including `test_circuit_spectrum.py`
- [ ] `results/manifest.json` complete, no orphans
- [ ] Every C1–C5 marked confirmed / refuted / inconclusive
- [ ] Clean-clone `repro-quick` succeeds
- [ ] Repository public, MIT licensed

**DoD:** commit tagged `v1.0-submission`.
