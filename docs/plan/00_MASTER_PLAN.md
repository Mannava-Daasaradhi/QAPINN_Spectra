# QAPINN-Spectra — Master Implementation Plan

**Audience:** the implementing agent (Sonnet) and the owner (daasa).
**Status:** authoritative build plan. Derived from `project.md` (the scientific spec). Where this plan
contradicts `project.md`, **this plan wins** — every deviation is listed in §4 with its reason.

**Constraints fixed for this build (answered by owner 2026-07-28):**

| Constraint | Value | Consequence |
|---|---|---|
| Time budget | **~2 weeks, full-time** | Scope cuts in §5. P5 dropped from the critical path. |
| Runtime | **Windows-native, `uv` venv** | No `make`; no `lightning.gpu`. Custom torch statevector sim is the fast path (§3, D2). |
| Reference [4] | Owner will paste citation | Task **T2.13** re-positions related work when it arrives; nothing else blocks on it. |

**Measured hardware (this machine):**
`RTX 4090 Laptop, 16 GB VRAM · 32 logical cores · 31.7 GB RAM · CUDA driver 13.0 · 173 GB free disk`
`uv` ✓ · `git` ✓ · `make` ✗ · Python 3.13.13 (miniconda base, no torch/pennylane/scipy).

---

## 1. How to use this plan

Read in this order. Do not skip §2 and §3 — they encode decisions that every later task depends on.

| Doc | Contents |
|---|---|
| `00_MASTER_PLAN.md` | ← you are here. Decisions, schedule, cut lines, DAG. |
| `01_CONVENTIONS.md` | **Read before writing any code.** Frequency convention, config schema, result schema, module API contracts, determinism rules, coding standards. |
| `02_PHASE0_foundations.md` | Scaffold, env, PDEs, reference solvers, classical PINN, training loop. |
| `03_PHASE1_instruments.md` | The six XAI instruments, validated on classical models only. |
| `04_PHASE2_theory_smcd.md` | Derivations, quantum simulator, circuits, SMCD, hybrid models. |
| `05_PHASE3_experiments.md` | Pre-registration, main matrix, three sweeps. |
| `06_PHASE4_honesty.md` | Negative controls, noise, barren-plateau frontier, claim adjudication. |
| `07_PHASE5_package.md` | Figures, FINDINGS, paper, slides, clean-clone reproduction. |
| `08_TASK_INDEX.md` | **Flat chronological checklist of all 88 tasks with IDs and dependencies.** |

**Working rules for the implementing agent:**

1. Execute tasks **in the order given by `08_TASK_INDEX.md`.** The order is a topological sort of the
   dependency DAG; do not reorder without checking `blocked_by`.
2. Every task states a **Definition of Done (DoD)** that is mechanically checkable — usually a passing
   pytest test or a produced artifact. A task is not done until its DoD check actually runs and passes.
3. **Never** mark a task done on the strength of "the code looks right". Run the test.
4. If a task's DoD cannot be met, **stop and report** rather than weakening the DoD. Several DoDs in
   Phase 2 are the scientific validity guarantees of the project (notably T2.6, T2.9).
5. Commit after each task with message `T<id>: <subject>`. One task = one commit.
6. Do not add dependencies beyond those in `pyproject.toml` (T0.2) without recording them there.

---

## 2. What this project actually is (one screen)

We claim that a data-re-uploading variational quantum circuit is **a Fourier feature map with a
prescribable frequency set**, and that you can therefore *construct* — not search for — a circuit
matched to a given PDE's solution spectrum. Then we instrument both a classical PINN and the hybrid
model with the same six XAI probes to prove the mechanism is spectral-bias relief inside the encoded
band, and to map the PDE classes where the quantum layer does nothing or actively hurts.

The whole project stands on one theorem (Schuld–Sweke–Meyer): for encoding gates `exp(−i x H)`, an
`L`-layer re-uploading circuit outputs a truncated Fourier series with frequency set
`Ω = { Σ_l (λ_a^{(l)} − λ_b^{(l)}) }`. **Task T2.6 numerically verifies this for the exact circuits we
ship.** If T2.6 fails, nothing downstream is meaningful.

The single most important figure in the paper is the **coverage-vs-error plot** (T3.7): SMCD spectral
coverage on the x-axis, final relative L2 error on the y-axis. A monotone decrease validates the
methodology. Anything else is still publishable and gets reported honestly.

---

## 3. Key technical decisions (with rationale)

These are binding. Each has an ID referenced from the task docs.

### D1 — Differentiation backend: **autograd through a statevector simulator, not `adjoint`.**

`project.md` §11 suggests `adjoint` differentiation. **This is wrong for a PINN and must not be used
for training.** A PINN residual needs `∂u/∂x` and `∂²u/∂x²` where `u` flows *through* the circuit.
Adjoint differentiation returns first-order gradients with respect to *parameters* only and does not
compose under higher-order autodiff. Options considered:

| Method | `∂_θ` | `∂_x` | `∂²_x` | Batched | Verdict |
|---|---|---|---|---|---|
| `adjoint` | ✓ | ✗ | ✗ | ✓ | Unusable for PINN residuals |
| Parameter-shift | ✓ | ✓ (input is an angle) | ✓ (2nd-order shift) | ✓ | Exact + hardware-faithful, but ~`4·|Ω|` circuit evals per residual. Too slow to train. |
| **Backprop through statevector** | ✓ | ✓ | ✓ | ✓ | **Chosen for training.** |

**Decision:** train with reverse-mode autodiff through a differentiable statevector simulation.
Parameter-shift is implemented anyway (T2.7) and used *only* in tests and one hardware-faithfulness
appendix run — which converts `project.md` §3.8's prose claim ("∂_x of the circuit is also a
parameter-shift and is exact on hardware") into a **verified** claim. That is a stronger deliverable.

### D2 — Custom batched torch statevector simulator as the fast path; PennyLane as the oracle.

For `n ≤ 8` qubits the state is ≤ 256 complex amplitudes — trivial arithmetic. PennyLane's per-call
Python overhead (~ms) would dominate and, multiplied by `~380 runs × 20k steps`, is fatal to a 2-week
timeline.

**Decision:** implement `src/qapinn/models/qsim.py` — a minimal batched statevector simulator
(RX/RY/RZ, CZ/CNOT, ⟨Z_i⟩ expectations) operating on a tensor of shape `[B, 2]*n`, running on CUDA,
fully autograd-native (so D1's higher-order derivatives come for free).

**Guard rail:** `tests/test_qsim_vs_pennylane.py` (T2.5) asserts agreement with PennyLane
`default.qubit` to `1e-10` on randomized circuits. PennyLane stays a hard dependency and remains the
reference implementation cited in the paper. We never claim a result the oracle disagrees with.

### D3 — Encoder must be **affine**, identity-initialised.

`project.md` §5.1 puts a classical encoder `E_φ` before the circuit. **If `E_φ` is a nonlinear MLP,
the circuit's frequency set `Ω` lives in encoded coordinates `z`, not in `x` — and the entire
Prop-2 → Prop-3 matching argument collapses**, because we would no longer know the model's frequency
content in the physical variable. This is a real hole in the spec and closing it is mandatory.

**Decision:** `E_φ(x) = A x + b` with `A` initialised to identity. Then `e^{iω·z} = e^{i(ωA)·x + iωb}`,
so the realised frequency set in physical coordinates is exactly `Ω·A` — still computable in closed
form, at every training step.

**Bonus finding this unlocks:** track `‖A − I‖` and the induced `Ω(t)` across training. "Spectral
drift of the learned coordinate map" is a genuine, cheap, novel XAI readout (instrument 7.7, T1.8).
A nonlinear-encoder variant is kept as an **ablation only** (`Q-serial-mlp`), reported with the
explicit caveat that coverage is undefined for it.

### D4 — Fixed (non-trainable) frequency scalings `ω_j`.

SMCD prescribes `ω_j`; making them trainable would (a) destroy determinism of the design card and
(b) invite the "Fourier locking" failure mode of arXiv:2607.11013. Scalings are buffers, not
parameters. Recorded in the design card.

### D5 — Corrected depth formula (**bug in `project.md` Alg. 1 step 4**).

`project.md` gives `L ← ⌈log_3(K/Δ)⌉`. This is **wrong**. With ternary scalings `ω_j = Δ·3^{j−1}` and
Pauli generators (eigenvalue differences `m_l ∈ {−1,0,+1}`), the reachable set is *balanced ternary*:

```
Ω = Δ · { −(3^L − 1)/2 , … , (3^L − 1)/2 }        (all integers in that range)
max|Ω| = Δ·(3^L − 1)/2
```

Requiring `max|Ω| ≥ K` gives the correct rule:

```
L ≥ log_3( 2K/Δ + 1 )        ⇒        L = ⌈ log_3( 2K/Δ + 1 ) ⌉
```

*Worked check (P1):* `K = 15π`, `Δ = π` ⇒ `3^L ≥ 31` ⇒ **L = 4** (the spec's formula gives
`⌈log_3 15⌉ = 3`, and `L=3` reaches only `±13π` — it would silently miss the `15π` mode and quietly
invalidate every coverage number). Implement the corrected form; `tests/test_smcd_depth.py` (T2.10)
pins it.

### D6 — Hard BC by default, soft BC as an explicit variant.

Alg. 1 step 9 mandates hard constraints (`u = B + D·N`), which removes the BC loss term entirely.
But §7.1 asks for *per-loss-block* NTK eigenvalue mass (residual vs BC vs IC), which requires a BC
loss to exist. **Decision:** config flag `bc_mode: hard | soft`. Main matrix runs `hard`. The NTK
loss-imbalance study (T3.6) runs `soft` on P1 and P4 so the classical block-imbalance story is
measurable. Both are reported.

### D7 — NTK by explicit Jacobian materialisation, chunked.

Probe set ≤ 512 points, parameter count ~2–6k ⇒ `J` is at most `512 × 6000` floats (~12 MB). Full
materialisation via `torch.func.jacrev` + `vmap` is fine and far simpler than JVP tricks. The spec's
memory concern applies at a scale we never reach. Chunk over probe points at 64/chunk to bound VRAM.

Block decomposition (Prop. 4) falls straight out of `PINNModel.param_groups()`:
`Θ_cl = J_cl J_clᵀ`, `Θ_q = J_q J_qᵀ`, `Θ_× = J_cl J_qᵀ`, and `Θ_hyb = Θ_cl + Θ_q + 2 Θ_×` is asserted
numerically in T1.3 — turning Prop. 4 from an assertion into a tested identity.

### D8 — Noise: analytic depolarizing + shot-noise surrogate; density matrix for validation only.

Full density-matrix simulation is `4^n` and ~256× more expensive — unaffordable across the matrix.

- **Depolarizing `p`:** global depolarizing acting `m` times scales any traceless observable by
  `(1−p)^m`. Implement as an exact, differentiable multiplicative factor.
- **Shot noise `N`:** add detached Gaussian noise of variance `(1 − ⟨Z⟩²)/N` to the expectation
  (straight-through). Correct to first order in the CLT regime; `N = 1024` is comfortably there.
- **Validation (T4.4):** reproduce a small subset on PennyLane `default.mixed` with *local*
  depolarizing and report the discrepancy honestly rather than assuming the surrogate is exact.

### D9 — Task runner: `tasks.py`, with a `Makefile` mirror.

`make` is absent on this machine. `python tasks.py <target>` is the primary interface; a `Makefile`
with identical target names is shipped so the reproducibility contract in `project.md` §11 still reads
naturally on Linux. Both call the same underlying module functions — no logic duplicated.

### D10 — Python **3.12**, `uv`-managed, project-local venv.

PennyLane 0.45.1 supports 3.11–3.14 and torch has 3.13 wheels, so 3.13 would work — but 3.12 is the
widest-compatible target for the whole scientific stack and nothing here needs 3.13 features. The
miniconda base install is left untouched. Torch from the `cu128` index (forward-compatible with the
installed CUDA 13.0 driver).

### D11 — Every run is fully specified by one config; every artifact is stamped.

`run_id = sha256(canonical_json(config))[:12]`. Each run writes `config.yaml`, `metrics.json`,
`design_card.json`, `xai/*.npz`, and a `provenance.json` holding git SHA, seed, resolved package
versions, hostname, wall-clock, and device. `results/manifest.json` maps **every figure → the runs
that produced it** (`project.md` §11).

### D12 — Frequency convention fixed once, project-wide.

**Angular frequency `ω`, complex exponential convention `e^{iωx}`.** `sin(15πx)` has `ω = 15π ≈ 47.12`,
*not* `k = 7.5` cycles. All of `smcd/`, `xai/spectral_error.py`, and every axis label use this. Mixing
conventions is the single most likely silent bug in this project — `01_CONVENTIONS.md` §2 restates it
and `tests/test_fft_convention.py` (T0.9) pins it.

### D13 — Performance spike before committing to the matrix.

**T0.21** benchmarks one `Q-serial` training step end-to-end and extrapolates full-matrix wall-clock.
If the projection exceeds 24 h, the matrix is cut per §5 *before* Phase 3 starts, not during it.

---

## 4. Deviations from `project.md` (complete list)

| # | `project.md` says | This plan does | Why |
|---|---|---|---|
| 1 | Use `adjoint` differentiation | Backprop through statevector sim | D1 — adjoint gives no `∂²_x` |
| 2 | Alg. 1 step 4: `L = ⌈log_3(K/Δ)⌉` | `L = ⌈log_3(2K/Δ + 1)⌉` | D5 — balanced-ternary range is `±(3^L−1)/2` |
| 3 | Encoder `E_φ: R^d→R^m` unspecified | Affine, identity-init | D3 — otherwise `Ω` is not expressible in physical coordinates |
| 4 | Alg. 1 step 9: hard BC | Hard default, soft variant | D6 — §7.1 needs a BC block to weigh |
| 5 | `Makefile` | `tasks.py` + Makefile mirror | D9 — no `make` on Windows |
| 6 | Depolarizing `p=1e-3` across matrix | Analytic surrogate; density-matrix on a subset | D8 — `4^n` cost |
| 7 | 5 PDEs × 7 × 5 × 3 noise = 525 runs | ~381 runs, §5 | 2-week budget |
| 8 | P5 (Allen–Cahn / wave) | **Cut** from critical path | 2-week budget; spec already marks it stretch |

None of these weaken a claimed contribution. Deviations 1–3 *strengthen* C1/C3 by removing ways the
result could be silently wrong.

---

## 5. Scope, run budget, and cut lines

### Experiment budget as planned (~381 runs)

| Block | Composition | Runs |
|---|---|---|
| Core matrix | 6 problem instances (P1, P2, P3, P4@k=4,10,20) × 7 families × 5 seeds | **210** |
| Noise study | {P1, P4@k=10} × {Q-serial, Q-random, C-FF} × 3 seeds × {shot-1024, depol-1e-3} | **36** |
| Coverage sweep | {P1, P4@k=10} × 6 coverage levels × 3 seeds | **36** |
| Depth/qubit sweep | P4@k=10 × L∈{2,3,4,5,6} × n∈{4,6,8} × 3 seeds | **45** |
| α sweep | P1 × α∈{0,.05,.1,.2,.4,.8} × {C-MLP, C-FF, Q-serial} × 3 seeds | **54** |

Projected wall-clock at 4 min/run with 6 parallel workers ≈ **4.2 h**; at a pessimistic 10 min/run
≈ **10.6 h**. Phase 3 is budgeted 2 days, which absorbs reruns.

### Cut lines, in the order they get cut if time runs short

1. **α sweep** → reduce to 3 α values, 3 seeds (−27 runs).
2. **Depth/qubit sweep** → drop `n=4` row (−15 runs).
3. **Noise study** → shot-noise only, drop depolarizing (−18 runs).
4. **Seeds** 5 → 3 on the core matrix (−84 runs). *Report the reduction explicitly in FINDINGS.*
5. **P3 Burgers** → drop `Q-parallel` and `Q-octave` families.
6. **P4@k=4** → drop (keep 10 and 20, which carry the "increasing with k" claim).

**Never cut:** the coverage sweep (it is the validation of C1), `C-RFF-matched` (it is the honest
baseline of C1's falsifier), or P2 (it is the negative control for C4).

### Minimum viable deliverable

If everything goes wrong, the smallest thing that still satisfies the assignment brief is:
P1 + P4@k=10, families {C-MLP, C-FF, C-RFF-matched, Q-serial, Q-random}, 3 seeds, hard BC, noiseless,
instruments 7.1 + 7.2 only, coverage sweep on P1. That is **~60 runs** and roughly 4 days of work.
It still delivers C1, C2, C3, C5 and a partial C4.

---

## 6. Phase overview and the 14-day schedule

| Phase | Days | Exit criterion (hard gate) |
|---|---|---|
| **0 — Foundations** | 1–2 | `pytest` green; MMS convergence test passes; `C-MLP` trains on P1; Burgers matches reference to published accuracy |
| **1 — Instruments** | 3–4 | **The spectral-bias staircase heatmap exists for `C-MLP` on P1.** If you cannot see spectral bias, nothing downstream means anything. |
| **2 — Theory + SMCD** | 5–7 | `test_circuit_spectrum.py` green — circuit FFT peaks land on predicted `Ω` to `1e-8`; all 7 families train end-to-end |
| **3 — Main experiments** | 8–9 | `docs/predictions.md` committed **before** the first run; coverage-vs-error plot exists, whatever it says |
| **4 — Honesty pass** | 10–11 | Every claim C1–C5 marked confirmed / refuted / inconclusive with the figure that decides it |
| **5 — Package** | 12–14 | Clean-clone `python tasks.py repro-quick` succeeds on a fresh directory |

### Day-by-day

| Day | Work |
|---|---|
| 1 | T0.1–T0.9: scaffold, env, task runner, config system, determinism, PDE modules, diff operators, FFT convention test |
| 2 | T0.10–T0.22: reference solvers (+MMS, Cole–Hopf), C-MLP, C-FF, losses, training loop, checkpointing, **perf spike (T0.21, D13)**, Phase-0 gate |
| 3 | T1.1–T1.5: NTK spectroscopy, per-frequency error trajectory, viz primitives — the two load-bearing instruments |
| 4 | T1.6–T1.13: attribution, Fisher/effective dim, gradient variance, probes, landscape, spectral drift; **Phase-1 gate** |
| 5 | T2.1–T2.7: derivations (Prop. 1 **first**, by hand), `qsim.py`, circuits, parameter-shift |
| 6 | T2.8–T2.14: `smcd/symbol.py`, `design.py`, `card.py`, hybrid models, all 7 families wired |
| 7 | T2.15–T2.18: full-pipeline smoke on every family × every PDE; **Phase-2 gate**; write `docs/predictions.md` |
| 8 | T3.1–T3.4: pre-registration commit, launch core matrix, monitor |
| 9 | T3.5–T3.10: three sweeps, coverage-vs-error plot, NTK spectra, frequency heatmaps; **Phase-3 gate** |
| 10 | T4.1–T4.6: `C-RFF-matched` ablation, noise study, barren-plateau frontier, density-matrix validation |
| 11 | T4.7–T4.10: claim adjudication, statistics (paired tests, median+IQR), negative-result map |
| 12 | T5.1–T5.5: all figures regenerable, `FINDINGS.md`, decision table |
| 13 | T5.6–T5.9: paper (12 pp), math section centred on Prop. 1–4 + Alg. 1 |
| 14 | T5.10–T5.15: slides, notebooks, README headline figure, `REPRODUCE.md`, clean-clone verification, buffer |

---

## 7. Dependency DAG (phase level)

```
P0 Foundations ──┬──> P1 Instruments ──┐
                 │                     ├──> P3 Experiments ──> P4 Honesty ──> P5 Package
                 └──> P2 Theory+SMCD ──┘
```

Within that, the critical path is:

```
T0.2 env → T0.6 PDE base → T0.8 PDEs → T0.12 reference solvers → T0.18 train loop → T0.21 perf spike
       → T1.2 NTK → T1.4 spectral error → T1.5 staircase [PHASE 1 GATE]
       → T2.3 qsim → T2.4 circuits → T2.6 circuit-spectrum test [HARD GATE]
       → T2.8 symbol → T2.9 design → T2.12 hybrid → [PHASE 2 GATE]
       → T3.1 pre-registration → T3.2 core matrix → T3.7 coverage plot
       → T4.7 claim adjudication → T5.2 FINDINGS → T5.6 paper
```

**Parallelisable side-branches** (safe to do while a long run executes): derivations (T2.1, T2.2),
instruments 7.3–7.6 (T1.6–T1.11), paper scaffolding (T5.6), slides (T5.10).

---

## 8. Risk register (build-level, extends `project.md` §13)

| Risk | Trigger to watch | Mitigation |
|---|---|---|
| Quantum sim too slow | T0.13 spike projects >24 h | Cut per §5; reduce collocation batch; reduce checkpoint count |
| `qsim` disagrees with PennyLane | T2.5 fails | **Stop.** Debug gate-application ordering / endianness before anything else. Do not "adjust tolerance". |
| Prop. 1 test fails | T2.6 fails | Almost always an FFT-grid or convention bug (D12), not a physics bug. Check sample rate ≥ 2·max Ω and exact periodicity first. |
| Encoder drift invalidates coverage | `‖A−I‖` grows large during training | Already instrumented (T1.8). Report drift; optionally freeze `A` in an ablation. |
| `C-RFF-matched` beats the quantum model | Phase 4 | **Legitimate result.** Paper is pre-framed (`project.md` §13) so this is the headline finding, not a failure. |
| Burgers reference solver wrong | MMS test passes but L2 is off | Cross-check Cole–Hopf quadrature against the pseudospectral solver (T0.12) — two independent methods must agree to 1e-6 |
| Barren plateaus kill Q training | Gradient variance < 1e-8 in T1.9 | Local observables + `L ≤ 6` + small-angle init are already in Alg. 1; if still flat, use octave-split (`Q-octave`) |
| Scope creep | Any new PDE or family appears | P5 is cut. Adding anything requires cutting something from §5 first. |

---

## 9. Definition of project done

- [ ] `paper/main.pdf` — 12 pp, math section centred on Prop. 1–4 + Alg. 1
- [ ] `slides/` — ~18 slides built from the same figures as the paper
- [ ] `FINDINGS.md` — numbered findings, each with supporting figure + confidence level, plus the decision table
- [ ] `README.md` — 60-second version + headline figure
- [ ] `docs/REPRODUCE.md` — verified by clean-clone run on a different directory
- [ ] `python tasks.py repro-quick` — ~15 min, reproduces both headline figures at reduced fidelity
- [ ] `pytest` — fully green, including `test_circuit_spectrum.py`
- [ ] `results/manifest.json` — every figure traced to its runs
- [ ] Every claim C1–C5 explicitly marked confirmed / refuted / inconclusive
