# QAPINN-Spectra

**Spectrum-Matched Quantum-Assisted PINNs: a constructive design methodology for problem-specific variational circuits, instrumented by explainable-AI diagnostics.**

Status: spec / pre-Phase-0 · Created 2026-07-20 · Owner: daasa

---

## 0. One-paragraph pitch

A classical PINN fails on exactly one thing, reliably and provably: it learns low frequencies fast and high frequencies almost not at all (spectral bias), and its NTK eigenspectrum decays roughly exponentially so the residual and boundary loss terms converge at wildly different rates. A data-re-uploading variational quantum circuit is *not* a black-box speedup — it is a **Fourier feature map whose accessible frequency set is exactly and prescribably determined by the eigenvalues of its data-encoding Hamiltonians**. Those two facts have never been joined into a design rule. This project joins them: given a PDE, we compute the target spectral support analytically (from the operator symbol, the forcing, and the boundary data), then **construct** the encoding generators, re-uploading depth, and qubit count so the circuit's frequency set covers it — no architecture search. We then use XAI not as decoration but as the *measurement instrument* that proves the mechanism: NTK eigenspectra, per-frequency error trajectories, integrated-gradient attribution over collocation points, Fisher/effective-dimension, and gradient-variance (barren plateau) tracking, computed identically for the classical PINN and the QAPINN so the delta is attributable to the quantum layer and nothing else.

**The honest headline we expect to report:** the quantum layer *helps* on high-wavenumber and multiscale problems, *does nothing* on smooth diffusive problems, and *hurts* when the circuit spectrum is mismatched to the PDE. A design rule plus a "when not to go quantum" recommendation is a stronger contribution than a cherry-picked win.

---

## 1. Assignment mapping

| Requirement | Where it is satisfied |
|---|---|
| Solve a set of PDEs with classical PINNs and QAPINNs | §4 PDE ladder (5 problems), §6 both model families, identical training harness |
| XAI techniques/metrics showing *how* learning happens in each | §7 six-instrument XAI suite, run on both families at matched checkpoints |
| Demonstrate how the VQC affects the classical PINN learning process | §7.1 NTK block decomposition + §7.2 per-frequency convergence + §8 ablation matrix; this is the causal claim, with negative controls |
| Methodology to construct problem-specific circuit + architecture for a PDE class, with mathematical basis | §5 **SMCD** (Spectrum-Matched Circuit Design), Prop. 1–4 + Algorithm 1 |
| Technical report 5–15 pp | §10 `paper/` |
| Source code repository | §9 layout |
| Reproducibility instructions | §11 |
| Presentation slides | §10 `slides/` |
| Summary of key findings + recommendations | §10 `FINDINGS.md`, decision table |

> **Open item — reference [4].** The brief cites a `[4]` behind a Scribd link that is login-walled and not fetchable. Paste the actual citation (title/authors) before Phase 2 so §3 related-work positions against it explicitly rather than against my inferred candidate (most likely the Sedykh et al. / Trahan-lineage hybrid-QPINN line, or arXiv 2606.04679 "when and where hybridization is effective?"). Everything else in this plan is independent of that resolution.

---

## 2. Contributions claimed (and how each is falsifiable)

**C1 — SMCD, a constructive PDE→circuit map.** Given a linear operator symbol and forcing/BC spectra, output the encoding generator eigenvalues `{λ}`, re-uploading depth `L`, qubit count `n`, and entangler pattern. Deterministic, no search.
*Falsifier:* a spectrum-matched circuit does not beat a size-matched random-frequency circuit on the same PDE.

**C2 — Quantum-NTK block decomposition.** Derive `Θ_hybrid = Θ_cl + Θ_q + 2Θ_×` for the hybrid model and show the quantum block contributes eigen-directions at the encoded frequencies, flattening the eigenvalue decay from exponential to algebraic over the matched band.
*Falsifier:* measured eigenspectra of the two families are statistically indistinguishable after size-matching.

**C3 — Spectral-bias relief is the *mechanism*, not just the outcome.** Per-frequency error curves must show the QAPINN's high-`k` modes converging at rates comparable to low-`k`, while the classical PINN's do not — and the improvement band must coincide with the encoded frequency set.
*Falsifier:* accuracy improves but the improvement is spread uniformly across `k`, or lands outside the encoded band ⇒ our explanation is wrong and we say so.

**C4 — A negative-result map with a decision rule.** Explicit characterization of PDE classes where hybridization is useless or harmful, with the mathematical reason (smoothing operators kill high-frequency content, so a high-frequency-rich circuit adds parameters and barren-plateau risk for nothing).

**C5 — XAI protocol for hybrid physics-informed models.** A reusable, model-agnostic instrument set with published metric definitions. Nobody has run a matched classical-vs-hybrid XAI comparison on PINNs; the existing XAI-for-PINN literature is SHAP-on-features, which is the shallow version of this question.

---

## 3. Background you must be able to derive from scratch on a whiteboard

This is the "demonstrates I know the fundamentals" spine. Every item below gets a short derivation in `docs/derivations/` and a one-slide version.

**Classical side**
1. Strong-form residual minimization; why a PINN is a collocation method with a neural ansatz, and how it differs from Galerkin/FEM (no weak form, no mesh, non-convex).
2. Reverse-mode AD gives *exact* derivatives (not finite differences); cost of `k`-th derivative and why `u_xx` in 2D is the real bottleneck.
3. NTK for a PINN: `Θ(x,x') = ∇_θ N[u_θ](x) · ∇_θ N[u_θ](x')ᵀ`; lazy-training regime; error dynamics `de/dt = −Θe` ⇒ mode `i` decays as `exp(−λ_i t)`. Loss-term imbalance = eigenvalue imbalance between the residual block and the boundary block.
4. Spectral bias / F-principle: for a ReLU or tanh MLP the NTK eigenvalues decay ~ `k^{-(d+1)}` (power law) against Fourier modes ⇒ high-`k` is learned last, slowly. Fourier-feature and random-feature fixes and why they are the classical baseline we must beat.
5. Hard vs soft boundary enforcement; the ansatz `u = B(x) + D(x)·N(x)` trick and why it removes one NTK block entirely.

**Quantum side**
6. Qubits, `R_x/R_y/R_z`, CNOT/CZ, expectation `f(x) = ⟨0|U†(x,θ) M U(x,θ)|0⟩`.
7. **The central theorem.** For encoding gates `exp(-i x H)` with Hamiltonian eigenvalues `{λ_1..λ_d}`, an `L`-layer re-uploading circuit's output is a truncated Fourier series `f(x) = Σ_{ω∈Ω} c_ω(θ) e^{iωx}` with `Ω = {Σ_l (λ_{j_l} − λ_{k_l})}`. Derive this for one qubit / one layer, then state the general case (Schuld–Sweke–Meyer 2021). **This is the mathematical basis of the entire methodology.** Everything in §5 is engineering on top of this.
8. Parameter-shift rule: `∂_θ f = ½[f(θ+π/2) − f(θ−π/2)]` — derive it; note that for PINNs we need `∂_x` of the circuit too, which is *also* a parameter-shift (input is an angle) and gives exact derivatives on hardware — a genuinely underused fact worth a slide.
9. Barren plateaus: `Var[∂_θ C]` decays exponentially in `n` for global cost + deep random circuits; mitigations we actually use (shallow `L`, local observables, layerwise init, small-angle init).
10. Effective dimension / Fisher information as capacity measure (Abbas et al.) — the quantum-native XAI metric.

---

## 4. The PDE ladder

Five problems, each isolating one mechanism. Every one has ground truth (analytic or high-order reference solver we write ourselves — never "PINN vs PINN").

| # | PDE | Domain | Why it is in the set | Expected verdict |
|---|---|---|---|---|
| **P1** | Poisson `−u'' = f`, `u = sin(πx) + α sin(15πx)` | `[0,1]` | Clean, analytic, two-scale. The spectral-bias microscope. | Quantum **helps** (targeted band) |
| **P2** | Heat `u_t = ν u_xx` | `[0,1]×[0,1]` | Parabolic smoothing kills high `k` exponentially. **Negative control.** | Quantum **no gain** — predicted a priori by SMCD |
| **P3** | Burgers `u_t + uu_x = (0.01/π)u_xx` | `[-1,1]×[0,1]` | Canonical PINN benchmark (Raissi). Nonlinearity + steep internal layer ⇒ broadband spectrum that *grows* in time. | Quantum helps at the shock; tests dynamic spectrum |
| **P4** | Helmholtz `Δu + k²u = f`, `k ∈ {4, 10, 20}` | `[-1,1]²` | The hero result. PINNs famously fail as `k` grows; spectral support is *known exactly* — SMCD's best case. | Quantum **helps, increasingly with k** |
| **P5** (stretch) | Allen–Cahn or 1D wave | — | Stiff / causality-violating; tests whether SMCD composes with causal weighting. | Open |

Reference solutions: P1 analytic; P2 analytic (separation of variables); P3 Cole–Hopf/Raissi dataset **and** our own spectral solver for cross-check; P4 analytic manufactured solution; P5 high-order FD + `scipy.integrate`.

---

## 5. SMCD — Spectrum-Matched Circuit Design (the methodology)

### 5.1 Setup

Hybrid ansatz (the *serial* form; §8 ablates parallel and quantum-in-the-middle):

```
x ─► [classical encoder E_φ: R^d → R^m]  ─► [VQC Q_θ with data re-uploading]  ─► [linear head w] ─► u
                     (learns coordinates)        (supplies the Fourier basis)      (learns coefficients)
```

The split is deliberate and is itself a claim: **the classical part learns the *warping*, the quantum part supplies the *basis*, the head learns the *coefficients*.** That is exactly the structure of a spectral method with a learned coordinate map, and it is why the XAI results should be readable.

### 5.2 The four propositions (what goes in the report's math section)

**Proposition 1 (Circuit spectrum).** An `L`-layer re-uploading circuit with encoding generators `H_j` acting on input feature `x_i` with scalings `ω_{j}` realizes
`f(x) = Σ_{n∈Ω} c_n(θ) e^{i⟨n,x⟩}`, `Ω = {Σ_{l=1}^{L} (λ^{(l)}_{a} − λ^{(l)}_{b})}`.
For Pauli generators with scaling `ω_j`, `Ω ⊇ {0, ±ω_1, …, ±Σω_j}`. *Corollary:* choosing `ω_j = ω_0 · 3^{j-1}` (ternary spacing) gives `|Ω| = O(3^L)` distinct frequencies with `L` layers — exponential spectral richness in depth. Choosing `ω_j = ω_0` gives only `Ω = {0,±ω_0,…,±Lω_0}` — linear. **The choice of scalings is the design knob and almost all QPINN papers leave it at 1.**

**Proposition 2 (PDE target spectrum).** For a linear constant-coefficient operator `L` with symbol `σ(k)`, the solution of `Lu = f` on a periodic/rectangular domain satisfies `û(k) = f̂(k)/σ(k)`. Hence the essential spectral support `S_ε = {k : |û(k)| > ε‖û‖}` is computable *before training* from `f̂` and `σ`. For Helmholtz `σ(k) = k² − |k|²` ⇒ support concentrates near `|k| ≈ k` (near-resonant), which is exactly the band a classical PINN cannot reach.

**Proposition 3 (Matching condition & resource bound).** If `S_ε ⊆ Ω` then the hybrid model can represent the solution to `ε` accuracy with a *linear* head — the remaining problem is convex in `w`. Resource bound: covering `S_ε` with max frequency `K` and resolution `Δ` requires `L ≥ log_3(K/ω_0)` layers with ternary scaling, and `n ≥ ⌈d⌉` qubits for `d` input dimensions (one wire per coordinate, plus ancilla wires for cross-terms). **This turns architecture choice into arithmetic.**

**Proposition 4 (NTK consequence).** For the hybrid model, `Θ_hyb = Θ_cl + Θ_q + 2Θ_×` where `Θ_q(x,x') = Σ_{n,n'∈Ω} ∂_θ c_n ∂_θ c_{n'} e^{i(⟨n,x⟩−⟨n',x'⟩)}`. Because `Ω` is discrete and prescribed, `Θ_q` has eigen-directions *at* the encoded frequencies with eigenvalues bounded below independently of `|n|` — i.e. the exponential/power-law eigenvalue decay in `|k|` is replaced by a flat response over `Ω`. **This is the formal statement of "the quantum layer removes spectral bias inside its band," and it is what §7.1 measures.**

### 5.3 Algorithm 1 — SMCD (this is the deliverable "methodology")

```
INPUT : PDE operator L, forcing f, boundary data g, domain Ω_x, tolerance ε, qubit budget n_max
OUTPUT: encoding generators+scalings {ω}, depth L, qubits n, entangler, observable, ansatz form

1  SYMBOL      σ(k) ← Fourier symbol of L   (linearize about a base state if nonlinear; for
                                             Burgers use the viscous-scale estimate k_max ~ 1/√ν)
2  TARGET      Ŝ ← {k : |f̂(k)/σ(k)| > ε}  ∪  spectral support of the boundary lift
                   (+ for time-dependent: sweep t, take the union; record growth rate)
3  BAND        K ← max|k| in Ŝ ;  Δ ← min spacing in Ŝ ;  d ← dim of x
4  DEPTH       L ← ⌈log_3(K/Δ)⌉  with ternary scalings ω_j = Δ·3^{j-1}
                   (fall back to linear scalings if Ŝ is a dense low band)
5  WIDTH       n ← d + n_cross   where n_cross covers required mixed-frequency terms
                   (Helmholtz 2D needs k_x±k_y ⇒ entangle the two coordinate wires)
6  CHECK       if n > n_max or L > L_max(barren-plateau budget):
                   split the band → multi-circuit ensemble (one circuit per octave) — see §5.4
7  ENTANGLER   ring CZ if cross-terms needed, else none (avoid gratuitous entanglement:
                   it costs gradient variance and buys nothing if Ŝ is separable)
8  OBSERVABLE  local: ⟨Z_0⟩ (or Σ_i⟨Z_i⟩/n) — never a global Pauli string (barren plateaus)
9  ANSATZ      hard-constrain BCs: u = B(x) + D(x)·[w·Q_θ(E_φ(x))]
10 INIT        small-angle θ ~ N(0, σ_0²), σ_0 = 0.1; identity-block init for the encoder
11 REPORT      emit a design card: (Ŝ, Ω, coverage = |Ŝ∩Ω|/|Ŝ|, L, n, param count, predicted
               NTK band) — this card is committed alongside every experiment
```

Step 11 matters: **every run ships a design card**, so the report can correlate predicted coverage with measured improvement. That correlation plot *is* the validation of the methodology and is the single most important figure in the paper.

### 5.4 Extension: octave-split circuit ensembles

When one circuit cannot cover `Ŝ` within the barren-plateau depth budget, use `M` shallow circuits, each matched to one octave of `Ŝ`, summed at the head. This is multi-resolution analysis with quantum basis functions and is a clean secondary novelty — it keeps every circuit shallow (trainable) while covering a wide band.

---

## 6. Model families (all size-matched)

| ID | Model | Purpose |
|---|---|---|
| `C-MLP` | tanh MLP PINN, standard | The baseline everyone reports |
| `C-FF` | Fourier-feature PINN (Tancik-style, random `B`) | **The fair classical baseline.** If we beat only `C-MLP` and not `C-FF`, the quantum layer bought nothing that random Fourier features don't. This baseline is non-negotiable. |
| `C-RFF-matched` | Fourier features *at the SMCD frequencies* | Isolates "matched spectrum" from "quantum" — the sharpest ablation in the project |
| `Q-serial` | Encoder → VQC → head (SMCD-designed) | Main model |
| `Q-parallel` | MLP ‖ VQC, summed | Alt topology (the HQPINN form in the literature) |
| `Q-random` | Same size, random/unit scalings | Ablates the *design*, keeps the quantum |
| `Q-octave` | §5.4 ensemble | For P4 at `k=20` |

Parameter counts matched within ±10% across families. Reported alongside FLOPs and wall-clock, because "fewer parameters" is a claim QPINN papers make loosely and we should make it precisely.

---

## 7. XAI instrument suite

Six instruments, run at identical checkpoints (`{0, 100, 500, 1k, 5k, 20k, final}` steps) for every model × PDE × seed. Two are the load-bearing ones (7.1, 7.2); the other four are corroborating and give the report breadth.

**7.1 NTK spectroscopy (primary).** Empirical NTK on a fixed probe set; track eigenvalue spectrum, its decay exponent, the condition number, and the per-loss-block eigenvalue mass (residual vs BC vs IC). Plot `λ_i` vs `i` on log-log for `C-MLP` and `Q-serial` on the same axes at the same step. **Predicted signature: a plateau in the QAPINN spectrum located at the encoded band.** Also track NTK drift `‖Θ_t − Θ_0‖/‖Θ_0‖` to check whether the hybrid stays in the lazy regime (the quantum layer may *break* lazy training — that would be a real finding either way).

**7.2 Per-frequency error trajectory (primary).** Project the error `u_θ − u*` onto Fourier modes each checkpoint; plot `|ê(k,t)|` as a heatmap (k × training-step). The classical PINN gives the textbook staircase (low `k` first). The QAPINN should show simultaneous decay across `Ω`. **Overlay the SMCD design card's `Ω` on the heatmap.** If the improved band and `Ω` coincide, C3 is proven visually in one figure.

**7.3 Collocation-point attribution.** Integrated Gradients / gradient×input of the residual loss w.r.t. input coordinates, aggregated into an attribution field over the domain. Shows *where* each model is "paying attention" — expected: classical spreads uniformly, hybrid concentrates on the steep-gradient region (Burgers shock, Helmholtz oscillation). Sanity-check attribution against the true residual field (a completeness check, since IG is known to be unstable — we report the correlation, not just the picture).

**7.4 Quantum-specific capacity: Fisher information spectrum & effective dimension.** Compute the empirical Fisher of the VQC block; report effective dimension vs the classical block. Distinguishes "more expressive" from "just more parameters."

**7.5 Trainability: gradient variance / barren plateau tracking.** `Var[∂_θ L]` vs qubit count and depth, measured, plotted, and compared to the `2^{-n}` prediction. This is the honest cost side of the ledger and belongs in the recommendations.

**7.6 Layerwise probing + loss landscape.** Linear probes on hidden/quantum-feature representations for how much of `u*` each layer already encodes (a CKA/probe-`R²` per layer), plus 2-D filtered loss-landscape slices for each family. Cheap, visually strong for slides.

**Discipline:** every XAI claim needs (a) a prediction stated *before* the run, in `docs/predictions.md`, and (b) a matched control. Post-hoc storytelling over a SHAP plot is exactly the failure mode the XAI-for-PINN literature is criticized for; we avoid it by pre-registering.

---

## 8. Experiment matrix

```
5 PDEs × 7 model families × 5 seeds × {noiseless, shot-noise 1024, depolarizing p=1e-3}
```
Plus targeted sweeps:
- **Coverage sweep** (the key validation): vary SMCD coverage `|Ŝ∩Ω|/|Ŝ|` ∈ {0.2 … 1.0} on P1/P4, plot final L2 error vs coverage. Monotone decrease ⇒ methodology validated.
- **Depth/qubit sweep** on P4 for the barren-plateau frontier.
- **α sweep** on P1 (amplitude of the high-frequency component) to find where quantum starts winning.

Metrics: relative L2, L∞, residual norm, per-frequency error, steps-to-tolerance, params, wall-clock, circuit evaluations, gradient-variance, NTK condition number.

Statistics: 5 seeds, report median + IQR, paired tests across families on the same seeds. No single-run bar charts.

---

## 9. Repository layout

```
QAPINN_Spectra/
├── project.md                  # this file
├── README.md                   # 60-second version + headline figure
├── FINDINGS.md                 # §10 deliverable: key findings + recommendations
├── pyproject.toml / uv.lock    # pinned env
├── Makefile                    # make repro-p1 … make repro-all, make paper, make slides
├── configs/                    # YAML per experiment; every run is fully specified by one file
│   ├── pde/{poisson,heat,burgers,helmholtz,allen_cahn}.yaml
│   ├── model/{c_mlp,c_ff,c_rff_matched,q_serial,q_parallel,q_random,q_octave}.yaml
│   └── exp/*.yaml
├── src/qapinn/
│   ├── pdes/                   # residual operators + analytic/reference solutions
│   ├── reference/              # OUR spectral & FD solvers (ground truth, not another NN)
│   ├── models/
│   │   ├── mlp.py  fourier_features.py
│   │   ├── circuits.py         # PennyLane templates, parameterized by SMCD output
│   │   └── hybrid.py           # serial / parallel / octave-ensemble
│   ├── smcd/
│   │   ├── symbol.py           # Prop. 2: PDE → target spectrum
│   │   ├── design.py           # Algorithm 1 → design card
│   │   └── card.py             # serialization + coverage metric
│   ├── train/                  # loop, hard-BC ansatz, LBFGS+Adam schedule, checkpointing
│   ├── xai/
│   │   ├── ntk.py  spectral_error.py  attribution.py
│   │   ├── fisher.py  gradvar.py  probes.py  landscape.py
│   └── viz/
├── scripts/                    # run_experiment.py, sweep.py, make_figures.py
├── tests/                      # pytest: symbol correctness, circuit-spectrum unit tests,
│                               # parameter-shift vs autodiff agreement, MMS convergence
├── notebooks/                  # 01_theory_walkthrough … 05_results (narrative, not source of truth)
├── docs/
│   ├── derivations/            # §3 items, one .md each, LaTeX
│   ├── predictions.md          # pre-registered predictions (timestamped, committed before runs)
│   └── REPRODUCE.md
├── results/                    # committed: design cards, metrics JSON, seeds. NOT raw checkpoints.
├── paper/                      # LaTeX, 5–15 pp
└── slides/                     # Marp or Beamer, built from the same figures
```

**Non-negotiables:** every figure regenerable by one `make` target from committed metrics; every run stamped with config hash + git SHA + seed; `tests/test_circuit_spectrum.py` numerically verifies Proposition 1 for the actual circuits used (FFT the circuit output, assert the peaks are at `Ω`) — that test *is* the proof that the implementation matches the theory, and it's the first thing a reviewer should be pointed to.

---

## 10. Deliverables

| Artifact | Spec |
|---|---|
| `paper/main.pdf` | 12 pp: Intro · Background (§3 condensed) · **Methodology/SMCD with Prop 1–4 + Alg 1** · XAI protocol · Results (coverage-vs-error plot, NTK spectra, frequency heatmaps) · Negative results · Limitations · Conclusion. Math section is the centerpiece — the brief explicitly asks for mathematical basis. |
| Repo | §9, public, MIT, with the headline figure in the README |
| `docs/REPRODUCE.md` | §11 |
| `slides/` | ~18 slides: problem → the one theorem → the design rule → the two XAI figures → the honest negative result → recommendations |
| `FINDINGS.md` | Numbered findings, each with the figure that supports it and a confidence level; a **decision table** ("PDE has property X ⇒ use/don't use a quantum layer, with these settings") |

---

## 11. Reproducibility contract

- `uv sync` (or `conda env create -f environment.yml`) — fully pinned; PennyLane + PyTorch, CPU-only path guaranteed to work.
- `make repro-quick` — P1 + P4(k=4), 1 seed, ~15 min CPU, reproduces the two headline figures at reduced fidelity.
- `make repro-all` — full matrix; document the actual wall-clock on stated hardware (expect hours-to-days; simulating VQC gradients is the cost driver — use `adjoint` differentiation and `default.qubit` batched).
- Determinism: global seed control, `torch.use_deterministic_algorithms(True)`, seeds recorded per run; a `results/manifest.json` mapping every figure → the runs that produced it.
- A `--smoke` flag on every script so a reviewer can verify the pipeline in 60 seconds.
- No hardware required. Optional: one small IBMQ/Braket run of the trained P1 circuit as a hardware-feasibility appendix — nice-to-have, not on the critical path.

---

## 12. Phases

| Phase | Work | Exit criterion |
|---|---|---|
| **0. Foundations** | §3 derivations written by hand; reference solvers for P1–P4; classical PINN reproduces Raissi's Burgers result | Burgers L2 error matches published value; MMS convergence test passes |
| **1. Instruments** | XAI suite implemented and validated on *classical* models only; spectral bias reproduced on P1 | The staircase heatmap exists for `C-MLP`. If you can't see spectral bias, nothing downstream means anything. |
| **2. Theory + SMCD** | Prop 1–4 written; `smcd/` implemented; `test_circuit_spectrum.py` green | Circuit FFT peaks land on predicted `Ω` to machine precision |
| **3. Main experiments** | Full matrix; predictions pre-registered *before* running | Coverage-vs-error plot exists, whatever it says |
| **4. Honesty pass** | Negative controls, `C-RFF-matched` ablation, noise study, barren-plateau frontier | Every claim in C1–C5 marked confirmed / refuted / inconclusive |
| **5. Package** | Paper, slides, FINDINGS, repro run from a clean clone on a different machine | Clean-clone `make repro-quick` succeeds |

---

## 13. Risks

| Risk | Mitigation |
|---|---|
| **`C-RFF-matched` beats the quantum model** — i.e. all the benefit is "matched frequencies," none is "quantum" | This is a *legitimate result* and gets reported as the paper's most useful finding. The methodology (Prop 2, Alg 1 steps 1–4) survives intact and becomes a classical design rule with a quantum instantiation. Frame the paper so this outcome is still a contribution. |
| Simulation cost explodes | Cap at 6–8 qubits, `adjoint` diff, batched `default.qubit`, octave-split instead of deep circuits |
| Barren plateaus kill training | Local observables, `L ≤ 6`, small-angle init, layerwise warm-up — all already in Alg 1 |
| NTK computation too expensive in 2D | Subsample probe set (≤512 points), use JVP-based empirical NTK, not full Jacobian materialization |
| Scope creep | P5 is explicitly a stretch goal; drop it without guilt |
| Reference [4] contradicts the framing | Resolve before Phase 2 (§1 open item) |

---

## 14. Immediate next actions

1. **Get reference [4]** — paste the real citation.
2. `git init`, scaffold §9, pin env, CI running `pytest` on push.
3. Write `docs/derivations/07_circuit_fourier_spectrum.md` (Prop 1) **first** — it is the load-bearing theorem, and writing it by hand before any code is the point of the exercise.
4. Implement `reference/` solvers + `C-MLP` on P1; reproduce the spectral-bias staircase. Phase 1's exit criterion is the real project start.

---

## 15. Sources consulted for positioning

- [The effect of data encoding on the expressive power of variational quantum ML models — Schuld, Sweke, Meyer (arXiv:2008.08605)](https://arxiv.org/abs/2008.08605) — Proposition 1's origin
- [Quantum models as Fourier series — PennyLane demo](https://pennylane.ai/demos/tutorial_expressivity_fourier_series) — reference implementation to validate against
- [Gradients and frequency profiles of quantum re-uploading models — Quantum (2024)](https://quantum-journal.org/papers/q-2024-11-14-1523/) — links spectrum to trainability; directly relevant to §7.5
- [Hybrid quantum physics-informed neural networks — Mach. Learn.: Sci. Technol.](https://iopscience.iop.org/article/10.1088/2632-2153/ad43b2/pdf) — likely lineage of ref [4]
- [Hybrid quantum-classical PINNs for nonlinear PDEs: when and where is hybridization effective? (arXiv:2606.04679)](https://arxiv.org/pdf/2606.04679) — closest prior work; **differentiate:** they characterize *when* empirically, we give a *constructive* design rule + XAI mechanism
- [Quantum PINNs for Maxwell's equations: circuit design, barren plateau mitigation (arXiv:2506.23246)](https://arxiv.org/pdf/2506.23246) — circuit-design prior art to cite and go beyond
- [A classical-quantum hybrid architecture for PINNs (arXiv:2511.07216)](https://arxiv.org/html/2511.07216v1)
- [NTK analysis to probe convergence in physics-informed solvers: PIKANs vs PINNs (arXiv:2506.07958)](https://arxiv.org/abs/2506.07958) — the template for §7.1; we do the same for quantum
- [Overcoming Fourier locking in data re-uploading classifiers via spectral homotopy (arXiv:2607.11013)](https://arxiv.org/html/2607.11013) — a known failure mode of trainable-scaling circuits; read before fixing `ω_j`
- [Fundamental flaws of PINNs and explainability methods in engineering systems](https://www.sciencedirect.com/science/article/pii/S0360835225008502) — the criticism §7's pre-registration discipline is designed to survive
