# Phase 1 — XAI Instruments (Days 3–4)

**Goal:** build and validate all seven measurement instruments **on classical models only**. Every
instrument must be trustworthy before a quantum model exists, otherwise a Phase-3 anomaly is
un-diagnosable — you would not know whether the instrument or the physics is lying.

**Exit gate (`project.md` §12 Phase 1):**
> **The spectral-bias staircase heatmap exists for `c_mlp` on P1.**
> If you cannot see spectral bias, nothing downstream means anything.

Concretely: the T1.5 heatmap must show the `ω = π` mode converging visibly earlier than the `ω = 15π`
mode, with a clear separation in steps-to-tolerance (target: ≥ 10× more steps for the high mode).

---

## Instrument register

| Instrument | `project.md` § | Module | Load-bearing? |
|---|---|---|---|
| NTK spectroscopy | 7.1 | `xai/ntk.py` | **Yes — primary** |
| Per-frequency error trajectory | 7.2 | `xai/spectral_error.py` | **Yes — primary** |
| Collocation attribution (IG) | 7.3 | `xai/attribution.py` | corroborating |
| Fisher / effective dimension | 7.4 | `xai/fisher.py` | corroborating |
| Gradient variance (barren plateau) | 7.5 | `xai/gradvar.py` | corroborating (cost ledger) |
| Layerwise probes + loss landscape | 7.6 | `xai/probes.py`, `xai/landscape.py` | corroborating |
| **Encoder spectral drift** | *new, from D3* | `xai/drift.py` | corroborating |

---

## T1.1 — Plot style and viz primitives

**Depends on:** T0.22
**Files:** `src/qapinn/viz/style.py`

One matplotlib style used by **every** figure in the project (paper and slides share figures, so
they must share style). Define: colourblind-safe palette with a **fixed family→colour map**, serif
font matching the paper, `figsize` presets (`single`, `double`, `slide`), consistent log-log helpers,
and `save_figure(fig, name)` which writes both `.pdf` (paper) and `.png` at 200 dpi (slides) into
`paper/figures/` and records the figure in `results/manifest.json`.

Fixed colour map: `c_mlp` grey, `c_ff` blue, `c_rff_matched` green, `q_serial` red, `q_parallel`
orange, `q_random` purple, `q_octave` brown. Never let matplotlib pick.

**DoD:** `save_figure` produces both files and appends a manifest entry containing the figure name and
the list of `run_id`s used.

---

## T1.2 — NTK computation core

**Depends on:** T0.19
**Files:** `src/qapinn/xai/ntk.py`, `tests/test_ntk.py`

Empirical NTK for a PINN. The key point: the NTK is taken on the **operator outputs**, not just `u`.

```python
def jacobian(model, pde, probe_x: Tensor, output: str,
             group: str = "all", chunk: int = 64) -> Tensor:
    """output: 'u' | 'residual' | 'bc'
       group : 'all' | 'classical' | 'quantum'   (uses model.param_groups())
       returns J: [P, N_params_in_group]"""

def ntk(model, pde, probe_x, output="residual", group="all") -> Tensor:   # [P,P]

def ntk_blocks(model, pde, probe_x) -> dict[str, Tensor]:
    """Prop. 4 decomposition. Returns
       {'hyb':Θ, 'cl':J_cl J_clᵀ, 'q':J_q J_qᵀ, 'cross':J_cl J_qᵀ}"""

def spectrum_stats(K: Tensor) -> dict:
    """{'eigenvalues': [...], 'decay_exponent': float, 'condition_number': float,
        'trace': float, 'effective_rank': float}
    decay_exponent = slope of a least-squares line fit to log λ_i vs log i over
    the middle 80% of the index range (drop the first and last 10% — the head is
    dominated by a few large modes and the tail by numerical noise)."""

def block_mass(model, pde, probe_r, probe_b) -> dict:
    """Per-loss-block eigenvalue mass: trace(Θ_rr), trace(Θ_bb), trace(Θ_ri).
    Only meaningful under bc_mode='soft' (D6)."""

def ntk_drift(K_t: Tensor, K_0: Tensor) -> float:    # ‖Θ_t − Θ_0‖_F / ‖Θ_0‖_F
```

Implementation: `torch.func.jacrev` over a functional form of the model
(`torch.func.functional_call`), `vmap`ped over probe points, chunked at 64 (D7). Probe set is
**fixed across the whole project** for comparability: a deterministic quasi-uniform set of
`P = 512` points from `pde.eval_grid`, subsampled with a fixed stride. Store the probe set once.

**DoD:** `tests/test_ntk.py` asserts three things:
1. **Analytic check.** For a *linear* model `u = w·φ(x)` with fixed features `φ`, the NTK equals
   `φ(x)φ(x')ᵀ` exactly (to `1e-10`).
2. **Prop. 4 identity.** `Θ_hyb == Θ_cl + Θ_q + 2·sym(Θ_cross)` to `1e-10` on a model with two
   parameter groups (use a dummy two-group classical model in Phase 1; the real hybrid arrives in
   Phase 2). *This converts Prop. 4 from an assertion into a tested identity.*
3. **PSD.** `Θ_cl` and `Θ_q` have no eigenvalue below `−1e-10`.

---

## T1.3 — NTK spectroscopy reporting + figure

**Depends on:** T1.2, T1.1
**Files:** `src/qapinn/xai/ntk.py` (reporting half), `scripts/make_figures.py` (fig: `ntk_spectrum`)

At each checkpoint, save `xai/ntk_step<N>.npz` with eigenvalues, block masses, condition number,
decay exponent, and drift-vs-step-0.

Figure `ntk_spectrum`: `λ_i` vs `i` on log-log, `c_mlp` and `q_serial` on the same axes at the same
step, with the **encoded band `Ω` shaded** (Phase 2 supplies `Ω`; leave the shading hook now).

> **Predicted signature (pre-register in T3.1):** a plateau in the QAPINN spectrum located at the
> encoded band, against a clean power-law decay for `c_mlp`.

**DoD:** the figure renders for `c_mlp` alone (two-family version arrives Phase 2), and the measured
`decay_exponent` for a tanh MLP on P1 is negative and in a plausible range (roughly `−1` to `−4`);
record the measured value in `docs/plan/BENCH.md` as the classical baseline.

---

## T1.4 — Per-frequency error trajectory

**Depends on:** T0.9, T0.19
**Files:** `src/qapinn/xai/spectral_error.py`, `tests/test_spectral_error.py`

```python
def per_frequency_error(u_pred: np.ndarray, u_exact: np.ndarray,
                        grid_shape: tuple[int, ...], dx: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (omega, |ê(ω)|).
    1-D: full FFT of the error on the uniform grid.
    2-D: FFT2, then RADIAL BINNING of |ê| into |ω| bins (default 64 bins), because the
         heatmap axis must stay 1-D. Also return the unbinned 2-D array for P4, where
         the four discrete target frequencies matter individually."""

def error_trajectory(run_dir: Path) -> np.ndarray:
    """[n_checkpoints, n_freq] assembled from the per-checkpoint arrays."""

def steps_to_tolerance_per_mode(traj, omega, tol=0.1) -> dict[float, int]:
    """For each target mode, the first checkpoint where |ê(ω)| falls below tol × its initial value.
    This is the number that quantifies the staircase."""
```

For time-dependent PDEs (P2, P3) take the FFT in **space only**, at a fixed set of time slices
`t ∈ {0.1, 0.5, 0.9}`, and report one heatmap per slice.

**DoD:** `tests/test_spectral_error.py` — construct a synthetic error field
`e = 0.5 sin(πx) + 0.2 sin(15πx)`; assert the returned spectrum has exactly two peaks, at `ω = π` and
`ω = 15π`, with amplitude ratio `0.5 : 0.2` to 1 %. Assert Parseval: `Σ|ê|² ≈ Σ|e|²` to `1e-10`.

---

## T1.5 — ★ THE SPECTRAL-BIAS STAIRCASE (Phase-1 gate figure)

**Depends on:** T1.4, T1.1
**Files:** `scripts/make_figures.py` (fig: `staircase_cmlp_p1`)

Train `c_mlp` on P1 with `α = 0.3`, full 20 k steps, 3 seeds. Produce the heatmap
`|ê(ω, t)|` with `ω` on the vertical axis and training step on the horizontal (log scale),
`log10` colour scale, median over seeds.

**This is the Phase-1 exit gate.** The figure must show the textbook staircase: the `ω = π` mode
collapsing early while `ω = 15π` persists.

Record in `docs/plan/BENCH.md`: `steps_to_tolerance_per_mode` for both modes and their ratio.
**Gate condition:** ratio ≥ 10. If the ratio is small, the network is too wide, `α` too large, or the
learning rate too high — tune the *baseline* until spectral bias is clearly visible, and record what
was needed. Do not proceed with an invisible baseline effect.

**DoD:** `paper/figures/staircase_cmlp_p1.pdf` exists and the ratio is recorded.

---

## T1.6 — Collocation-point attribution (Integrated Gradients)

**Depends on:** T0.19
**Files:** `src/qapinn/xai/attribution.py`, `tests/test_attribution.py`

```python
def integrated_gradients(model, pde, x: Tensor, baseline: Tensor | None = None,
                         n_steps: int = 64, target: str = "residual") -> Tensor:
    """IG of the target scalar w.r.t. input coordinates. Baseline defaults to the domain centre.
    Returns [B, d] attributions."""

def attribution_field(model, pde, grid_n: int = 128) -> np.ndarray
def completeness_error(model, pde, x, attrs) -> float:
    """IG's completeness axiom: Σ attributions ≈ F(x) − F(baseline). Returns the relative
    violation. IG is known to be unstable — we REPORT this number, we do not hide it."""
def attribution_residual_correlation(attrs_field, residual_field) -> float:
    """Spearman correlation between the attribution field and the true residual field.
    project.md §7.3 requires the correlation, not just the picture."""
```

**DoD:** `tests/test_attribution.py` — on a linear model, IG attributions equal
`(x − baseline) · w` exactly; `completeness_error < 1e-6` for `n_steps = 256`.

---

## T1.7 — Fisher information & effective dimension

**Depends on:** T0.19
**Files:** `src/qapinn/xai/fisher.py`, `tests/test_fisher.py`

```python
def empirical_fisher(model, pde, x, group="all") -> Tensor:
    """F = (1/B) Σ_b g_b g_bᵀ where g_b = ∇_θ residual(x_b). Shape [N,N].
    For N > 4000, return only the eigenvalue spectrum via a randomised SVD of J (never
    materialise F) — J is [B,N] and eigenvalues of F are singular values of J squared / B."""

def effective_dimension(F_eigs: np.ndarray, n_data: int, gamma: float = 1.0) -> float:
    """Abbas et al. effective dimension:
       d_eff = 2 · log( (1/V) ∫ sqrt(det(I + (γ n / 2π log n) F̂)) dθ ) / log(γ n / 2π log n)
    Use the standard single-point (MLE) approximation:
       d_eff ≈ 2 · Σ_i log(1 + c λ_i) / log(c),   c = γ n / (2π log n),  λ normalised so mean(λ)=1.
    Document this approximation explicitly in the paper — do not present it as the full integral."""
```

**DoD:** `tests/test_fisher.py` — for a linear model with `k` independent features, `d_eff` recovers
`k` within 10 % as `n_data → large`. Assert the Fisher eigenvalue spectrum is non-negative.

---

## T1.8 — Encoder spectral drift (new instrument, from D3)

**Depends on:** T0.19
**Files:** `src/qapinn/xai/drift.py`, `tests/test_drift.py`

Because the encoder is affine (D3), the model's realised frequencies in physical coordinates are
`Ω·A(t)`. Track how they move.

```python
def encoder_drift(model) -> dict:
    """{'frobenius': ‖A − I‖_F, 'realised_omega': Ω·A, 'coverage_now': float}
    Classical families return {} — this instrument is quantum/FF-specific."""
```
For Fourier-feature families, `A` is implicit (the `B` matrix is fixed) so drift is 0 by construction;
report it anyway for symmetry.

**DoD:** with `A` frozen at identity, `frobenius == 0` and `realised_omega == Ω`. After manually
perturbing `A` by a known scale factor `s`, `realised_omega == s·Ω`.

---

## T1.9 — Gradient variance / barren-plateau tracking

**Depends on:** T0.19
**Files:** `src/qapinn/xai/gradvar.py`, `tests/test_gradvar.py`

```python
def gradient_variance(model, pde, n_samples: int = 100, group: str = "quantum") -> dict:
    """Re-initialise the group's parameters n_samples times from the init distribution,
    compute ∇_θ L each time on a FIXED batch, return
    {'var_mean': mean over coordinates of Var[∂_θ L],
     'var_per_param': [...], 'n_qubits': n, 'n_layers': L}"""

def barren_plateau_fit(results: list[dict]) -> dict:
    """Fit log(var) = a − b·n over qubit counts. Return {'slope_b': ..., 'r2': ...}.
    The theoretical prediction for a global cost + deep random circuit is b = log(2)
    (i.e. Var ~ 2^{-n}).  We compare our LOCAL-observable, SHALLOW circuits against it —
    the expectation is that we are well ABOVE the barren-plateau line, and that is the
    point of the cost-ledger section."""
```

**DoD:** on a classical MLP the variance is roughly `n`-independent (slope ≈ 0), confirming the
instrument does not manufacture a plateau where none exists.

---

## T1.10 — Layerwise linear probes

**Depends on:** T0.19
**Files:** `src/qapinn/xai/probes.py`, `tests/test_probes.py`

```python
def layer_probe_r2(model, pde, grid, layer_outputs: dict[str, Tensor]) -> dict[str, float]:
    """Ridge-regress u* on each layer's activations; return R² per layer.
    Answers 'how much of the solution is already linearly decodable here?'"""

def cka(X: Tensor, Y: Tensor) -> float:
    """Linear centred kernel alignment between two representations."""
```
Hook layer outputs with `register_forward_hook`; for hybrid models the quantum-feature vector is one
of the layers (that is the interesting row of the table).

**DoD:** probe `R²` on the *input* layer for P1 is low (a linear function of `x` cannot represent
`sin(15πx)`), and `R²` on the final pre-head layer is high (> 0.95) for a trained model. `cka(X, X) == 1`.

---

## T1.11 — Loss landscape slices

**Depends on:** T0.19
**Files:** `src/qapinn/xai/landscape.py`

Filter-normalised 2-D random-direction slices (Li et al.). Two random direction vectors, normalised
per-layer to the norm of the corresponding trained weights; evaluate loss on a `25 × 25` grid at
`±1` in each direction.

**DoD:** produces a `[25,25]` array whose minimum sits at the centre (the trained point) for a
converged model.

---

## T1.12 — XAI hook registry wired into the training loop

**Depends on:** T1.2, T1.4, T1.6, T1.7, T1.8, T1.9, T1.10, T1.11
**Files:** `src/qapinn/xai/__init__.py`, edit `src/qapinn/train/loop.py`

```python
INSTRUMENTS = {"ntk":..., "specerr":..., "attribution":..., "fisher":...,
               "drift":..., "gradvar":..., "probes":..., "landscape":...}
def run_instruments(model, pde, cfg, step, out_dir, which: tuple[str,...]) -> None
```
The training loop calls this at every checkpoint in `cfg.train.checkpoints`. Config gains
`train.instruments: tuple[str,...]` defaulting to all. Expensive instruments (`landscape`, `gradvar`)
default to **final checkpoint only** — record that choice in the config, not in code.

**DoD:** a `--smoke` run with all instruments enabled completes in < 120 s and writes one `.npz`
per instrument per checkpoint.

---

## T1.13 — Phase 1 gate check

**Depends on:** T1.5, T1.12

Confirm:
1. `pytest` green including all instrument tests.
2. `paper/figures/staircase_cmlp_p1.pdf` shows the staircase, ratio ≥ 10, recorded in `BENCH.md`.
3. Every instrument runs on `c_mlp`, `c_ff`, `c_rff_matched` without error on all four PDEs.
4. NTK decay exponent for `c_mlp` recorded as the classical baseline.

**DoD:** commit tagged `phase1-complete`. Note in `BENCH.md` the wall-clock cost per checkpoint of
the full instrument suite — this feeds the Phase-3 budget.
