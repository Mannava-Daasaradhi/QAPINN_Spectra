# Phase 2 — Theory, Quantum Layer, and SMCD (Days 5–7)

**Goal:** the load-bearing theorem, the circuits that realise it, and the constructive PDE→circuit map.

**Exit gate (`project.md` §12 Phase 2):**
> `tests/test_circuit_spectrum.py` green — circuit FFT peaks land on the predicted `Ω` to machine precision.

That test **is** the proof that the implementation matches the theory, and it is the first thing a
reviewer should be pointed at (`project.md` §9). It is a hard gate: if it fails, do not proceed.

---

## The mathematics this phase implements

### Proposition 1 — circuit spectrum

For an encoding gate `exp(−i x H)` with generator `H` having eigenvalues `{λ_1..λ_d}`, an `L`-layer
data-re-uploading circuit outputs

```
f(x) = Σ_{ω ∈ Ω} c_ω(θ) e^{iωx} ,    Ω = { Σ_{l=1}^{L} (λ_a^{(l)} − λ_b^{(l)}) }
```

For a **Pauli** generator scaled by `ω_l` — i.e. gate `RZ(ω_l x) = exp(−i ω_l x Z/2)` — the generator
eigenvalues are `±ω_l/2`, so the eigenvalue **differences** are `{−ω_l, 0, +ω_l}`. Hence

```
Ω = { Σ_l m_l ω_l  :  m_l ∈ {−1, 0, +1} }
```

> **The factor of 2.** Generator eigenvalues are `±ω_l/2`; realised frequencies are `±ω_l`.
> Getting this wrong halves every frequency in the project. `tests/test_fft_convention.py` (T0.9)
> and `tests/test_circuit_spectrum.py` (T2.6) both pin it.

**Ternary scalings.** With `ω_l = Δ·3^{l−1}`, the sum `Σ_l m_l 3^{l−1}` is the *balanced ternary*
representation, which is a **bijection** onto the integers in `[−(3^L−1)/2, (3^L−1)/2]`. Therefore

```
Ω = Δ · { −(3^L−1)/2 , … , −1, 0, 1, … , (3^L−1)/2 }        |Ω| = 3^L  exactly, no degeneracy
max|Ω| = Δ (3^L − 1)/2
```

**Linear scalings.** With `ω_l = ω_0` for all `l`, `Ω = ω_0·{−L,…,L}`, `|Ω| = 2L+1` — heavily
degenerate. This contrast (`3^L` vs `2L+1`) is the design knob that `project.md` §5.2 says almost all
QPINN papers leave at 1, and it is directly testable (T2.6).

**Corrected depth rule (D5 — `project.md` Alg. 1 step 4 is wrong).**
```
max|Ω| ≥ K   ⇔   Δ(3^L − 1)/2 ≥ K   ⇔   L ≥ log_3( 2K/Δ + 1 )
L = ⌈ log_3( 2K/Δ + 1 ) ⌉
```
*P1 check:* `K = 15π`, `Δ = π` ⇒ `3^L ≥ 31` ⇒ `L = 4`. The spec's `⌈log_3(K/Δ)⌉ = 3` reaches only
`±13π` and would silently miss the target mode.

### Proposition 2 — PDE target spectrum
`û(ω) = f̂(ω)/σ(ω)`, so the essential support `S_ε` is computable **before training**.

### Proposition 3 — matching condition
If `S_ε ⊆ Ω`, the hybrid can represent the solution to `ε` with a *linear* head, so the residual
problem is convex in `w`. T0.16's DoD already dry-runs this classically.

### Proposition 4 — NTK consequence
`Θ_hyb = Θ_cl + Θ_q + 2Θ_×`. Already implemented and **tested as an identity** in T1.2.

---

## T2.1 — ★ Write the Prop. 1 derivation by hand, first

**Depends on:** T1.13
**Files:** `docs/derivations/07_circuit_fourier_spectrum.md`

`project.md` §14 is emphatic: write this **before any quantum code**, because doing it by hand is the
point of the exercise. Derive, in LaTeX:
1. One qubit, one layer: `⟨0|U†(x,θ) Z U(x,θ)|0⟩` expanded to show `{−ω, 0, +ω}`.
2. One qubit, `L` layers: the sum-of-differences set.
3. General statement (Schuld–Sweke–Meyer 2021) with the multi-qubit / multi-dimensional case.
4. The ternary corollary, including the balanced-ternary bijection and **the corrected depth rule**.
5. The factor-of-2 note.

**DoD:** the document exists, contains all five items, and the ternary corollary's arithmetic is
worked for `L = 1,2,3,4` with the explicit frequency sets listed.

---

## T2.2 — The remaining background derivations

**Depends on:** T2.1 (can run in parallel with T2.3–T2.7)
**Files:** `docs/derivations/01…10_*.md`

One short `.md` per item of `project.md` §3 (items 1–6, 8–10; item 7 is T2.1). Each gets a one-slide
version later (T5.10). These are owner-facing "I can derive this on a whiteboard" artifacts — the
implementing agent should draft them, and the owner should verify each before the paper is written.

**DoD:** ten files exist, each with a derivation (not a summary), each ≤ 2 pages.

---

## T2.3 — ★ Batched statevector simulator (`qsim.py`)

**Depends on:** T1.13
**Files:** `src/qapinn/models/qsim.py`, `tests/test_qsim.py`

The performance foundation (D2). Minimal, autograd-native, CUDA-resident.

```python
class StateVectorSim:
    """State shape [B, 2]*n, complex128. Qubit 0 is the FIRST trailing axis.

    Required ops (nothing more — keep this file small and auditable):
      zeros_state(batch, n) -> Tensor
      apply_1q(state, U: [2,2] or [B,2,2], wire) -> Tensor
      apply_cz(state, w0, w1) -> Tensor
      apply_cnot(state, control, target) -> Tensor
      expval_z(state, wire) -> [B]
      expval_z_mean(state) -> [B]
    """
def rx(theta): ...   # [B,2,2] batched rotation matrices
def ry(theta): ...
def rz(theta): ...
```

**Implementation notes (these are where the bugs live):**
- `apply_1q`: `torch.movedim` the target wire axis to the end, reshape to `[..., 2]`, `einsum` with
  the (possibly batched) `2×2`, move back. Do **not** build `2^n × 2^n` matrices.
- Batched gate angles: encoding gates have per-sample angles (`ω x_b`), so `U` must broadcast over
  the batch. Rotation gates for `θ` are shared across the batch. Support both.
- `apply_cz` is diagonal: multiply the `|11⟩` sub-block by `−1`. No matmul needed.
- Keep everything complex128 and differentiable — never call `.detach()`, `.item()`, or `.numpy()`
  inside the simulator.

**DoD:** `tests/test_qsim.py` asserts:
1. Norm preservation: `‖state‖² == 1` to `1e-12` after a random circuit.
2. Known values: `RY(π/2)` on `|0⟩` gives `⟨Z⟩ = 0`; `RX(π)` gives `⟨Z⟩ = −1`.
3. Bell state via `RY(π/2)` + `CNOT` gives `⟨Z_0⟩ = ⟨Z_1⟩ = 0` and correlation 1.
4. **Second derivatives flow:** `torch.autograd.grad` twice through `expval_z` w.r.t. an input angle
   returns a finite tensor. This is D1's requirement and must be tested explicitly.

---

## T2.4 — Re-uploading circuit module

**Depends on:** T2.3
**Files:** `src/qapinn/models/circuits.py`

```python
class ReuploadCircuit(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int,
                 scalings: np.ndarray,          # [n_layers, n_qubits] ANGULAR, buffer not parameter (D4)
                 wire_to_dim: tuple[int, ...],  # which input coordinate each wire encodes
                 entangler: str = "ring_cz",    # "ring_cz" | "none"
                 observable: str = "z0",        # "z0" | "z_mean"   (local only — Alg.1 step 8)
                 init_sigma: float = 0.1):      # small-angle init  (Alg.1 step 10)
        ...
    def forward(self, z: Tensor) -> Tensor:     # [B, d] -> [B, 1]
    def frequencies(self) -> np.ndarray:        # Ω, computed from scalings, per Prop. 1
```

Layer structure (fixed — do not vary this without updating T2.6):
```
for l in range(L):
    for q in range(n):  RZ( scalings[l,q] * z[:, wire_to_dim[q]] )   # encoding
    for q in range(n):  RY(theta[l,q,0]) ; RZ(theta[l,q,1])          # trainable
    if entangler == "ring_cz":  CZ(q, (q+1) % n) for all q
final: RY/RZ trainable layer, then measure the observable
```
Prepend a `RY(π/2)` on every wire at the start so the encoding `RZ` acts on a superposition (an `RZ`
on `|0⟩` is a global phase and would produce a constant output — **this is the single most common
"my circuit has no frequency content" bug**).

`frequencies()` enumerates `{Σ_l m_l ω_l : m_l ∈ {−1,0,1}}` per wire-dimension and, when the
entangler is present, the cross sums across wires mapped to different input dimensions.

**Global-observable ban:** `observable` must be local (Alg. 1 step 8). Raise on anything else.

**DoD:** `frequencies()` for `L=3`, ternary `Δ=π`, one wire returns exactly the 27 values
`π·{−13,…,13}`; for linear scalings returns the 7 values `π·{−3,…,3}`.

---

## T2.5 — Oracle agreement: `qsim` vs PennyLane

**Depends on:** T2.4
**Files:** `tests/test_qsim_vs_pennylane.py`

Build the identical circuit in PennyLane (`default.qubit`, torch interface) and compare outputs on
100 random `(x, θ)` draws for `n ∈ {2,4,6}`, `L ∈ {1,3,5}`, both entangler settings.

**DoD:** max absolute difference **< 1e-10**.
**If this fails:** the cause is almost always qubit-ordering/endianness or CZ wire indexing. Fix the
simulator — **never loosen the tolerance.** PennyLane is the oracle we cite in the paper (D2).

---

## T2.6 — ★★ HARD GATE: `test_circuit_spectrum.py` (Prop. 1 verified numerically)

**Depends on:** T2.5
**Files:** `tests/test_circuit_spectrum.py`

The protocol exploits the fact that the circuit output is an **exact trigonometric polynomial**, so a
correctly-sampled FFT is exact — which is why the tolerance can be `1e-8`, not `1e-2`.

```
1. Build a circuit with ternary scalings, Δ known, L layers, random θ.
2. All frequencies are integer multiples of Δ ⇒ f is periodic with period T = 2π/Δ.
3. Sample N = 4·max_index + 1 points on [0, T) — strictly above Nyquist, exactly one period,
   endpoint EXCLUDED. Zero spectral leakage by construction.
4. FFT. Map bin indices to angular frequency via omega_grid (T0.9).
5. ASSERT: every bin whose ω ∉ Ω has |coefficient| < 1e-8.
6. ASSERT: at least 80% of bins with ω ∈ Ω have |coefficient| > 1e-6  (a random θ can null a
   few coefficients by chance, so do not demand all of them; average over 5 random θ draws).
```

Required test cases:
- `L ∈ {1,2,3,4}`, ternary, 1 wire → assert `|Ω| = 3^L` and the support matches exactly.
- Linear scalings → assert support is `ω_0·{−L..L}` and **nothing outside it**.
- 2 wires, 2 input dims, `entangler="none"` → assert `⟨Z_0⟩` contains **only** wire-0 frequencies
  (no cross terms). *This is the numerical justification for Alg. 1 step 7's "avoid gratuitous
  entanglement".*
- 2 wires, `entangler="ring_cz"` → assert cross terms `ω_x ± ω_y` **do** appear.

**DoD:** all cases pass. Commit tagged `prop1-verified`.
**This is the project's scientific foundation. Do not proceed past it on a failing test.**

---

## T2.7 — Parameter-shift derivatives (verification + hardware faithfulness)

**Depends on:** T2.4
**Files:** `src/qapinn/models/pshift.py`, `tests/test_pshift.py`

```python
def psr_grad_theta(circuit, z, theta, idx) -> Tensor:      # ½[f(θ+π/2) − f(θ−π/2)]
def psr_grad_input(circuit, z, dim) -> Tensor:             # ∂_x via the SAME rule (input is an angle)
def psr_grad2_input(circuit, z, dim) -> Tensor:            # second-order shift rule
```

The second-order rule for a two-eigenvalue generator with scaling `ω`:
`∂²_x f = (ω²/2)·[ f(x+π/ω) − 2f(x) + f(x−π/ω) ]` — derive it in `docs/derivations/08_parameter_shift.md`
and state its validity conditions. When several encoding gates share the input, apply the rule
per-gate and sum (product rule), or use the multi-term generalisation; document which you used.

**DoD:** `tests/test_pshift.py` asserts parameter-shift and autodiff agree to `1e-9` for `∂_θ`, `∂_x`,
and `∂²_x`. This **verifies** `project.md` §3.8's claim rather than merely asserting it, and licenses
the appendix statement that the PINN residual is exactly computable on hardware.

---

## T2.8 — SMCD Part 1: target spectrum (Prop. 2)

**Depends on:** T2.6
**Files:** `src/qapinn/smcd/symbol.py`, `tests/test_symbol.py`

```python
@dataclass(frozen=True)
class TargetSpectrum:
    omega: np.ndarray       # [M, d] angular frequencies
    weight: np.ndarray      # [M] energy weight (see below)
    eps: float
    support: np.ndarray     # [M] bool mask, the set S_eps
    K: float                # max |omega| within the support
    delta: float            # min nonzero spacing within the support
    source: str             # "analytic" | "empirical"

def analytic_spectrum(pde, eps) -> TargetSpectrum | None
def empirical_spectrum(pde, eps, n_grid=256) -> TargetSpectrum
```

**Energy weight — the definition that makes P2 a quantitative negative control.**
- Steady problems (P1, P4): `w(ω) = |û(ω)|`.
- Time-dependent (P2, P3): `w(ω) = ‖û(ω, ·)‖_{L²(0,T)}` — the *time-integrated* amplitude.

This matters. For P2 (heat), `ω = 15π` has `w ≈ 0.020` against `w(π) ≈ 0.79`: **2.5 % of the energy**,
and it is concentrated at `t ≈ 0`. So SMCD predicts, *before any training*, that the maximum
achievable benefit from covering the high mode on P2 is ≈ 2.5 % of the error. That is a **quantitative
a-priori negative prediction**, which is a far stronger form of contribution C4 than "smoothing kills
high frequencies". Pre-register this number in T3.1.

`support = {ω : w(ω) > eps · max(w)}`.

The empirical path FFTs the **reference solution** (T0.13) — used for P3 (nonlinear, no closed-form
symbol) and as a cross-check everywhere else.

**DoD:** `tests/test_symbol.py` asserts
1. P1 analytic support is exactly `{π, 15π}` with weights `{1, α}`.
2. P4 (`k=10`) analytic support is exactly the four points `(±3π, ±1π)`.
3. **Analytic and empirical agree** for P1, P2, P4 — same support, weights matching to 1 %. (This
   validates the empirical path, which is the *only* path available for P3.)
4. P2's high-mode weight ratio is ≈ 0.025 (assert within 20 %).

---

## T2.9 — ★ SMCD Part 2: Algorithm 1

**Depends on:** T2.8
**Files:** `src/qapinn/smcd/design.py`, `tests/test_smcd_design.py`

Implement `project.md` §5.3 Algorithm 1, all 11 steps, **with the D5 correction at step 4**.

```python
def smcd(pde, eps: float = 1e-3, n_max: int = 8, L_max: int = 6,
         scaling_mode: str = "ternary",
         coverage_target: float | None = None) -> DesignCard
```

| Step | Implementation |
|---|---|
| 1 SYMBOL | `pde.symbol`; for Burgers use the viscous-scale estimate **and** the empirical spectrum, take the union, record both in the card |
| 2 TARGET | `analytic_spectrum` if available else `empirical_spectrum`; union over `t` for time-dependent |
| 3 BAND | `K = max|ω|` in support; `Δ = min nonzero spacing`; `d = pde.dim` |
| 4 DEPTH | **`L = ⌈log_3(2K/Δ + 1)⌉`** (D5); fall back to `linear` scalings when the support is a dense low band (heuristic: `K/Δ < 4`) |
| 5 WIDTH | `n = d + n_cross`; `n_cross = 1` when the support contains genuine mixed-frequency terms (P4), else 0 |
| 6 CHECK | if `n > n_max` or `L > L_max` → octave split (§5.4, T2.15); record that the split was triggered |
| 7 ENTANGLER | `ring_cz` iff cross terms are needed, else `none` |
| 8 OBSERVABLE | `z0` (or `z_mean`); global Pauli strings are rejected by T2.4 |
| 9 ANSATZ | hard-BC (D6 default) |
| 10 INIT | `θ ~ N(0, 0.1²)`; encoder `A = I` (D3) |
| 11 REPORT | emit the design card |

**`coverage_target` (for the Phase-3 coverage sweep).** When set, *deliberately de-tune* the design to
hit approximately that coverage, by reducing `L` and/or multiplying `Δ` by a mismatch factor. The
function must return the **achieved** weighted coverage in the card, not the requested one.

**DoD:** `tests/test_smcd_design.py` asserts
1. P1 → `L = 4`, `n = 1`, `entangler = "none"`, weighted coverage `= 1.0`.
2. P4 `k=10` → `n = 3` (2 dims + 1 cross), `entangler = "ring_cz"`, coverage `= 1.0`.
3. P2 → the card's `predicted_benefit` field is < 0.05 (the a-priori negative prediction).
4. `coverage_target=0.5` returns a card with achieved coverage in `[0.4, 0.6]`.
5. **Determinism:** two calls with the same inputs give byte-identical cards.

---

## T2.10 — SMCD Part 3: design card & coverage metric

**Depends on:** T2.9
**Files:** `src/qapinn/smcd/card.py`, `tests/test_smcd_depth.py`

```python
@dataclass(frozen=True)
class DesignCard:
    pde: str; eps: float
    target_omega: list; target_weight: list          # Ŝ and its weights
    omega_set: list                                  # Ω
    coverage: float                                  # |Ŝ ∩ Ω| / |Ŝ|            (count)
    coverage_weighted: float                         # Σ_{Ŝ∩Ω} w / Σ_Ŝ w        (PRIMARY)
    predicted_benefit: float                         # see below
    n_qubits: int; n_layers: int; scalings: list; wire_to_dim: list
    entangler: str; observable: str; param_count: int
    predicted_ntk_band: tuple[float, float]
    octave_split: bool; notes: str
    def to_json(self) / from_json(cls, ...)

def coverage(S: TargetSpectrum, Omega: np.ndarray, rtol: float = 1e-6) -> tuple[float, float]
```

Matching uses a **relative tolerance** `rtol` (frequencies are floats; `15π` from a symbol and `15π`
from a scaling product will differ in the last bits). Never use `==`.

`predicted_benefit` — the pre-registered heuristic: the fraction of solution energy lying **above the
classical reach frequency** `ω_knee` *and* inside `Ω`. `ω_knee` is a config constant estimated from
the Phase-1 `c_mlp` baseline on P1 (T1.5) and **frozen before Phase 3**.

**DoD:** `tests/test_smcd_depth.py` pins the D5 formula: for `(K/Δ) ∈ {1, 4, 13, 14, 40, 41}` assert
`L ∈ {1, 2, 3, 4, 4, 5}` respectively, and assert `max|Ω| ≥ K` in every case. Card round-trips
through JSON unchanged.

---

## T2.11 — Noise models

**Depends on:** T2.4
**Files:** `src/qapinn/models/noise.py`, `tests/test_noise.py`

Per D8:
```python
class ShotNoise(nn.Module):
    """expval -> expval + detach(N(0, (1 - expval²)/n_shots)). Straight-through:
    the noise term is detached so gradients pass unmodified. Valid in the CLT regime;
    n_shots=1024 is comfortably there."""

class GlobalDepolarizing(nn.Module):
    """expval -> (1-p)^m · expval, m = number of noisy layers. Exact for the global
    depolarizing channel on any traceless observable, and differentiable."""
```

**DoD:** `tests/test_noise.py` — shot noise has the right empirical variance over 10 000 draws
(within 5 %); depolarizing with `p=0` is the identity; gradients flow through both.
Honest-labelling requirement: both classes carry a docstring stating that they are **surrogates**,
and T4.4 quantifies the gap against a density-matrix simulation.

---

## T2.12 — ★ Hybrid models

**Depends on:** T2.4, T2.10
**Files:** `src/qapinn/models/hybrid.py`, `configs/model/q_*.yaml`

```python
class AffineEncoder(nn.Module):
    """z = A x + b, A initialised to I (D3). Exposes A for xai/drift.py."""

class SerialHybrid(PINNModel):
    """x -> AffineEncoder -> ReuploadCircuit -> Linear head -> u.
    param_groups: {'classical': [A, b, head], 'quantum': [circuit θ]}
    realised_frequencies: Ω @ A   (physical coordinates — this is the whole point of D3)"""

class ParallelHybrid(PINNModel):
    """u = MLP(x) + w·Circuit(AffineEncoder(x)). The HQPINN form from the literature."""

class OctaveEnsemble(PINNModel):
    """u = Σ_m w_m · Circuit_m(Enc_m(x)), each circuit matched to one octave of Ŝ (§5.4)."""
```

Family wiring:
- `q_serial` — `SerialHybrid`, `scaling_mode="ternary"`, scalings **from the design card**.
- `q_random` — `SerialHybrid`, `scaling_mode="random"` or `"unit"`, same `n`, `L`, same parameter
  count. **Ablates the design, keeps the quantum.** This is C1's falsifier.
- `q_parallel` — `ParallelHybrid`.
- `q_octave` — `OctaveEnsemble` (T2.15).

**DoD:** each family forwards `[B,d] → [B,1]`; `param_groups()["quantum"]` is non-empty;
`realised_frequencies()` at init equals `Ω` exactly (since `A = I`); a 50-step training run on P1
reduces the loss.

---

## T2.13 — Reference [4] re-positioning

**Depends on:** owner supplying the citation (`project.md` §1 open item)
**Files:** `docs/related_work.md`

When the citation arrives, write the positioning paragraph explicitly against it. If it has not
arrived by the start of Phase 3, position against the inferred candidates (Sedykh et al. hybrid-QPINN
line; arXiv 2606.04679) and add a visible note in the paper that the reference was inaccessible.
**Non-blocking** — nothing else depends on this task.

**DoD:** `docs/related_work.md` states our differentiation in one paragraph: prior work characterises
*when* hybridisation helps, empirically; we give a *constructive* design rule plus an XAI mechanism.

---

## T2.14 — Wire `c_rff_matched` to the design card

**Depends on:** T2.10, T0.16
**Files:** edit `src/qapinn/models/fourier_features.py`

`c_rff_matched` builds its `B` matrix from `DesignCard.omega_set` (restricted to the target support).
This is **the sharpest ablation in the project** (`project.md` §6): it isolates "matched spectrum"
from "quantum". If `c_rff_matched` matches `q_serial`, the benefit is spectral matching, not quantum —
a legitimate and important result (`project.md` §13).

**DoD:** for P1, `c_rff_matched.realised_frequencies()` contains `{π, 15π}`; its parameter count is
matched to `q_serial` within 10 %.

---

## T2.15 — Octave-split ensemble (`q_octave`)

**Depends on:** T2.12
**Files:** `src/qapinn/models/hybrid.py` (OctaveEnsemble), `src/qapinn/smcd/design.py` (split logic)

Per `project.md` §5.4: when one circuit cannot cover `Ŝ` within the barren-plateau depth budget, use
`M` shallow circuits, each matched to one octave of `Ŝ`, summed at the head. Split `Ŝ` into octaves
`[2^j Δ, 2^{j+1} Δ)`; design one circuit per non-empty octave; each stays shallow and trainable.

Needed for P4 at `k=20`.

**DoD:** on P4 `k=20`, `smcd(..., L_max=4)` triggers the split, produces ≥ 2 circuits, each with
`L ≤ 4`, and the union of their `Ω` covers `Ŝ` with weighted coverage 1.0.

---

## T2.16 — Cross-family size matching

**Depends on:** T2.12, T2.14
**Files:** `src/qapinn/models/base.py` (extend), `tests/test_size_matching.py`

Per `project.md` §6, parameter counts must match within ±10 % across all seven families. The
**quantum model is the reference**: SMCD fixes `n`, `L` ⇒ `q_serial`'s parameter count; every other
family is then matched to it.

**DoD:** for each of the six problem instances, all seven families' `n_params()` lie within ±10 % of
`q_serial`'s. Emit a table into `results/size_matching.json` for the paper's appendix, alongside
FLOPs and measured wall-clock (`project.md` §6 requires these reported together).

---

## T2.17 — Full-pipeline smoke across the whole matrix

**Depends on:** T2.16, T1.12
**Files:** `tests/test_pipeline_smoke.py`, `tasks.py` (`smoke` target)

Run `train(..., smoke=True)` for **every (problem instance × family)** pair = 6 × 7 = 42 combinations,
with all instruments enabled.

**DoD:** all 42 complete without error and write complete artifact directories. Total time < 20 min.
This is the last thing that can catch an integration bug cheaply before the Phase-3 matrix launches.

---

## T2.18 — Phase 2 gate check

**Depends on:** T2.6, T2.9, T2.17

Confirm:
1. `tests/test_circuit_spectrum.py` green (the hard gate).
2. `tests/test_qsim_vs_pennylane.py` green at `1e-10`.
3. Prop. 4 identity test (T1.2) green **on a real hybrid model** now that one exists — re-run it.
4. Parameter-shift ↔ autodiff agreement green.
5. All 42 smoke combinations pass.
6. Design cards produced and committed for all six problem instances.

**DoD:** commit tagged `phase2-complete`. Record in `BENCH.md` the measured seconds/step for
`q_serial` on P1 and P4 — this replaces T0.21's estimate with the real number and **re-checks the
matrix budget** before Phase 3 spends two days on it.
