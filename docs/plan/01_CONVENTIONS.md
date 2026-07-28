# Conventions & Module API Contracts

**Read this before writing any code.** Every signature here is binding: later tasks assume these exact
names, shapes, and semantics. If a task description and this document disagree, this document wins.

---

## 1. Repository layout (as built)

```
QAPINN_Spectra/
├── project.md                 # scientific spec (do not edit)
├── README.md                  # T5.12
├── FINDINGS.md                # T5.2
├── pyproject.toml, uv.lock    # T0.2
├── tasks.py                   # T0.3   primary task runner
├── Makefile                   # T0.3   mirror of tasks.py targets
├── configs/
│   ├── pde/{poisson,heat,burgers,helmholtz}.yaml
│   ├── model/{c_mlp,c_ff,c_rff_matched,q_serial,q_parallel,q_random,q_octave}.yaml
│   └── exp/{core_matrix,coverage_sweep,depth_sweep,alpha_sweep,noise_study}.yaml
├── src/qapinn/
│   ├── __init__.py
│   ├── config.py              # T0.4  schema + loading + hashing
│   ├── seeding.py             # T0.5  determinism
│   ├── pdes/
│   │   ├── base.py            # T0.6  PDE ABC, Domain
│   │   ├── diffops.py         # T0.7  autograd derivative helpers
│   │   ├── poisson.py heat.py burgers.py helmholtz.py     # T0.8
│   ├── reference/
│   │   ├── analytic.py        # T0.10
│   │   ├── spectral.py        # T0.11  Fourier pseudospectral solver
│   │   └── colehopf.py        # T0.12  Burgers exact via quadrature
│   ├── models/
│   │   ├── base.py            # T0.14  PINNModel ABC
│   │   ├── mlp.py             # T0.15
│   │   ├── fourier_features.py# T0.16
│   │   ├── qsim.py            # T2.3   batched statevector simulator
│   │   ├── circuits.py        # T2.4   re-uploading circuit
│   │   ├── pshift.py          # T2.7   parameter-shift derivatives
│   │   └── hybrid.py          # T2.12  serial / parallel / octave
│   ├── smcd/
│   │   ├── symbol.py          # T2.8   Prop. 2: PDE -> target spectrum
│   │   ├── design.py          # T2.9   Algorithm 1
│   │   └── card.py            # T2.10  design card + coverage
│   ├── train/
│   │   ├── losses.py          # T0.17
│   │   ├── loop.py            # T0.18
│   │   └── checkpoint.py      # T0.19
│   ├── xai/
│   │   ├── ntk.py             # T1.2
│   │   ├── spectral_error.py  # T1.4
│   │   ├── attribution.py     # T1.6
│   │   ├── fisher.py          # T1.7
│   │   ├── gradvar.py         # T1.9
│   │   ├── probes.py          # T1.10
│   │   ├── landscape.py       # T1.11
│   │   └── drift.py           # T1.8   spectral drift of the affine encoder
│   ├── viz/
│   │   ├── style.py           # T1.1   single matplotlib style, used everywhere
│   │   └── figures.py         # T5.1
│   └── runner.py              # T3.2   parallel run orchestration
├── scripts/{run_experiment.py, sweep.py, make_figures.py}
├── tests/
├── notebooks/
├── docs/
│   ├── plan/                  # these documents
│   ├── derivations/           # T2.1, T2.2
│   ├── predictions.md         # T3.1  PRE-REGISTERED, committed before any run
│   ├── ENV_RESOLVED.md        # T0.2  resolved package versions
│   └── REPRODUCE.md           # T5.13
├── results/                   # design cards, metrics JSON, seeds. NOT raw checkpoints.
├── paper/
└── slides/
```

**Hard rules:** nothing but the listed files at repo root. No working files, no scratch tests at root.
Keep every source file **under 500 lines** — if a module grows past that, split it.

---

## 2. THE FREQUENCY CONVENTION (D12)

> **Angular frequency `ω`. Complex exponential convention `e^{iωx}`.**

- `sin(15πx)` has `ω = 15π ≈ 47.124`. It is **not** `k = 7.5`.
- Forward transform: `û(ω) = ∫ u(x) e^{−iωx} dx`, discretised with `numpy.fft.fft`.
- Frequency grid: `omega = 2π * numpy.fft.fftfreq(N, d=dx)`. **Note the `2π`** — `fftfreq` returns
  cycles per unit, we always multiply by `2π`. There is exactly one helper for this and everything
  must use it:

```python
# src/qapinn/xai/spectral_error.py
def omega_grid(n: int, dx: float) -> np.ndarray:
    """Angular-frequency grid matching numpy.fft.fft output ordering."""
    return 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
```

- In 2D, `ω = (ω_x, ω_y)` and `|ω| = sqrt(ω_x² + ω_y²)`.
- Circuit frequencies `Ω` are angular. Encoding gate `RZ(ω x) = exp(−i ω x Z / 2)` has generator
  eigenvalues `±ω/2`, so eigenvalue **differences** are `{−ω, 0, +ω}`. This factor of 2 between
  "generator eigenvalue" and "realised frequency" is the second most likely silent bug. `T0.9` pins it.

Every plot axis showing frequency is labelled `ω` (rad/unit), never `k`.

---

## 3. Tensor shape contract

| Quantity | Shape | Notes |
|---|---|---|
| Input coordinates `x` | `[B, d]` | `d = 1` (P1), `2` (P2, P3, P4). float64 by default. |
| Model output `u` | `[B, 1]` | always 2-D, never squeezed to `[B]` |
| Circuit output | `[B, n_obs]` | `n_obs` = number of measured observables |
| Statevector | `[B, 2, 2, ..., 2]` | `n` trailing axes of size 2, complex128; qubit 0 is the **first** trailing axis |
| Jacobian `J` | `[P, N]` | `P` = probe points, `N` = parameters |
| NTK block | `[P, P]` | symmetric PSD |

**dtype policy:** `float64` throughout for physics and NTK (PINN residuals are ill-conditioned in
float32 and second derivatives amplify the error). Complex arithmetic in `complex128`. Set globally:

```python
torch.set_default_dtype(torch.float64)
```

**device policy:** a single `qapinn.device` resolved once from config (`cuda` if available else `cpu`),
passed explicitly to every tensor construction. Never call `.cuda()` inline.

---

## 4. Config schema

Configs are YAML, loaded into frozen dataclasses (use `pydantic.dataclasses` for validation).
A full experiment config is the composition of three files plus overrides.

```python
# src/qapinn/config.py
@dataclass(frozen=True)
class PDEConfig:
    name: str                      # "poisson" | "heat" | "burgers" | "helmholtz"
    params: dict[str, float]       # e.g. {"alpha": 0.3} or {"k": 10.0, "a1": 3, "a2": 1}
    n_collocation: int = 4096
    n_boundary: int = 512
    n_eval: int = 1024             # uniform grid for L2 / spectral error

@dataclass(frozen=True)
class ModelConfig:
    family: str                    # "c_mlp"|"c_ff"|"c_rff_matched"|"q_serial"|"q_parallel"|"q_random"|"q_octave"
    widths: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    # Fourier-feature families
    n_features: int | None = None
    ff_sigma: float | None = None          # c_ff only: B ~ N(0, sigma^2)
    # quantum families
    n_qubits: int | None = None
    n_layers: int | None = None
    scaling_mode: str | None = None        # "ternary" | "linear" | "random" | "unit"
    entangler: str | None = None           # "ring_cz" | "none"
    observable: str | None = None          # "z0" | "z_mean"
    encoder: str = "affine"                # "affine" (D3) | "mlp" (ablation only)
    target_params: int | None = None       # size-matching target; see §7

@dataclass(frozen=True)
class TrainConfig:
    steps_adam: int = 20000
    steps_lbfgs: int = 2000
    lr: float = 1e-3
    lr_schedule: str = "cosine"
    bc_mode: str = "hard"                  # "hard" | "soft"   (D6)
    lambda_residual: float = 1.0
    lambda_bc: float = 1.0                 # used only when bc_mode == "soft"
    lambda_ic: float = 1.0
    checkpoints: tuple[int, ...] = (0, 100, 500, 1000, 5000, 20000, -1)   # -1 = final
    noise: str = "none"                    # "none" | "shot_1024" | "depol_1e-3"

@dataclass(frozen=True)
class ExpConfig:
    pde: PDEConfig
    model: ModelConfig
    train: TrainConfig
    seed: int
    smcd_eps: float = 1e-3
    smcd_coverage_target: float | None = None   # coverage sweep: deliberately de-tune the design
    tag: str = ""

    @property
    def run_id(self) -> str:
        """sha256 of canonical JSON, first 12 hex chars. D11."""
```

`run_id` must be computed from a **canonical** JSON dump: `sort_keys=True, separators=(",",":")`,
floats formatted with `repr`. Two configs that differ only in key order must produce the same id.

---

## 5. PDE contract

```python
# src/qapinn/pdes/base.py
@dataclass(frozen=True)
class Domain:
    bounds: tuple[tuple[float, float], ...]   # ((x_lo,x_hi), (t_lo,t_hi))
    names: tuple[str, ...]                    # ("x",) | ("x","t") | ("x","y")
    time_axis: int | None                     # index of the time coordinate, None if steady

class PDE(ABC):
    name: str
    domain: Domain
    dim: int
    params: dict[str, float]

    # --- physics -------------------------------------------------------
    @abstractmethod
    def residual(self, x: Tensor, u: Tensor) -> Tensor:
        """x:[B,d] (requires_grad=True), u:[B,1] -> residual:[B,1].
        Uses diffops helpers; must call them with create_graph=True."""

    @abstractmethod
    def forcing(self, x: Tensor) -> Tensor:      # [B,d] -> [B,1]

    @abstractmethod
    def exact(self, x: Tensor) -> Tensor:        # [B,d] -> [B,1]  ground truth

    # --- hard boundary ansatz  u = B(x) + D(x) * N(x) -------------------
    @abstractmethod
    def bc_lift(self, x: Tensor) -> Tensor:      # B(x): [B,d] -> [B,1]

    @abstractmethod
    def bc_mask(self, x: Tensor) -> Tensor:      # D(x): [B,d] -> [B,1], vanishes on the constrained set

    def apply_hard_bc(self, x: Tensor, n_out: Tensor) -> Tensor:
        return self.bc_lift(x) + self.bc_mask(x) * n_out

    # --- soft-BC variant (D6) -------------------------------------------
    @abstractmethod
    def boundary_residual(self, x_b: Tensor, u_b: Tensor) -> Tensor:

    # --- SMCD -----------------------------------------------------------
    @abstractmethod
    def symbol(self, omega: np.ndarray) -> np.ndarray:
        """Fourier symbol sigma(omega) of the linear (or linearised) operator.
        omega: [M,d] angular frequencies -> [M] complex."""

    def target_spectrum_analytic(self, eps: float) -> "TargetSpectrum | None":
        """Return None if no closed form exists; SMCD then falls back to the
        empirical path (from the reference solution). See T2.8."""

    # --- sampling --------------------------------------------------------
    def sample_collocation(self, n: int, gen: torch.Generator) -> Tensor
    def sample_boundary(self, n: int, gen: torch.Generator) -> Tensor
    def eval_grid(self, n: int) -> Tensor        # deterministic uniform grid, for L2 + FFT
```

### The four PDE instances, fully specified

Implement exactly these. All constants are binding — they are chosen so the manufactured solutions
satisfy the boundary conditions *exactly* and so P4 sits near resonance.

**P1 — Poisson.** `−u'' = f` on `[0,1]`, `u(0)=u(1)=0`.
```
u*(x) = sin(πx) + α sin(15πx)                        α default 0.3
f(x)  = π² sin(πx) + α (15π)² sin(15πx)
B(x)  = 0 ;  D(x) = x(1−x)
symbol σ(ω) = ω²        (since −u'' ↔ ω² û)
target spectrum: discrete, S = {π, 15π} with weights {1, α}
```

**P2 — Heat (NEGATIVE CONTROL).** `u_t = ν u_xx` on `[0,1]×[0,1]`, `ν = 0.05`.
```
u(x,0) = sin(πx) + α sin(15πx)   ;   u(0,t)=u(1,t)=0
u*(x,t) = sin(πx) e^{−ν π² t} + α sin(15πx) e^{−ν (15π)² t}
B(x,t) = sin(πx) + α sin(15πx)   ;   D(x,t) = t · x(1−x)
```
Note `ν(15π)² ≈ 111`, so the high-frequency mode is `e^{−111}` at `t=1` — dead. **SMCD must predict
"no benefit" for this PDE before any training happens.** That prediction is contribution C4.

**P3 — Burgers.** `u_t + u u_x = (0.01/π) u_xx` on `[−1,1]×[0,1]`.
```
u(x,0) = −sin(πx)  ;  u(±1,t) = 0
B(x,t) = −sin(πx)  ;  D(x,t) = t (1 − x²)
```
Check: at `t=0`, `u = −sin(πx)` ✓ IC. At `x=±1`, `sin(±π)=0` and `D=0` ⇒ `u=0` ✓ BC.
Nonlinear ⇒ no closed-form symbol. Target spectrum comes from the **empirical** path: FFT the
reference solution at each `t`, take the union over `t`, record the growth rate (`project.md` Alg. 1
step 2). Reference solution: Cole–Hopf quadrature (T0.12) cross-checked against pseudospectral (T0.11).

**P4 — Helmholtz (HERO RESULT).** `Δu + k²u = f` on `[−1,1]²`, `u = 0` on `∂Ω`.
Manufactured solution `u* = sin(a₁πx) sin(a₂πy)` with **integer** `a₁,a₂` (so `u*` vanishes on the
boundary exactly), chosen near resonance `π²(a₁²+a₂²) ≈ k²`:

| `k` | `(a₁, a₂)` | `π²(a₁²+a₂²)` | `k²` |
|---|---|---|---|
| 4 | (1, 1) | 19.74 | 16 |
| 10 | (3, 1) | 98.7 | 100 |
| 20 | (6, 2) | 394.8 | 400 |

```
f = (k² − π²(a₁²+a₂²)) u*
B = 0 ;  D(x,y) = (1−x²)(1−y²)
symbol σ(ω) = k² − |ω|²
target spectrum: four discrete points (±a₁π, ±a₂π)
```
Requires cross-frequency terms `ω_x ± ω_y` ⇒ entangler `ring_cz` (Alg. 1 step 7).

---

## 6. Differential operator helpers

```python
# src/qapinn/pdes/diffops.py
def d1(u: Tensor, x: Tensor, i: int) -> Tensor:
    """∂u/∂x_i.  u:[B,1], x:[B,d] with requires_grad -> [B,1]. create_graph=True always."""

def d2(u: Tensor, x: Tensor, i: int, j: int) -> Tensor:
    """∂²u/∂x_i∂x_j -> [B,1]"""

def laplacian(u: Tensor, x: Tensor, dims: tuple[int, ...] | None = None) -> Tensor:
    """Σ_i ∂²u/∂x_i² over dims (default: all spatial dims)."""
```

All three use `torch.autograd.grad(..., create_graph=True, retain_graph=True)` so higher-order and
parameter gradients both survive. **Never** use finite differences anywhere in this project.

---

## 7. Model contract

```python
# src/qapinn/models/base.py
class PINNModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """[B,d] -> [B,1]. This is the RAW network output N(x), BEFORE the hard-BC ansatz.
        The training loop applies pde.apply_hard_bc()."""

    @abstractmethod
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Keys are exactly {'classical', 'quantum'}. 'quantum' is [] for classical families.
        Required by the NTK block decomposition (D7) — Prop. 4 is tested through this."""

    def n_params(self) -> int
    def realised_frequencies(self) -> np.ndarray | None:
        """Angular frequencies the model can currently represent, in PHYSICAL coordinates.
        Fourier-feature families: 2π·B rows. Quantum families: Ω·A where A is the affine
        encoder matrix (D3). None for c_mlp. Recomputed on demand — it drifts during training."""
```

**Size matching (`project.md` §6: within ±10%).** Implement:

```python
# src/qapinn/models/base.py
def match_param_count(family: str, target: int, tol: float = 0.10, **kw) -> ModelConfig:
    """Binary-search the MLP width (or n_features) so n_params lands within tol of target.
    Raises if unreachable. The target is set by the SMCD-designed q_serial model, so every
    family is matched TO the quantum model, not the other way round."""
```

---

## 8. Result artifacts

Each run writes to `results/runs/<run_id>/`:

| File | Contents |
|---|---|
| `config.yaml` | the fully resolved `ExpConfig` |
| `provenance.json` | git SHA, seed, package versions, hostname, device, wall-clock, `run_id` |
| `design_card.json` | SMCD output (quantum families) or `null` |
| `metrics.json` | scalar metrics, see below |
| `history.parquet` | per-step loss, lr, grad-norm |
| `xai/ntk_step<N>.npz` | eigenvalues, block masses, condition number, drift |
| `xai/specerr.npz` | `[n_checkpoints, n_freq]` error magnitude heatmap data |
| `xai/*.npz` | one per instrument |

`metrics.json` schema (flat, all floats — this is what every figure reads):
```json
{
  "rel_l2": 0.0, "l_inf": 0.0, "residual_norm": 0.0,
  "steps_to_tol_1e-2": 0, "steps_to_tol_1e-3": 0,
  "n_params": 0, "wall_clock_s": 0.0, "circuit_evals": 0,
  "grad_var_final": 0.0, "ntk_cond": 0.0, "ntk_decay_exponent": 0.0,
  "smcd_coverage": 0.0, "smcd_coverage_weighted": 0.0,
  "encoder_drift": 0.0
}
```

**Never commit raw checkpoints.** `results/` holds design cards, metrics, and the small `.npz`
arrays only (`project.md` §9).

---

## 9. Determinism contract

```python
# src/qapinn/seeding.py
def set_global_seed(seed: int) -> torch.Generator:
    """Seeds python random, numpy, torch (cpu+cuda); sets
    torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG=:4096:8.
    Returns a generator to be threaded explicitly through all sampling."""
```

Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before** torch is imported (do it in `tasks.py`).
All collocation sampling takes the returned generator explicitly — never rely on global RNG state.
Two runs with the same config must produce bit-identical `metrics.json`; `tests/test_determinism.py`
(T0.20) asserts this on a 50-step run.

---

## 10. Coding standards

- Type hints on every public function. `from __future__ import annotations` at the top of each module.
- Docstrings state shapes: `"""x: [B,d] -> [B,1]"""`.
- Validate at boundaries (config load, PDE construction, circuit construction) — assert shapes and
  ranges there, not in inner loops.
- No magic numbers in `src/`; everything tunable lives in a config dataclass with a default.
- `pytest` only, no unittest. Tests are fast by default; anything >5 s is marked `@pytest.mark.slow`.
- Logging via `logging`, not `print`, except in `tasks.py` and `scripts/`.
- Every script accepts `--smoke` which reduces steps/points so the pipeline runs in <60 s
  (`project.md` §11).

---

## 11. Naming that must stay stable

These strings appear in configs, filenames, figures, and the paper. Fix them now.

| Concept | Canonical string |
|---|---|
| Model families | `c_mlp`, `c_ff`, `c_rff_matched`, `q_serial`, `q_parallel`, `q_random`, `q_octave` |
| PDEs | `poisson`, `heat`, `burgers`, `helmholtz` |
| Problem instances | `poisson`, `heat`, `burgers`, `helmholtz_k4`, `helmholtz_k10`, `helmholtz_k20` |
| Noise settings | `none`, `shot_1024`, `depol_1e-3` |
| BC modes | `hard`, `soft` |
| Scaling modes | `ternary`, `linear`, `random`, `unit` |
| Claims | `C1`…`C5` (used in `FINDINGS.md` and the adjudication table) |
