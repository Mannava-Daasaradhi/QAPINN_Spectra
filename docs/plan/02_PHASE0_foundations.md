# Phase 0 — Foundations (Days 1–2)

**Goal:** a working, deterministic, tested classical PINN stack with trustworthy ground truth.
No quantum code in this phase. No XAI in this phase.

**Exit gate (all must hold):**
1. `pytest` fully green.
2. MMS convergence test passes for the pseudospectral reference solver (observed order ≥ 4).
3. `c_mlp` trains on P1 and reaches rel-L2 < 5e-3 on the low-frequency-only case (α = 0).
4. Burgers reference solver agrees with Cole–Hopf quadrature to < 1e-6.
5. Performance spike (T0.21) projects full-matrix wall-clock < 24 h, or the matrix is cut.

---

## T0.1 — Initialise repository and directory skeleton

**Depends on:** —
**Files:** `.gitignore`, `.gitattributes`, all directories from `01_CONVENTIONS.md` §1 with `.gitkeep`

Run `git init`. Create the full directory tree. `.gitignore` must exclude: `.venv/`, `__pycache__/`,
`*.pyc`, `results/runs/*/xai/*.npz` is **kept** (small) but `results/runs/*/checkpoints/` excluded,
`.pytest_cache/`, `paper/*.aux|*.log|*.out`, `.claude-flow/`.

Set `.gitattributes` to force `text eol=lf` for `*.py`, `*.md`, `*.yaml` so the clean-clone check in
T5.14 is not defeated by CRLF.

**DoD:** `git status` clean after an initial commit; `python -c "import os; print(os.path.isdir('src/qapinn'))"` → True.

---

## T0.2 — Pin the environment with `uv`

**Depends on:** T0.1
**Files:** `pyproject.toml`, `uv.lock`, `docs/ENV_RESOLVED.md`

Per D10, target **Python 3.12** in a project-local venv; do not touch the miniconda base install.

```powershell
uv python install 3.12
uv venv --python 3.12
uv add torch --index https://download.pytorch.org/whl/cu128
uv add pennylane numpy scipy matplotlib pyyaml pydantic pandas pyarrow pytest tqdm
uv add --dev ruff pytest-cov
```

`cu128` wheels are forward-compatible with the installed CUDA 13.0 driver (D10).

Write `docs/ENV_RESOLVED.md` recording the **actual resolved versions** of torch, pennylane, numpy,
scipy and the CUDA build — do not hand-write expected versions, capture them:

```python
import torch, pennylane, numpy, scipy, sys
print(sys.version, torch.__version__, torch.version.cuda, torch.cuda.is_available(),
      pennylane.__version__, numpy.__version__, scipy.__version__)
```

**DoD:** `uv run python -c "import torch,pennylane;print(torch.cuda.is_available())"` prints `True`,
and `docs/ENV_RESOLVED.md` exists with real captured versions.

**If CUDA is False:** stop and report. Everything downstream is sized on GPU throughput (D13).

---

## T0.3 — Task runner (`tasks.py`) + `Makefile` mirror

**Depends on:** T0.2
**Files:** `tasks.py`, `Makefile`

`tasks.py` uses stdlib `argparse` with subcommands. It must set
`os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` **before importing torch** (§9 of conventions).

Targets (stubs now, filled in by later tasks):
`test`, `lint`, `smoke`, `repro-quick`, `repro-all`, `run` (single config), `sweep`, `figures`,
`paper`, `slides`, `clean`.

`Makefile` declares the same target names, each delegating: `python tasks.py <target>`. No logic is
duplicated between the two.

**DoD:** `python tasks.py test` runs pytest and exits 0 (with zero tests collected initially).

---

## T0.4 — Config schema, loading, composition, hashing

**Depends on:** T0.2
**Files:** `src/qapinn/config.py`, `tests/test_config.py`

Implement `PDEConfig`, `ModelConfig`, `TrainConfig`, `ExpConfig` exactly as in
`01_CONVENTIONS.md` §4, using `pydantic.dataclasses.dataclass(frozen=True)`.

```python
def load_config(pde: str, model: str, *, overrides: dict | None = None,
                seed: int = 0) -> ExpConfig
def canonical_json(cfg: ExpConfig) -> str
def run_id(cfg: ExpConfig) -> str          # sha256(canonical_json)[:12]
```

Composition: read `configs/pde/<pde>.yaml` and `configs/model/<model>.yaml`, deep-merge, apply
`overrides` (dotted keys, e.g. `{"train.steps_adam": 100}`), validate, freeze.

**DoD:** `tests/test_config.py` asserts (a) two configs differing only in YAML key order have the
**same** `run_id`; (b) changing `seed` changes `run_id`; (c) an unknown field raises a validation error;
(d) dotted overrides apply correctly.

---

## T0.5 — Determinism utilities

**Depends on:** T0.2
**Files:** `src/qapinn/seeding.py`, `src/qapinn/__init__.py` (device resolution)

Implement `set_global_seed(seed) -> torch.Generator` per `01_CONVENTIONS.md` §9, and

```python
def resolve_device(prefer: str = "auto") -> torch.device
```

Also set `torch.set_default_dtype(torch.float64)` in `qapinn/__init__.py`.

**DoD:** calling `set_global_seed(0)` twice and sampling `torch.randn` through the returned
generator produces bit-identical tensors.

---

## T0.6 — PDE abstract base + `Domain`

**Depends on:** T0.4, T0.5
**Files:** `src/qapinn/pdes/base.py`

Implement `Domain` and the `PDE` ABC exactly as specified in `01_CONVENTIONS.md` §5, including
`apply_hard_bc`, `sample_collocation`, `sample_boundary`, `eval_grid`.

`sample_collocation` draws uniformly in the box from the explicit generator. `eval_grid` returns a
**deterministic tensor-product uniform grid** — this grid is what all FFT-based spectral analysis
uses, so it must be uniform and, for periodic-in-x problems, exclude the duplicate endpoint.

**DoD:** an in-test dummy subclass instantiates; `eval_grid(64)` for `d=2` returns shape `[4096, 2]`;
`sample_collocation` is reproducible under a fixed generator.

---

## T0.7 — Autograd differential operators

**Depends on:** T0.6
**Files:** `src/qapinn/pdes/diffops.py`, `tests/test_diffops.py`

Implement `d1`, `d2`, `laplacian` per `01_CONVENTIONS.md` §6. All must pass `create_graph=True`.

**DoD:** `tests/test_diffops.py` checks against closed forms on `u = sin(3x)cos(2y)`:
`d1` → `3cos(3x)cos(2y)`, `d2(0,0)` → `−9 sin(3x)cos(2y)`, `laplacian` → `−13 u`, each to `1e-10`.
Also assert that gradients flow to a parameter *through* `d2` (i.e. `create_graph` really is on) by
backpropagating a scalar function of `d2` to a leaf parameter and checking `.grad` is non-`None`.

---

## T0.8 — The four PDE implementations

**Depends on:** T0.7
**Files:** `src/qapinn/pdes/poisson.py`, `heat.py`, `burgers.py`, `helmholtz.py`,
`configs/pde/*.yaml`, `tests/test_pdes.py`

Implement P1–P4 exactly per `01_CONVENTIONS.md` §5, including all four constants tables. Register
them in a factory `qapinn.pdes.build(cfg: PDEConfig) -> PDE`.

Note for P4: `a1`, `a2` are config parameters and the config files must carry the `(k, a1, a2)`
triples `(4,1,1)`, `(10,3,1)`, `(20,6,2)`.

**DoD:** `tests/test_pdes.py` asserts, for each PDE:
1. **The exact solution satisfies its own residual to `1e-8`.** (Feed `pde.exact` through
   `pde.residual` on random interior points.) This single test catches almost every sign error.
2. `apply_hard_bc` reproduces the boundary/initial data exactly: sample points on the constrained set,
   pass an arbitrary random `n_out`, and assert the result equals the prescribed data to `1e-12`.
3. `forcing` matches the analytically derived `f` for P1, P2, P4.

---

## T0.9 — FFT convention test (D12 guard)

**Depends on:** T0.8
**Files:** `src/qapinn/xai/spectral_error.py` (only `omega_grid` for now), `tests/test_fft_convention.py`

Implement `omega_grid(n, dx)` per `01_CONVENTIONS.md` §2.

**DoD:** FFT of `sin(15πx)` sampled on `[0,1)` with `N=256` has its peak at `|ω| = 15π ± dω/2`.
Assert the *angular* value, not a cycle count. Also assert the generator-eigenvalue factor of 2:
a helper `pauli_freq(omega_scaling) -> float` returns `omega_scaling` (not `omega_scaling/2`) with a
comment pointing at `01_CONVENTIONS.md` §2.

---

## T0.10 — Analytic reference solutions

**Depends on:** T0.8
**Files:** `src/qapinn/reference/analytic.py`, `tests/test_reference_analytic.py`

Closed forms for P1, P2, P4 (already given as `pde.exact`; this module wraps them for the reference
API and adds the P2 separation-of-variables derivation as executable code with more modes than the
two seeded ones, so it can also serve general initial data).

```python
def poisson_exact(x, alpha) -> np.ndarray
def heat_exact(x, t, nu, modes: list[tuple[int,float]]) -> np.ndarray
def helmholtz_exact(x, y, a1, a2) -> np.ndarray
```

**DoD:** each agrees with the corresponding `pde.exact` to `1e-12` on a random point set.

---

## T0.11 — Fourier pseudospectral reference solver + MMS test

**Depends on:** T0.10
**Files:** `src/qapinn/reference/spectral.py`, `tests/test_mms.py`

A Fourier pseudospectral solver used as an independent ground truth for Burgers and as the MMS
convergence vehicle. Spatial discretisation: FFT with 2/3 dealiasing. Time stepping: RK4 with an
integrating factor for the viscous term (or ETDRK4 — either is acceptable, RK4+IF is simpler).

```python
def solve_burgers_spectral(nu: float, n_x: int, n_t: int, t_final: float,
                           u0: Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    """returns (x, t, u) with u of shape [n_t, n_x]"""
```

**Method of manufactured solutions:** add a forcing term so a chosen smooth `u*` is the exact
solution, then measure the error as `n_x` refines.

**DoD:** `tests/test_mms.py` shows **spectral convergence in space** (error drops by ≥ 2 orders of
magnitude between `n_x=32` and `n_x=64` on a smooth manufactured solution) and **4th-order convergence
in time** (observed order ≥ 3.8 as `dt` halves). Mark `@pytest.mark.slow` if it exceeds 5 s.

---

## T0.12 — Cole–Hopf exact Burgers solution (independent cross-check)

**Depends on:** T0.11
**Files:** `src/qapinn/reference/colehopf.py`, `tests/test_burgers_crosscheck.py`

For `u_t + u u_x = ν u_xx` with `u(x,0) = −sin(πx)`, the Cole–Hopf transform gives an exact solution
as a ratio of integrals. Evaluate with high-order quadrature (`scipy.integrate.quad` per point, or
Gauss–Hermite since the kernel is Gaussian — Gauss–Hermite is far faster and is preferred).

```python
def burgers_colehopf(x: np.ndarray, t: np.ndarray, nu: float, n_quad: int = 200) -> np.ndarray
```

**DoD:** `tests/test_burgers_crosscheck.py` asserts the Cole–Hopf solution and the pseudospectral
solution (T0.11) agree to **< 1e-6** in max-norm on a `[−1,1] × [0,1]` grid at `ν = 0.01/π`.
Two independent methods agreeing is what makes this ground truth, not a guess.
Cache the result to `results/reference/burgers_ref.npz` — it is expensive and never changes.

---

## T0.13 — Reference solution caching + registry

**Depends on:** T0.12
**Files:** `src/qapinn/reference/__init__.py`

```python
def reference_solution(pde: PDE, grid: np.ndarray) -> np.ndarray
```
Dispatches: analytic where available (P1, P2, P4), cached Cole–Hopf for P3. Caches to
`results/reference/<pde_name>_<grid_hash>.npz`.

**DoD:** calling twice hits the cache the second time (assert via file mtime or a call counter).

---

## T0.14 — `PINNModel` base + size matching

**Depends on:** T0.5
**Files:** `src/qapinn/models/base.py`, `tests/test_param_matching.py`

Implement the `PINNModel` ABC and `match_param_count` exactly per `01_CONVENTIONS.md` §7.

**DoD:** `match_param_count("c_mlp", target=3000)` yields a model whose `n_params()` is within 10 %
of 3000; an impossible target raises.

---

## T0.15 — `c_mlp` (tanh MLP PINN)

**Depends on:** T0.14
**Files:** `src/qapinn/models/mlp.py`, `configs/model/c_mlp.yaml`

Plain tanh MLP, Xavier-normal init, configurable widths. `param_groups()` returns
`{"classical": [...], "quantum": []}`. `realised_frequencies()` returns `None`.

**DoD:** forward on `[B,d]` gives `[B,1]`; `param_groups()["quantum"] == []`.

---

## T0.16 — `c_ff` and `c_rff_matched` (Fourier-feature PINNs)

**Depends on:** T0.15
**Files:** `src/qapinn/models/fourier_features.py`, `configs/model/c_ff.yaml`, `configs/model/c_rff_matched.yaml`

```python
class FourierFeatures(nn.Module):
    """B: [d, m] buffer (NOT a parameter). forward: x -> [sin(Bx), cos(Bx)] : [B, 2m].
    NOTE: B holds ANGULAR frequencies directly (D12) — do not apply an extra 2π."""
```
- `c_ff`: `B ~ N(0, σ²)` (Tancik-style), `σ` from config. **This is the fair classical baseline and
  is non-negotiable** (`project.md` §6).
- `c_rff_matched`: `B` rows are set **to the SMCD frequencies** `Ω`. Until Phase 2 exists, accept an
  explicit frequency list in config; T2.14 wires it to the design card.

`realised_frequencies()` returns the rows of `B`.

**DoD:** for `c_rff_matched` with `B = [[π],[15π]]`, a least-squares fit of the linear head to
`sin(πx) + 0.3 sin(15πx)` on a dense grid reaches rel-L2 < 1e-8 — i.e. the matched basis really does
span the target exactly. This test is the classical dry-run of Prop. 3.

---

## T0.17 — Loss assembly

**Depends on:** T0.8, T0.15
**Files:** `src/qapinn/train/losses.py`

```python
def pinn_loss(model, pde, batch, cfg: TrainConfig) -> tuple[Tensor, dict[str, float]]
```
- `bc_mode == "hard"`: `u = pde.apply_hard_bc(x, model(x))`; loss = `λ_r · MSE(residual)` only.
- `bc_mode == "soft"`: `u = model(x)` raw; loss = `λ_r·MSE(res) + λ_b·MSE(bc) + λ_i·MSE(ic)`.

Return the scalar loss and a dict of the individual (detached) terms for logging.

**DoD:** with `bc_mode="hard"` on P1, feeding a model that exactly returns `(u* − B)/D` gives a loss
below `1e-10`.

---

## T0.18 — Training loop

**Depends on:** T0.17
**Files:** `src/qapinn/train/loop.py`

```python
def train(cfg: ExpConfig, *, smoke: bool = False) -> RunResult
```
Sequence: seed → build PDE → build model → (Phase 2: build SMCD design card) → Adam with cosine
schedule for `steps_adam` → LBFGS (`strong_wolfe`) for `steps_lbfgs` → final evaluation.

Resample collocation points every 100 steps (fixed schedule, from the seeded generator).
At each checkpoint in `cfg.train.checkpoints`, invoke the registered XAI hooks (Phase 1 fills these;
for now the hook list is empty) and record metrics.

Metrics computed on `pde.eval_grid(cfg.pde.n_eval)`: `rel_l2`, `l_inf`, `residual_norm`,
`steps_to_tol_*`, plus wall-clock. Write all artifacts per `01_CONVENTIONS.md` §8.

`smoke=True` overrides to `steps_adam=50, steps_lbfgs=10, n_collocation=256, checkpoints=(0,-1)`.

**DoD:** `python tasks.py run --pde poisson --model c_mlp --smoke` completes in < 60 s and writes a
complete `results/runs/<id>/` directory with all required files.

---

## T0.19 — Checkpointing & provenance

**Depends on:** T0.18
**Files:** `src/qapinn/train/checkpoint.py`

Save model state at each configured checkpoint to `results/runs/<id>/checkpoints/` (git-ignored),
because Phase 1 instruments run **at checkpoints** and must be re-runnable without retraining.
Write `provenance.json` per D11 (git SHA via `subprocess`, package versions, device, hostname).

**DoD:** a checkpoint reloads into a fresh model and reproduces the same `rel_l2` to `1e-12`.

---

## T0.20 — Determinism test

**Depends on:** T0.19
**Files:** `tests/test_determinism.py`

**DoD:** two `train()` calls with identical configs (50 steps) produce **bit-identical**
`metrics.json`. If this fails on GPU, find the non-deterministic op — do not relax the test to a
tolerance; the reproducibility contract in `project.md` §11 depends on it.

---

## T0.21 — PERFORMANCE SPIKE (gate for the whole plan, D13)

**Depends on:** T0.18
**Files:** `scripts/bench_step.py`, `docs/plan/BENCH.md`

Measure and record:
1. `c_mlp` on P1: seconds per Adam step at `n_collocation=4096`.
2. `c_mlp` on P4 (2-D, second derivatives): seconds per step.
3. **A stand-in for the quantum cost**: a `c_ff` model with `n_features=128` plus a synthetic
   `[B, 256]` complex matmul chain of depth `L=4, n=6` — this approximates the statevector
   arithmetic that T2.3 will do, before `qsim.py` exists.

Extrapolate: `total_hours = 381 runs × steps × s_per_step / (3600 × n_parallel)` at `n_parallel = 6`.

**DoD:** `docs/plan/BENCH.md` contains the measured numbers and the projection.
**Gate:** if projected wall-clock > 24 h, apply the §5 cut lines of the master plan **now** and record
which cuts were taken. Do not proceed into Phase 1 with an unaffordable matrix.

---

## T0.22 — Phase 0 gate check

**Depends on:** T0.9, T0.11, T0.12, T0.20, T0.21

Run `python tasks.py test` and confirm all five exit criteria at the top of this document.
Train `c_mlp` on P1 with `α = 0` (low frequency only) for 20 k steps and confirm rel-L2 < 5e-3 —
this proves the harness can solve an easy problem before we ask it to solve a hard one.

**DoD:** a short note appended to `docs/plan/BENCH.md` recording the five gate results.
Commit tagged `phase0-complete`.
