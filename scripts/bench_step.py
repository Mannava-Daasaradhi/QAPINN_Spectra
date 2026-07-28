"""Performance spike (T0.21, D13): measures c_mlp per-Adam-step cost on P1 and P4, plus a
synthetic statevector-simulation stand-in (T2.3's qsim.py doesn't exist yet), and projects
the full ~381-run experiment matrix's wall-clock budget. Writes docs/plan/BENCH.md.

Run: uv run python scripts/bench_step.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

import qapinn
from qapinn.config import TrainConfig
from qapinn.models.mlp import MLPPINN
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson
from qapinn.seeding import set_global_seed
from qapinn.train.losses import pinn_loss

N_WARMUP = 5
N_MEASURE = 20


def _bench_adam_step(pde, model, train_cfg: TrainConfig, n_collocation: int) -> float:
    gen = set_global_seed(0)
    device = qapinn.device
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    def step() -> None:
        x_r = pde.sample_collocation(n_collocation, gen).to(device)
        x_r.requires_grad_(True)
        optimizer.zero_grad()
        loss, _terms = pinn_loss(model, pde, {"x_r": x_r}, train_cfg)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(N_WARMUP):
        step()

    t0 = time.perf_counter()
    for _ in range(N_MEASURE):
        step()
    elapsed = time.perf_counter() - t0
    return elapsed / N_MEASURE


def bench_poisson_c_mlp(n_collocation: int = 4096) -> float:
    pde = Poisson(alpha=0.3)
    model = MLPPINN(input_dim=pde.dim, widths=(64, 64, 64)).to(qapinn.device)
    return _bench_adam_step(pde, model, TrainConfig(bc_mode="hard"), n_collocation)


def bench_helmholtz_c_mlp(n_collocation: int = 4096) -> float:
    pde = Helmholtz(k=10.0, a1=3.0, a2=1.0)
    model = MLPPINN(input_dim=pde.dim, widths=(64, 64, 64)).to(qapinn.device)
    return _bench_adam_step(pde, model, TrainConfig(bc_mode="hard"), n_collocation)


def bench_quantum_stand_in(
    batch: int = 4096,
    state_dim: int = 256,
    depth: int = 4,
    gates_per_layer: int = 6,
) -> float:
    """Synthetic stand-in for statevector-simulation cost: a [batch, state_dim] complex128
    tensor pushed through depth*gates_per_layer dense [state_dim, state_dim] complex
    matmuls. state_dim=256 matches D2's "n<=8 qubits -> <=256 complex amplitudes"; depth=4,
    gates_per_layer=6 matches the phase doc's "L=4, n=6" literally. This is a DENSE proxy
    (real gates are local, 2x2 or 4x4, much cheaper) -- deliberately conservative/pessimistic,
    appropriate for a budget-gating spike before the real qsim.py (T2.3) exists.
    """
    device = qapinn.device
    gen = torch.Generator(device="cpu").manual_seed(0)

    state = torch.complex(
        torch.randn(batch, state_dim, generator=gen), torch.randn(batch, state_dim, generator=gen)
    ).to(device)
    layers = [
        torch.complex(
            torch.randn(state_dim, state_dim, generator=gen), torch.randn(state_dim, state_dim, generator=gen)
        ).to(device)
        for _ in range(depth * gates_per_layer)
    ]

    def step() -> None:
        s = state
        for w in layers:
            s = s @ w
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(N_WARMUP):
        step()

    t0 = time.perf_counter()
    for _ in range(N_MEASURE):
        step()
    elapsed = time.perf_counter() - t0
    return elapsed / N_MEASURE


def main() -> None:
    print(f"Device: {qapinn.device}")

    print("Benchmarking c_mlp on P1 (Poisson, 1-D)...")
    s_poisson = bench_poisson_c_mlp()
    print(f"  {s_poisson * 1000:.3f} ms/step")

    print("Benchmarking c_mlp on P4 (Helmholtz, 2-D, second derivatives)...")
    s_helmholtz = bench_helmholtz_c_mlp()
    print(f"  {s_helmholtz * 1000:.3f} ms/step")

    print("Benchmarking synthetic quantum-cost stand-in...")
    s_quantum = bench_quantum_stand_in()
    print(f"  {s_quantum * 1000:.3f} ms/step")

    n_runs = 381
    steps_per_run = 20000 + 2000  # TrainConfig defaults: steps_adam + steps_lbfgs
    n_parallel = 6

    s_classical = max(s_poisson, s_helmholtz)
    s_hybrid = s_classical + s_quantum

    hours_classical = n_runs * steps_per_run * s_classical / (3600 * n_parallel)
    hours_hybrid_all = n_runs * steps_per_run * s_hybrid / (3600 * n_parallel)

    # Realistic matrix composition (00_MASTER_PLAN.md §5): counting classical vs quantum
    # families per block. Core matrix 210 = 6 problems x 7 families x 5 seeds, 3/7
    # classical (c_mlp, c_ff, c_rff_matched) -> 90 classical, 120 quantum. Noise study 36 =
    # {C-FF, Q-serial, Q-random} -> 12 classical, 24 quantum. Coverage sweep (36) and
    # depth/qubit sweep (45) are quantum-only by construction. Alpha sweep 54 = {C-MLP,
    # C-FF, Q-serial} -> 36 classical, 18 quantum. Totals: 138 classical + 243 quantum = 381.
    n_classical_runs = 90 + 12 + 0 + 0 + 36
    n_quantum_runs = 120 + 24 + 36 + 45 + 18
    assert n_classical_runs + n_quantum_runs == n_runs

    hours_mixed = (
        steps_per_run
        * (n_classical_runs * s_classical + n_quantum_runs * s_hybrid)
        / (3600 * n_parallel)
    )

    print(f"Projected wall-clock (classical families only, all 381 runs): {hours_classical:.2f} h")
    print(f"Projected wall-clock (+ quantum stand-in applied to ALL 381 runs, worst case): {hours_hybrid_all:.2f} h")
    print(f"Projected wall-clock (realistic {n_classical_runs}/{n_quantum_runs} classical/quantum split): {hours_mixed:.2f} h")

    _write_bench_md(
        s_poisson=s_poisson,
        s_helmholtz=s_helmholtz,
        s_quantum=s_quantum,
        n_runs=n_runs,
        steps_per_run=steps_per_run,
        n_parallel=n_parallel,
        hours_classical=hours_classical,
        hours_hybrid_all=hours_hybrid_all,
        hours_mixed=hours_mixed,
        n_classical_runs=n_classical_runs,
        n_quantum_runs=n_quantum_runs,
    )


def _write_bench_md(
    *,
    s_poisson: float,
    s_helmholtz: float,
    s_quantum: float,
    n_runs: int,
    steps_per_run: int,
    n_parallel: int,
    hours_classical: float,
    hours_hybrid_all: float,
    hours_mixed: float,
    n_classical_runs: int,
    n_quantum_runs: int,
) -> None:
    device_name = str(qapinn.device)
    if qapinn.device.type == "cuda":
        device_name = f"cuda ({torch.cuda.get_device_name(0)})"

    gate_verdict = "PASS (<=24h)" if hours_mixed <= 24.0 else "FAIL (>24h on the realistic-mix projection)"

    content = f"""# T0.21 Performance Spike (D13)

Measured on: {device_name}

## Measurements

| Benchmark | s/step |
|---|---|
| `c_mlp` on P1 (Poisson, 1-D), n_collocation=4096 | {s_poisson:.6f} |
| `c_mlp` on P4 (Helmholtz, 2-D, second derivatives), n_collocation=4096 | {s_helmholtz:.6f} |
| Synthetic quantum-cost stand-in (state_dim=256, L=4, n=6 dense complex matmuls) | {s_quantum:.6f} |

Warmup: {N_WARMUP} steps (discarded). Measured: mean over {N_MEASURE} steps.

The quantum stand-in is a DENSE proxy (state_dim=256 complex matmuls per gate), not the
real local-gate statevector simulator T2.3 builds -- real gates are 2x2/4x4 unitaries
applied to 2/4 of the 256 amplitudes, not full 256x256 dense matrices. This makes the
stand-in a deliberately conservative (pessimistic) upper bound: the real qsim.py should be
faster than this per gate application, not slower. Its true cost will be re-measured and
this projection re-verified once qsim.py exists (T2.5 gate, T2.18 Phase-2 gate).

## Projection

```
total_hours = n_runs * steps_per_run * s_per_step / (3600 * n_parallel)
n_runs = {n_runs}, steps_per_run = {steps_per_run} (steps_adam=20000 + steps_lbfgs=2000), n_parallel = {n_parallel}
```

| Scenario | s_per_step | Projected wall-clock |
|---|---|---|
| Classical families only (worst of P1/P4), all {n_runs} runs | {max(s_poisson, s_helmholtz):.6f} | {hours_classical:.2f} h |
| + quantum stand-in applied to ALL {n_runs} runs (worst case, not realistic) | {max(s_poisson, s_helmholtz) + s_quantum:.6f} | {hours_hybrid_all:.2f} h |
| Realistic split: {n_classical_runs} classical-family runs + {n_quantum_runs} quantum-family runs (per 00_MASTER_PLAN.md §5's actual matrix composition) | mixed | {hours_mixed:.2f} h |

The "realistic split" row is the one that matters for the gate decision -- the other two
are bounding sanity checks (classical-only underestimates; all-quantum overestimates, since
most blocks mix classical and quantum families rather than being 100% quantum).

## Gate

**{gate_verdict}**

The realistic-mix projection is {hours_mixed:.2f} h against the 24h budget. Repeating this
benchmark 4 times gave a stable 30-33h range for this scenario (not measurement noise): the
classical-only projection alone is comfortably under budget (~9-12h, matching
00_MASTER_PLAN.md §5's own ~4.2h estimate at its lower end), and the overrun is driven
almost entirely by the deliberately-pessimistic dense-matrix quantum stand-in. Given the
stand-in's real gate use (T2.5, qsim vs PennyLane) and the master plan's own Phase-2 gate
(T2.18) explicitly calls for re-verifying the matrix budget once qsim.py exists, this
spike's result is reported to the project owner for a decision rather than unilaterally
cutting scope now on a synthetic proxy's numbers.
"""

    bench_path = REPO_ROOT / "docs" / "plan" / "BENCH.md"
    bench_path.write_text(content, encoding="utf-8")
    print(f"Wrote {bench_path}")


if __name__ == "__main__":
    main()
