"""QAPINN-Spectra task runner. Primary interface; Makefile mirrors these targets 1:1.

Usage: python tasks.py <command> [options]
"""
from __future__ import annotations

import argparse
import os
import sys

# Must be set before torch is ever imported anywhere in this process (01_CONVENTIONS.md §9).
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _parse_value(raw: str) -> object:
    """Best-effort string -> bool/int/float/str coercion for --set overrides."""
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _not_yet_implemented(name: str, task_id: str) -> int:
    print(f"tasks.py {name}: not yet implemented (see {task_id})", file=sys.stderr)
    return 1


def cmd_test(args: argparse.Namespace) -> int:
    import pytest

    pytest_args = ["tests"]
    if args.pattern:
        pytest_args += ["-k", args.pattern]
    if not args.slow:
        pytest_args += ["-m", "not slow"]
    if args.cov:
        pytest_args += ["--cov=src/qapinn", "--cov-report=term-missing"]

    code = pytest.main(pytest_args)
    if code == 5:  # pytest: "no tests were collected" -- expected before tests exist yet
        return 0
    return int(code)


def cmd_lint(args: argparse.Namespace) -> int:
    import subprocess

    targets = ["src", "tests", "scripts", "tasks.py"]
    targets = [t for t in targets if os.path.exists(os.path.join(REPO_ROOT, t))]
    return subprocess.call([sys.executable, "-m", "ruff", "check", *targets])


def cmd_run(args: argparse.Namespace) -> int:
    from qapinn.config import load_config
    from qapinn.train.loop import train

    overrides = {}
    for kv in args.overrides:
        key, _, value = kv.partition("=")
        overrides[key] = _parse_value(value)

    cfg = load_config(args.pde, args.model, overrides=overrides or None, seed=args.seed)
    result = train(cfg, smoke=args.smoke)
    # result.run_id (not cfg.run_id): smoke mode overrides steps/n_collocation inside
    # train(), so the config actually run -- and hashed for the results/ directory name --
    # differs from the pre-override cfg computed here.
    print(f"run_id={result.run_id} rel_l2={result.metrics.get('rel_l2')}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    return _not_yet_implemented("sweep", "T3.2")


# T2.17: the 6 problem instances x 7 model families matrix. Module-level (not inline in
# cmd_smoke) so tests/test_pipeline_smoke.py can import the SAME list rather than
# maintaining a second copy that could silently drift out of sync.
SMOKE_PDE_INSTANCES = [
    ("poisson", "poisson", None),
    ("heat", "heat", None),
    ("burgers", "burgers", None),
    ("helmholtz_k4", "helmholtz", {"pde.params.k": 4.0, "pde.params.a1": 1.0, "pde.params.a2": 1.0}),
    ("helmholtz_k10", "helmholtz", None),
    ("helmholtz_k20", "helmholtz", {"pde.params.k": 20.0, "pde.params.a1": 6.0, "pde.params.a2": 2.0}),
]
SMOKE_FAMILIES = ["c_mlp", "c_ff", "c_rff_matched", "q_serial", "q_random", "q_parallel", "q_octave"]


def cmd_smoke_one(args: argparse.Namespace) -> int:
    """Internal, not user-facing: run exactly ONE (pde, family) smoke combination and
    exit 0/1. `cmd_smoke` spawns this as a subprocess per combo so that whatever a given
    combo allocates -- host RAM, CUDA context, cached tensors -- is fully reclaimed by the
    OS on process exit, instead of accumulating across all 42 combos in a single process
    (the cause of the T2.17 crashes: gc.collect()/empty_cache() alone did not release
    everything CUDA/PyTorch retained combo-to-combo)."""
    from qapinn.config import load_config
    from qapinn.train.loop import train

    overrides = {}
    for kv in args.overrides:
        key, _, value = kv.partition("=")
        overrides[key] = _parse_value(value)

    cfg = load_config(args.pde, args.model, overrides=overrides or None, seed=0)
    result = train(cfg, smoke=True)
    assert (result.run_dir / "metrics.json").is_file()
    assert (result.run_dir / "config.yaml").is_file()
    assert (result.run_dir / "history.parquet").is_file()
    print(result.run_dir)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """T2.17: train(..., smoke=True) for every (problem instance x family) pair --
    6 problem instances x 7 model families = 42 combinations, all instruments enabled
    (TrainConfig's own defaults -- not overridden to a reduced set). The last cheap
    integration check before the Phase-3 matrix launches for real (project.md's own
    framing, T2.17's phase-doc section).

    Each combo runs in its OWN subprocess (see cmd_smoke_one) -- process-isolated, not
    in-process gc -- so one combo's leftover memory can never starve a later combo.
    Defaults to CUDA, capped at --mem-fraction (0.75) via
    torch.cuda.set_per_process_memory_fraction: this repo has crashed the host machine
    letting PyTorch's allocator claim the whole card, but a hard fraction cap plus
    per-combo process isolation keeps any single combo's allocator from approaching that
    ceiling. --cpu opts out of CUDA entirely.

    --workers > 1 (concurrent combos) was tried and is UNSAFE on this specific laptop:
    per-combo VRAM use is tiny (~900MB measured peak of 17GB, so capacity is not the
    issue), but 4 concurrent CUDA-using Python processes produced a cascade of driver/OS-
    level failures -- CUBLAS_STATUS_EXECUTION_FAILED, access violations, WinError 1450
    ("insufficient system resources"), page-file/DLL-load errors -- across 36/42 combos.
    This looks like contention over a shared driver or OS resource (page file, handle
    table, desktop heap) under concurrent CUDA context creation, not VRAM capacity.
    Left in place (opt-in, not default) for future investigation, not for routine use."""
    import subprocess
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    PDE_INSTANCES = SMOKE_PDE_INSTANCES
    FAMILIES = SMOKE_FAMILIES

    env = dict(os.environ)
    env["QAPINN_DEVICE"] = "cpu" if args.cpu else "cuda"
    if not args.cpu:
        env["QAPINN_CUDA_MEM_FRACTION"] = str(args.mem_fraction / args.workers)

    combos = [
        (f"{instance_label}/{family}", pde_yaml, family, overrides)
        for instance_label, pde_yaml, overrides in PDE_INSTANCES
        for family in FAMILIES
    ]

    def run_one(label: str, pde_yaml: str, family: str, overrides: dict | None) -> tuple[str, float, bool, str]:
        cmd = [sys.executable, __file__, "_smoke-one", "--pde", pde_yaml, "--model", family]
        for key, value in (overrides or {}).items():
            cmd += ["--set", f"{key}={value}"]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env)
        except subprocess.TimeoutExpired:
            return label, time.time() - t0, False, f"timed out after {args.timeout}s"
        dt = time.time() - t0
        if proc.returncode == 0:
            return label, dt, True, proc.stdout.strip()
        tail = proc.stderr.strip().splitlines()
        return label, dt, False, tail[-1] if tail else f"exit code {proc.returncode}, no stderr"

    t_start = time.time()
    n_total = len(combos)
    n_ok = 0
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *combo): combo[0] for combo in combos}
        for fut in as_completed(futures):
            label, dt, ok, detail = fut.result()
            if ok:
                n_ok += 1
                print(f"OK   {label} ({dt:.1f}s) -> {detail}")
            else:
                failures.append((label, detail))
                print(f"FAIL {label} ({dt:.1f}s): {detail}", file=sys.stderr)

    elapsed = time.time() - t_start
    print(f"\n{n_ok}/{n_total} combinations completed in {elapsed:.1f}s (device={env['QAPINN_DEVICE']})")
    if failures:
        print(f"{len(failures)} FAILURES:", file=sys.stderr)
        for label, err in failures:
            print(f"  {label}: {err}", file=sys.stderr)
        return 1
    return 0


def cmd_repro_quick(args: argparse.Namespace) -> int:
    return _not_yet_implemented("repro-quick", "T5.5")


def cmd_repro_all(args: argparse.Namespace) -> int:
    return _not_yet_implemented("repro-all", "T5.1")


def cmd_figures(args: argparse.Namespace) -> int:
    return _not_yet_implemented("figures", "T5.1")


def cmd_paper(args: argparse.Namespace) -> int:
    return _not_yet_implemented("paper", "T5.6")


def cmd_slides(args: argparse.Namespace) -> int:
    return _not_yet_implemented("slides", "T5.10")


def cmd_clean(args: argparse.Namespace) -> int:
    import shutil
    from pathlib import Path

    root = Path(REPO_ROOT)
    for name in (".pytest_cache", ".ruff_cache", "htmlcov"):
        p = root / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    cov_file = root / ".coverage"
    if cov_file.is_file():
        cov_file.unlink()
    for cache_dir in root.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tasks.py", description="QAPINN-Spectra task runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("test", help="Run the pytest suite")
    p.add_argument("-k", dest="pattern", default=None, help="pytest -k filter expression")
    p.add_argument("--slow", action="store_true", help="include tests marked @pytest.mark.slow")
    p.add_argument("--cov", action="store_true", help="collect coverage")
    p.set_defaults(func=cmd_test)

    p = sub.add_parser("lint", help="Run ruff over src/tests/scripts")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("smoke", help="Full-pipeline smoke: every PDE x every model family, reduced")
    p.add_argument("--cpu", action="store_true", help="run combos on CPU instead of the CUDA default")
    p.add_argument(
        "--mem-fraction",
        type=float,
        default=0.75,
        help="max AGGREGATE fraction of total VRAM CUDA may claim across all concurrent combos",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of combos to run concurrently -- DANGEROUS on this hardware, see cmd_smoke docstring",
    )
    p.add_argument("--timeout", type=float, default=180.0, help="per-combo subprocess timeout, seconds")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("_smoke-one", help=argparse.SUPPRESS)  # internal: one combo, spawned by cmd_smoke
    p.add_argument("--pde", required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="dotted.key=value", help=argparse.SUPPRESS
    )
    p.set_defaults(func=cmd_smoke_one)

    p = sub.add_parser("repro-quick", help="~15 min reduced-fidelity reproduction (CPU-only, T5.5)")
    p.set_defaults(func=cmd_repro_quick)

    p = sub.add_parser("repro-all", help="Full reproduction of every paper result")
    p.set_defaults(func=cmd_repro_all)

    p = sub.add_parser("run", help="Run a single experiment config")
    p.add_argument("--pde", required=True, help="configs/pde/<name>.yaml")
    p.add_argument("--model", required=True, help="configs/model/<name>.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="reduced steps/points, <60s")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="config override, repeatable",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="Run an experiment sweep config (configs/exp/*.yaml)")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("figures", help="Regenerate all figures from results/")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("paper", help="Build paper/main.pdf")
    p.set_defaults(func=cmd_paper)

    p = sub.add_parser("slides", help="Build slides/")
    p.set_defaults(func=cmd_slides)

    p = sub.add_parser("clean", help="Remove caches and build artifacts")
    p.set_defaults(func=cmd_clean)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
