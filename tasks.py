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
    from qapinn.train.loop import train

    from qapinn.config import load_config

    overrides = {}
    for kv in args.overrides:
        key, _, value = kv.partition("=")
        overrides[key] = _parse_value(value)

    cfg = load_config(args.pde, args.model, overrides=overrides or None, seed=args.seed)
    result = train(cfg, smoke=args.smoke)
    print(f"run_id={cfg.run_id} rel_l2={result.metrics.get('rel_l2')}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    return _not_yet_implemented("sweep", "T3.2")


def cmd_smoke(args: argparse.Namespace) -> int:
    return _not_yet_implemented("smoke", "T2.17")


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
    p.set_defaults(func=cmd_smoke)

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
