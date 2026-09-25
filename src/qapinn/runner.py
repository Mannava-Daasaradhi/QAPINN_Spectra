"""T3.2: run orchestration -- expand configs/exp/*.yaml into individual ExpConfigs and
execute them with process isolation and resume support (01_CONVENTIONS.md, project.md
Section 5.3 step 9).
"""
from __future__ import annotations

import dataclasses
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

import qapinn.train.loop as _loop
from qapinn.config import ExpConfig, load_config
from qapinn.train.loop import apply_smoke_overrides

# The 6 problem instances shared across every Phase 2/3 use (T2.17's smoke matrix, every
# T3.x sweep): label -> (pde yaml stem, config overrides). Single source of truth --
# tasks.py's own SMOKE_PDE_INSTANCES re-exports this exact object so the two names can
# never silently drift apart.
PROBLEM_INSTANCES: list[tuple[str, str, dict | None]] = [
    ("poisson", "poisson", None),
    ("heat", "heat", None),
    ("burgers", "burgers", None),
    ("helmholtz_k4", "helmholtz", {"pde.params.k": 4.0, "pde.params.a1": 1.0, "pde.params.a2": 1.0}),
    ("helmholtz_k10", "helmholtz", None),
    ("helmholtz_k20", "helmholtz", {"pde.params.k": 20.0, "pde.params.a1": 6.0, "pde.params.a2": 2.0}),
]
_PROBLEM_BY_LABEL = {label: (pde_yaml, overrides) for label, pde_yaml, overrides in PROBLEM_INSTANCES}


def _merge_overrides(*dicts: dict | None) -> dict:
    merged: dict = {}
    for d in dicts:
        if d:
            merged.update(d)
    return merged


def enumerate_runs(exp_cfg_path: Path) -> list[ExpConfig]:
    """Expand an experiment YAML into the cartesian product of problems x families x
    seeds x any declared extra `axes`, one ExpConfig per combination. Schema:

        problems: [instance_label, ...]     # from PROBLEM_INSTANCES
        families: [family_name, ...]        # configs/model/<name>.yaml
        seeds: [int, ...]
        train: {key: value, ...}            # optional, shorthand for train.<key> overrides
        axes:                               # optional, each ADDS a cartesian dimension
          - key: dotted.override.key        # any load_config-style dotted override key
            values: [...]

    `axes` generalizes coverage_sweep's coverage_targets, depth_sweep's n_layers x
    n_qubits, alpha_sweep's alpha, and noise_study's noise into one mechanism, rather than
    a special-cased branch per experiment type -- all five configs/exp/*.yaml files (T3.3)
    are expressible this way.
    """
    return [cfg for _label, cfg in enumerate_labeled_runs(exp_cfg_path)]


def enumerate_labeled_runs(exp_cfg_path: Path) -> list[tuple[str, ExpConfig]]:
    """`enumerate_runs`, with each config paired with its problem-instance label
    ("helmholtz_k10", not the PDE name "helmholtz", which three instances share). Figure
    generation and adjudication both group runs by this label, so they must get it from
    the same place rather than re-deriving it."""
    with Path(exp_cfg_path).open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    problems = spec["problems"]
    families = spec["families"]
    seeds = spec["seeds"]
    train_overrides = {f"train.{k}": v for k, v in spec.get("train", {}).items()}
    axes = spec.get("axes", [])

    axis_combos: list[dict] = [{}]
    for axis in axes:
        axis_combos = [dict(combo, **{axis["key"]: v}) for combo in axis_combos for v in axis["values"]]

    configs: list[tuple[str, ExpConfig]] = []
    for problem in problems:
        if problem not in _PROBLEM_BY_LABEL:
            raise ValueError(f"unknown problem instance {problem!r}; expected one of {sorted(_PROBLEM_BY_LABEL)}")
        pde_yaml, pde_overrides = _PROBLEM_BY_LABEL[problem]
        for family in families:
            for seed in seeds:
                for axis_combo in axis_combos:
                    overrides = _merge_overrides(pde_overrides, train_overrides, axis_combo)
                    configs.append((problem, load_config(pde_yaml, family, overrides=overrides or None, seed=seed)))
    return configs


@dataclasses.dataclass
class RunAllResult:
    n_total: int
    n_ok: int
    n_skipped: int
    n_failed: int
    failures: list[tuple[str, str]]  # (run_id, error summary)


def _effective_run_id(cfg: ExpConfig, smoke: bool) -> str:
    """The run_id `train(cfg, smoke=smoke)` will ACTUALLY use. `smoke=True` mutates the
    config internally (apply_smoke_overrides, T2.17) before hashing it for run_id/run_dir
    -- using `cfg.run_id` directly here would check/report the wrong directory for every
    smoke run, silently breaking both resume (never finds a prior smoke run, re-runs it
    every time) and error.json placement."""
    return apply_smoke_overrides(cfg).run_id if smoke else cfg.run_id


def _run_one(cfg: ExpConfig, smoke: bool) -> tuple[str, bool, str]:
    """Executed in its OWN worker process (run_all's max_tasks_per_child=1) -- a crashed
    run writes error.json and reports failure without killing the pool or losing
    visibility into the other runs' results."""
    from qapinn.train.loop import train

    run_id = _effective_run_id(cfg, smoke)
    run_dir = _loop.RESULTS_ROOT / run_id
    try:
        result = train(cfg, smoke=smoke)
        return run_id, True, str(result.run_dir)
    except Exception as e:  # noqa: BLE001 -- isolate this run's failure, report, don't propagate
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "error.json").open("w", encoding="utf-8") as f:
            json.dump({"error": repr(e), "traceback": traceback.format_exc()}, f, indent=2)
        return run_id, False, repr(e)


def run_all(cfgs: list[ExpConfig], n_workers: int = 1, resume: bool = True, smoke: bool = False) -> RunAllResult:
    """Execute every config in `cfgs`. Skips (resume=True) any run whose
    results/runs/<run_id>/metrics.json already exists, so an interrupted sweep can be
    re-launched and only pick up what's left.

    Each run gets its OWN worker process (ProcessPoolExecutor's max_tasks_per_child=1) --
    genuine OS-level process isolation per run, the pattern T2.17 found necessary (a bare
    gc.collect()/empty_cache() between in-process runs did NOT prevent that task's crash;
    full process teardown did).

    n_workers defaults to 1, NOT the phase doc's originally-specified 6: T2.17 directly
    measured that 4 concurrent CUDA-using Python processes on this laptop produce a
    cascade of driver/OS-level failures (CUBLAS_STATUS_EXECUTION_FAILED, access
    violations, Windows WinError 1450 "insufficient system resources") even though
    per-run VRAM use is tiny (~1GB). Sequential (n_workers=1) is the only mode PROVEN safe
    on this hardware (BENCH.md's "Post-T2.18 Owner Decisions" section) -- pass a higher
    n_workers only on hardware where concurrent CUDA contexts have been separately
    verified safe.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    to_run = []
    n_skipped = 0
    for cfg in cfgs:
        # Looked up on the module at call time, not bound at import, so a redirected
        # QAPINN_RESULTS_DIR / monkeypatched RESULTS_ROOT (tests) is honoured here too.
        run_dir = _loop.RESULTS_ROOT / _effective_run_id(cfg, smoke)
        if resume and (run_dir / "metrics.json").is_file():
            n_skipped += 1
            continue
        to_run.append(cfg)

    n_ok = 0
    failures: list[tuple[str, str]] = []

    if to_run:
        with ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=1) as pool:
            futures = {pool.submit(_run_one, cfg, smoke): _effective_run_id(cfg, smoke) for cfg in to_run}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="sweep"):
                run_id, ok, message = fut.result()
                if ok:
                    n_ok += 1
                else:
                    failures.append((run_id, message))

    return RunAllResult(n_total=len(cfgs), n_ok=n_ok, n_skipped=n_skipped, n_failed=len(failures), failures=failures)
