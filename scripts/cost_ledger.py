"""T4.6 DoD: `results/cost_ledger.json` + a paper-ready table -- per family per problem,
parameter count, wall-clock/step, total wall-clock, and rel-L2 achieved. Per project.md S6,
this must also report error at MATCHED WALL-CLOCK, not just at matched parameters, so a
"fewer parameters" claim cannot silently smuggle in a much larger time cost.

Every core_matrix run trains for the SAME step budget regardless of family (steps_adam +
steps_lbfgs is fixed across all 7 families in configs/exp/core_matrix.yaml) -- so "matched
steps" is automatic, but per-step wall-clock is NOT: a quantum-simulated circuit can take far
longer per step than a classical MLP of similar parameter count. `wall_clock_ratio_vs_fastest`
operationalises "error at matched wall-clock" as the real, measured cost multiplier relative to
the cheapest family for the same problem -- e.g. "family X reaches its rel_l2 while taking 5.2x
as long as the fastest family for that problem" -- which is exactly the disparity project.md S6
warns against hiding.

`circuit_evals` and `flops_per_step` are reported as `null`, NOT fabricated: neither is
instrumented anywhere in the codebase yet (same gap flagged in BENCH.md's "metrics.json's dead
placeholder fields" section for T4.5 -- circuit_evals is hardcoded to 0 in every metrics.json,
and no FLOP counter exists for either the classical or quantum-simulated families). A literal
error-vs-wall-clock CURVE (resampling one family's trajectory at another's stopping time) is
also not derivable -- NOT because checkpoints are sparse (production runs actually checkpoint
at steps (0, 100, 500, 1000, 5000, 20000, final); only `smoke=True` runs collapse this to
(0, final), `train/loop.py`'s own `apply_smoke_overrides`), but because NO wall-clock
timestamp is recorded per checkpoint or per step anywhere -- `metrics.json`/`provenance.json`
only hold ONE final total `wall_clock_s`, and `history.parquet` logs `step`/`loss`/`lr`/
`grad_norm` with no timestamp column either. Both gaps are documented here, not hidden.

Many `results/runs/*` directories are stray smoke/prep-test artifacts from ad-hoc testing
throughout this project, or real full-scale runs of a DIFFERENT experiment (depth_sweep,
noise_study, ...) -- not core_matrix production runs. `enumerate_core_matrix_run_ids` uses
`qapinn.runner`'s own deterministic config expansion (the same mechanism `tasks.py sweep`
itself uses to compute run_ids) as the authoritative membership test, the same approach
`tests/test_verify_preregistration.py`'s real-data test already relies on -- this is stronger
than filtering on e.g. `steps_adam == 20000`, which cannot distinguish core_matrix runs from
other full-scale sweeps that happen to share the same step budget. It also preserves each
`PROBLEM_INSTANCES` LABEL (e.g. `helmholtz_k4`) rather than the raw PDE name (`helmholtz`),
so the three helmholtz variants in core_matrix.yaml group separately instead of collapsing.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import yaml

from qapinn.config import load_config
from qapinn.runner import PROBLEM_INSTANCES

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results" / "runs"
CORE_MATRIX_CFG = REPO_ROOT / "configs" / "exp" / "core_matrix.yaml"
_PROBLEM_BY_LABEL = {label: (pde_yaml, overrides) for label, pde_yaml, overrides in PROBLEM_INSTANCES}


def enumerate_core_matrix_run_ids(exp_cfg_path: Path = CORE_MATRIX_CFG) -> dict:
    """run_id -> {'problem': label, 'family': ..., 'seed': ...} for every run this
    experiment config expands to. Mirrors `qapinn.runner.enumerate_runs`'s cartesian
    expansion but keeps the problem LABEL instead of collapsing it into `cfg.pde.name`.
    Does not support the `axes` extension (core_matrix.yaml uses none)."""
    with Path(exp_cfg_path).open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    train_overrides = {f"train.{k}": v for k, v in spec.get("train", {}).items()}

    mapping = {}
    for label in spec["problems"]:
        pde_yaml, pde_overrides = _PROBLEM_BY_LABEL[label]
        overrides = {**(pde_overrides or {}), **train_overrides} or None
        for family in spec["families"]:
            for seed in spec["seeds"]:
                cfg = load_config(pde_yaml, family, overrides=overrides, seed=seed)
                mapping[cfg.run_id] = {"problem": label, "family": family, "seed": seed}
    return mapping


def load_run_record(run_dir: Path) -> dict | None:
    """One flat record for a run, or None if its config/metrics are missing (e.g. still
    in progress). `problem` is the raw PDE name from config.yaml -- callers that need the
    core_matrix LABEL (e.g. distinguishing helmholtz_k4/k10/k20) should override it via
    `build_cost_ledger`'s `label_by_run_id`."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    if not config_path.is_file() or not metrics_path.is_file():
        return None
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_cfg = cfg.get("train", {})
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    n_steps = train_cfg["steps_adam"] + train_cfg["steps_lbfgs"]
    wall_clock_s = metrics["wall_clock_s"]
    return {
        "run_id": run_dir.name,
        "family": cfg["model"]["family"],
        "problem": cfg["pde"]["name"],
        "seed": cfg["seed"],
        "n_params": metrics["n_params"],
        "wall_clock_s": wall_clock_s,
        "wall_clock_s_per_step": wall_clock_s / n_steps,
        "rel_l2": metrics["rel_l2"],
        "circuit_evals": None,  # not instrumented -- see module docstring
        "flops_per_step": None,  # not instrumented -- see module docstring
    }


def build_cost_ledger(run_dirs: list, label_by_run_id: dict | None = None) -> dict:
    """Aggregates `load_run_record` over seeds into one entry per (family, problem),
    with medians and `wall_clock_ratio_vs_fastest` (this family's median wall-clock
    divided by the fastest family's median wall-clock, for the SAME problem -- computed
    over whatever families/problems are present so far, not gated on the full sweep).

    `label_by_run_id`, when given, overrides each record's grouping key with
    `label_by_run_id[run_id]['problem']` (see `enumerate_core_matrix_run_ids`); records
    whose run_id isn't in the mapping are excluded -- this is how non-core_matrix runs
    (stray smoke tests, other experiments' full-scale runs) get filtered out."""
    records = []
    for d in run_dirs:
        r = load_run_record(d)
        if r is None:
            continue
        if label_by_run_id is not None:
            label = label_by_run_id.get(r["run_id"])
            if label is None:
                continue
            r = {**r, "problem": label["problem"]}
        records.append(r)

    grouped: dict = {}
    for r in records:
        grouped.setdefault((r["family"], r["problem"]), []).append(r)

    entries = []
    for (family, problem), rs in grouped.items():
        entries.append(
            {
                "family": family,
                "problem": problem,
                "n_seeds": len(rs),
                "n_params": rs[0]["n_params"],
                "wall_clock_s_median": statistics.median(r["wall_clock_s"] for r in rs),
                "wall_clock_s_per_step_median": statistics.median(r["wall_clock_s_per_step"] for r in rs),
                "rel_l2_median": statistics.median(r["rel_l2"] for r in rs),
                "circuit_evals": None,
                "flops_per_step": None,
            }
        )

    by_problem: dict = {}
    for e in entries:
        by_problem.setdefault(e["problem"], []).append(e)
    for es in by_problem.values():
        fastest = min(e["wall_clock_s_median"] for e in es)
        for e in es:
            e["wall_clock_ratio_vs_fastest"] = e["wall_clock_s_median"] / fastest

    entries.sort(key=lambda e: (e["problem"], e["family"]))
    return {
        "entries": entries,
        "n_production_runs": len(records),
        "families_present": sorted({e["family"] for e in entries}),
        "problems_present": sorted(by_problem.keys()),
    }


def render_markdown_table(ledger: dict) -> str:
    """A paper-ready table: one row per (problem, family), sorted the same way as
    `entries`. `circuit_evals`/`flops_per_step` are rendered as the literal string
    "n/a (uninstrumented)" -- never a fabricated number -- so the gap stays visible
    wherever this table is read, not just in this module's docstring."""
    header = (
        "| Problem | Family | n_seeds | n_params | wall-clock/step (s) | "
        "total wall-clock (s) | rel_l2 (median) | wall-clock ratio vs fastest | "
        "circuit_evals | FLOPs/step |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for e in ledger["entries"]:
        rows.append(
            "| {problem} | {family} | {n_seeds} | {n_params} | {wcps:.4f} | {wc:.2f} | "
            "{rel_l2:.4f} | {ratio:.2f}x | n/a (uninstrumented) | n/a (uninstrumented) |".format(
                problem=e["problem"],
                family=e["family"],
                n_seeds=e["n_seeds"],
                n_params=e["n_params"],
                wcps=e["wall_clock_s_per_step_median"],
                wc=e["wall_clock_s_median"],
                rel_l2=e["rel_l2_median"],
                ratio=e["wall_clock_ratio_vs_fastest"],
            )
        )
    return "\n".join(rows)


def main() -> None:
    label_by_run_id = enumerate_core_matrix_run_ids()
    run_dirs = sorted(
        RUNS_DIR / run_id for run_id in label_by_run_id if (RUNS_DIR / run_id / "metrics.json").is_file()
    )
    ledger = build_cost_ledger(run_dirs, label_by_run_id=label_by_run_id)
    out_path = REPO_ROOT / "results" / "cost_ledger.json"
    out_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({ledger['n_production_runs']} production runs, "
          f"{len(ledger['entries'])} (family, problem) entries)")
    print()
    print(render_markdown_table(ledger))


if __name__ == "__main__":
    main()
