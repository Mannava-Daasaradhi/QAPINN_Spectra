"""T4.7's mechanical substance: check each of `docs/predictions.md`'s PR-1...PR-12 against
real run data, using the EXACT pre-registered thresholds (never invented ones) and T4.1's
`paired_comparison`.

Does NOT produce the C1-C5 narrative verdicts T4.7's own adjudication table asks for --
those synthesize MULTIPLE PRs plus other evidence (T4.2-T4.6) into one written sentence
per claim, a human judgment call (same reasoning already established for T4.2's/T4.3's
own deferred write-ups -- "write the conclusion the data supports"). What this DOES
produce is the mechanical, falsifiable substance every PR's verdict is actually built on,
so T4.7 has exact numbers to cite rather than needing to re-derive them by hand.

Verdict vocabulary: CONFIRMED / REFUTED / INCONCLUSIVE / INSUFFICIENT_DATA.
- REFUTED: the pre-registered falsifier condition (`docs/predictions.md`'s own
  "Falsifiers" section, verbatim) is met. Every falsifier there is written as a plain
  ratio/threshold check, not a p-value check, so that's what decides REFUTED here too.
- CONFIRMED: threshold met AND (where a paired comparison applies) `p_value <= 0.0625` --
  the best achievable at n=5 (`project.md` SS8) -- i.e. a "beats on every seed" signal,
  not just a favourable median that could flip with one different seed.
- INCONCLUSIVE: threshold nominally met but the seeds disagree (`p_value > 0.0625`) --
  T4.7's own rule ("a verdict of inconclusive is acceptable and must be used where n=5
  cannot separate the alternatives") applied per-PR rather than only at the C1-C5 level.
- INSUFFICIENT_DATA: the runs this PR needs don't exist yet (e.g. `depth_sweep`,
  `coverage_sweep`, `helmholtz_k4`/`k20` not yet run) -- reported explicitly, never guessed.

PR-7/PR-8 depend on T3.4 (core matrix) only, not T3.5, so unlike PR-9/PR-10/PR-5 they are
checkable as soon as ANY problem has both `c_mlp` and `q_serial` runs -- this mirrors
`configs/exp/05_PHASE3_experiments.md`'s own T3.8/T3.9 dependency (`T3.4`, not `T3.5`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from qapinn.stats import paired_comparison

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results" / "runs"
DESIGN_CARDS_PATH = REPO_ROOT / "results" / "design_cards.json"
N5_P_FLOOR = 0.0625  # smallest achievable two-sided Wilcoxon p at n=5 (project.md SS8)
PR12_DRIFT_THRESHOLD = 0.2
PR11_CHECKPOINT_STEP = 5000


def _rel_l2_by_seed(run_dirs: list) -> list[dict]:
    out = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        out.append({"seed": cfg["seed"], "rel_l2": metrics["rel_l2"]})
    return out


def _shared_seed_count(runs_a: list[dict], runs_b: list[dict]) -> int:
    return len({r["seed"] for r in runs_a} & {r["seed"] for r in runs_b})


def _beats_by_ratio(run_dirs_a: list, run_dirs_b: list, threshold: float) -> dict:
    """'a beats b by >= threshold x lower median rel_l2' (PR-1/PR-2/PR-6's shape).
    ratio = median_b / median_a (how many times bigger b's error is than a's)."""
    runs_a, runs_b = _rel_l2_by_seed(run_dirs_a), _rel_l2_by_seed(run_dirs_b)
    if _shared_seed_count(runs_a, runs_b) < 2:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "fewer than 2 shared seeds"}

    stats = paired_comparison(runs_a, runs_b, metric="rel_l2")
    ratio = stats["median_b"] / stats["median_a"] if stats["median_a"] != 0 else float("inf")
    result = {"measured_ratio": ratio, "threshold": threshold, "stats": stats}
    if ratio < threshold:
        result["verdict"] = "REFUTED"
    elif stats["p_value"] <= N5_P_FLOOR:
        result["verdict"] = "CONFIRMED"
    else:
        result["verdict"] = "INCONCLUSIVE"
    return result


def _approximately_equal(run_dirs_a: list, run_dirs_b: list, band: float) -> dict:
    """'a ~= b, ratio within [1/band, band] either way' (PR-3's shape) -- a pure
    two-sided equivalence check, no p-value nuance (predictions.md states none for PR-3)."""
    runs_a, runs_b = _rel_l2_by_seed(run_dirs_a), _rel_l2_by_seed(run_dirs_b)
    if _shared_seed_count(runs_a, runs_b) < 2:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "fewer than 2 shared seeds"}

    stats = paired_comparison(runs_a, runs_b, metric="rel_l2")
    ratio = stats["median_a"] / stats["median_b"] if stats["median_b"] != 0 else float("inf")
    within_band = (1.0 / band) <= ratio <= band
    return {
        "verdict": "CONFIRMED" if within_band else "REFUTED",
        "measured_ratio": ratio,
        "band": [1.0 / band, band],
        "stats": stats,
    }


def check_pr1(run_dirs_by_family: dict) -> dict:
    """q_serial beats c_mlp on P1 (poisson) by >= 2x lower median rel-L2."""
    if not run_dirs_by_family.get("q_serial") or not run_dirs_by_family.get("c_mlp"):
        return {"verdict": "INSUFFICIENT_DATA", "reason": "missing q_serial or c_mlp runs"}
    return _beats_by_ratio(run_dirs_by_family["q_serial"], run_dirs_by_family["c_mlp"], threshold=2.0)


def check_pr2(run_dirs_by_family: dict) -> dict:
    """q_serial beats c_ff on P1 (poisson) by >= 1.3x lower median rel-L2."""
    if not run_dirs_by_family.get("q_serial") or not run_dirs_by_family.get("c_ff"):
        return {"verdict": "INSUFFICIENT_DATA", "reason": "missing q_serial or c_ff runs"}
    return _beats_by_ratio(run_dirs_by_family["q_serial"], run_dirs_by_family["c_ff"], threshold=1.3)


def check_pr3(run_dirs_by_family: dict) -> dict:
    """q_serial ~= c_rff_matched on P1 (poisson), ratio within 1.3x either way."""
    if not run_dirs_by_family.get("q_serial") or not run_dirs_by_family.get("c_rff_matched"):
        return {"verdict": "INSUFFICIENT_DATA", "reason": "missing q_serial or c_rff_matched runs"}
    return _approximately_equal(run_dirs_by_family["q_serial"], run_dirs_by_family["c_rff_matched"], band=1.3)


def check_pr4(run_dirs_by_family_heat: dict) -> dict:
    """No family beats c_mlp by more than 5% on P2 (heat) -- C4's core claim."""
    if not run_dirs_by_family_heat.get("c_mlp"):
        return {"verdict": "INSUFFICIENT_DATA", "reason": "missing c_mlp runs on heat"}
    c_mlp_runs = _rel_l2_by_seed(run_dirs_by_family_heat["c_mlp"])
    median_c_mlp = float(np.median([r["rel_l2"] for r in c_mlp_runs]))

    benefits = {}
    for family, dirs in run_dirs_by_family_heat.items():
        if family == "c_mlp" or not dirs:
            continue
        median_family = float(np.median([r["rel_l2"] for r in _rel_l2_by_seed(dirs)]))
        benefits[family] = (median_c_mlp - median_family) / median_c_mlp

    violators = {f: b for f, b in benefits.items() if b > 0.05}
    return {
        "verdict": "REFUTED" if violators else "CONFIRMED",
        "benefits": benefits,
        "violators": violators,
        "families_checked": sorted(benefits),
    }


def check_pr5(run_dirs_by_k: dict) -> dict:
    """P4 (helmholtz) advantage grows with k: c_mlp/q_serial rel-L2 ratio strictly
    increases across k=4,10,20. Needs all three k instances; reports INSUFFICIENT_DATA
    (not a guess) for any k missing c_mlp or q_serial runs."""
    ratios = {}
    for k_label in ("helmholtz_k4", "helmholtz_k10", "helmholtz_k20"):
        by_family = run_dirs_by_k.get(k_label, {})
        if not by_family.get("c_mlp") or not by_family.get("q_serial"):
            return {"verdict": "INSUFFICIENT_DATA", "reason": f"missing c_mlp or q_serial runs for {k_label}"}
        c_mlp_med = float(np.median([r["rel_l2"] for r in _rel_l2_by_seed(by_family["c_mlp"])]))
        q_serial_med = float(np.median([r["rel_l2"] for r in _rel_l2_by_seed(by_family["q_serial"])]))
        ratios[k_label] = c_mlp_med / q_serial_med if q_serial_med != 0 else float("inf")

    ordered = [ratios["helmholtz_k4"], ratios["helmholtz_k10"], ratios["helmholtz_k20"]]
    increasing = ordered[0] < ordered[1] < ordered[2]
    return {"verdict": "CONFIRMED" if increasing else "REFUTED", "ratios": ratios}


def check_pr6(run_dirs_by_family: dict) -> dict:
    """q_serial beats q_random at matched size by >= 1.5x lower median rel-L2 -- C1's
    own falsifier, verbatim."""
    if not run_dirs_by_family.get("q_serial") or not run_dirs_by_family.get("q_random"):
        return {"verdict": "INSUFFICIENT_DATA", "reason": "missing q_serial or q_random runs"}
    return _beats_by_ratio(run_dirs_by_family["q_serial"], run_dirs_by_family["q_random"], threshold=1.5)


def check_pr7(run_dir_c_mlp, run_dir_q_serial, problem: str, step: int) -> dict:
    """NTK spectrum of q_serial is flatter INSIDE the encoded band Omega than outside,
    by >= 0.5 (decay-exponent gap). Reads both runs' already-committed
    xai/ntk_step{step}.npz -- NOT make_ntk_spectrum_comparison, which reconstructs live
    models from results/runs/*/checkpoints/*.pt. checkpoints/ is deliberately gitignored
    (.gitignore's own stated policy: keep only config.yaml/metrics.json/design_card.json/
    xai/*.npz, exclude raw checkpoints), so calling the checkpoint-reloading version here
    made this check silently unreproducible from a clean clone -- found by actually
    running T5.14's clean-clone verification, not by inspection. See
    make_ntk_spectrum_comparison_from_npz's own docstring: it exists specifically to be
    the T5.1 DoD-compliant, committed-data-only substitute for exactly this call site.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from make_figures import make_ntk_spectrum_comparison_from_npz

    if not DESIGN_CARDS_PATH.is_file():
        return {"verdict": "INSUFFICIENT_DATA", "reason": "results/design_cards.json missing"}
    cards = json.loads(DESIGN_CARDS_PATH.read_text(encoding="utf-8"))
    if problem not in cards:
        return {"verdict": "INSUFFICIENT_DATA", "reason": f"no design card for {problem!r}"}
    omega_set = cards[problem]["omega_set"]

    result = make_ntk_spectrum_comparison_from_npz(
        run_dir_c_mlp, run_dir_q_serial, step, omega_set, figure_name=f"ntk_spectrum_{problem}_pr7"
    )
    inside, outside = result["decay_exponent_inside_band"], result["decay_exponent_outside_band"]
    if math.isnan(inside) or math.isnan(outside):
        return {
            "verdict": "INSUFFICIENT_DATA",
            "threshold": 0.5,
            "reason": (
                "decay exponent undefined for at least one of inside/outside the band -- "
                "fewer than 2 positive eigenvalues in that index range "
                "(_decay_exponent_in_index_range's own floor, not a NaN-vs-threshold "
                "coincidence: NaN >= 0.5 is False in Python, which would otherwise silently "
                "report REFUTED with no stated reason)"
            ),
            **result,
        }
    gap = abs(outside) - abs(inside)
    return {
        "verdict": "CONFIRMED" if gap >= 0.5 else "REFUTED",
        "measured_gap": gap,
        "threshold": 0.5,
        **result,
    }


def check_pr8(run_dir_c_mlp, run_dir_q_serial, problem: str) -> dict:
    """Per-frequency error improvement coincides with Omega: >= 70% of the improvement
    lands inside the encoded band.

    Checks both runs' `specerr.npz` for the crash-retry duplicate-step corruption found
    while building this check against real data (`scripts/check_specerr_integrity.py`)
    BEFORE calling `make_freq_heatmap_figure` -- that function's own mismatched-array
    guard often catches this by accident (a contaminated run usually has a different
    step COUNT than a clean one), but not always (two runs with the same duplication
    pattern would still "match"), and its error message ("must share the same eval grid")
    is misleading for what is actually single-run data corruption, not a real grid
    mismatch. Reported as INSUFFICIENT_DATA (not a crash) with the specific contaminated
    run named."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from check_specerr_integrity import specerr_has_duplicate_steps
    from make_figures import make_freq_heatmap_figure

    if not DESIGN_CARDS_PATH.is_file():
        return {"verdict": "INSUFFICIENT_DATA", "reason": "results/design_cards.json missing"}
    cards = json.loads(DESIGN_CARDS_PATH.read_text(encoding="utf-8"))
    if problem not in cards:
        return {"verdict": "INSUFFICIENT_DATA", "reason": f"no design card for {problem!r}"}
    omega_set = cards[problem]["omega_set"]

    for run_dir in (run_dir_c_mlp, run_dir_q_serial):
        if specerr_has_duplicate_steps(run_dir):
            return {
                "verdict": "INSUFFICIENT_DATA",
                "reason": f"specerr.npz corrupted by a crash/retry (duplicate step entries): {run_dir}",
            }

    result = make_freq_heatmap_figure(problem, run_dir_c_mlp, run_dir_q_serial, omega_set)
    # make_freq_heatmap_figure returns overlap_fraction=NaN by design when
    # total_improvement <= 0 (q_serial does not beat c_mlp at ANY frequency -- a 0/0
    # undefined ratio, not a computation failure). `NaN >= 0.7` is False in Python, so
    # this WOULD silently fall through to REFUTED anyway -- but via an accident of
    # float comparison semantics, with no stated reason and a verdict indistinguishable
    # from "some improvement exists but is poorly localized to Omega". Made explicit so
    # the reported reason matches what's actually true: PR-8's own premise (q_serial
    # shows a per-frequency improvement to attribute to Omega) does not hold here at all.
    if result["total_improvement"] <= 0:
        return {
            "verdict": "REFUTED",
            "threshold": 0.7,
            "reason": (
                "q_serial does not outperform c_mlp at any frequency "
                f"(total_improvement={result['total_improvement']!r}) -- PR-8's premise "
                "that an improvement exists to attribute to Omega does not hold, "
                "independent of the 70% threshold"
            ),
            **result,
        }
    return {
        "verdict": "CONFIRMED" if result["overlap_fraction"] >= 0.7 else "REFUTED",
        "threshold": 0.7,
        **result,
    }


def check_pr9(run_dirs_by_problem: dict) -> dict:
    """Final L2 error decreases monotonically with SMCD coverage: Spearman rho <= -0.7.
    Needs coverage_sweep runs (T3.5), not yet launched -- reports INSUFFICIENT_DATA until
    they exist; the check itself (make_coverage_vs_error_figure) is already implemented
    and tested (T3.7 prep, BENCH.md)."""
    if not run_dirs_by_problem:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "no coverage_sweep runs found"}
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from make_figures import make_coverage_vs_error_figure

    result = make_coverage_vs_error_figure(run_dirs_by_problem)
    worst_rho = max(r["spearman_rho"] for r in result.values())
    return {"verdict": "CONFIRMED" if worst_rho <= -0.7 else "REFUTED", "threshold": -0.7, "by_problem": result}


def check_pr10(run_dirs: list) -> dict:
    """Gradient variance decays with n but stays above the 2^-n line:
    barren_plateau_fit.slope_b < log(2). Needs depth_sweep runs (T3.5), not yet launched
    -- the check itself (make_barren_frontier_figure) is already implemented and tested
    (T4.5, BENCH.md)."""
    if not run_dirs:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "no depth_sweep runs found"}
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from make_figures import make_barren_frontier_figure

    result = make_barren_frontier_figure(run_dirs)
    if result.get("pr10_holds") is None:
        return {"verdict": "INSUFFICIENT_DATA", "threshold": math.log(2.0), **result}
    return {"verdict": "CONFIRMED" if result["pr10_holds"] else "REFUTED", "threshold": math.log(2.0), **result}


def check_pr11(run_dirs_hybrid: list, run_dirs_classical: list, step: int = PR11_CHECKPOINT_STEP) -> dict:
    """Hybrid leaves the lazy/NTK-linear regime earlier than classical: NTK drift
    ||Theta_t - Theta_0|| / ||Theta_0|| at 5k steps is LARGER for hybrid."""

    def _drifts(run_dirs):
        out = []
        for run_dir in run_dirs:
            npz_path = Path(run_dir) / "xai" / f"ntk_step{step}.npz"
            if npz_path.is_file():
                out.append(float(np.load(npz_path)["drift"]))
        return out

    hybrid_drifts, classical_drifts = _drifts(run_dirs_hybrid), _drifts(run_dirs_classical)
    if not hybrid_drifts or not classical_drifts:
        return {"verdict": "INSUFFICIENT_DATA", "reason": f"no readable ntk_step{step}.npz for one or both groups"}

    median_hybrid, median_classical = float(np.median(hybrid_drifts)), float(np.median(classical_drifts))
    return {
        "verdict": "CONFIRMED" if median_hybrid > median_classical else "REFUTED",
        "median_hybrid_drift": median_hybrid,
        "median_classical_drift": median_classical,
        "n_hybrid": len(hybrid_drifts),
        "n_classical": len(classical_drifts),
    }


def check_pr12(run_dirs_quantum: list) -> dict:
    """Encoder drift stays small enough that coverage remains valid: ||A - I||_F < 0.2
    at the final step. Only meaningful for quantum families (drift_step<N>.npz doesn't
    exist for classical families -- xai/__init__.py::_run_drift's own early return)."""
    frobenius_values = []
    for run_dir in run_dirs_quantum:
        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        final_step = cfg["train"]["steps_adam"] + cfg["train"]["steps_lbfgs"]
        npz_path = run_dir / "xai" / f"drift_step{final_step}.npz"
        if npz_path.is_file():
            frobenius_values.append(float(np.load(npz_path)["frobenius"]))

    if not frobenius_values:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "no readable drift_step<final>.npz"}

    median_frobenius = float(np.median(frobenius_values))
    return {
        "verdict": "CONFIRMED" if median_frobenius < PR12_DRIFT_THRESHOLD else "REFUTED",
        "median_frobenius": median_frobenius,
        "threshold": PR12_DRIFT_THRESHOLD,
        "n_runs": len(frobenius_values),
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from cost_ledger import enumerate_core_matrix_run_ids

    from qapinn.runner import enumerate_labeled_runs, enumerate_runs

    label_by_run_id = enumerate_core_matrix_run_ids()
    by_problem_family: dict = {}
    for run_id, info in label_by_run_id.items():
        run_dir = RUNS_DIR / run_id
        if not (run_dir / "metrics.json").is_file():
            continue
        by_problem_family.setdefault(info["problem"], {}).setdefault(info["family"], []).append(run_dir)

    poisson = by_problem_family.get("poisson", {})
    heat = by_problem_family.get("heat", {})
    helmholtz = {k: by_problem_family.get(k, {}) for k in ("helmholtz_k4", "helmholtz_k10", "helmholtz_k20")}
    helmholtz_k10 = helmholtz.get("helmholtz_k10", {})

    report = {
        "PR-1": check_pr1(poisson),
        "PR-2": check_pr2(poisson),
        "PR-3": check_pr3(poisson),
        "PR-4": check_pr4(heat),
        "PR-5": check_pr5(helmholtz),
        "PR-6 (poisson)": check_pr6(poisson),
        "PR-6 (heat)": check_pr6(heat),
    }

    # PR-7/PR-8: core_matrix only (T3.4), c_mlp vs q_serial at the final checkpoint.
    for problem_label, family_dirs in (("poisson", poisson), ("helmholtz_k10", helmholtz_k10)):
        if family_dirs.get("c_mlp") and family_dirs.get("q_serial"):
            # core_matrix's own final checkpoint step varies by run (20000+2000 at full
            # budget), so read it from config.yaml rather than assuming a literal number
            # (same pattern check_pr12 already uses).
            def _final_step(run_dir):
                cfg = yaml.safe_load((Path(run_dir) / "config.yaml").read_text(encoding="utf-8"))
                return cfg["train"]["steps_adam"] + cfg["train"]["steps_lbfgs"]

            step = _final_step(family_dirs["c_mlp"][0])
            report[f"PR-7 ({problem_label})"] = check_pr7(
                family_dirs["c_mlp"][0], family_dirs["q_serial"][0], problem_label, step
            )
            report[f"PR-8 ({problem_label})"] = check_pr8(
                family_dirs["c_mlp"][0], family_dirs["q_serial"][0], problem_label
            )

    # PR-9: coverage_sweep (T3.5), already complete -- group its own run_ids by problem
    # instance label, the same grouping (and so the same figure) `tasks.py figures` uses.
    coverage_cfg = REPO_ROOT / "configs" / "exp" / "coverage_sweep.yaml"
    if coverage_cfg.is_file():
        cov_by_problem: dict = {}
        for problem_label, cfg in enumerate_labeled_runs(coverage_cfg):
            run_dir = RUNS_DIR / cfg.run_id
            if (run_dir / "metrics.json").is_file():
                cov_by_problem.setdefault(problem_label, []).append(run_dir)
        report["PR-9"] = check_pr9(cov_by_problem)

    # PR-10: depth_sweep (T3.5) -- reports INSUFFICIENT_DATA cleanly if not complete yet.
    depth_cfg = REPO_ROOT / "configs" / "exp" / "depth_sweep.yaml"
    if depth_cfg.is_file():
        depth_cfgs = enumerate_runs(depth_cfg)
        depth_dirs = [RUNS_DIR / cfg.run_id for cfg in depth_cfgs if (RUNS_DIR / cfg.run_id / "metrics.json").is_file()]
        report["PR-10"] = check_pr10(depth_dirs)

    # PR-11/PR-12: core_matrix NTK-drift / encoder-drift npz, poisson q_serial vs c_mlp.
    if poisson.get("q_serial") and poisson.get("c_mlp"):
        report["PR-11"] = check_pr11(poisson["q_serial"], poisson["c_mlp"])
        report["PR-12"] = check_pr12(poisson["q_serial"])

    print(json.dumps(report, indent=2, default=str))
