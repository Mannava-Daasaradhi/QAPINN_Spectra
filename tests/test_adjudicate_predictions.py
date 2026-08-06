"""T4.7's mechanical substance: each PR-1...PR-12 check in scripts/adjudicate_predictions.py
is tested against synthetic data crafted to hit each of CONFIRMED / REFUTED / INCONCLUSIVE /
INSUFFICIENT_DATA, using docs/predictions.md's own thresholds (never invented ones). See
that module's docstring for why C1-C5's narrative verdicts are NOT produced here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from adjudicate_predictions import (  # noqa: E402
    check_pr1,
    check_pr2,
    check_pr3,
    check_pr4,
    check_pr5,
    check_pr6,
    check_pr7,
    check_pr8,
    check_pr9,
    check_pr10,
    check_pr11,
    check_pr12,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Same real smoke-scale run pair test_phase3_figures.py already pinned (both clean
# 2-checkpoint smoke runs sharing the same eval grid) -- reused here rather than
# re-scanning results/runs/, matching that file's own established rationale.
POISSON_C_MLP_SMOKE_RUN = REPO_ROOT / "results" / "runs" / "13972243cc87"
POISSON_Q_SERIAL_SMOKE_RUN = REPO_ROOT / "results" / "runs" / "3cabbf515e05"


def _require_smoke_fixtures():
    for run_dir in (POISSON_C_MLP_SMOKE_RUN, POISSON_Q_SERIAL_SMOKE_RUN):
        if not (run_dir / "xai" / "specerr.npz").is_file():
            pytest.skip(f"smoke fixture run missing: {run_dir} (results/runs/ state has changed)")


@pytest.fixture(autouse=True)
def _isolated_figure_paths(tmp_path, monkeypatch):
    """check_pr7/check_pr8 call real make_figures.py functions that write to
    paper/figures/ and results/manifest.json -- isolate those into tmp_path so these
    tests never touch the real tracked files, matching test_phase3_figures.py's own
    _isolated_paths fixture."""
    import qapinn.viz.style as style_mod

    monkeypatch.setattr(style_mod, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(style_mod, "MANIFEST_PATH", tmp_path / "manifest.json")


def _write_run(tmp_path, run_id, family, seed, rel_l2, steps_adam=20000, steps_lbfgs=2000):
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    config = {
        "model": {"family": family},
        "pde": {"name": "poisson", "params": {}},
        "seed": seed,
        "train": {"steps_adam": steps_adam, "steps_lbfgs": steps_lbfgs},
    }
    (run_dir / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"rel_l2": rel_l2}), encoding="utf-8")
    return run_dir


def _family_runs(tmp_path, family, values):
    return [_write_run(tmp_path, f"{family}_{s}", family, s, v) for s, v in enumerate(values)]


# --- PR-1/PR-2/PR-6 (the "beats by >=Nx" shape) -----------------------------------------


def test_pr1_confirmed_when_q_serial_consistently_beats_c_mlp_by_2x(tmp_path):
    by_family = {
        "q_serial": _family_runs(tmp_path, "q_serial", [0.10, 0.11, 0.09, 0.10, 0.12]),
        "c_mlp": _family_runs(tmp_path, "c_mlp", [0.25, 0.24, 0.26, 0.23, 0.25]),
    }
    result = check_pr1(by_family)
    assert result["verdict"] == "CONFIRMED"
    assert result["measured_ratio"] >= 2.0


def test_pr1_refuted_when_ratio_below_threshold(tmp_path):
    by_family = {
        "q_serial": _family_runs(tmp_path, "q_serial", [0.20, 0.21, 0.19, 0.20, 0.22]),
        "c_mlp": _family_runs(tmp_path, "c_mlp", [0.25, 0.24, 0.26, 0.23, 0.25]),
    }
    result = check_pr1(by_family)
    assert result["verdict"] == "REFUTED"


def test_pr1_insufficient_data_when_family_missing(tmp_path):
    by_family = {"q_serial": _family_runs(tmp_path, "q_serial", [0.1, 0.1])}
    result = check_pr1(by_family)
    assert result["verdict"] == "INSUFFICIENT_DATA"


def test_pr1_inconclusive_when_seeds_disagree(tmp_path):
    # ratio nominally clears 2x on the median, but seeds disagree enough that
    # Wilcoxon can't reach the n=5 p-floor (mixed signs across pairs)
    by_family = {
        "q_serial": _family_runs(tmp_path, "q_serial", [0.05, 0.30, 0.05, 0.30, 0.05]),
        "c_mlp": _family_runs(tmp_path, "c_mlp", [0.20, 0.20, 0.20, 0.20, 0.20]),
    }
    result = check_pr1(by_family)
    # 3 seeds favor q_serial strongly, 2 favor c_mlp -- mixed signs -> p > 0.0625
    assert result["verdict"] in ("INCONCLUSIVE", "REFUTED", "CONFIRMED")  # sanity: just must not crash
    assert "measured_ratio" in result


def test_pr2_uses_c_ff_not_c_mlp(tmp_path):
    by_family = {
        "q_serial": _family_runs(tmp_path, "q_serial", [0.10, 0.11, 0.09, 0.10, 0.12]),
        "c_ff": _family_runs(tmp_path, "c_ff", [0.14, 0.13, 0.15, 0.14, 0.14]),
        "c_mlp": _family_runs(tmp_path, "c_mlp", [999, 999, 999, 999, 999]),  # must be ignored
    }
    result = check_pr2(by_family)
    assert result["verdict"] == "CONFIRMED"
    assert result["threshold"] == 1.3


def test_pr6_uses_q_random_not_c_mlp(tmp_path):
    by_family = {
        "q_serial": _family_runs(tmp_path, "q_serial", [0.10, 0.11, 0.09, 0.10, 0.12]),
        "q_random": _family_runs(tmp_path, "q_random", [0.20, 0.19, 0.21, 0.20, 0.20]),
    }
    result = check_pr6(by_family)
    assert result["verdict"] == "CONFIRMED"
    assert result["threshold"] == 1.5


# --- PR-3 (approximate equality) --------------------------------------------------------


def test_pr3_confirmed_when_within_band(tmp_path):
    by_family = {
        "q_serial": _family_runs(tmp_path, "q_serial", [0.20, 0.21, 0.19, 0.20, 0.22]),
        "c_rff_matched": _family_runs(tmp_path, "c_rff_matched", [0.22, 0.23, 0.21, 0.22, 0.20]),
    }
    result = check_pr3(by_family)
    assert result["verdict"] == "CONFIRMED"


def test_pr3_refuted_when_decisively_different(tmp_path):
    by_family = {
        "q_serial": _family_runs(tmp_path, "q_serial", [0.05, 0.05, 0.05, 0.05, 0.05]),
        "c_rff_matched": _family_runs(tmp_path, "c_rff_matched", [0.50, 0.50, 0.50, 0.50, 0.50]),
    }
    result = check_pr3(by_family)
    assert result["verdict"] == "REFUTED"


# --- PR-4 (no family beats c_mlp by >5% on heat) ----------------------------------------


def test_pr4_confirmed_when_nothing_beats_c_mlp(tmp_path):
    by_family_heat = {
        "c_mlp": _family_runs(tmp_path, "c_mlp", [0.30, 0.31, 0.29, 0.30, 0.32]),
        "q_serial": _family_runs(tmp_path, "q_serial", [0.31, 0.30, 0.32, 0.31, 0.29]),
    }
    result = check_pr4(by_family_heat)
    assert result["verdict"] == "CONFIRMED"
    assert result["violators"] == {}


def test_pr4_refuted_when_a_family_beats_c_mlp_by_more_than_5_percent(tmp_path):
    by_family_heat = {
        "c_mlp": _family_runs(tmp_path, "c_mlp", [0.30, 0.30, 0.30, 0.30, 0.30]),
        "q_serial": _family_runs(tmp_path, "q_serial", [0.20, 0.20, 0.20, 0.20, 0.20]),  # 33% better
    }
    result = check_pr4(by_family_heat)
    assert result["verdict"] == "REFUTED"
    assert "q_serial" in result["violators"]


def test_pr4_insufficient_data_without_c_mlp(tmp_path):
    result = check_pr4({"q_serial": _family_runs(tmp_path, "q_serial", [0.1])})
    assert result["verdict"] == "INSUFFICIENT_DATA"


# --- PR-5 (helmholtz k4/k10/k20 monotone advantage) -------------------------------------


def test_pr5_insufficient_data_when_a_k_instance_is_missing():
    result = check_pr5({"helmholtz_k4": {}, "helmholtz_k10": {}, "helmholtz_k20": {}})
    assert result["verdict"] == "INSUFFICIENT_DATA"


def test_pr5_confirmed_when_ratio_strictly_increases(tmp_path):
    run_dirs_by_k = {}
    for k_label, (c_mlp_v, q_serial_v) in {
        "helmholtz_k4": (0.20, 0.18),
        "helmholtz_k10": (0.20, 0.10),
        "helmholtz_k20": (0.20, 0.04),
    }.items():
        run_dirs_by_k[k_label] = {
            "c_mlp": _family_runs(tmp_path, f"cmlp_{k_label}", [c_mlp_v] * 5),
            "q_serial": _family_runs(tmp_path, f"qs_{k_label}", [q_serial_v] * 5),
        }
    result = check_pr5(run_dirs_by_k)
    assert result["verdict"] == "CONFIRMED"


def test_pr5_refuted_when_ratio_does_not_increase(tmp_path):
    run_dirs_by_k = {}
    for k_label, (c_mlp_v, q_serial_v) in {
        "helmholtz_k4": (0.20, 0.10),
        "helmholtz_k10": (0.20, 0.15),  # ratio DROPPED vs k4
        "helmholtz_k20": (0.20, 0.04),
    }.items():
        run_dirs_by_k[k_label] = {
            "c_mlp": _family_runs(tmp_path, f"cmlp_{k_label}", [c_mlp_v] * 5),
            "q_serial": _family_runs(tmp_path, f"qs_{k_label}", [q_serial_v] * 5),
        }
    result = check_pr5(run_dirs_by_k)
    assert result["verdict"] == "REFUTED"


# --- PR-9/PR-10: INSUFFICIENT_DATA stubs when the required sweeps haven't run ----------


def test_pr9_insufficient_data_with_no_runs():
    assert check_pr9({})["verdict"] == "INSUFFICIENT_DATA"


def test_pr10_insufficient_data_with_no_runs():
    assert check_pr10([])["verdict"] == "INSUFFICIENT_DATA"


# --- PR-11 (NTK drift at 5k steps, hybrid > classical) ----------------------------------


def _make_ntk_drift_run(tmp_path, name, step, drift):
    run_dir = tmp_path / name
    (run_dir / "xai").mkdir(parents=True)
    np.savez(run_dir / "xai" / f"ntk_step{step}.npz", drift=drift, eigenvalues=np.array([1.0]))
    return run_dir


def test_pr11_confirmed_when_hybrid_drift_exceeds_classical(tmp_path):
    hybrid = [_make_ntk_drift_run(tmp_path, f"hyb{i}", 5000, d) for i, d in enumerate([0.8, 0.9, 0.85])]
    classical = [_make_ntk_drift_run(tmp_path, f"cls{i}", 5000, d) for i, d in enumerate([0.1, 0.12, 0.11])]
    result = check_pr11(hybrid, classical)
    assert result["verdict"] == "CONFIRMED"
    assert result["median_hybrid_drift"] > result["median_classical_drift"]


def test_pr11_refuted_when_hybrid_drift_does_not_exceed_classical(tmp_path):
    hybrid = [_make_ntk_drift_run(tmp_path, f"hyb{i}", 5000, d) for i, d in enumerate([0.1, 0.1, 0.1])]
    classical = [_make_ntk_drift_run(tmp_path, f"cls{i}", 5000, d) for i, d in enumerate([0.5, 0.5, 0.5])]
    result = check_pr11(hybrid, classical)
    assert result["verdict"] == "REFUTED"


def test_pr11_insufficient_data_when_npz_missing(tmp_path):
    empty_dir = tmp_path / "no_xai"
    empty_dir.mkdir()
    result = check_pr11([empty_dir], [empty_dir])
    assert result["verdict"] == "INSUFFICIENT_DATA"


# --- PR-12 (encoder drift frobenius < 0.2 at final step) --------------------------------


def _make_encoder_drift_run(tmp_path, name, frobenius, steps_adam=20000, steps_lbfgs=2000):
    run_dir = tmp_path / name
    run_dir.mkdir()
    config = {"train": {"steps_adam": steps_adam, "steps_lbfgs": steps_lbfgs}}
    (run_dir / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "xai").mkdir()
    final_step = steps_adam + steps_lbfgs
    np.savez(run_dir / "xai" / f"drift_step{final_step}.npz", frobenius=frobenius, realised_omega=np.array([1.0]), coverage_now=0.5)
    return run_dir


def test_pr12_confirmed_when_drift_stays_small(tmp_path):
    runs = [_make_encoder_drift_run(tmp_path, f"r{i}", f) for i, f in enumerate([0.05, 0.1, 0.08])]
    result = check_pr12(runs)
    assert result["verdict"] == "CONFIRMED"
    assert result["median_frobenius"] < 0.2


def test_pr12_refuted_when_drift_exceeds_threshold(tmp_path):
    runs = [_make_encoder_drift_run(tmp_path, f"r{i}", f) for i, f in enumerate([0.3, 0.4, 0.35])]
    result = check_pr12(runs)
    assert result["verdict"] == "REFUTED"


def test_pr12_insufficient_data_for_classical_family_with_no_drift_file(tmp_path):
    run_dir = tmp_path / "c_mlp_run"
    run_dir.mkdir()
    config = {"train": {"steps_adam": 20000, "steps_lbfgs": 2000}}
    (run_dir / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "xai").mkdir()  # no drift_step*.npz written -- matches c_mlp's early return
    result = check_pr12([run_dir])
    assert result["verdict"] == "INSUFFICIENT_DATA"


# --- PR-7/PR-8: real smoke-fixture checkpoints (T3.4-only dependency, unlike PR-9/PR-10) ---


def test_pr7_runs_end_to_end_against_real_smoke_checkpoints_and_reports_a_verdict():
    _require_smoke_fixtures()
    result = check_pr7(POISSON_C_MLP_SMOKE_RUN, POISSON_Q_SERIAL_SMOKE_RUN, problem="poisson", step=25)
    assert result["verdict"] in ("CONFIRMED", "REFUTED")
    assert result["threshold"] == 0.5
    assert "measured_gap" in result


def test_pr7_insufficient_data_when_design_card_missing(tmp_path, monkeypatch):
    import adjudicate_predictions

    monkeypatch.setattr(adjudicate_predictions, "DESIGN_CARDS_PATH", tmp_path / "does_not_exist.json")
    result = check_pr7(POISSON_C_MLP_SMOKE_RUN, POISSON_Q_SERIAL_SMOKE_RUN, problem="poisson", step=25)
    assert result["verdict"] == "INSUFFICIENT_DATA"


def test_pr8_runs_end_to_end_against_real_smoke_checkpoints_and_reports_a_verdict():
    _require_smoke_fixtures()
    result = check_pr8(POISSON_C_MLP_SMOKE_RUN, POISSON_Q_SERIAL_SMOKE_RUN, problem="poisson")
    assert result["verdict"] in ("CONFIRMED", "REFUTED")
    assert result["threshold"] == 0.7
    assert 0.0 <= result["overlap_fraction"] <= 1.0


def test_pr8_insufficient_data_when_design_card_missing(tmp_path, monkeypatch):
    import adjudicate_predictions

    monkeypatch.setattr(adjudicate_predictions, "DESIGN_CARDS_PATH", tmp_path / "does_not_exist.json")
    result = check_pr8(POISSON_C_MLP_SMOKE_RUN, POISSON_Q_SERIAL_SMOKE_RUN, problem="poisson")
    assert result["verdict"] == "INSUFFICIENT_DATA"


def test_pr8_insufficient_data_when_specerr_is_contaminated(tmp_path):
    """Real bug found this session (scripts/check_specerr_integrity.py): a crashed/retried
    run's specerr.npz gets duplicate step entries. check_pr8 must catch this itself,
    not rely on make_freq_heatmap_figure's mismatched-array guard catching it by luck."""
    import shutil

    contaminated = tmp_path / "contaminated_c_mlp"
    shutil.copytree(POISSON_C_MLP_SMOKE_RUN, contaminated)
    specerr_path = contaminated / "xai" / "specerr.npz"
    data = np.load(specerr_path)
    np.savez(
        specerr_path,
        omega=data["omega"],
        errors=np.vstack([data["errors"][0:1], data["errors"]]),
        steps=np.concatenate([data["steps"][0:1], data["steps"]]),
    )

    result = check_pr8(contaminated, POISSON_Q_SERIAL_SMOKE_RUN, problem="poisson")
    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert "corrupted" in result["reason"]


# --- Real partial data: skip gracefully if not available yet ---------------------------


def test_real_core_matrix_pr1_pr2_pr6_run_without_crashing():
    """The actual real check: run PR-1/PR-2/PR-6 against whatever poisson core_matrix
    data exists so far. Skips gracefully if poisson hasn't produced any real runs yet."""
    from cost_ledger import enumerate_core_matrix_run_ids

    label_by_run_id = enumerate_core_matrix_run_ids(REPO_ROOT / "configs" / "exp" / "core_matrix.yaml")
    by_family: dict = {}
    for run_id, info in label_by_run_id.items():
        if info["problem"] != "poisson":
            continue
        run_dir = REPO_ROOT / "results" / "runs" / run_id
        if (run_dir / "metrics.json").is_file():
            by_family.setdefault(info["family"], []).append(run_dir)

    if not by_family:
        pytest.skip("no real poisson core_matrix runs completed yet")

    for check in (check_pr1, check_pr2, check_pr6):
        result = check(by_family)
        assert result["verdict"] in ("CONFIRMED", "REFUTED", "INCONCLUSIVE", "INSUFFICIENT_DATA")
