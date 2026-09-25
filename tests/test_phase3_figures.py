"""T3.6-T3.9 prep DoD: the three new figure-orchestration functions in
scripts/make_figures.py (make_coverage_vs_error_figure, make_ntk_spectrum_comparison,
make_freq_heatmap_figure) run end-to-end without crashing and produce the expected
artifacts. These are NOT the real T3.7/T3.8/T3.9 DoDs (those need actual T3.4/T3.5 run
data, which does not exist yet at prep time) -- they verify the plumbing against real
smoke-scale checkpoints (T3.8/T3.9) and synthetic-but-realistic ExpConfigs (T3.7), per
BENCH.md's "T3.6/T3.8/T3.9 -- prep work" section.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from make_figures import (
    _omega_lines_from_design_card,
    make_ablation_matched_figure,
    make_barren_frontier_figure,
    make_coverage_vs_error_figure,
    make_decision_map_figure,
    make_freq_heatmap_figure,
    make_ntk_spectrum_comparison,
    read_grad_var_final,
)

import qapinn.viz.style as style_mod
from qapinn.config import load_config
from qapinn.runner import enumerate_runs

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_CARDS_PATH = REPO_ROOT / "results" / "design_cards.json"

# Real smoke-scale run pairs (poisson, both clean 2-checkpoint smoke runs sharing the
# same eval grid) -- found once via a one-off scan of results/runs/*/config.yaml for
# n_collocation==256 and non-duplicated xai/specerr.npz steps; pinned here as fixture
# run_ids rather than re-scanning results/runs/ on every test run.
POISSON_C_MLP_SMOKE_RUN = REPO_ROOT / "results" / "runs" / "13972243cc87"
POISSON_Q_SERIAL_SMOKE_RUN = REPO_ROOT / "results" / "runs" / "3cabbf515e05"


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(style_mod, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(style_mod, "MANIFEST_PATH", tmp_path / "manifest.json")


def _require_smoke_fixtures():
    for run_dir in (POISSON_C_MLP_SMOKE_RUN, POISSON_Q_SERIAL_SMOKE_RUN):
        if not (run_dir / "xai" / "specerr.npz").is_file():
            pytest.skip(f"smoke fixture run missing: {run_dir} (results/runs/ state has changed)")


def _require_smoke_checkpoints():
    # checkpoints/ is gitignored: only the machine that trained the fixtures has them.
    for run_dir in (POISSON_C_MLP_SMOKE_RUN, POISSON_Q_SERIAL_SMOKE_RUN):
        if not (run_dir / "checkpoints" / "step_0.pt").is_file():
            pytest.skip(f"{run_dir.name}/checkpoints/ not present (gitignored)")


def test_freq_heatmap_figure_renders_with_omega_overlay_and_pr8_metric():
    _require_smoke_fixtures()
    cards = json.loads(DESIGN_CARDS_PATH.read_text(encoding="utf-8"))
    omega_set = cards["poisson"]["omega_set"]

    result = make_freq_heatmap_figure(
        "poisson", POISSON_C_MLP_SMOKE_RUN, POISSON_Q_SERIAL_SMOKE_RUN, omega_set
    )

    assert style_mod.FIGURES_DIR.joinpath("freq_heatmap_poisson.pdf").is_file()
    assert style_mod.FIGURES_DIR.joinpath("freq_heatmap_poisson.png").is_file()
    assert 0.0 <= result["overlap_fraction"] <= 1.0
    assert result["total_improvement"] > 0.0  # q_serial beats c_mlp in this smoke fixture


def test_omega_lines_keep_their_sign_on_a_signed_frequency_axis():
    omega_set = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    assert _omega_lines_from_design_card(omega_set, signed_axis=True).tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert _omega_lines_from_design_card(omega_set).tolist() == [2.0, 1.0, 0.0, 1.0, 2.0]
    # multi-component vectors always map to their norm (the radially binned axis)
    assert _omega_lines_from_design_card([[3.0, 4.0]], signed_axis=True).tolist() == [5.0]


def _write_specerr(run_dir: Path, omega: np.ndarray, errors: np.ndarray) -> Path:
    (run_dir / "xai").mkdir(parents=True)
    np.savez(run_dir / "xai" / "specerr.npz", omega=omega, errors=errors, steps=np.array([0, 10]))
    return run_dir


def test_pr8_overlap_counts_both_halves_of_a_symmetric_spectrum(tmp_path):
    # Regression for the post-submission PR-8 erratum: Omega is symmetric (+w and -w) and
    # so is a real field's error spectrum on Poisson's two-sided FFT axis. An improvement
    # that sits entirely inside Omega must score 1.0, not the ~0.5-plus-DC that folding
    # Omega onto |w| produced (v1.0 reported 71.3% for what is really 99.9%).
    omega = np.arange(-8.0, 9.0)  # signed axis, unit bins
    baseline = np.ones((2, omega.size))
    improved = baseline.copy()
    improved[1, np.abs(omega) <= 2] = 0.0  # improvement only at |w| <= 2, mirrored
    run_a = _write_specerr(tmp_path / "a", omega, baseline)
    run_b = _write_specerr(tmp_path / "b", omega, improved)

    result = make_freq_heatmap_figure("poisson", run_a, run_b, [[w] for w in (-2.0, -1.0, 0.0, 1.0, 2.0)])

    assert result["total_improvement"] == pytest.approx(5.0)
    assert result["overlap_fraction"] == pytest.approx(1.0)


def test_freq_heatmap_figure_rejects_mismatched_eval_grids(tmp_path):
    # Runs on different eval grids (e.g. two different PDEs) must be rejected rather than
    # silently plotted on one shared frequency axis. Synthetic runs, so the check never
    # depends on which development runs happen to be present in results/runs/.
    run_a = _write_specerr(tmp_path / "a", np.arange(-8.0, 9.0), np.ones((2, 17)))
    run_b = _write_specerr(tmp_path / "b", np.arange(0.0, 17.0), np.ones((2, 17)))

    with pytest.raises(ValueError, match="must share the same eval grid"):
        make_freq_heatmap_figure("poisson", run_a, run_b, [[1.0]])


@pytest.mark.slow
def test_ntk_spectrum_comparison_renders_with_band_and_inside_outside_decay():
    _require_smoke_fixtures()
    _require_smoke_checkpoints()
    cards = json.loads(DESIGN_CARDS_PATH.read_text(encoding="utf-8"))
    omega_set = cards["poisson"]["omega_set"]

    result = make_ntk_spectrum_comparison(
        POISSON_C_MLP_SMOKE_RUN,
        POISSON_Q_SERIAL_SMOKE_RUN,
        step=0,
        omega_set=omega_set,
        figure_name="ntk_spectrum_poisson_test",
    )

    assert style_mod.FIGURES_DIR.joinpath("ntk_spectrum_poisson_test.pdf").is_file()
    assert set(result["decay_exponent_whole"].keys()) == {"c_mlp", "q_serial"}
    assert result["band_size"] == len(omega_set)
    # both decay exponents must be real numbers (not NaN) given >=2 points on each side
    assert result["decay_exponent_inside_band"] == result["decay_exponent_inside_band"]
    assert result["decay_exponent_outside_band"] == result["decay_exponent_outside_band"]


def test_ntk_spectrum_comparison_rejects_mismatched_pdes():
    _require_smoke_fixtures()
    cards = json.loads(DESIGN_CARDS_PATH.read_text(encoding="utf-8"))
    # Any committed heat run works: the PDE mismatch is detected from config.yaml before
    # a (gitignored) checkpoint is ever loaded.
    heat_q_serial = next(
        c for c in enumerate_runs(REPO_ROOT / "configs" / "exp" / "core_matrix.yaml")
        if c.pde.name == "heat" and c.model.family == "q_serial"
    )
    other_pde_run = REPO_ROOT / "results" / "runs" / heat_q_serial.run_id

    with pytest.raises(ValueError, match="must be on the same PDE"):
        make_ntk_spectrum_comparison(
            POISSON_C_MLP_SMOKE_RUN,
            other_pde_run,
            step=0,
            omega_set=cards["poisson"]["omega_set"],
            figure_name="ntk_spectrum_mismatch_test",
        )


def test_coverage_vs_error_figure_groups_by_achieved_coverage_and_computes_spearman(tmp_path):
    targets_and_errors = [
        (0.2, 0.50), (0.2, 0.55),
        (0.4, 0.30), (0.4, 0.28),
        (0.6, 0.20), (0.6, 0.19),
        (0.9, 0.05), (0.9, 0.06),
        (1.0, 0.04), (1.0, 0.045),
    ]
    run_dirs = []
    for i, (target, err) in enumerate(targets_and_errors):
        cfg = load_config("poisson", "q_serial", seed=i, overrides={"smcd_coverage_target": target})
        run_dir = tmp_path / f"run{i}"
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=True), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(json.dumps({"rel_l2": err}), encoding="utf-8")
        run_dirs.append(run_dir)

    result = make_coverage_vs_error_figure({"poisson": run_dirs})

    assert style_mod.FIGURES_DIR.joinpath("coverage_vs_error.pdf").is_file()
    assert result["poisson"]["n_points"] == 10
    # engineered so higher coverage -> lower error: rho must be strongly negative
    assert result["poisson"]["spearman_rho"] <= -0.7
    assert result["poisson"]["spearman_p"] < 0.05


def test_coverage_vs_error_figure_reads_achieved_not_requested_coverage(tmp_path):
    """metrics.json's own smcd_coverage_weighted field is a dead placeholder (always
    0.0, BENCH.md) -- this must NOT be what the figure plots. A metrics.json with a
    poisoned smcd_coverage_weighted=0.0 for every run must still produce a real,
    non-degenerate spread of achieved-coverage x-values (recomputed via smcd(), not
    read from the dead field)."""
    cfg = load_config("poisson", "q_serial", seed=0, overrides={"smcd_coverage_target": 1.0})
    run_dir = tmp_path / "run0"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=True), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"rel_l2": 0.04, "smcd_coverage_weighted": 0.0}), encoding="utf-8"
    )

    result = make_coverage_vs_error_figure({"poisson": [run_dir]})
    assert result["poisson"]["n_points"] == 1


def _make_barren_run(tmp_path, name, n_qubits, n_layers, grad_var, seed=0):
    cfg = load_config(
        "poisson", "q_serial", seed=seed, overrides={"model.n_qubits": n_qubits, "model.n_layers": n_layers}
    )
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=True), encoding="utf-8")
    xai_dir = run_dir / "xai"
    xai_dir.mkdir()
    final_step = cfg.train.steps_adam + cfg.train.steps_lbfgs
    np.savez(
        xai_dir / f"gradvar_step{final_step}.npz",
        var_mean=grad_var,
        var_per_param=np.array([grad_var]),
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    return run_dir


def test_read_grad_var_final_reads_the_correct_final_checkpoint(tmp_path):
    run_dir = _make_barren_run(tmp_path, "run0", n_qubits=4, n_layers=2, grad_var=0.0123)
    assert read_grad_var_final(run_dir) == pytest.approx(0.0123)


def test_read_grad_var_final_returns_none_when_missing(tmp_path):
    cfg = load_config("poisson", "c_mlp", seed=0)
    run_dir = tmp_path / "run_classical"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=True), encoding="utf-8")
    assert read_grad_var_final(run_dir) is None


def test_barren_frontier_figure_renders_and_finds_a_frontier(tmp_path):
    # grad-var shrinking with n_qubits (the barren-plateau signature), two layer counts.
    sizes = [
        ("r0", 2, 2, 0.05), ("r1", 4, 2, 0.01), ("r2", 6, 2, 0.002),
        ("r3", 2, 4, 0.03), ("r4", 4, 4, 0.005), ("r5", 6, 4, 0.0008),
    ]
    run_dirs = [_make_barren_run(tmp_path, name, n_q, n_l, gv) for name, n_q, n_l, gv in sizes]

    result = make_barren_frontier_figure(run_dirs)

    assert style_mod.FIGURES_DIR.joinpath("barren_frontier.pdf").is_file()
    assert result["frontier_n_qubits"] is not None
    assert result["frontier_grad_var"] > 1e-10
    assert set(result["sizes_checked"]) == {(2, 2), (4, 2), (6, 2), (2, 4), (4, 4), (6, 4)}


def test_barren_frontier_figure_excludes_below_floor_sizes_from_frontier(tmp_path):
    run_dirs = [
        _make_barren_run(tmp_path, "ok", n_qubits=4, n_layers=2, grad_var=0.01),
        _make_barren_run(tmp_path, "collapsed", n_qubits=8, n_layers=2, grad_var=1e-12, seed=1),
    ]
    result = make_barren_frontier_figure(run_dirs)
    # the n=8 point is below the 1e-10 barren floor -- must not be reported as the frontier
    assert result["frontier_n_qubits"] == 4


def test_barren_frontier_figure_computes_pr10_slope_b(tmp_path):
    """PR-10 (docs/predictions.md): the deciding metric is barren_plateau_fit.slope_b
    vs log(2), not the practical frontier -- both must be reported, not just one."""
    import numpy as np

    true_b = 0.1  # well below log(2)~=0.693 -- classical-like, shallow-circuit regime
    ns = [2, 4, 6, 8, 10]
    run_dirs = [
        _make_barren_run(tmp_path, f"r{n}", n_qubits=n, n_layers=2, grad_var=float(np.exp(-true_b * n)))
        for n in ns
    ]
    result = make_barren_frontier_figure(run_dirs)
    assert result["slope_b"] == pytest.approx(true_b, abs=0.05)
    assert result["fit_r2"] > 0.99
    assert result["pr10_holds"] is True


def test_barren_frontier_figure_pr10_holds_false_when_slope_exceeds_log2(tmp_path):
    import numpy as np

    true_b = 1.5  # steeper than log(2) -- a genuine barren plateau
    ns = [2, 4, 6, 8, 10]
    run_dirs = [
        _make_barren_run(tmp_path, f"r{n}", n_qubits=n, n_layers=2, grad_var=float(np.exp(-true_b * n)))
        for n in ns
    ]
    result = make_barren_frontier_figure(run_dirs)
    assert result["pr10_holds"] is False


def test_barren_frontier_figure_reports_degenerate_fit_when_n_qubits_constant(tmp_path):
    # mirrors configs/exp/depth_sweep.yaml's current grid: n_qubits fixed at 6, only
    # n_layers varies -- the frontier itself is still computable, but slope_b/pr10_holds
    # must NOT be silently reported from a fit against a constant independent variable.
    run_dirs = [
        _make_barren_run(tmp_path, f"r{layer}", n_qubits=6, n_layers=layer, grad_var=1.0 / layer)
        for layer in (2, 3, 4, 5)
    ]
    result = make_barren_frontier_figure(run_dirs)
    assert result["frontier_n_qubits"] == 6
    assert result["slope_b"] is None
    assert result["pr10_holds"] is None
    assert result["fit_error"] == "degenerate_fit"


def test_barren_frontier_figure_raises_with_no_readable_grad_var(tmp_path):
    cfg = load_config("poisson", "c_mlp", seed=0)
    run_dir = tmp_path / "run_classical"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="nothing to plot"):
        make_barren_frontier_figure([run_dir])


def _make_ablation_run(tmp_path, name, problem, family, seed, rel_l2):
    cfg = load_config(problem, family, seed=seed)
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=True), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"rel_l2": rel_l2}), encoding="utf-8")
    return run_dir


def test_ablation_matched_figure_renders_and_computes_pairwise_stats(tmp_path):
    by_family = {
        "q_serial": [0.30, 0.32, 0.28, 0.31, 0.29],
        "c_rff_matched": [0.31, 0.33, 0.27, 0.30, 0.32],  # ~= q_serial by construction
        "q_random": [0.90, 0.95, 0.88, 0.92, 0.91],  # much worse
        "c_ff": [0.80, 0.82, 0.78, 0.81, 0.79],
    }
    run_dirs_by_family = {
        fam: [_make_ablation_run(tmp_path, f"{fam}_{s}", "poisson", fam, s, v) for s, v in enumerate(vals)]
        for fam, vals in by_family.items()
    }

    result = make_ablation_matched_figure({"poisson": run_dirs_by_family})

    assert style_mod.FIGURES_DIR.joinpath("ablation_matched.pdf").is_file()
    stats = result["poisson"]
    assert stats["n_seeds"] == {fam: 5 for fam in by_family}
    comparisons = stats["comparisons"]
    assert comparisons["q_serial_vs_c_rff_matched"]["n_pairs"] == 5
    # q_random is much worse than c_rff_matched on every seed -> large positive Cliff's delta
    assert comparisons["c_rff_matched_vs_q_random"]["cliffs_delta"] < -0.9


def test_ablation_matched_figure_reports_insufficient_data_below_two_shared_seeds(tmp_path):
    run_dirs_by_family = {
        "q_serial": [_make_ablation_run(tmp_path, "qs0", "poisson", "q_serial", 0, 0.3)],
        "c_rff_matched": [_make_ablation_run(tmp_path, "rff0", "poisson", "c_rff_matched", 0, 0.31)],
        "q_random": [],
        "c_ff": [],
    }
    result = make_ablation_matched_figure({"poisson": run_dirs_by_family})
    comparisons = result["poisson"]["comparisons"]
    assert comparisons["q_serial_vs_c_rff_matched"] == "insufficient_data"  # only 1 shared seed
    assert comparisons["c_rff_matched_vs_q_random"] == "insufficient_data"  # 0 q_random runs


def test_ablation_matched_figure_handles_multiple_problems_independently(tmp_path):
    def runs_for(problem, seed_offset):
        return {
            fam: [
                _make_ablation_run(tmp_path, f"{problem}_{fam}_{s}", problem, fam, s + seed_offset, 0.3 + 0.01 * s)
                for s in range(2)
            ]
            for fam in _ABLATION_FAMILIES_UNDER_TEST
        }

    run_dirs_by_problem = {"poisson": runs_for("poisson", 0), "heat": runs_for("heat", 10)}
    result = make_ablation_matched_figure(run_dirs_by_problem)
    assert set(result.keys()) == {"poisson", "heat"}
    assert style_mod.FIGURES_DIR.joinpath("ablation_matched.pdf").is_file()


_ABLATION_FAMILIES_UNDER_TEST = ("q_serial", "c_rff_matched", "q_random", "c_ff")


def test_decision_map_figure_computes_predicted_and_measured_benefit(tmp_path):
    from qapinn.pdes import build as build_pde
    from qapinn.smcd.design import smcd

    c_mlp_runs = [_make_ablation_run(tmp_path, f"cmlp_{s}", "poisson", "c_mlp", s, v) for s, v in enumerate([0.5, 0.4, 0.6])]
    q_serial_runs = [_make_ablation_run(tmp_path, f"qs_{s}", "poisson", "q_serial", s, v) for s, v in enumerate([0.2, 0.3, 0.25])]

    result = make_decision_map_figure({"poisson": {"c_mlp": c_mlp_runs, "q_serial": q_serial_runs}})

    assert style_mod.FIGURES_DIR.joinpath("decision_map.pdf").is_file()
    entry = result["poisson"]
    # medians: c_mlp=0.5, q_serial=0.25 -> measured_benefit = (0.5-0.25)/0.5 = 0.5
    assert entry["measured_benefit"] == pytest.approx(0.5)
    assert entry["n_seeds"] == {"c_mlp": 3, "q_serial": 3}

    cfg = load_config("poisson", "q_serial", seed=0)
    pde = build_pde(cfg.pde)
    expected_card = smcd(pde, eps=cfg.smcd_eps, coverage_target=cfg.smcd_coverage_target)
    assert entry["predicted_benefit"] == pytest.approx(expected_card.predicted_benefit)


def test_decision_map_figure_omits_problems_missing_required_families(tmp_path):
    c_mlp_only = [_make_ablation_run(tmp_path, "cmlp0", "poisson", "c_mlp", 0, 0.5)]
    both = {
        "c_mlp": [_make_ablation_run(tmp_path, "cmlp1", "heat", "c_mlp", 0, 0.5)],
        "q_serial": [_make_ablation_run(tmp_path, "qs1", "heat", "q_serial", 0, 0.3)],
    }
    result = make_decision_map_figure({"poisson": {"c_mlp": c_mlp_only}, "heat": both})
    assert "poisson" not in result
    assert "heat" in result


def test_decision_map_figure_raises_when_nothing_to_plot(tmp_path):
    c_mlp_only = [_make_ablation_run(tmp_path, "cmlp0", "poisson", "c_mlp", 0, 0.5)]
    with pytest.raises(ValueError, match="nothing to plot"):
        make_decision_map_figure({"poisson": {"c_mlp": c_mlp_only}})
