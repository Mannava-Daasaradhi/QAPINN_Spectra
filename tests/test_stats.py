"""T4.1 DoD: Wilcoxon matches scipy.stats.wilcoxon on a known input; bootstrap CI covers
the true median at the nominal rate on synthetic data.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from qapinn.stats import bootstrap_ci, cliffs_delta, holm_bonferroni, paired_comparison


def _runs(seeds, values, metric="rel_l2"):
    return [{"seed": s, metric: v} for s, v in zip(seeds, values)]


def test_paired_comparison_wilcoxon_matches_scipy_directly():
    seeds = [0, 1, 2, 3, 4]
    vals_a = [0.5, 0.3, 0.8, 0.2, 0.6]
    vals_b = [0.4, 0.35, 0.7, 0.25, 0.5]

    result = paired_comparison(_runs(seeds, vals_a), _runs(seeds, vals_b), metric="rel_l2")
    expected = scipy_stats.wilcoxon(vals_a, vals_b)

    assert result["wilcoxon_stat"] == pytest.approx(float(expected.statistic))
    assert result["p_value"] == pytest.approx(float(expected.pvalue))
    assert result["n_pairs"] == 5
    assert result["median_a"] == pytest.approx(float(np.median(vals_a)))
    assert result["median_b"] == pytest.approx(float(np.median(vals_b)))


def test_paired_comparison_only_uses_shared_seeds():
    runs_a = _runs([0, 1, 2, 3], [0.5, 0.3, 0.8, 0.2])
    runs_b = _runs([1, 2, 3, 4], [0.4, 0.7, 0.25, 0.9])  # seed 0 missing, seed 4 extra

    result = paired_comparison(runs_a, runs_b, metric="rel_l2")
    assert result["n_pairs"] == 3  # only seeds 1, 2, 3 are shared


def test_paired_comparison_raises_below_two_shared_seeds():
    runs_a = _runs([0], [0.5])
    runs_b = _runs([0], [0.4])
    with pytest.raises(ValueError):
        paired_comparison(runs_a, runs_b, metric="rel_l2")


def test_paired_comparison_identical_values_reports_no_evidence_of_difference():
    seeds = [0, 1, 2, 3, 4]
    vals = [0.5, 0.3, 0.8, 0.2, 0.6]
    result = paired_comparison(_runs(seeds, vals), _runs(seeds, vals), metric="rel_l2")
    assert result["p_value"] == 1.0
    assert result["cliffs_delta"] == 0.0


def test_paired_comparison_n5_p_floor_is_honest():
    """project.md SS8: at n=5 the smallest achievable two-sided Wilcoxon p is 0.0625 --
    even a family that beats another on EVERY seed cannot reach p<0.05."""
    seeds = [0, 1, 2, 3, 4]
    vals_a = [0.1, 0.1, 0.1, 0.1, 0.1]  # strictly LOWER error on every seed ("better")
    vals_b = [0.9, 0.9, 0.9, 0.9, 0.9]
    result = paired_comparison(_runs(seeds, vals_a), _runs(seeds, vals_b), metric="rel_l2")
    assert result["p_value"] >= 0.0625 - 1e-9
    # a < b on every single pairing -> cliffs_delta(a, b) = -1 (standard convention:
    # positive delta means a tends to be GREATER than b, not "a wins" in some task sense)
    assert result["cliffs_delta"] == pytest.approx(-1.0)


def test_cliffs_delta_known_values():
    # a always greater than b -> delta = +1
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    # a always less than b -> delta = -1
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == pytest.approx(-1.0)
    # fully overlapping identical sets -> delta = 0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_bootstrap_ci_covers_true_median_at_nominal_rate():
    rng = np.random.default_rng(42)
    true_median = 0.0
    n_trials = 200
    hits = 0
    for trial in range(n_trials):
        sample = rng.normal(loc=true_median, scale=1.0, size=30)
        lo, hi = bootstrap_ci(sample, statistic=np.median, n_boot=500, alpha=0.05, seed=trial)
        if lo <= true_median <= hi:
            hits += 1
    coverage = hits / n_trials
    # nominal is 95%; Monte Carlo noise over 200 trials means +-3% (2*sqrt(0.05*0.95/200))
    # is expected -- allow a wider band (85-100%) to keep this non-flaky while still
    # catching a genuinely broken implementation (e.g. way off like 50%).
    assert 0.85 <= coverage <= 1.0


def test_bootstrap_ci_reproducible_with_same_seed():
    values = [0.1, 0.5, 0.3, 0.9, 0.2, 0.4]
    ci_a = bootstrap_ci(values, seed=7)
    ci_b = bootstrap_ci(values, seed=7)
    assert ci_a == ci_b


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_holm_bonferroni_matches_hand_computed_example():
    # Classic textbook example: 4 raw p-values, sorted ascending.
    raw = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    adjusted = holm_bonferroni(raw)
    # sorted ascending: d=0.005, a=0.01, c=0.03, b=0.04 -> multipliers 4,3,2,1
    assert adjusted["d"] == pytest.approx(0.02)
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)  # max(0.04*1, running_max=0.06) -> 0.06


def test_holm_bonferroni_enforces_monotonicity_and_clips_to_one():
    raw = {"x": 0.9, "y": 0.99, "z": 0.999}
    adjusted = holm_bonferroni(raw)
    values_in_rank_order = [adjusted[k] for k in sorted(raw, key=lambda k: raw[k])]
    assert values_in_rank_order == sorted(values_in_rank_order)
    assert all(0.0 <= v <= 1.0 for v in adjusted.values())


def test_holm_bonferroni_single_pvalue_unchanged():
    assert holm_bonferroni({"only": 0.03}) == pytest.approx({"only": 0.03})
