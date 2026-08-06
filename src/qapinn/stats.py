"""T4.1: statistics layer (project.md SS8). 5 seeds -> report median + IQR, paired tests
across families ON THE SAME SEEDS, Wilcoxon signed-rank (not a t-test: n=5, no normality
assumption), Cliff's delta as the effect size since a p-value alone is thin at n=5, and
Holm-Bonferroni across the pre-registered PR-set for the multiple-comparison correction.

n=5 honesty (project.md SS8): the smallest achievable two-sided Wilcoxon p at n=5 is
0.0625 -- no single paired comparison in this project can ever reach p<0.05. This is a
property of the sample size, not a bug; report effect sizes and cross-problem direction
consistency, not significance stars.

Post-cut update (2026-08-06, state in FINDINGS.md): the Aug 7 deadline cut core_matrix's
seed count from the planned 5 down to 2 (08_TASK_INDEX.md's cut-line #4, applied twice).
`adjudicate_predictions.py`'s CONFIRMED gate (`p_value <= 0.0625`, kept UNCHANGED at its
original pre-registered n=5 value, not loosened post-hoc to compensate for the smaller
sample) is now structurally unreachable: at n=2 paired differences, the smallest possible
two-sided Wilcoxon p is 0.5 (verified directly -- both differences agreeing in sign is
the single most extreme n=2 outcome, and scipy.stats.wilcoxon returns p=0.5 for it, not
0.0625). Every `_beats_by_ratio`-based check (PR-1, PR-2, PR-6) can therefore only ever
land on REFUTED or INCONCLUSIVE now, never CONFIRMED, independent of the true effect
size -- a real, honest consequence of the seed cut, not a statistics-layer bug. Report
this explicitly rather than let a silent "still INCONCLUSIVE" read as inconclusive DATA
when it is actually inconclusive-by-construction at this sample size.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def cliffs_delta(a, b) -> float:
    """(#{a_i > b_j} - #{a_i < b_j}) / (len(a) * len(b)), in [-1, 1]. Non-parametric
    effect size: does not assume normality, unlike Cohen's d -- matches the Wilcoxon
    test's own assumptions."""
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a.reshape(-1, 1) - b.reshape(1, -1)
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    return (gt - lt) / (len(a) * len(b))


def paired_comparison(runs_a: list[dict], runs_b: list[dict], metric: str) -> dict:
    """Pairs `runs_a`/`runs_b` by their shared `'seed'` key (project.md SS8: paired
    tests across families on the SAME seeds, not independent samples) and compares
    `metric`. Returns:

        {'median_a', 'median_b', 'iqr_a', 'iqr_b', 'ratio_median',
         'wilcoxon_stat', 'p_value', 'n_pairs', 'cliffs_delta'}

    Raises ValueError if there are no shared seeds, or fewer than 2 (Wilcoxon is
    undefined below that). If every paired difference is exactly zero,
    `scipy.stats.wilcoxon` itself raises -- reported here as p_value=1.0,
    wilcoxon_stat=0.0 (no evidence of any difference) rather than propagating that
    exception, since "identical on every seed" is a legitimate, informative outcome.
    """
    by_seed_a = {r["seed"]: r[metric] for r in runs_a}
    by_seed_b = {r["seed"]: r[metric] for r in runs_b}
    common_seeds = sorted(set(by_seed_a) & set(by_seed_b))
    if len(common_seeds) < 2:
        raise ValueError(
            f"paired_comparison needs >=2 shared seeds between runs_a and runs_b, got {len(common_seeds)}"
        )

    vals_a = np.array([by_seed_a[s] for s in common_seeds], dtype=float)
    vals_b = np.array([by_seed_b[s] for s in common_seeds], dtype=float)

    median_a = float(np.median(vals_a))
    median_b = float(np.median(vals_b))
    iqr_a = float(np.percentile(vals_a, 75) - np.percentile(vals_a, 25))
    iqr_b = float(np.percentile(vals_b, 75) - np.percentile(vals_b, 25))
    ratio_median = median_a / median_b if median_b != 0 else float("inf")

    if np.allclose(vals_a, vals_b):
        wilcoxon_stat, p_value = 0.0, 1.0
    else:
        result = scipy_stats.wilcoxon(vals_a, vals_b)
        wilcoxon_stat, p_value = float(result.statistic), float(result.pvalue)

    return {
        "median_a": median_a,
        "median_b": median_b,
        "iqr_a": iqr_a,
        "iqr_b": iqr_b,
        "ratio_median": ratio_median,
        "wilcoxon_stat": wilcoxon_stat,
        "p_value": p_value,
        "n_pairs": len(common_seeds),
        "cliffs_delta": cliffs_delta(vals_a, vals_b),
    }


def bootstrap_ci(
    values,
    statistic=np.median,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `statistic(values)` (default: the median). `seed`
    makes this reproducible across repeated calls on the same input -- callers wanting
    genuinely independent draws (e.g. a coverage simulation) should vary `seed`
    explicitly, not rely on this defaulting to fresh randomness."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("bootstrap_ci requires at least one value")
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = statistic(sample)
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction (project.md SS8: applied across the 12
    pre-registered predictions). Adjusted p-values are monotonically non-decreasing in
    rank and each clipped to <=1.0; report both raw and corrected, per the phase doc."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for i, (key, p) in enumerate(items):
        adj = min((m - i) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[key] = running_max
    return adjusted
