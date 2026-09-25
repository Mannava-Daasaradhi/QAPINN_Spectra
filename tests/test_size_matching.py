"""T2.16 DoD: for each of the six problem instances, all seven families' n_params() lie
within +-10% of q_serial's. Emit a table into results/size_matching.json for the paper's
appendix, alongside FLOPs and measured wall-clock (project.md Section 6 requires these
reported together).

Two KNOWN, documented exceptions to the +-10% requirement (see BENCH.md's T2.16 section
for the full writeup) -- both are genuine structural findings verified before writing any
code, not bugs papered over:
  1. c_rff_matched on Burgers (83% off): its own empirical target support (16 points,
     eps=1e-3) already needs more capacity (33 params) than q_serial's entire circuit
     there (14) -- T2.14 already documented this; "contains the target support" wins
     over "matches the size" when they structurally conflict.
  2. q_parallel on Heat/Burgers (11.11% off, just over the line): ParallelHybrid's own
     FIXED overhead (encoder + quantum circuit + scalar weight) for a time-dependent PDE
     (whose encoder must be sized to pde.dim, not the circuit's smaller spatial-only
     dimensionality -- found via this task, see hybrid.py) is already so close to
     q_serial's own tiny parameter budget that even the mathematically SMALLEST possible
     MLP branch (Linear(pde.dim, 1), zero hidden layers) still slightly overshoots.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from qapinn.models.base import size_matching_report
from qapinn.pdes.burgers import Burgers
from qapinn.pdes.heat import Heat
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson

TOL = 0.10

INSTANCES = [
    ("poisson", Poisson(alpha=0.3)),
    ("heat", Heat(alpha=0.3, nu=0.05)),
    ("burgers", Burgers()),
    ("helmholtz_k4", Helmholtz(k=4.0, a1=1.0, a2=1.0)),
    ("helmholtz_k10", Helmholtz(k=10.0, a1=3.0, a2=1.0)),
    ("helmholtz_k20", Helmholtz(k=20.0, a1=6.0, a2=2.0)),
]

KNOWN_MISSES = {
    ("burgers", "c_rff_matched"),
    ("heat", "q_parallel"),
    ("burgers", "q_parallel"),
}

EXPECTED_FAMILIES = {"q_serial", "c_mlp", "c_ff", "c_rff_matched", "q_random", "q_parallel", "q_octave"}


@pytest.fixture(scope="module")
def report():
    return size_matching_report(INSTANCES, tol=TOL)


def test_report_covers_six_instances_seven_families(report):
    assert set(report.keys()) == {name for name, _ in INSTANCES}
    for instance, families in report.items():
        assert set(families.keys()) == EXPECTED_FAMILIES, instance


def test_all_families_within_tolerance_except_known_exceptions(report):
    failures = []
    for instance, families in report.items():
        for family, m in families.items():
            if (instance, family) in KNOWN_MISSES:
                continue
            if m["rel_diff_from_q_serial"] > TOL:
                failures.append((instance, family, m["rel_diff_from_q_serial"]))
    assert not failures, f"unexpected tolerance misses: {failures}"


def test_known_misses_are_still_present_and_not_worse_than_expected():
    """Guards against silent drift: if a "known miss" starts passing, that's fine (an
    improvement); if a family we expect to match suddenly misses, or a known miss gets
    dramatically worse, that's worth noticing rather than the exception list quietly
    absorbing a new regression."""
    report_local = size_matching_report(INSTANCES, tol=TOL)
    heat = report_local["heat"]["q_parallel"]["rel_diff_from_q_serial"]
    burgers_parallel = report_local["burgers"]["q_parallel"]["rel_diff_from_q_serial"]
    burgers_rff = report_local["burgers"]["c_rff_matched"]["rel_diff_from_q_serial"]

    assert heat < 0.20, f"heat q_parallel drifted worse than expected: {heat}"
    assert burgers_parallel < 0.20, f"burgers q_parallel drifted worse than expected: {burgers_parallel}"
    assert burgers_rff < 1.5, f"burgers c_rff_matched drifted worse than expected: {burgers_rff}"


def _without_timings(report: dict) -> dict:
    return {
        instance: {family: {k: v for k, v in m.items() if k != "wall_clock_s"} for family, m in families.items()}
        for instance, families in report.items()
    }


def test_emits_results_json(report, tmp_path):
    # Written to a temp dir: the committed results/size_matching.json is an artifact, and
    # its wall_clock_s fields change on every run, so a test must not rewrite it.
    out_path = tmp_path / "size_matching.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    assert out_path.is_file()
    with out_path.open(encoding="utf-8") as f:
        reloaded = json.load(f)
    assert set(reloaded.keys()) == {name for name, _ in INSTANCES}


def test_committed_results_json_matches_current_code(report):
    # Every field except the machine-dependent timings must still agree with what the
    # code computes today; a mismatch means the committed table has gone stale.
    committed_path = Path(__file__).resolve().parent.parent / "results" / "size_matching.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    assert _without_timings(committed) == _without_timings(report)
