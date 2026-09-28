# Pre-registered predictions: the groundwater showcase (v1 models)

Committed, and pushed to GitHub, **before** any run of `configs/exp/groundwater_v1.yaml`.
`scripts/verify_preregistration.py --groundwater` checks that the commit adding this
file is an ancestor of every such run's recorded `git_sha`.

## The problem

Steady groundwater flow in a confined aquifer between two rivers 2 km apart, under
farmland irrigated by six leaking canals (`configs/pde/groundwater.yaml`,
`src/qapinn/pdes/groundwater.py`). Scenario values are within textbook ranges (Freeze &
Cherry 1979; Todd & Mays 2005), not calibrated to one site. The exact solution is closed
form and matches an independent finite-difference solver to 5e-8 m
(`tests/test_groundwater.py`). A field is waterlogged where the water table is within
1.5 m of the land surface (surface at 9 m, so above 7.5 m): exactly 713 m of the 2 km.

## The models and protocol

v1's families, unchanged: SMCD's designed circuit `q_serial` (1 qubit, 4 layers, 14
parameters), its random-frequency twin `q_random`, the frequency-matched classical
`c_rff_matched`, v1's `c_mlp` (~8,500 parameters), and `c_mlp_matched`, a `c_mlp` with 13
parameters (within 10% of `q_serial`). Seeds 10-14, never used before; the full
20000 Adam + 2000 L-BFGS budget; v1's hard boundary ansatz and learning rate.
Adjudicated by `scripts/adjudicate_groundwater.py --models v1`.

## Predictions

| ID | Prediction | Deciding metric | Threshold |
|---|---|---|---|
| G-1 | `q_serial` solves the problem accurately | median rel-L2 against the exact solution | ≤ 1% |
| G-2 | `q_serial` beats a same-size classical network | median rel-L2 ratio `c_mlp_matched` / `q_serial` | ≥ 1.3×, with v1's PR-1 statistical rule |
| G-3 | The SMCD design beats a random design of the same size | median rel-L2 ratio `q_random` / `q_serial` | ≥ 1.5×, v1's PR-6 rule |
| G-4 | `q_serial`'s improvement over `c_mlp` lies on the designed frequencies | v1's PR-8 overlap | ≥ 70% inside Ω |
| G-5 | `q_serial` gets the practical answer right | median waterlogged length vs the exact 713 m | within 5% |

Falsifiers: each claim is REFUTED when its metric misses its threshold, and
INCONCLUSIVE where v1's statistical rule says five seeds cannot separate the two models
(G-2, G-3). No claim is dropped or reworded after the runs.

**Stated in advance.** On v1's Poisson problem, the closest relative of this one,
`q_serial` lost to every classical baseline (FINDINGS.md), and the report's diagnostics
show why: its loss stalls near 3e4 and it loses the slow wave while fitting the fast
one. G-1 and G-2 are therefore expected to fail. G-3 and G-4 test the design rule
itself, which held on v1's heat problem (PR-6: 6.5× over the random design, all five
seeds).
