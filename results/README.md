# results/

Every number and figure in this repository is computed from the files here; nothing
reloads a trained model. Start with `runs_index.csv` to find a run.

| Path | Contents |
|---|---|
| `runs/<run_id>/` | One training run. `run_id` is the first 12 hex characters of the SHA-256 of the run's canonical config, so a config always maps to the same directory. |
| `runs_index.csv` | Every run that carries data: experiment group, problem, family, seed, step budget, final rel-L2. Regenerate with `python scripts/index_runs.py`. |
| `design_cards.json` | SMCD's design card per problem: realised frequency set Ω, encoding scalings, depth, qubits, coverage, predicted benefit (`scripts/make_design_cards.py`). |
| `size_matching.json` | Parameter-matched (±10%) configurations for every family (T2.16). The core matrix trained each family at its default size instead; see FINDINGS.md erratum E6. |
| `cost_ledger.json` | Parameters, FLOPs and wall-clock per (problem, family), T4.6. |
| `manifest.json` | Which runs each figure in `paper/figures/` was drawn from. |

## Inside a run directory

| File | Contents |
|---|---|
| `config.yaml` | The complete configuration; `python tasks.py run` or `sweep` with the same settings reproduces the same `run_id`. |
| `metrics.json` | Final metrics: rel-L2, L∞, residual norm, parameter count, wall-clock. |
| `history.parquet` | Loss and metrics over training. |
| `provenance.json` | Git commit, device, host, package versions. |
| `design_card.json` | Quantum families only: the circuit actually built (scalings, Ω, qubits, depth). |
| `xai/*.npz` | Instrument outputs per checkpoint: `ntk`, `specerr`, `attribution`, `fisher`, `drift`, `gradvar`, `probes`, `landscape`, `block_mass`. |
| `error.json` | Present when an attempt crashed and was retried; `metrics.json` is from the successful retry. |

Not committed: `runs/*/checkpoints/` (model weights) and `reference/` (a cache of
reference solutions, rebuilt on demand).

## Groups in `runs_index.csv`

| Group | Runs | What they are |
|---|---:|---|
| `core_matrix`, `coverage_sweep`, `depth_sweep`, `alpha_sweep`, `noise_study`, `soft_bc_ntk` | 181 | The six pre-registered experiments, `configs/exp/<group>.yaml`. |
| `baseline_tuning`, `baseline_tuning_confirm` | 16 | T4.9's baseline-fairness check (FINDINGS.md F14), run after adjudication. |
| `core_matrix_extra_seed` | 92 | Seeds 2–4 of the core matrix, finished before the seed count was cut to {0, 1} under the deadline. No reported number uses them. |
| `coverage_sweep_superseded` | 33 | Coverage-sweep runs at earlier step budgets (20000+2000 and 4500+500) that were superseded when the budget was cut. FINDINGS.md erratum E4 cites the full-budget ones. |
| `t1.5_staircase` | 3 | The spectral-bias staircase gate runs (T1.5, `paper/figures/staircase_cmlp_p1`). |
| `test_fixture` | 2 | Smoke-scale runs pinned by the test suite. |

Directories under `runs/` that are not in the index are development smoke runs or
incomplete runs from Phases 0–2. No figure, verdict or test reads them.
