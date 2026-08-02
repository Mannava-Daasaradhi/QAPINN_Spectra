"""T0.19 DoD: a checkpoint reloads into a fresh model and reproduces the same rel_l2 to
1e-12."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import qapinn
import qapinn.models as models_pkg
import qapinn.pdes as pdes_pkg
import qapinn.reference as reference_pkg
import qapinn.train.loop as loop_mod
from qapinn.config import load_config
from qapinn.train.checkpoint import load_checkpoint
from qapinn.train.loop import train


@pytest.fixture(autouse=True)
def _isolated_results_root(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "RESULTS_ROOT", tmp_path)


def test_checkpoint_reload_reproduces_rel_l2():
    cfg = load_config("poisson", "c_mlp", seed=0)
    result = train(cfg, smoke=True)

    ckpt_dir = result.run_dir / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    assert len(ckpt_files) >= 2  # smoke checkpoints=(0, -1) -> at least step 0 and the final step
    final_ckpt = ckpt_files[-1]

    pde = pdes_pkg.build(result.cfg.pde)
    fresh_model = models_pkg.build(result.cfg.model, input_dim=pde.dim).to(qapinn.device)
    load_checkpoint(fresh_model, final_ckpt)

    eval_grid = pde.eval_grid(result.cfg.pde.n_eval).to(qapinn.device)
    with torch.no_grad():
        u_pred = pde.apply_hard_bc(eval_grid, fresh_model(eval_grid))
    u_pred_np = u_pred.detach().cpu().numpy().reshape(-1)

    u_ref = reference_pkg.reference_solution(pde, eval_grid.detach().cpu().numpy()).reshape(-1)
    rel_l2_reloaded = float(np.linalg.norm(u_pred_np - u_ref) / np.linalg.norm(u_ref))

    assert abs(rel_l2_reloaded - result.metrics["rel_l2"]) < 1e-12


def test_checkpoints_saved_at_every_requested_step():
    cfg = load_config("poisson", "c_mlp", seed=0)
    result = train(cfg, smoke=True)  # smoke overrides checkpoints to (0, -1)

    ckpt_dir = result.run_dir / "checkpoints"
    steps = sorted(int(p.stem.split("_")[1]) for p in ckpt_dir.glob("step_*.pt"))
    assert steps[0] == 0
    assert steps[-1] == 25  # steps_adam=20 + steps_lbfgs=5 (T2.17: cut from 50+10)


def test_provenance_has_git_sha_and_versions():
    cfg = load_config("poisson", "c_mlp", seed=0)
    result = train(cfg, smoke=True)

    with (result.run_dir / "provenance.json").open() as f:
        provenance = json.load(f)

    assert provenance["run_id"] == result.run_id
    assert provenance["seed"] == 0
    assert "hostname" in provenance
    assert "python_version" in provenance
    assert set(provenance["versions"].keys()) == {"torch", "pennylane", "numpy", "scipy"}
    # git_sha may be None outside a git repo, but the key must always be present
    assert "git_sha" in provenance
