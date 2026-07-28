"""T0.4 DoD: run_id is key-order invariant, seed-sensitive, rejects unknown fields,
and dotted overrides apply correctly."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

import qapinn.config as config_mod
from qapinn.config import ExpConfig, ModelConfig, PDEConfig, TrainConfig, load_config, run_id

PDE_YAML_A = textwrap.dedent(
    """\
    name: poisson
    params:
      alpha: 0.3
    n_collocation: 4096
    n_boundary: 512
    n_eval: 1024
    """
)

# Same content as PDE_YAML_A, deliberately reordered keys.
PDE_YAML_B = textwrap.dedent(
    """\
    n_eval: 1024
    params:
      alpha: 0.3
    name: poisson
    n_boundary: 512
    n_collocation: 4096
    """
)

MODEL_YAML = textwrap.dedent(
    """\
    family: c_mlp
    widths: [64, 64, 64]
    activation: tanh
    """
)


def _write_configs(root: Path, pde_yaml: str, model_yaml: str, name: str = "toy") -> None:
    (root / "pde").mkdir(parents=True, exist_ok=True)
    (root / "model").mkdir(parents=True, exist_ok=True)
    (root / "pde" / f"{name}.yaml").write_text(pde_yaml, encoding="utf-8")
    (root / "model" / f"{name}.yaml").write_text(model_yaml, encoding="utf-8")


def test_run_id_invariant_to_yaml_key_order(tmp_path, monkeypatch):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_configs(root_a, PDE_YAML_A, MODEL_YAML)
    _write_configs(root_b, PDE_YAML_B, MODEL_YAML)

    monkeypatch.setattr(config_mod, "CONFIG_ROOT", root_a)
    cfg_a = load_config("toy", "toy", seed=0)
    monkeypatch.setattr(config_mod, "CONFIG_ROOT", root_b)
    cfg_b = load_config("toy", "toy", seed=0)

    assert cfg_a.run_id == cfg_b.run_id


def test_seed_changes_run_id(tmp_path, monkeypatch):
    _write_configs(tmp_path, PDE_YAML_A, MODEL_YAML)
    monkeypatch.setattr(config_mod, "CONFIG_ROOT", tmp_path)

    cfg0 = load_config("toy", "toy", seed=0)
    cfg1 = load_config("toy", "toy", seed=1)
    assert cfg0.run_id != cfg1.run_id


def test_unknown_field_raises(tmp_path, monkeypatch):
    bad_pde_yaml = PDE_YAML_A + "not_a_real_field: 123\n"
    _write_configs(tmp_path, bad_pde_yaml, MODEL_YAML)
    monkeypatch.setattr(config_mod, "CONFIG_ROOT", tmp_path)

    with pytest.raises(ValidationError):
        load_config("toy", "toy", seed=0)


def test_dotted_overrides_apply(tmp_path, monkeypatch):
    _write_configs(tmp_path, PDE_YAML_A, MODEL_YAML)
    monkeypatch.setattr(config_mod, "CONFIG_ROOT", tmp_path)

    cfg = load_config(
        "toy",
        "toy",
        seed=0,
        overrides={"train.steps_adam": 100, "model.n_qubits": 4, "tag": "unit-test"},
    )
    assert cfg.train.steps_adam == 100
    assert cfg.model.n_qubits == 4
    assert cfg.tag == "unit-test"
    # untouched fields keep their defaults / composed values
    assert cfg.train.steps_lbfgs == 2000
    assert cfg.pde.name == "poisson"


def test_run_id_is_sha256_prefix():
    cfg = ExpConfig(
        pde=PDEConfig(name="poisson", params={"alpha": 0.3}),
        model=ModelConfig(family="c_mlp"),
        train=TrainConfig(),
        seed=0,
    )
    rid = run_id(cfg)
    assert len(rid) == 12
    assert all(c in "0123456789abcdef" for c in rid)
