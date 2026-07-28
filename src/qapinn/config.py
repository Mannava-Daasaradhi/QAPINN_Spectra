"""Config schema, composition, and content-addressed run_id hashing (01_CONVENTIONS.md §4)."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"

_STRICT = ConfigDict(extra="forbid")


@pydantic_dataclass(config=_STRICT, frozen=True)
class PDEConfig:
    name: str
    params: dict[str, float]
    n_collocation: int = 4096
    n_boundary: int = 512
    n_eval: int = 1024


@pydantic_dataclass(config=_STRICT, frozen=True)
class ModelConfig:
    family: str
    widths: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    # Fourier-feature families
    n_features: int | None = None
    ff_sigma: float | None = None
    frequencies: tuple[float, ...] | None = None  # c_rff_matched only, pre-Phase-2
    # placeholder (T0.16) -- explicit target-spectrum frequencies until T2.14 wires this
    # to the real SMCD design card
    # quantum families
    n_qubits: int | None = None
    n_layers: int | None = None
    scaling_mode: str | None = None
    entangler: str | None = None
    observable: str | None = None
    encoder: str = "affine"
    target_params: int | None = None


@pydantic_dataclass(config=_STRICT, frozen=True)
class TrainConfig:
    steps_adam: int = 20000
    steps_lbfgs: int = 2000
    lr: float = 1e-3
    lr_schedule: str = "cosine"
    bc_mode: str = "hard"
    lambda_residual: float = 1.0
    lambda_bc: float = 1.0
    lambda_ic: float = 1.0
    checkpoints: tuple[int, ...] = (0, 100, 500, 1000, 5000, 20000, -1)
    noise: str = "none"


@pydantic_dataclass(config=_STRICT, frozen=True)
class ExpConfig:
    pde: PDEConfig
    model: ModelConfig
    train: TrainConfig
    seed: int
    smcd_eps: float = 1e-3
    smcd_coverage_target: float | None = None
    tag: str = ""

    @property
    def run_id(self) -> str:
        """sha256 of canonical JSON, first 12 hex chars. D11."""
        return run_id(self)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _apply_dotted_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply {'train.steps_adam': 100} style dotted-key overrides onto a nested dict."""
    result = copy.deepcopy(data)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        cursor = result
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
            cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = value
    return result


def load_config(
    pde: str,
    model: str,
    *,
    overrides: dict[str, Any] | None = None,
    seed: int = 0,
) -> ExpConfig:
    """Compose configs/pde/<pde>.yaml + configs/model/<model>.yaml (+ dotted overrides).

    pde/model yaml files hold exactly the fields of PDEConfig/ModelConfig respectively
    (no wrapping "pde:"/"model:" key). TrainConfig starts from its defaults and is
    customised only via `overrides` (e.g. {"train.bc_mode": "soft"}).
    """
    pde_data = _load_yaml(CONFIG_ROOT / "pde" / f"{pde}.yaml")
    model_data = _load_yaml(CONFIG_ROOT / "model" / f"{model}.yaml")

    data: dict[str, Any] = {
        "pde": pde_data,
        "model": model_data,
        "train": {},
        "seed": seed,
    }
    if overrides:
        data = _apply_dotted_overrides(data, overrides)

    return ExpConfig(**data)


def canonical_json(cfg: ExpConfig) -> str:
    """Deterministic JSON: sorted keys, no extra whitespace. Two configs differing only
    in source YAML key order (or dict insertion order) hash identically (D11)."""
    data = dataclasses.asdict(cfg)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def run_id(cfg: ExpConfig) -> str:
    """sha256(canonical_json(cfg))[:12]. D11."""
    digest = hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()
    return digest[:12]
