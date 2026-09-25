"""T2.18 gate item 6: produce and commit the SMCD design card for each of this project's
six problem instances (poisson, heat, burgers, helmholtz_k4/k10/k20). Reuses
tasks.SMOKE_PDE_INSTANCES so this list and the smoke matrix's own instance list (T2.17)
cannot silently drift apart.

Run: uv run python scripts/make_design_cards.py
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qapinn.config import load_config
from qapinn.pdes import build as build_pde
from qapinn.smcd.design import smcd
from tasks import SMOKE_PDE_INSTANCES


def main() -> None:
    cards = {}
    for instance_label, pde_yaml, overrides in SMOKE_PDE_INSTANCES:
        cfg = load_config(pde_yaml, "q_serial", overrides=overrides, seed=0)
        pde = build_pde(cfg.pde)
        card = smcd(pde, eps=cfg.smcd_eps, coverage_target=cfg.smcd_coverage_target)
        cards[instance_label] = asdict(card)
        print(f"{instance_label}: n_qubits={card.n_qubits} n_layers={card.n_layers} coverage={card.coverage:.4f}")

    out_path = REPO_ROOT / "results" / "design_cards.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cards, f, indent=2, sort_keys=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
