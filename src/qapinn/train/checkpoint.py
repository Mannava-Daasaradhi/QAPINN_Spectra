"""Checkpointing + provenance (01_CONVENTIONS.md §8, D11, T0.19)."""
from __future__ import annotations

import platform
import socket
import subprocess
from pathlib import Path

import numpy
import pennylane
import scipy
import torch

from qapinn.models.base import PINNModel


def save_checkpoint(model: PINNModel, run_dir: Path, step: int) -> Path:
    """Save model.state_dict() to results/runs/<id>/checkpoints/step_<N>.pt (git-ignored --
    Phase 1 instruments (T1.x) run at checkpoints and must be re-runnable without retraining)."""
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step}.pt"
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model: PINNModel, path: Path) -> PINNModel:
    """Load a saved state_dict into `model` in place (onto whatever device model already
    lives on); returns the same model for chaining."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def build_provenance(run_id: str, seed: int, device: str, wall_clock_s: float) -> dict:
    """Full provenance.json per D11: git SHA, seed, resolved package versions, device,
    hostname, wall-clock, run_id."""
    return {
        "run_id": run_id,
        "seed": seed,
        "device": device,
        "wall_clock_s": wall_clock_s,
        "git_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "versions": {
            "torch": torch.__version__,
            "pennylane": pennylane.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
    }
