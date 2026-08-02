"""Determinism utilities (01_CONVENTIONS.md §9)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> torch.Generator:
    """Seeds python random, numpy, torch (cpu+cuda); enables deterministic algorithms.

    Returns a torch.Generator that must be threaded explicitly through all sampling
    calls (collocation/boundary sampling, etc.) -- never rely on global RNG state.

    CUBLAS_WORKSPACE_CONFIG must already be set to ":4096:8" in the environment before
    torch initialises any CUDA context (tasks.py sets it at process start; tests/conftest.py
    does the same for bare `pytest` invocations).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(prefer: str = "auto") -> torch.device:
    """Resolve the compute device once. prefer: "auto" | "cpu" | "cuda".

    QAPINN_DEVICE env var (checked only when prefer="auto") overrides autodetection --
    lets process-isolated batch runs (e.g. tasks.py smoke) force CPU without touching
    call sites, since qapinn.device is resolved once at import time (01_CONVENTIONS.md §3).
    """
    if prefer == "auto" and os.environ.get("QAPINN_DEVICE"):
        prefer = os.environ["QAPINN_DEVICE"]
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested via prefer='cuda' but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")
    if prefer != "auto":
        raise ValueError(
            f"unknown device preference: {prefer!r} (expected 'auto', 'cpu', or 'cuda')"
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
