"""QAPINN-Spectra: quantum-classical hybrid PINNs via spectrum-matched circuit design."""
from __future__ import annotations

import os

import torch

from qapinn.seeding import resolve_device

# dtype policy (01_CONVENTIONS.md §3): PINN residuals are ill-conditioned in float32 and
# second derivatives amplify the error -- float64 throughout for physics and NTK.
torch.set_default_dtype(torch.float64)

# device policy (01_CONVENTIONS.md §3): resolved once, passed explicitly to every tensor
# construction. Never call .cuda() inline.
device: torch.device = resolve_device()

# Optional hard VRAM cap (fraction of total device memory, e.g. "0.75"). Opt-in via env
# var only -- this repo's GPU has crashed the host machine outright when PyTorch's
# allocator was left free to claim the whole card, so batch drivers that hit this
# repeatedly (tasks.py smoke) set it explicitly rather than this being a silent global
# default that would also throttle unrelated real training runs.
if device.type == "cuda":
    _frac = os.environ.get("QAPINN_CUDA_MEM_FRACTION")
    if _frac:
        torch.cuda.set_per_process_memory_fraction(float(_frac))
