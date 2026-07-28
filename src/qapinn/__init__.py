"""QAPINN-Spectra: quantum-classical hybrid PINNs via spectrum-matched circuit design."""
from __future__ import annotations

import torch

from qapinn.seeding import resolve_device

# dtype policy (01_CONVENTIONS.md §3): PINN residuals are ill-conditioned in float32 and
# second derivatives amplify the error -- float64 throughout for physics and NTK.
torch.set_default_dtype(torch.float64)

# device policy (01_CONVENTIONS.md §3): resolved once, passed explicitly to every tensor
# construction. Never call .cuda() inline.
device: torch.device = resolve_device()
