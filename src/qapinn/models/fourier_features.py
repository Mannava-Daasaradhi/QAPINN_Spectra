"""Fourier-feature PINNs: c_ff and c_rff_matched (T0.16).

c_ff: B ~ N(0, sigma^2) (Tancik-style) -- the fair classical baseline, non-negotiable
(project.md §6). c_rff_matched: B rows set to the SMCD target frequencies directly; until
Phase 2 exists, taken from an explicit config list (ModelConfig.frequencies) -- T2.14 wires
this to the real design card.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from qapinn.models.base import PINNModel


class FourierFeatures(nn.Module):
    """B: [m, d] buffer (NOT a parameter) -- one row per Fourier feature, holding ANGULAR
    frequencies directly in PHYSICAL coordinates (D12; no extra 2*pi anywhere).

    forward: x:[batch,d] -> [sin(x @ B^T), cos(x @ B^T)] : [batch, 2m].
    """

    def __init__(self, B: Tensor):
        super().__init__()
        self.register_buffer("B", B)

    @property
    def n_features(self) -> int:
        return self.B.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        proj = x @ self.B.T  # [batch, m]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [batch, 2m]


class FourierFeaturePINN(PINNModel):
    """Linear head on top of fixed Fourier features. Fair classical baseline (c_ff) / the
    classical dry-run of Prop. 3 (c_rff_matched)."""

    def __init__(self, B: Tensor):
        super().__init__()
        self.features = FourierFeatures(B)
        self.head = nn.Linear(2 * self.features.n_features, 1)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"classical": list(self.parameters()), "quantum": []}

    def realised_frequencies(self) -> np.ndarray | None:
        return self.features.B.detach().cpu().numpy()


def make_c_ff(
    input_dim: int, n_features: int, ff_sigma: float, gen: torch.Generator | None = None
) -> FourierFeaturePINN:
    """c_ff: B ~ N(0, sigma^2), Tancik-style. The fair classical baseline (non-negotiable,
    project.md §6)."""
    B = torch.randn(n_features, input_dim, generator=gen) * ff_sigma
    return FourierFeaturePINN(B)


def make_c_rff_matched(frequencies: Tensor | np.ndarray | list) -> FourierFeaturePINN:
    """c_rff_matched: B rows set to the SMCD target frequencies directly (1-D case: a flat
    list of scalar angular frequencies; T2.14 wires the real multi-dim design card)."""
    B = torch.as_tensor(frequencies, dtype=torch.get_default_dtype())
    if B.ndim == 1:
        B = B.unsqueeze(-1)
    return FourierFeaturePINN(B)
