"""c_mlp: plain tanh MLP PINN, Xavier-normal init, configurable widths (T0.15)."""
from __future__ import annotations

import numpy as np
from torch import Tensor, nn

from qapinn.models.base import PINNModel


class MLPPINN(PINNModel):
    def __init__(self, input_dim: int, widths: tuple[int, ...] = (64, 64, 64), activation: str = "tanh"):
        super().__init__()
        if activation != "tanh":
            raise ValueError(f"MLPPINN only supports activation='tanh', got {activation!r}")

        sizes = [input_dim, *widths, 1]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
        self.layers = nn.ModuleList(layers)
        # Separate hookable modules (not inline torch.tanh calls) so layer *activations*
        # (post-nonlinearity, T1.10's linear probes) are visible to register_forward_hook.
        self.activations = nn.ModuleList([nn.Tanh() for _ in range(len(layers) - 1)])

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activations[i](h)
        return h

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"classical": list(self.parameters()), "quantum": []}

    def realised_frequencies(self) -> np.ndarray | None:
        return None
