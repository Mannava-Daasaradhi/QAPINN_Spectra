"""Noise surrogates (D8, T2.11). Both classes are SURROGATES -- cheap, closed-form
stand-ins for real quantum-hardware noise, NOT density-matrix simulations. T4.4
quantifies the gap between these surrogates and a real density-matrix simulation; do not
present numbers produced through these as physically exact.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ShotNoise(nn.Module):
    """expval -> expval + detach(N(0, (1 - expval^2) / n_shots)). Straight-through: the
    noise term is detached so gradients pass through unmodified. `1 - expval^2` is the
    exact variance of a single +-1-valued Pauli measurement with mean `expval`; dividing
    by n_shots is the CLT variance of the sample mean over n_shots such measurements --
    valid in the CLT regime, and n_shots=1024 (the default) is comfortably there.
    """

    def __init__(self, n_shots: int = 1024):
        super().__init__()
        if n_shots <= 0:
            raise ValueError(f"n_shots must be positive, got {n_shots}")
        self.n_shots = n_shots

    def forward(self, expval: Tensor, generator: torch.Generator | None = None) -> Tensor:
        var = torch.clamp(1.0 - expval**2, min=0.0) / self.n_shots
        if generator is None:
            noise = torch.randn(expval.shape, dtype=expval.dtype, device=expval.device)
        else:
            # A generator's device must match torch.randn's own device argument (a CPU
            # generator against a CUDA output tensor raises); generate on the generator's
            # own device, then move to match expval, as elsewhere in this codebase (T1.12).
            noise = torch.randn(expval.shape, generator=generator, dtype=expval.dtype, device=generator.device)
            noise = noise.to(expval.device)
        return expval + (noise * var.sqrt()).detach()


class GlobalDepolarizing(nn.Module):
    """expval -> (1-p)^m * expval, m = number of noisy layers. Exact (not sampled) for
    the global depolarizing channel acting on any traceless observable (Pauli-Z included),
    and differentiable (a plain rescaling)."""

    def __init__(self, p: float, m: int):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if m < 0:
            raise ValueError(f"m must be nonnegative, got {m}")
        self.p = p
        self.m = m

    def forward(self, expval: Tensor) -> Tensor:
        return ((1.0 - self.p) ** self.m) * expval


def build_noise_model(noise: str, n_layers: int) -> nn.Module | None:
    """Parses `TrainConfig.noise` ('none' | 'shot_<n_shots>' | 'depol_<p>', T3.3's
    `noise_study.yaml`) into a `ShotNoise`/`GlobalDepolarizing` instance, or None for
    'none' (T2.11 built these classes but nothing constructed them from the config string
    until now). `n_layers`: the circuit's own layer count, used as GlobalDepolarizing's
    `m` -- one noisy layer per circuit layer."""
    if noise == "none":
        return None
    if noise.startswith("shot_"):
        return ShotNoise(n_shots=int(noise[len("shot_") :]))
    if noise.startswith("depol_"):
        return GlobalDepolarizing(p=float(noise[len("depol_") :]), m=n_layers)
    raise ValueError(f"unrecognised noise spec {noise!r}; expected 'none', 'shot_<n>', or 'depol_<p>'")
