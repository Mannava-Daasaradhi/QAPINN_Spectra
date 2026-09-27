"""v2 boundary ansatz (bc_mode: hard_affine), models.base.AffineBCWrapper."""
from __future__ import annotations

import math

import pytest
import torch

from qapinn.config import load_config
from qapinn.models.base import AffineBCWrapper
from qapinn.models.mlp import MLPPINN
from qapinn.pdes import build as build_pde

_PROBLEMS = {
    "poisson": ("poisson", None),
    "heat": ("heat", None),
    "helmholtz_k10": ("helmholtz", None),
    "groundwater": ("groundwater", None),
}


def _pde(label):
    pde_yaml, overrides = _PROBLEMS[label]
    pde = build_pde(load_config(pde_yaml, "c_mlp", overrides=overrides).pde)
    pde.bc_ansatz = "affine"
    return pde


def _constrained_points(pde, n=64):
    """Points on every Dirichlet face (spatial axes) and on t = t0 (time axis)."""
    gen = torch.Generator().manual_seed(0)
    pts = []
    for axis, (lo, hi) in enumerate(pde.domain.bounds):
        faces = (lo,) if axis == pde.domain.time_axis else (lo, hi)
        for value in faces:
            x = torch.stack(
                [torch.rand(n, generator=gen, dtype=torch.float64) * (b - a) + a for a, b in pde.domain.bounds], dim=1
            )
            x[:, axis] = value
            pts.append(x)
    return torch.cat(pts)


@pytest.mark.parametrize("label", sorted(_PROBLEMS))
def test_any_network_satisfies_the_constraints_exactly(label):
    pde = _pde(label)
    torch.manual_seed(1)
    wrapped = AffineBCWrapper(MLPPINN(input_dim=pde.dim, widths=(16, 16)).double(), pde)
    x = _constrained_points(pde)
    u = pde.apply_hard_bc(x, wrapped(x))
    torch.testing.assert_close(u, pde.exact(x).to(u.dtype), atol=1e-10, rtol=0)


def test_network_equal_to_the_solution_is_left_unchanged():
    """The point of the ansatz: if N = u - B, then u is reproduced exactly, so a
    band-limited solution stays representable (v1's mask ansatz divides it by D)."""
    pde = _pde("poisson")

    class Exact(MLPPINN):
        def forward(self, x):
            xx = x[:, 0:1]
            return torch.sin(math.pi * xx) + 0.3 * torch.sin(15 * math.pi * xx)

    wrapped = AffineBCWrapper(Exact(input_dim=1, widths=(2,)), pde)
    x = torch.linspace(0, 1, 101, dtype=torch.float64).unsqueeze(-1)
    torch.testing.assert_close(pde.apply_hard_bc(x, wrapped(x)), pde.exact(x), atol=1e-12, rtol=0)


def test_wrapper_delegates_to_the_model():
    cfg = load_config("poisson", "q_serial")
    pde = build_pde(cfg.pde)
    from qapinn.models import build

    net = build(cfg.model, input_dim=1, pde=pde)
    wrapped = AffineBCWrapper(net, pde)
    assert wrapped.n_params() == net.n_params()
    assert wrapped.param_groups() == net.param_groups()
    assert wrapped.encoder_A is net.encoder_A
    assert wrapped.circuit is net.circuit
    assert not hasattr(AffineBCWrapper(MLPPINN(input_dim=1, widths=(4,)), pde), "encoder_A")


def test_hard_affine_trains_end_to_end(tmp_path, monkeypatch):
    from qapinn.train import loop

    monkeypatch.setattr(loop, "RESULTS_ROOT", tmp_path)
    cfg = load_config("heat", "c_mlp", overrides={"train.bc_mode": "hard_affine"})
    result = loop.train(cfg, smoke=True)
    assert math.isfinite(result.metrics["rel_l2"])
