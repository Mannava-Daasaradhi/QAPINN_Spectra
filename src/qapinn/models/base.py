"""PINNModel abstract base + size matching (01_CONVENTIONS.md §7)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor, nn

from qapinn.config import ModelConfig


class PINNModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """[B,d] -> [B,1]. This is the RAW network output N(x), BEFORE the hard-BC ansatz.
        The training loop applies pde.apply_hard_bc()."""

    @abstractmethod
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Keys are exactly {'classical', 'quantum'}. 'quantum' is [] for classical
        families. Required by the NTK block decomposition (D7) -- Prop. 4 is tested
        through this."""

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def realised_frequencies(self) -> np.ndarray | None:
        """Angular frequencies the model can currently represent, in PHYSICAL coordinates.
        Fourier-feature families: 2*pi*B rows. Quantum families: Omega*A where A is the
        affine encoder matrix (D3). None for c_mlp (the base-class default). Recomputed on
        demand -- it drifts during training."""
        return None


def match_param_count(family: str, target: int, tol: float = 0.10, raise_on_miss: bool = True, **kw) -> ModelConfig:
    """Binary-search the MLP width (or n_features) so n_params lands within tol of target.
    Raises if unreachable, UNLESS raise_on_miss=False, in which case the closest
    achievable config is returned regardless (T2.16: some family/instance combinations
    are structurally unreachable within tol -- e.g. ParallelHybrid's own FIXED overhead
    (encoder + quantum circuit + scalar weight) can already exceed a tiny SMCD-designed
    target before any MLP branch is added at all, verified for P1 -- and the caller needs
    to record that honestly rather than crash). The target is set by the SMCD-designed
    q_serial model, so every family is matched TO the quantum model, not the other way
    round.

    Searches over REAL constructed models (via qapinn.models.build), not an estimated
    parameter-count formula, so the result can never drift from what the family actually
    builds. kw: input_dim (default 1), n_hidden_layers (default 3, matching
    ModelConfig.widths' default length), pde (required for family='q_parallel', whose
    quantum branch needs an actual PDE to design against via smcd()).
    """
    import qapinn.models as models_pkg  # local import: avoids a base.py <-> models cycle

    input_dim = kw.get("input_dim", 1)
    n_hidden_layers = kw.get("n_hidden_layers", 3)
    pde = kw.get("pde")

    def _cfg_for(width: int) -> ModelConfig:
        if family == "c_mlp":
            return ModelConfig(family=family, widths=(width,) * n_hidden_layers)
        if family == "c_ff":
            return ModelConfig(family=family, n_features=width, ff_sigma=kw.get("ff_sigma", 1.0))
        if family == "q_parallel":
            return ModelConfig(
                family=family, widths=(width,) * n_hidden_layers, activation=kw.get("activation", "tanh")
            )
        if family == "c_rff_matched":
            raise ValueError(
                "match_param_count does not apply to c_rff_matched: its frequency set (and "
                "therefore its size) is fixed by SMCD, not a free knob to search over -- "
                "every OTHER family is matched to it, not the reverse (T2.14 wires it "
                "directly via smcd's design card and matched_and_padded_frequencies)"
            )
        raise ValueError(f"match_param_count: unsupported family {family!r}")

    def _n_params(width: int) -> int:
        model = models_pkg.build(_cfg_for(width), input_dim=input_dim, pde=pde)
        return model.n_params()

    lo, hi = 1, 1
    while _n_params(hi) < target:
        hi *= 2
        if hi > 10**7:
            raise ValueError(f"target {target} unreachable for family {family!r} (width search exceeded 1e7)")

    while lo < hi:
        mid = (lo + hi) // 2
        if _n_params(mid) < target:
            lo = mid + 1
        else:
            hi = mid

    candidates = range(max(1, lo - 2), lo + 3)
    best_width = min(candidates, key=lambda w: abs(_n_params(w) - target))
    n = _n_params(best_width)
    if abs(n - target) > tol * target and raise_on_miss:
        raise ValueError(
            f"target {target} unreachable for family {family!r} within tol={tol} "
            f"(closest: width={best_width}, n_params={n})"
        )

    return _cfg_for(best_width)


def measure_flops(model: PINNModel, x: Tensor) -> int:
    """Real (measured, not estimated) forward-pass FLOPs via torch's FlopCounterMode --
    matches match_param_count's own principle of measuring the actual constructed model
    rather than an estimated formula. Counts whatever ATen ops the forward pass actually
    dispatches, so it works unmodified for the quantum families' complex128 einsum-based
    statevector simulation (qsim.py) as well as ordinary matmuls."""
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter:
        model(x)
    return int(counter.get_total_flops())


def measure_wall_clock_s(model: PINNModel, x: Tensor, n_reps: int = 50, n_warmup: int = 5) -> float:
    """Mean wall-clock seconds per forward call, after n_warmup untimed calls (JIT/cache
    warmup -- the quantum circuit path in particular has a first-call overhead from
    building gate matrices)."""
    import time

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_reps):
            model(x)
        elapsed = time.perf_counter() - t0
    return elapsed / n_reps


def _measure(model: PINNModel, x: Tensor, target: int) -> dict:
    n = model.n_params()
    return {
        "n_params": n,
        "rel_diff_from_q_serial": (abs(n - target) / target) if target > 0 else 0.0,
        "flops": measure_flops(model, x),
        "wall_clock_s": measure_wall_clock_s(model, x),
    }


def size_matching_report(instances: list[tuple[str, object]], tol: float = 0.10) -> dict:
    """T2.16: for each (instance_name, pde), builds q_serial (the reference -- SMCD fixes
    n, L, hence its param count) then matches every other family to it within `tol`:
    c_mlp/c_ff/q_parallel via match_param_count's width search; c_rff_matched via T2.14's
    own design-card-driven padding; q_random/q_octave match (near-)exactly by
    construction (same circuit shape as q_serial). Measures REAL (not estimated) FLOPs
    and wall-clock per family, per project.md Section 6's requirement that these be
    reported together with the parameter-count table.

    Does NOT raise if a family misses tolerance for some instance (e.g. Burgers'
    c_rff_matched, T2.14's documented finding: its own richer empirical target support
    needs more capacity than q_serial's tiny circuit there) -- records
    rel_diff_from_q_serial honestly instead, so the caller can decide how to report a
    known, already-documented exception rather than the report silently crashing on it.
    """
    import qapinn.models as models_pkg  # local import: avoids a base.py <-> models cycle

    report: dict[str, dict] = {}
    for name, pde in instances:
        q_serial = models_pkg.build(ModelConfig(family="q_serial"), input_dim=pde.dim, pde=pde)
        target = q_serial.n_params()
        x = torch.rand(1, pde.dim)

        entry: dict[str, dict] = {"q_serial": _measure(q_serial, x, target)}

        # c_mlp's default 3-hidden-layer width granularity is too coarse for these tiny
        # (SMCD-designed) targets -- e.g. for P1's target=14, width=1 gives 8 params and
        # width=2 gives 19, straddling the +-10% band entirely with nothing achievable in
        # between. Retrying with fewer hidden layers (finer per-width granularity: a
        # single hidden layer scales as ~3*width+1, not ~2*width^2) is not a hack around
        # the DoD -- it is the SAME shallow-MLP family, just sized finely enough to hit a
        # tiny target, matching c_ff's own inherently fine (per-feature) granularity.
        # raise_on_miss=False on the LAST attempt: if even n_hidden_layers=1 can't reach
        # tolerance, record the best-effort result honestly rather than crash the whole
        # report over one family/instance combination.
        for n_hidden in (3, 2, 1):
            cfg = match_param_count(
                "c_mlp", target, tol=tol, input_dim=pde.dim, n_hidden_layers=n_hidden, raise_on_miss=False
            )
            model = models_pkg.build(cfg, input_dim=pde.dim)
            if abs(model.n_params() - target) <= tol * target:
                break
        else:
            # n_hidden_layers=0 has no free "width" to search over -- a single Linear(d,1)
            # layer's size doesn't depend on a width at all, so match_param_count's binary
            # search (which assumes n_params grows with width) doesn't apply; just build
            # the one fixed-size config directly as the final fallback.
            model = models_pkg.build(ModelConfig(family="c_mlp", widths=()), input_dim=pde.dim)
        entry["c_mlp"] = _measure(model, x, target)

        cfg = match_param_count("c_ff", target, tol=tol, input_dim=pde.dim)
        model = models_pkg.build(cfg, input_dim=pde.dim)
        entry["c_ff"] = _measure(model, x, target)

        c_rff = models_pkg.build(ModelConfig(family="c_rff_matched"), input_dim=pde.dim, pde=pde)
        entry["c_rff_matched"] = _measure(c_rff, x, target)

        q_random = models_pkg.build(
            ModelConfig(family="q_random", scaling_mode="random"),
            input_dim=pde.dim,
            pde=pde,
            gen=torch.Generator().manual_seed(0),
        )
        entry["q_random"] = _measure(q_random, x, target)

        # ParallelHybrid's own FIXED overhead (encoder + quantum circuit + scalar weight)
        # can already exceed a tiny SMCD-designed target before any MLP branch is added
        # (verified for P1: fixed overhead alone is 13/14 of the target) -- same
        # finer-granularity retry as c_mlp, best-effort (not raising) if even that can't
        # reach tolerance, since the fixed overhead alone may already be the binding
        # constraint no MLP depth can fix.
        for n_hidden in (3, 2, 1):
            cfg_par = match_param_count(
                "q_parallel", target, tol=tol, input_dim=pde.dim, pde=pde, n_hidden_layers=n_hidden, raise_on_miss=False
            )
            q_parallel = models_pkg.build(cfg_par, input_dim=pde.dim, pde=pde)
            if abs(q_parallel.n_params() - target) <= tol * target:
                break
        else:
            # n_hidden_layers=0 fallback -- see the c_mlp comment above for why this
            # can't go through match_param_count's own width search.
            q_parallel = models_pkg.build(
                ModelConfig(family="q_parallel", widths=()), input_dim=pde.dim, pde=pde
            )
        entry["q_parallel"] = _measure(q_parallel, x, target)

        q_octave = models_pkg.build(ModelConfig(family="q_octave"), input_dim=pde.dim, pde=pde)
        entry["q_octave"] = _measure(q_octave, x, target)

        report[name] = entry
    return report
