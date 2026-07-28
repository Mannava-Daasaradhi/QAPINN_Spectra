"""T0.5 DoD: same seed -> bit-identical sampling through the returned generator."""
from __future__ import annotations

import torch

from qapinn.seeding import resolve_device, set_global_seed


def test_set_global_seed_reproduces_bit_identical_tensors():
    gen1 = set_global_seed(0)
    a = torch.randn(1000, generator=gen1)

    gen2 = set_global_seed(0)
    b = torch.randn(1000, generator=gen2)

    assert torch.equal(a, b)


def test_set_global_seed_different_seeds_differ():
    gen1 = set_global_seed(0)
    a = torch.randn(1000, generator=gen1)

    gen2 = set_global_seed(1)
    b = torch.randn(1000, generator=gen2)

    assert not torch.equal(a, b)


def test_resolve_device_auto_matches_cuda_availability():
    dev = resolve_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.type == expected


def test_resolve_device_cpu_forced():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_invalid_raises():
    import pytest

    with pytest.raises(ValueError):
        resolve_device("tpu")
