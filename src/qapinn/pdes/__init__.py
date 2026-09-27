"""PDE registry / factory (01_CONVENTIONS.md §5)."""
from __future__ import annotations

from qapinn.config import PDEConfig
from qapinn.pdes.base import PDE, Domain
from qapinn.pdes.burgers import Burgers
from qapinn.pdes.groundwater import Groundwater
from qapinn.pdes.heat import Heat
from qapinn.pdes.helmholtz import Helmholtz
from qapinn.pdes.poisson import Poisson

_REGISTRY: dict[str, type[PDE]] = {
    "poisson": Poisson,
    "heat": Heat,
    "burgers": Burgers,
    "helmholtz": Helmholtz,
    "groundwater": Groundwater,
}


def build(cfg: PDEConfig) -> PDE:
    """Instantiate the PDE named in cfg.name, passing cfg.params as keyword arguments."""
    try:
        cls = _REGISTRY[cfg.name]
    except KeyError as e:
        raise ValueError(
            f"unknown PDE name {cfg.name!r}; registered: {sorted(_REGISTRY)}"
        ) from e
    return cls(**cfg.params)


__all__ = ["PDE", "Burgers", "Domain", "Groundwater", "Heat", "Helmholtz", "Poisson", "build"]
