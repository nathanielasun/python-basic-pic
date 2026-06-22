"""Initial velocity distributions for macroparticle populations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from .constants import EV_TO_J


def _as_vector3(value: float | tuple[float, float, float]) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full(3, float(value), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _resolve_energy_j(
    rng: np.random.Generator,
    *,
    energy_j: float | None,
    energy_ev: float | None,
    energy_ev_range: tuple[float, float] | None,
) -> float:
    if energy_j is not None:
        return float(energy_j)
    if energy_ev is not None:
        return float(energy_ev) * EV_TO_J
    if energy_ev_range is not None:
        lo, hi = energy_ev_range
        return float(rng.uniform(lo, hi)) * EV_TO_J
    raise ValueError("one of energy_j, energy_ev, or energy_ev_range is required")


@dataclass(frozen=True)
class UniformVelocityDistribution:
    """Independent uniform components in ``[v_min, v_max)``."""

    v_min: float | tuple[float, float, float] = 0.0
    v_max: float | tuple[float, float, float] = 0.0


@dataclass(frozen=True)
class MaxwellianVelocityDistribution:
    """Isotropic Maxwellian with mean kinetic energy ``E = 3/2 m sigma^2``."""

    energy_j: float | None = None
    energy_ev: float | None = None
    energy_ev_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class MaxwellianDriftVelocityDistribution:
    """Maxwellian thermal spread plus a bulk drift along ``drift_direction``."""

    energy_j: float | None = None
    energy_ev: float | None = None
    energy_ev_range: tuple[float, float] | None = None
    drift: float = 0.0
    drift_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)


VelocityDistribution = Union[
    UniformVelocityDistribution,
    MaxwellianVelocityDistribution,
    MaxwellianDriftVelocityDistribution,
]


def maxwellian_sigma(energy_j: float, mass: float) -> float:
    """Standard deviation per axis for mean kinetic energy ``E = 3/2 m sigma^2``."""
    return float(np.sqrt(2.0 * energy_j / (3.0 * mass)))


def sample_velocities(
    rng: np.random.Generator,
    distribution: VelocityDistribution,
    mass: float,
    count: int,
) -> np.ndarray:
    """Draw ``count`` initial velocity vectors for particles of ``mass``."""
    if isinstance(distribution, UniformVelocityDistribution):
        v_min = _as_vector3(distribution.v_min)
        v_max = _as_vector3(distribution.v_max)
        return rng.uniform(v_min, v_max, size=(count, 3))

    if isinstance(
        distribution,
        (MaxwellianVelocityDistribution, MaxwellianDriftVelocityDistribution),
    ):
        energy_j = _resolve_energy_j(
            rng,
            energy_j=distribution.energy_j,
            energy_ev=distribution.energy_ev,
            energy_ev_range=distribution.energy_ev_range,
        )
        sigma = maxwellian_sigma(energy_j, mass)
        vel = rng.normal(0.0, sigma, size=(count, 3))
        if isinstance(distribution, MaxwellianDriftVelocityDistribution):
            direction = _as_vector3(distribution.drift_direction)
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                raise ValueError("drift_direction must be non-zero")
            vel += (distribution.drift / norm) * direction
        return vel

    raise TypeError(f"unsupported velocity distribution: {type(distribution)!r}")
