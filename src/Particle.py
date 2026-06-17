"""
Author: Nathaniel Sun
Date: 2026-06-16
Description:
    Particle class for 3D PIC simulation.
    Tracks phase space (position, velocity), charge/mass, species kind, and
    internal excitation/ionization state for atomic collision models.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class ExcitationClass(StrEnum):
    """Broad classification of a particle's internal electronic state."""

    GROUND = "ground"
    EXCITED = "excited"
    METASTABLE = "metastable"
    IONIZED = "ionized"


@dataclass(frozen=True, slots=True)
class ExcitationState:
    """
    Internal excitation specification for a macro-particle.

    Attributes:
        ionization: Ionization stage (0 = neutral, 1 = singly ionized, ...).
        level: Excitation index within the current ion (0 = ground).
        internal_energy: Internal energy above the ground level of this ion (eV).
        metastable: Whether the state is a long-lived excited (metastable) level.
    """

    ionization: int = 0
    level: int = 0
    internal_energy: float = 0.0
    metastable: bool = False

    def __post_init__(self) -> None:
        if self.ionization < 0:
            raise ValueError("ionization must be non-negative")
        if self.level < 0:
            raise ValueError("level must be non-negative")
        if self.internal_energy < 0.0:
            raise ValueError("internal_energy must be non-negative")

    @classmethod
    def ground(cls) -> ExcitationState:
        return cls()

    @classmethod
    def ionized(cls, ionization: int = 1, internal_energy: float = 0.0) -> ExcitationState:
        if ionization < 1:
            raise ValueError("ionization must be at least 1 for ionized states")
        return cls(ionization=ionization, level=0, internal_energy=internal_energy, metastable=False)

    @classmethod
    def excited(
        cls,
        level: int,
        internal_energy: float,
        *,
        ionization: int = 0,
        metastable: bool = False,
    ) -> ExcitationState:
        if level < 1:
            raise ValueError("level must be at least 1 for excited states")
        return cls(
            ionization=ionization,
            level=level,
            internal_energy=internal_energy,
            metastable=metastable,
        )

    @property
    def excitation_class(self) -> ExcitationClass:
        if self.ionization > 0 and self.level == 0 and self.internal_energy == 0.0:
            return ExcitationClass.IONIZED
        if self.level == 0 and self.internal_energy == 0.0:
            return ExcitationClass.GROUND
        if self.metastable:
            return ExcitationClass.METASTABLE
        return ExcitationClass.EXCITED

    @property
    def is_ground(self) -> bool:
        return self.excitation_class == ExcitationClass.GROUND

    @property
    def is_ionized(self) -> bool:
        return self.ionization > 0

    @property
    def is_excited(self) -> bool:
        return self.level > 0 or self.internal_energy > 0.0

    def with_updates(self, **kwargs: object) -> ExcitationState:
        return replace(self, **kwargs)


def boris_push(
    vel: NDArray[np.floating],
    E: NDArray[np.floating],
    B: NDArray[np.floating],
    q: float,
    m: float,
    dt: float,
) -> NDArray[np.float64]:
    """
    Boris pusher for the Lorentz force: dv/dt = (q/m)(E + v x B).

    With B = 0 this reduces to a centered electric kick (leapfrog-compatible).
    """
    qmdt = (q / m) * dt
    E_arr = np.asarray(E, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)

    v_minus = np.asarray(vel, dtype=np.float64) + qmdt * E_arr / 2.0

    t = qmdt * B_arr / 2.0
    s = 2.0 * t / (1.0 + float(np.dot(t, t)))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    return v_plus + qmdt * E_arr / 2.0


class Particle:
    def __init__(
        self,
        kind: str,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        q: float,
        m: float,
        excitation: ExcitationState | None = None,
    ) -> None:
        self.kind = kind
        self.pos = np.array([x, y, z], dtype=np.float64)
        self.vel = np.array([vx, vy, vz], dtype=np.float64)
        self.q = q
        self.m = m
        self.excitation = excitation if excitation is not None else ExcitationState.ground()

    def move(self, dt: float) -> None:
        self.pos += self.vel * dt

    def boris_push(
        self,
        E: NDArray[np.floating],
        B: NDArray[np.floating],
        dt: float,
    ) -> None:
        self.vel = boris_push(self.vel, E, B, self.q, self.m, dt)

    def get_position(self) -> NDArray[np.float64]:
        return self.pos

    def get_velocity(self) -> NDArray[np.float64]:
        return self.vel

    def get_charge(self) -> float:
        return self.q

    def get_mass(self) -> float:
        return self.m

    def get_kind(self) -> str:
        return self.kind

    def get_excitation(self) -> ExcitationState:
        return self.excitation

    def get_excitation_class(self) -> ExcitationClass:
        return self.excitation.excitation_class

    def get_ionization(self) -> int:
        return self.excitation.ionization

    def get_excitation_level(self) -> int:
        return self.excitation.level

    def get_internal_energy(self) -> float:
        return self.excitation.internal_energy

    def is_ground_state(self) -> bool:
        return self.excitation.is_ground

    def is_ionized(self) -> bool:
        return self.excitation.is_ionized

    def is_excited(self) -> bool:
        return self.excitation.is_excited

    def is_metastable(self) -> bool:
        return self.excitation.metastable

    def set_position(self, pos: NDArray[np.floating]) -> None:
        self.pos = np.asarray(pos, dtype=np.float64)

    def set_velocity(self, vel: NDArray[np.floating]) -> None:
        self.vel = np.asarray(vel, dtype=np.float64)

    def set_charge(self, q: float) -> None:
        self.q = q

    def set_mass(self, m: float) -> None:
        self.m = m

    def set_kind(self, kind: str) -> None:
        self.kind = kind

    def set_excitation(self, excitation: ExcitationState) -> None:
        self.excitation = excitation

    def reset_to_ground(self) -> None:
        self.excitation = ExcitationState.ground()

    def excite(
        self,
        level: int,
        internal_energy: float,
        *,
        metastable: bool = False,
    ) -> None:
        """Promote to an excited state within the current ionization stage."""
        self.excitation = ExcitationState.excited(
            level,
            internal_energy,
            ionization=self.excitation.ionization,
            metastable=metastable,
        )

    def ionize(
        self,
        ionization: int | None = None,
        *,
        internal_energy: float = 0.0,
        q: float | None = None,
        m: float | None = None,
    ) -> None:
        """
        Advance ionization stage and reset electronic level to ground.

        If ionization is omitted, increment by one. Optionally update q and m
        when the particle becomes a new ion species.
        """
        stage = self.excitation.ionization + 1 if ionization is None else ionization
        self.excitation = ExcitationState.ionized(stage, internal_energy=internal_energy)
        if q is not None:
            self.q = q
        if m is not None:
            self.m = m

    def deexcite(self) -> None:
        """Return to the ground state of the current ionization stage."""
        self.excitation = ExcitationState(ionization=self.excitation.ionization)

    def recombine(
        self,
        ionization: int | None = None,
        *,
        q: float | None = None,
        m: float | None = None,
    ) -> None:
        """Reduce ionization stage and reset to ground."""
        if self.excitation.ionization == 0:
            return
        stage = self.excitation.ionization - 1 if ionization is None else ionization
        if stage < 0:
            raise ValueError("recombined ionization stage must be non-negative")
        self.excitation = ExcitationState(ionization=stage)
        if q is not None:
            self.q = q
        if m is not None:
            self.m = m
