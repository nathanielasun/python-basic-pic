"""Ion species metadata used when building quasi-neutral plasmas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IonSpecies:
    name: str
    mass: float
    charge_number: int = 1
