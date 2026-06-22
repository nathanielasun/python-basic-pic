"""Macroparticle layout and quasi-neutral plasma initialization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import E_CHARGE
from .distributions import VelocityDistribution, sample_velocities


@dataclass(frozen=True)
class MacroparticlePopulation:
    positions: np.ndarray
    velocities: np.ndarray
    charge: float


@dataclass(frozen=True)
class QuasiNeutralPlasma:
    electrons: MacroparticlePopulation
    ions: MacroparticlePopulation
    n_macro: int
    macro_charge: float


def macro_count(macros_per_cell: int, n_cells: int) -> int:
    return macros_per_cell * n_cells**3


def macro_charge_from_density(n_density: float, volume: float, n_macro: int) -> float:
    return n_density * E_CHARGE * volume / n_macro


def uniform_positions(
    rng: np.random.Generator,
    domain_length: float,
    count: int,
) -> np.ndarray:
    return rng.uniform(0.0, domain_length, size=(count, 3))


def initialize_species(
    rng: np.random.Generator,
    *,
    count: int,
    mass: float,
    charge: float,
    domain_length: float,
    velocity_distribution: VelocityDistribution,
) -> MacroparticlePopulation:
    return MacroparticlePopulation(
        positions=uniform_positions(rng, domain_length, count),
        velocities=sample_velocities(rng, velocity_distribution, mass, count),
        charge=charge,
    )


def initialize_quasi_neutral_plasma(
    rng: np.random.Generator,
    *,
    n_macro: int,
    n_density: float,
    domain_length: float,
    electron_mass: float,
    ion_mass: float,
    ion_charge_number: int,
    electron_velocity: VelocityDistribution,
    ion_velocity: VelocityDistribution,
) -> QuasiNeutralPlasma:
    volume = domain_length**3
    macro_charge = macro_charge_from_density(n_density, volume, n_macro)
    q_e = -macro_charge
    q_i = ion_charge_number * macro_charge

    electrons = initialize_species(
        rng,
        count=n_macro,
        mass=electron_mass,
        charge=q_e,
        domain_length=domain_length,
        velocity_distribution=electron_velocity,
    )
    ions = initialize_species(
        rng,
        count=n_macro,
        mass=ion_mass,
        charge=q_i,
        domain_length=domain_length,
        velocity_distribution=ion_velocity,
    )
    return QuasiNeutralPlasma(
        electrons=electrons,
        ions=ions,
        n_macro=n_macro,
        macro_charge=macro_charge,
    )
