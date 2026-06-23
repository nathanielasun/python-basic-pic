"""Tests for macroparticle initialization and velocity distributions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from particles import (
    M_E,
    MaxwellianDriftVelocityDistribution,
    MaxwellianVelocityDistribution,
    UniformVelocityDistribution,
    initialize_quasi_neutral_plasma,
    macro_charge_from_density,
    macro_count,
    maxwellian_sigma,
    sample_velocities,
)


class TestVelocityDistributions(unittest.TestCase):
    def test_uniform_velocity_bounds(self) -> None:
        rng = np.random.default_rng(0)
        vel = sample_velocities(
            rng,
            UniformVelocityDistribution(v_min=-1.0, v_max=1.0),
            M_E,
            1000,
        )
        self.assertEqual(vel.shape, (1000, 3))
        self.assertTrue(np.all(vel >= -1.0))
        self.assertTrue(np.all(vel < 1.0))

    def test_maxwellian_sigma(self) -> None:
        rng = np.random.default_rng(42)
        mass = M_E
        energy_j = 3.0 * 1.602176634e-19
        count = 50_000
        vel = sample_velocities(
            rng,
            MaxwellianVelocityDistribution(energy_j=energy_j),
            mass,
            count,
        )
        mean_v2 = float(np.mean(np.sum(vel * vel, axis=1)))
        expected_v2 = 2.0 * energy_j / mass
        self.assertLess(abs(mean_v2 - expected_v2) / expected_v2, 0.05)

    def test_maxwellian_drift_mean(self) -> None:
        rng = np.random.default_rng(7)
        drift = 2.0e6
        count = 20_000
        vel = sample_velocities(
            rng,
            MaxwellianDriftVelocityDistribution(
                energy_ev=1.0,
                drift=drift,
                drift_direction=(0.0, 0.0, 1.0),
            ),
            M_E,
            count,
        )
        mean_vz = float(np.mean(vel[:, 2]))
        self.assertLess(abs(mean_vz - drift) / drift, 0.05)
        self.assertLess(abs(float(np.mean(vel[:, 0]))), 1e4)
        self.assertLess(abs(float(np.mean(vel[:, 1]))), 1e4)

    def test_maxwellian_sigma_helper(self) -> None:
        energy_j = 5.0 * 1.602176634e-19
        sigma = maxwellian_sigma(energy_j, M_E)
        self.assertAlmostEqual(sigma, np.sqrt(2.0 * energy_j / (3.0 * M_E)))


class TestInitialization(unittest.TestCase):
    def test_macro_count_and_charge(self) -> None:
        self.assertEqual(macro_count(2, 16), 8192)
        charge = macro_charge_from_density(2e20, 1.0e-18, 8192)
        self.assertAlmostEqual(charge, 2e20 * 1.602176634e-19 * 1.0e-18 / 8192)

    def test_quasi_neutral_plasma(self) -> None:
        rng = np.random.default_rng(1)
        plasma = initialize_quasi_neutral_plasma(
            rng,
            n_macro=128,
            n_density=1e20,
            domain_length=1e-6,
            electron_mass=M_E,
            ion_mass=100.0 * M_E,
            ion_charge_number=1,
            electron_velocity=MaxwellianVelocityDistribution(energy_ev=3.0),
            ion_velocity=MaxwellianVelocityDistribution(energy_ev=1.0),
        )
        self.assertEqual(plasma.electrons.positions.shape, (128, 3))
        self.assertEqual(plasma.ions.velocities.shape, (128, 3))
        self.assertAlmostEqual(plasma.electrons.charge, -plasma.macro_charge)
        self.assertAlmostEqual(plasma.ions.charge, plasma.macro_charge)
        self.assertTrue(np.all(plasma.electrons.positions >= 0.0))
        self.assertTrue(np.all(plasma.electrons.positions < 1e-6))


if __name__ == "__main__":
    unittest.main()
