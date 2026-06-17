"""Tests for wave frames and polarized electric fields."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ElectricFields import ElectricFields
from field_frame import WaveFrame


class TestElectricFieldsWaveFrame(unittest.TestCase):
    def test_linear_sinusoid_along_local_z(self) -> None:
        efield = ElectricFields.sinusoidal_linear(E0=2.0, omega=1.0, k_magnitude=1.0)
        pos = np.array([0.0, 0.0, math.pi / 4])
        field = efield.at(pos, t=0.0)
        self.assertAlmostEqual(float(field[0]), math.sqrt(2.0), places=10)
        self.assertAlmostEqual(float(field[1]), 0.0, places=10)

    def test_elliptical_defaults_to_linear(self) -> None:
        linear = ElectricFields.sinusoidal_linear(E0=1.5, omega=2.0, k_magnitude=0.5)
        elliptical = ElectricFields.sinusoidal_elliptical(
            E0=1.5,
            omega=2.0,
            k_magnitude=0.5,
            delta=0.0,
        )
        pos = np.array([1.0, 0.0, 0.5])
        t = 0.25
        self.assertTrue(np.allclose(linear.at(pos, t), elliptical.at(pos, t), rtol=1e-12))

    def test_circular_polarization_unit_norm(self) -> None:
        efield = ElectricFields.sinusoidal_elliptical(
            E0=1.0,
            omega=1.0,
            k_magnitude=1.0,
            psi=math.pi / 4,
            delta=math.pi / 2,
        )
        pos = np.zeros(3)
        samples = [efield.at(pos, t=phase / 8.0) for phase in range(8)]
        norms = [float(np.linalg.norm(sample)) for sample in samples]
        self.assertTrue(np.allclose(norms, [1.0] * 8, rtol=1e-10))

    def test_transform_rotates_oblique_incidence(self) -> None:
        local = ElectricFields.sinusoidal_linear(E0=1.0, omega=1.0, k_magnitude=1.0)
        frame = WaveFrame.from_spherical(theta=math.pi / 4, phi=0.0)
        efield = ElectricFields.transform(local, frame)

        pos_lab = np.array([0.0, 0.0, 1.0])
        r_local = frame.position_to_local(pos_lab)
        expected = frame.vector_to_lab(local.at(r_local, t=0.0))
        self.assertTrue(np.allclose(efield.at(pos_lab, t=0.0), expected, rtol=1e-10))

    def test_spherical_incident_matches_direction(self) -> None:
        theta = math.pi / 3
        phi = math.pi / 6
        efield = ElectricFields.plane_wave_incident(
            E0=3.0,
            omega=5.0,
            theta=theta,
            phi=phi,
            k_magnitude=2.0,
        )
        pos = np.zeros(3)
        field = efield.at(pos, t=0.0)
        self.assertAlmostEqual(float(np.linalg.norm(field)), 3.0, places=10)
        self.assertTrue(np.allclose(field / np.linalg.norm(field), efield.frame.e1, rtol=1e-10))

    def test_polar_transform_is_time_independent(self) -> None:
        local = ElectricFields.sinusoidal_elliptical(
            E0=1.0,
            omega=4.0,
            k_magnitude=1.0,
            psi=math.pi / 4,
            delta=math.pi / 2,
        )
        frame = WaveFrame.from_spherical(theta=math.pi / 5, phi=math.pi / 7)
        efield = ElectricFields.transform(local, frame)
        pos = np.array([0.2, -0.3, 0.8])

        lab_t0 = efield.at(pos, t=0.0)
        lab_t1 = efield.at(pos, t=1.25)
        self.assertFalse(np.allclose(lab_t0, lab_t1))

        r_local = frame.position_to_local(pos)
        manual_t0 = frame.vector_to_lab(local.at(r_local, t=0.0))
        manual_t1 = frame.vector_to_lab(local.at(r_local, t=1.25))
        self.assertTrue(np.allclose(lab_t0, manual_t0, rtol=1e-12))
        self.assertTrue(np.allclose(lab_t1, manual_t1, rtol=1e-12))

    def test_sum_with_transformed_field(self) -> None:
        bias = ElectricFields.uniform([0.0, 0.0, 1.0])
        local = ElectricFields.sinusoidal_linear(E0=1.0, omega=1.0, k_magnitude=1.0)
        frame = WaveFrame.from_spherical(theta=math.pi / 4, phi=0.0)
        total = bias + ElectricFields.transform(local, frame)
        pos = np.zeros(3)
        expected = bias.at(pos, 0.0) + ElectricFields.transform(local, frame).at(pos, 0.0)
        self.assertTrue(np.allclose(total.at(pos, 0.0), expected, rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
