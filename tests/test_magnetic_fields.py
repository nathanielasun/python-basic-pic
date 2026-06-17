"""Tests for wave frames and polarized magnetic fields."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from MagneticFields import MagneticFields
from field_frame import WaveFrame

# CSV fixtures under data/ — version-controlled test inputs; do not remove.
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MAGNETIC_UNIFORM_Z_CSV = _DATA_DIR / "magnetic_uniform_z.csv"


class TestMagneticFieldsCsvIO(unittest.TestCase):
    """Field I/O tests using ``data/magnetic_uniform_z.csv`` (test fixture; keep file)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.csv_path = MAGNETIC_UNIFORM_Z_CSV
        if not cls.csv_path.is_file():
            raise unittest.SkipTest(f"test fixture missing: {cls.csv_path}")

    def test_from_csv_loads_structured_uniform_bz(self) -> None:
        bfield = MagneticFields.from_csv(self.csv_path)
        self.assertTrue(bfield.dataset.is_structured)
        self.assertFalse(bfield.dataset.is_time_dependent)
        self.assertEqual(bfield.dataset.metadata.get("test_fixture"), "true")
        self.assertEqual(bfield.dataset.metadata.get("component"), "magnetic")

        field = bfield.at(np.array([0.0, 0.0, 0.0]), t=0.0)
        self.assertAlmostEqual(float(field[0]), 0.0)
        self.assertAlmostEqual(float(field[1]), 0.0)
        self.assertAlmostEqual(float(field[2]), 0.5)

        corner = bfield.at(np.array([1.0, 1.0, 1.0]), t=0.0)
        self.assertTrue(np.allclose(corner, [0.0, 0.0, 0.5], rtol=1e-12))

    def test_csv_uniform_field_at_interior_point(self) -> None:
        bfield = MagneticFields.from_csv(self.csv_path)
        field = bfield.at(np.array([0.5, 0.5, 0.5]), t=0.0)
        self.assertTrue(np.allclose(field, [0.0, 0.0, 0.5], rtol=1e-10))

    def test_csv_on_grid_matches_samples(self) -> None:
        bfield = MagneticFields.from_csv(self.csv_path)
        coords = np.array([0.0, 1.0])
        bx, by, bz = bfield.on_grid(coords, coords, coords, t=0.0)
        self.assertTrue(np.allclose(bx, 0.0))
        self.assertTrue(np.allclose(by, 0.0))
        self.assertTrue(np.allclose(bz, 0.5))

    def test_read_from_csv_replaces_field(self) -> None:
        bfield = MagneticFields.zero()
        bfield.read_from_csv(self.csv_path)
        field = bfield.at(np.array([1.0, 0.0, 1.0]), t=0.0)
        self.assertTrue(np.allclose(field, [0.0, 0.0, 0.5], rtol=1e-12))


class TestMagneticFieldsWaveFrame(unittest.TestCase):
    def test_linear_sinusoid_along_local_z(self) -> None:
        bfield = MagneticFields.sinusoidal_linear(B0=2.0, omega=1.0, k_magnitude=1.0)
        pos = np.array([0.0, 0.0, math.pi / 4])
        field = bfield.at(pos, t=0.0)
        self.assertAlmostEqual(float(field[0]), math.sqrt(2.0), places=10)
        self.assertAlmostEqual(float(field[1]), 0.0, places=10)

    def test_elliptical_defaults_to_linear(self) -> None:
        linear = MagneticFields.sinusoidal_linear(B0=1.5, omega=2.0, k_magnitude=0.5)
        elliptical = MagneticFields.sinusoidal_elliptical(
            B0=1.5,
            omega=2.0,
            k_magnitude=0.5,
            delta=0.0,
        )
        pos = np.array([1.0, 0.0, 0.5])
        t = 0.25
        self.assertTrue(np.allclose(linear.at(pos, t), elliptical.at(pos, t), rtol=1e-12))

    def test_circular_polarization_unit_norm(self) -> None:
        bfield = MagneticFields.sinusoidal_elliptical(
            B0=1.0,
            omega=1.0,
            k_magnitude=1.0,
            psi=math.pi / 4,
            delta=math.pi / 2,
        )
        pos = np.zeros(3)
        samples = [bfield.at(pos, t=phase / 8.0) for phase in range(8)]
        norms = [float(np.linalg.norm(sample)) for sample in samples]
        self.assertTrue(np.allclose(norms, [1.0] * 8, rtol=1e-10))

    def test_transform_rotates_oblique_incidence(self) -> None:
        local = MagneticFields.sinusoidal_linear(B0=1.0, omega=1.0, k_magnitude=1.0)
        frame = WaveFrame.from_spherical(theta=math.pi / 4, phi=0.0)
        bfield = MagneticFields.transform(local, frame)

        pos_lab = np.array([0.0, 0.0, 1.0])
        r_local = frame.position_to_local(pos_lab)
        expected = frame.vector_to_lab(local.at(r_local, t=0.0))
        self.assertTrue(np.allclose(bfield.at(pos_lab, t=0.0), expected, rtol=1e-10))

    def test_spherical_incident_matches_direction(self) -> None:
        theta = math.pi / 3
        phi = math.pi / 6
        bfield = MagneticFields.plane_wave_incident(
            B0=3.0,
            omega=5.0,
            theta=theta,
            phi=phi,
            k_magnitude=2.0,
        )
        pos = np.zeros(3)
        field = bfield.at(pos, t=0.0)
        self.assertAlmostEqual(float(np.linalg.norm(field)), 3.0, places=10)
        self.assertTrue(np.allclose(field / np.linalg.norm(field), bfield.frame.e1, rtol=1e-10))

    def test_polar_transform_is_time_independent(self) -> None:
        local = MagneticFields.sinusoidal_elliptical(
            B0=1.0,
            omega=4.0,
            k_magnitude=1.0,
            psi=math.pi / 4,
            delta=math.pi / 2,
        )
        frame = WaveFrame.from_spherical(theta=math.pi / 5, phi=math.pi / 7)
        bfield = MagneticFields.transform(local, frame)
        pos = np.array([0.2, -0.3, 0.8])

        lab_t0 = bfield.at(pos, t=0.0)
        lab_t1 = bfield.at(pos, t=1.25)
        self.assertFalse(np.allclose(lab_t0, lab_t1))

        r_local = frame.position_to_local(pos)
        manual_t0 = frame.vector_to_lab(local.at(r_local, t=0.0))
        manual_t1 = frame.vector_to_lab(local.at(r_local, t=1.25))
        self.assertTrue(np.allclose(lab_t0, manual_t0, rtol=1e-12))
        self.assertTrue(np.allclose(lab_t1, manual_t1, rtol=1e-12))

    def test_mirror_local_and_transform(self) -> None:
        local = MagneticFields.mirror_local(B0=0.5, scale_length=2.0)
        pos_local = np.array([0.0, 0.0, 1.0])
        self.assertAlmostEqual(float(local.at(pos_local, 0.0)[2]), 0.5 * (1.0 + 0.25), places=10)

        frame = WaveFrame.from_spherical(theta=math.pi / 4, phi=0.0)
        bfield = MagneticFields.transform(local, frame)
        pos_lab = np.array([0.0, 0.0, 1.0])
        r_local = frame.position_to_local(pos_lab)
        expected = frame.vector_to_lab(local.at(r_local, 0.0))
        self.assertTrue(np.allclose(bfield.at(pos_lab, 0.0), expected, rtol=1e-10))

    def test_sum_with_transformed_field(self) -> None:
        bias = MagneticFields.uniform([0.0, 0.0, 1.0])
        local = MagneticFields.sinusoidal_linear(B0=1.0, omega=1.0, k_magnitude=1.0)
        frame = WaveFrame.from_spherical(theta=math.pi / 4, phi=0.0)
        total = bias + MagneticFields.transform(local, frame)
        pos = np.zeros(3)
        expected = bias.at(pos, 0.0) + MagneticFields.transform(local, frame).at(pos, 0.0)
        self.assertTrue(np.allclose(total.at(pos, 0.0), expected, rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
