"""Tests for explicit PIC particle pushers."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from Pushers import Pushers, boris_push, boris_push_batch, lorentz_gamma_from_velocity


class TestPushers(unittest.TestCase):
    def test_boris_pure_electric_matches_analytic_kick(self) -> None:
        v0 = np.array([0.1, -0.2, 0.0])
        E = np.array([2.0, 0.0, 0.0])
        q, m, dt = 1.0, 1.0, 0.01
        v1 = boris_push(v0, E, np.zeros(3), q, m, dt)
        expected = v0 + (q / m) * E * dt
        self.assertTrue(np.allclose(v1, expected, rtol=1e-12))

    def test_boris_batch_matches_scalar(self) -> None:
        rng = np.random.default_rng(4)
        vel = rng.normal(size=(10, 3))
        efield = rng.normal(size=(10, 3))
        bfield = np.zeros((10, 3))
        q, m, dt = -1.0, 2.0, 1e-12
        batch = boris_push_batch(vel, efield, bfield, q, m, dt)
        dispatcher = Pushers.push_batch("boris", vel, efield, bfield, q, m, dt)
        self.assertTrue(np.allclose(batch, dispatcher, rtol=1e-12))
        for i in range(10):
            scalar = boris_push(vel[i], efield[i], bfield[i], q, m, dt)
            self.assertTrue(np.allclose(batch[i], scalar, rtol=1e-12))

    def test_push_batch_scalar_fallback_matches_scalar(self) -> None:
        rng = np.random.default_rng(7)
        vel = rng.normal(scale=0.05, size=(6, 3))
        efield = rng.normal(scale=0.1, size=(6, 3))
        bfield = np.array([[0.0, 0.0, 0.8]] * 6)
        q, m, dt, c = 1.0, 1.0, 0.005, 1.0
        for kind in ("boris_relativistic", "vay", "higuera_cary"):
            batch = Pushers.push_batch(kind, vel, efield, bfield, q, m, dt, c=c)
            for i in range(6):
                scalar = Pushers.push(kind, vel[i], efield[i], bfield[i], q, m, dt, c=c)
                self.assertTrue(np.allclose(batch[i], scalar, rtol=1e-12, atol=1e-12))

    def test_boris_uniform_b_preserves_speed(self) -> None:
        q, m, dt = 1.0, 1.0, 0.05
        B = np.array([0.0, 0.0, 1.0])
        v0 = np.array([1.0, 0.0, 0.0])
        v1 = Pushers.boris(v0, np.zeros(3), B, q, m, dt)
        self.assertAlmostEqual(float(np.linalg.norm(v1)), float(np.linalg.norm(v0)), places=12)

    def test_relativistic_boris_nonrel_limit(self) -> None:
        v0 = np.array([0.01, 0.0, 0.0])
        E = np.array([0.5, 0.0, 0.0])
        B = np.array([0.0, 0.0, 0.2])
        q, m, dt, c = 1.0, 1.0, 0.01, 1.0
        v_classical = Pushers.boris(v0, E, B, q, m, dt)
        v_relativistic = Pushers.boris_relativistic(v0, E, B, q, m, dt, c=c)
        self.assertTrue(np.allclose(v_classical, v_relativistic, rtol=1e-3, atol=1e-6))

    def test_vay_pure_b_preserves_speed(self) -> None:
        q, m, dt, c = 1.0, 1.0, 0.01, 1.0
        B = np.array([0.0, 0.0, 1.0])
        v0 = np.array([0.3, 0.0, 0.0])
        v1 = Pushers.vay(v0, np.zeros(3), B, q, m, dt, c=c)
        self.assertAlmostEqual(float(np.linalg.norm(v1)), float(np.linalg.norm(v0)), places=10)

    def test_higuera_cary_pure_b_preserves_speed(self) -> None:
        q, m, dt, c = 1.0, 1.0, 0.01, 1.0
        B = np.array([0.0, 0.0, 1.0])
        v0 = np.array([0.0, 0.3, 0.0])
        v1 = Pushers.higuera_cary(v0, np.zeros(3), B, q, m, dt, c=c)
        self.assertAlmostEqual(float(np.linalg.norm(v1)), float(np.linalg.norm(v0)), places=10)

    def test_vay_gyro_orbit_closes(self) -> None:
        q, m, dt, c = 1.0, 1.0, 0.001, 1.0
        B = np.array([0.0, 0.0, 1.0])
        v0 = np.array([0.05, 0.0, 0.0])
        vel = v0.copy()
        steps = int(round(2.0 * math.pi / (q * B[2] / m * dt)))
        for _ in range(steps):
            vel = Pushers.vay(vel, np.zeros(3), B, q, m, dt, c=c)
        self.assertTrue(np.allclose(vel, v0, rtol=0.02, atol=0.001))

    def test_higuera_cary_gyro_orbit_closes(self) -> None:
        q, m, dt, c = 1.0, 1.0, 0.001, 1.0
        B = np.array([0.0, 0.0, 1.0])
        v0 = np.array([0.0, 0.05, 0.0])
        vel = v0.copy()
        steps = int(round(2.0 * math.pi / (q * B[2] / m * dt)))
        for _ in range(steps):
            vel = Pushers.higuera_cary(vel, np.zeros(3), B, q, m, dt, c=c)
        self.assertTrue(np.allclose(vel, v0, rtol=0.02, atol=0.001))

    def test_vay_close_to_relativistic_boris(self) -> None:
        v0 = np.array([0.15, 0.05, 0.0])
        E = np.array([0.2, -0.1, 0.0])
        B = np.array([0.0, 0.0, 0.8])
        q, m, dt, c = 1.0, 1.0, 0.005, 1.0
        v_boris = Pushers.boris_relativistic(v0, E, B, q, m, dt, c=c)
        v_vay = Pushers.vay(v0, E, B, q, m, dt, c=c)
        self.assertTrue(np.allclose(v_vay, v_boris, rtol=1e-5, atol=1e-5))

    def test_dispatcher(self) -> None:
        v0 = np.array([0.1, 0.0, 0.0])
        E = np.zeros(3)
        B = np.zeros(3)
        v = Pushers.push("boris", v0, E, B, 1.0, 1.0, 0.01)
        self.assertTrue(np.allclose(v, v0))

    def test_gamma_helper(self) -> None:
        v = np.array([0.5, 0.0, 0.0])
        gamma = lorentz_gamma_from_velocity(v, c=1.0)
        self.assertAlmostEqual(gamma, 1.0 / math.sqrt(0.75), places=12)


if __name__ == "__main__":
    unittest.main()
