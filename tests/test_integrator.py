"""Leapfrog integrator tests for the electrostatic PIC driver."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_SRC = Path(__file__).resolve().parents[1] / "src"
for path in (_EXAMPLES, _SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import E_CHARGE, M_E
from fields import ElectricFields


def _leapfrog_uniform(pos0: float, vel0: np.ndarray, e0: float, dt: float, n_steps: int) -> float:
    """Reference leapfrog with external E at position time t = (step-1)*dt."""
    pos = np.array([[pos0, pos0, pos0]], dtype=np.float64)
    vel = vel0.copy()
    q_over_m = -E_CHARGE / M_E
    efield = ElectricFields.uniform([e0, 0.0, 0.0])

    e_ext = efield.at_batch(pos, 0.0)
    vel = vel + q_over_m * (-0.5 * dt) * e_ext

    domain = 1.0e-5
    for step in range(1, n_steps + 1):
        t_kick = (step - 1) * dt
        e_total = efield.at_batch(pos, t_kick)
        vel = vel + q_over_m * dt * e_total
        pos = pos + vel * dt
        pos %= domain

    return float(pos[0, 0])


def _leapfrog_harmonic_cos(
    pos0: float,
    vel0: np.ndarray,
    e0: float,
    omega: float,
    dt: float,
    n_steps: int,
    *,
    t_offset_steps: float,
) -> float:
    """
    Leapfrog under E_x = E0 cos(omega t).

    Kick time t = (step - 1 + t_offset_steps) * dt.
    t_offset_steps=0 matches production; t_offset_steps=-0.5 is the old mis-centered formula.
    """
    pos = np.array([[pos0, pos0, pos0]], dtype=np.float64)
    vel = vel0.copy()
    q_over_m = -E_CHARGE / M_E

    e_at_zero = np.array([e0, 0.0, 0.0])
    vel = vel + q_over_m * (-0.5 * dt) * e_at_zero

    domain = 1.0e-5
    for step in range(1, n_steps + 1):
        t_kick = (step - 1 + t_offset_steps) * dt
        e_total = np.array([e0 * np.cos(omega * t_kick), 0.0, 0.0])
        vel = vel + q_over_m * dt * e_total
        pos = pos + vel * dt
        pos %= domain

    return float(pos[0, 0])


class TestLeapfrogIntegrator(unittest.TestCase):
    def test_uniform_e_displacement(self) -> None:
        """Single electron under constant E: x ≈ ½ a t² after leapfrog init + N steps."""
        e0 = 1.0e6
        dt = 1.0e-15
        n_steps = 200
        domain = 1.0e-5
        pos0 = domain / 2

        x_sim = _leapfrog_uniform(pos0, np.zeros((1, 3)), e0, dt, n_steps)

        t_total = n_steps * dt
        q_over_m = -E_CHARGE / M_E
        x_analytic = pos0 + 0.5 * (q_over_m * e0) * t_total**2
        self.assertAlmostEqual(x_sim, x_analytic, delta=abs(x_analytic - pos0) * 0.05)

    def test_harmonic_e_displacement(self) -> None:
        """E(t) = E0 cos(omega t): compare displacement to (qE0/m omega^2)(1 - cos omega t)."""
        e0 = 1.0e6
        dt = 1.0e-15
        omega = 0.05 / dt
        n_steps = 200
        domain = 1.0e-5
        pos0 = domain / 2

        x_sim = _leapfrog_harmonic_cos(
            pos0, np.zeros((1, 3)), e0, omega, dt, n_steps, t_offset_steps=0.0
        )

        t_total = n_steps * dt
        q_over_m = -E_CHARGE / M_E
        x_analytic = pos0 + (q_over_m * e0 / omega**2) * (1.0 - np.cos(omega * t_total))
        disp_sim = x_sim - pos0
        disp_analytic = x_analytic - pos0
        rel_err = abs(disp_sim - disp_analytic) / max(abs(disp_analytic), 1e-30)
        self.assertLess(rel_err, 0.10)

    def test_wrong_half_step_timing_deviates(self) -> None:
        """Old t=(step-0.5)dt timing should be less accurate than t=(step-1)dt for harmonic E."""
        e0 = 1.0e6
        dt = 1.0e-15
        omega = 0.05 / dt
        n_steps = 200
        domain = 1.0e-5
        pos0 = domain / 2
        vel0 = np.zeros((1, 3))

        x_correct = _leapfrog_harmonic_cos(
            pos0, vel0, e0, omega, dt, n_steps, t_offset_steps=0.0
        )
        x_wrong = _leapfrog_harmonic_cos(
            pos0, vel0, e0, omega, dt, n_steps, t_offset_steps=-0.5
        )

        t_total = n_steps * dt
        q_over_m = -E_CHARGE / M_E
        x_analytic = pos0 + (q_over_m * e0 / omega**2) * (1.0 - np.cos(omega * t_total))

        err_correct = abs(x_correct - x_analytic)
        err_wrong = abs(x_wrong - x_analytic)
        self.assertLess(err_correct, err_wrong)
        self.assertGreater(err_wrong, err_correct * 10.0)


if __name__ == "__main__":
    unittest.main()
