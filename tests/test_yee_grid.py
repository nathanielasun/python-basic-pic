"""Smoke tests for the dormant Yee EM grid."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grids import YeeGrid


class TestYeeGrid(unittest.TestCase):
    def test_mu0_scales_curl_update(self) -> None:
        dt = 0.01
        ex_samples = []
        for mu0 in (1.0, 4.0):
            grid = YeeGrid(8, 8, 8, mu0=mu0, eps0=1.0, boundary="periodic")
            ng = grid.ng
            y = np.arange(grid.ny + 1, dtype=np.float64) * grid.dy
            bz_profile = np.sin(2.0 * np.pi * y / grid.Ly)
            for i in range(grid.nx + 1):
                for k in range(grid.nz):
                    grid.Bz[ng + i, ng : ng + grid.ny + 1, ng + k] = bz_profile
            grid.Ex.fill(0.0)
            grid.update_e(dt)
            ex_samples.append(float(grid.Ex[ng + 1, ng + 1, ng]))

        self.assertAlmostEqual(ex_samples[0], 4.0 * ex_samples[1], delta=1e-10 * abs(ex_samples[0]))

    def test_vacuum_field_energy_finite(self) -> None:
        grid = YeeGrid(4, 4, 4, boundary="periodic")
        ng = grid.ng
        grid.Ex[ng : ng + grid.nx + 1, ng : ng + grid.ny, ng : ng + grid.nz].fill(1.0)
        grid.By[ng : ng + grid.nx + 1, ng : ng + grid.ny, ng : ng + grid.nz].fill(0.5)
        grid.update_b(0.001)
        grid.update_e(0.001)
        e_norm = float(np.sum(grid.Ex**2) + np.sum(grid.Ey**2) + np.sum(grid.Ez**2))
        b_norm = float(np.sum(grid.Bx**2) + np.sum(grid.By**2) + np.sum(grid.Bz**2))
        self.assertTrue(np.isfinite(e_norm + b_norm))
        self.assertGreater(e_norm + b_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
