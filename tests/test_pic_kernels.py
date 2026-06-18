"""Tests for Numba PIC kernels vs NumPy reference implementations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ElectrostaticGrid import ElectrostaticGrid

try:
    from pic_kernels import (
        HAS_NUMBA,
        deposit_cic_periodic,
        electric_kick_b0,
        gather_e_cic_periodic,
        warmup_kernels,
        wrap_positions_periodic,
    )
except ImportError:
    HAS_NUMBA = False


def _make_grid(n_cells: int = 8) -> ElectrostaticGrid:
    return ElectrostaticGrid(
        n_cells,
        n_cells,
        n_cells,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        boundary="periodic",
        particle_backend="numpy",
    )


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestPicKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warmup_kernels()

    def test_deposit_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        grid = _make_grid()
        n_particles = 200
        positions = rng.uniform(0.0, grid.Lx, size=(n_particles, 3))
        charges = rng.uniform(-1.0, 1.0, size=n_particles)
        cell_volume = grid.dx * grid.dy * grid.dz
        values = charges / cell_volume

        grid_np = _make_grid()
        grid_nb = _make_grid()
        pos = grid.position_in_domain_batch(positions)

        grid_np._deposit_scalar_batch(grid_np.rho, pos, values, (0.0, 0.0, 0.0))
        deposit_cic_periodic(
            grid_nb.rho,
            pos,
            values,
            grid_nb.dx,
            grid_nb.dy,
            grid_nb.dz,
            grid_nb.nx,
            grid_nb.ny,
            grid_nb.nz,
            grid_nb.ng,
        )

        np.testing.assert_allclose(grid_nb.rho, grid_np.rho, rtol=1e-10, atol=1e-12)

    def test_gather_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        grid = _make_grid()
        grid.Ex[...] = rng.normal(size=grid.Ex.shape)
        grid.Ey[...] = rng.normal(size=grid.Ey.shape)
        grid.Ez[...] = rng.normal(size=grid.Ez.shape)

        n_particles = 150
        positions = rng.uniform(0.0, grid.Lx, size=(n_particles, 3))
        pos = grid.position_in_domain_batch(positions)

        ex = grid._gather_scalar_batch(grid.Ex, pos, (0.0, 0.0, 0.0))
        ey = grid._gather_scalar_batch(grid.Ey, pos, (0.0, 0.0, 0.0))
        ez = grid._gather_scalar_batch(grid.Ez, pos, (0.0, 0.0, 0.0))
        e_np = np.column_stack([ex, ey, ez])

        e_nb = gather_e_cic_periodic(
            grid.Ex,
            grid.Ey,
            grid.Ez,
            pos,
            grid.dx,
            grid.dy,
            grid.dz,
            grid.nx,
            grid.ny,
            grid.nz,
            grid.ng,
        )

        np.testing.assert_allclose(e_nb, e_np, rtol=1e-10, atol=1e-12)

    def test_electric_kick_matches_numpy(self) -> None:
        rng = np.random.default_rng(2)
        n = 100
        vel_ref = rng.normal(size=(n, 3))
        vel_nb = vel_ref.copy()
        efield = rng.normal(size=(n, 3))
        q_over_m = -1.76e11
        dt = 1e-14

        vel_expected = vel_ref + q_over_m * dt * efield
        electric_kick_b0(vel_nb, efield, q_over_m, dt)

        np.testing.assert_allclose(vel_nb, vel_expected, rtol=1e-12, atol=1e-15)

    def test_wrap_periodic(self) -> None:
        pos = np.array(
            [
                [-0.5, 1.5, 2.5],
                [3.0, 4.0, 5.0],
            ],
            dtype=np.float64,
        )
        expected = pos.copy()
        expected[:, 0] %= 3.0
        expected[:, 1] %= 4.0
        expected[:, 2] %= 5.0

        wrap_positions_periodic(pos, 3.0, 4.0, 5.0)
        np.testing.assert_allclose(pos, expected)

    def test_maxwellian_sigma(self) -> None:
        """Mean |v|^2 should equal 2 E / m for sigma = sqrt(2 E / (3 m))."""
        rng = np.random.default_rng(42)
        mass = 9.1093837015e-31
        energy_j = 3.0 * 1.602176634e-19
        count = 50_000
        sigma = np.sqrt(2.0 * energy_j / (3.0 * mass))
        vel = rng.normal(0.0, sigma, size=(count, 3))

        mean_v2 = float(np.mean(np.sum(vel * vel, axis=1)))
        expected_v2 = 2.0 * energy_j / mass
        self.assertLess(abs(mean_v2 - expected_v2) / expected_v2, 0.05)

    def test_grid_numba_backend_dispatch(self) -> None:
        grid = ElectrostaticGrid(4, 4, 4, particle_backend="numba")
        self.assertEqual(grid.particle_backend, "numba")

        rng = np.random.default_rng(3)
        positions = rng.uniform(0.0, grid.Lx, size=(20, 3))
        charges = rng.uniform(-1.0, 1.0, size=20)
        grid.deposit_rho_cic_batch(positions, charges)
        e = grid.gather_e_cic_batch(positions)
        self.assertEqual(e.shape, (20, 3))


if __name__ == "__main__":
    unittest.main()
