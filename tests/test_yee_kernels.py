"""Tests for Yee staggered Numba kernels vs YeeGrid numpy reference."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grids import YeeGrid

try:
    from grids.pic_kernels import (
        HAS_NUMBA,
        gather_b_yee_cic_periodic,
        gather_e_yee_cic_periodic,
        warmup_kernels,
    )
except ImportError:
    HAS_NUMBA = False


def _make_yee_grid(n_cells: int = 8) -> YeeGrid:
    return YeeGrid(
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
class TestYeeKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warmup_kernels()

    def test_gather_e_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        grid = _make_yee_grid()
        grid.Ex.fill(1.0)
        grid.Ey.fill(2.0)
        grid.Ez.fill(3.0)
        n_particles = 150
        positions = rng.uniform(0.0, grid.Lx, size=(n_particles, 3))
        pos = grid.position_in_domain_batch(positions)

        ref = grid.gather_e_cic_batch(pos)
        numba_out = gather_e_yee_cic_periodic(
            grid.Ex, grid.Ey, grid.Ez, pos,
            grid.dx, grid.dy, grid.dz, grid.nx, grid.ny, grid.nz, grid.ng,
        )
        np.testing.assert_allclose(numba_out, ref, rtol=1e-10, atol=1e-12)

    def test_gather_b_matches_numpy(self) -> None:
        rng = np.random.default_rng(2)
        grid = _make_yee_grid()
        grid.Bx.fill(0.5)
        grid.By.fill(1.5)
        grid.Bz.fill(2.5)
        positions = rng.uniform(0.0, grid.Lx, size=(80, 3))
        pos = grid.position_in_domain_batch(positions)

        ref = grid.gather_b_cic_batch(pos)
        numba_out = gather_b_yee_cic_periodic(
            grid.Bx, grid.By, grid.Bz, pos,
            grid.dx, grid.dy, grid.dz, grid.nx, grid.ny, grid.nz, grid.ng,
        )
        np.testing.assert_allclose(numba_out, ref, rtol=1e-10, atol=1e-12)

    def test_esirkepov_continuity(self) -> None:
        grid = _make_yee_grid(8)
        rng = np.random.default_rng(3)
        # Stay away from domain edges so periodic face bookkeeping does not dominate.
        pos_old = rng.uniform(1.5, 6.5, size=(40, 3))
        pos_new = pos_old + rng.uniform(-0.35, 0.35, size=(40, 3))
        pos_old = grid.position_in_domain_batch(pos_old)
        pos_new = grid.position_in_domain_batch(pos_new)
        charges = rng.uniform(-1.0, 1.0, size=40)
        dt = 1.0

        grid.rho.fill(0.0)
        grid.deposit_rho_cic_batch(pos_old, charges)
        rho_old = grid.rho.copy()

        grid.rho.fill(0.0)
        grid.deposit_rho_cic_batch(pos_new, charges)
        rho_new = grid.rho.copy()
        drho_dt = (rho_new - rho_old) / dt

        grid_nb = YeeGrid(8, 8, 8, boundary="periodic", particle_backend="numba")
        grid_nb.deposit_j_esirkepov_cic_batch(pos_old, pos_new, charges)

        ng = grid.ng
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        div_j = np.zeros((nx, ny, nz), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ix = ng + i
                    jy = ng + j
                    kz = ng + k
                    djx = (
                        grid_nb.Jx[ix + 1, jy, kz] - grid_nb.Jx[ix, jy, kz]
                    ) / grid.dx
                    djy = (
                        grid_nb.Jy[ix, jy + 1, kz] - grid_nb.Jy[ix, jy, kz]
                    ) / grid.dy
                    djz = (
                        grid_nb.Jz[ix, jy, kz + 1] - grid_nb.Jz[ix, jy, kz]
                    ) / grid.dz
                    div_j[i, j, k] = djx + djy + djz

        interior = slice(ng, ng + nx)
        residual = div_j + drho_dt[interior, interior, interior]
        scale = np.max(np.abs(drho_dt[interior, interior, interior])) + 1e-30
        self.assertLess(float(np.max(np.abs(residual))) / scale, 2.0)

    def test_esirkepov_batch_matches_scalar(self) -> None:
        rng = np.random.default_rng(5)
        grid = _make_yee_grid(8)
        pos_old = rng.uniform(1.0, 7.0, size=(25, 3))
        pos_new = pos_old + rng.uniform(-0.3, 0.3, size=(25, 3))
        charges = rng.uniform(-2.0, 2.0, size=25)

        grid_scalar = YeeGrid(8, 8, 8, boundary="periodic", particle_backend="numpy")
        grid_scalar.zero_currents()
        grid_scalar.deposit_j_esirkepov_cic_batch(pos_old, pos_new, charges)

        grid_batch = YeeGrid(8, 8, 8, boundary="periodic", particle_backend="numba")
        grid_batch.zero_currents()
        grid_batch.deposit_j_esirkepov_cic_batch(pos_old, pos_new, charges)

        np.testing.assert_allclose(grid_batch.Jx, grid_scalar.Jx, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(grid_batch.Jy, grid_scalar.Jy, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(grid_batch.Jz, grid_scalar.Jz, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
