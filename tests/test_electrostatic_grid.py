"""Electrostatic grid physics tests (spectral E, Poisson consistency)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grids import ElectrostaticGrid


class TestElectrostaticGridPoisson(unittest.TestCase):
    def test_poisson_and_spectral_e_consistent(self) -> None:
        nx = ny = nz = 16
        dx = dy = dz = 1.0e-7
        eps0 = 8.8541878128e-12
        grid = ElectrostaticGrid(nx, ny, nz, dx=dx, dy=dy, dz=dz, eps0=eps0, boundary="periodic")

        ix, iy, iz = grid.interior_slice
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        kx0 = 2.0 * np.pi / (nx * dx)
        ky0 = 2.0 * np.pi / (ny * dy)
        kz0 = 2.0 * np.pi / (nz * dz)
        rho_int = 1.0e-6 * np.sin(kx0 * xx) * np.sin(ky0 * yy) * np.sin(kz0 * zz)
        grid.rho[ix, iy, iz] = rho_int

        grid.solve_fields()

        phi_int = grid.phi[ix, iy, iz]
        ex_int = grid.Ex[ix, iy, iz]
        rho_k = np.fft.fftn(rho_int)
        phi_k = np.fft.fftn(phi_int)
        ex_k = np.fft.fftn(ex_int)
        k2 = grid._k2_grid()
        kx, ky, kz = grid._k_wave_grids()
        mask = k2 > 0.0

        phi_expected = -rho_k[mask] / (eps0 * k2[mask])
        np.testing.assert_allclose(phi_k[mask], phi_expected, rtol=1e-10, atol=1e-20)

        ex_expected = -1j * kx[:, np.newaxis, np.newaxis] * phi_k
        np.testing.assert_allclose(ex_k, ex_expected, rtol=1e-10, atol=1e-10)

        div_k = 1j * (
            kx[:, np.newaxis, np.newaxis] * ex_k
            + ky[np.newaxis, :, np.newaxis] * np.fft.fftn(grid.Ey[ix, iy, iz])
            + kz[np.newaxis, np.newaxis, :] * np.fft.fftn(grid.Ez[ix, iy, iz])
        )
        div_residual = np.abs(div_k[mask] - k2[mask] * phi_k[mask])
        self.assertLess(float(np.max(div_residual)), 1e-6)

    def test_reflecting_rho_mean_removed(self) -> None:
        grid = ElectrostaticGrid(4, 4, 4, boundary="reflecting", particle_backend="numpy")
        ix, iy, iz = grid.interior_slice
        grid.rho[ix, iy, iz] = 1.0
        grid.solve_poisson()
        phi_int = grid.phi[ix, iy, iz]
        self.assertLess(float(np.std(phi_int)), 1.0)

    def test_deposit_in_place_matches_copy(self) -> None:
        rng = np.random.default_rng(7)
        grid = ElectrostaticGrid(8, 8, 8, dx=1.0, dy=1.0, dz=1.0, boundary="periodic", particle_backend="numpy")
        n = 40
        positions = rng.uniform(0.0, grid.Lx, size=(n, 3)).astype(np.float64)
        charges = rng.normal(size=n)

        grid_copy = ElectrostaticGrid(8, 8, 8, dx=1.0, dy=1.0, dz=1.0, boundary="periodic", particle_backend="numpy")
        grid_inplace = ElectrostaticGrid(8, 8, 8, dx=1.0, dy=1.0, dz=1.0, boundary="periodic", particle_backend="numpy")

        pos_copy = positions.copy()
        pos_inplace = positions.copy()

        grid_copy.deposit_rho_cic_batch(pos_copy, charges, in_place=False)
        grid_inplace.deposit_rho_cic_batch(pos_inplace, charges, in_place=True)

        np.testing.assert_allclose(grid_inplace.rho, grid_copy.rho, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
