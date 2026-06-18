"""
Author: Nathaniel Sun
Date: 2026-06-17
Description:
    3D grid-node (vertex-staggered) electrostatic PIC grid with CIC deposit/gather
    at integer grid nodes i·Δx (offset ``(0,0,0)``).

    Fields:
      - rho, phi: grid nodes (nx, ny, nz interior + ng guard cells)
      - Ex, Ey, Ez: grid nodes, from E = -grad(phi)

    Poisson equation:
      - periodic: FFT
      - anode: Dirichlet phi = anode_potential on all faces (sparse direct solve)
      - reflecting: Neumann dphi/dn = 0 on all faces (sparse direct solve, phi pinned;
        interior rho must have zero mean for a unique solution)

    Physical domain: x in [0, Lx), y in [0, Ly), z in [0, Lz) with L = n * d.
"""

from __future__ import annotations

from typing import Callable, Literal, TYPE_CHECKING

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from grid_common import periodic_field as _periodic_field

if TYPE_CHECKING:
    from Particle import Particle

BoundaryKind = Literal["periodic", "anode", "reflecting"]
ParticleBackend = Literal["numpy", "numba"]


def _shape_cell(nx: int, ny: int, nz: int, ng: int) -> tuple[int, int, int]:
    return nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


class ElectrostaticGrid:
    """Grid-node electrostatic PIC grid with periodic, anode, or reflecting walls."""

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        ng: int = 1,
        boundary: BoundaryKind = "periodic",
        eps0: float = 1.0,
        anode_potential: float = 0.0,
        particle_backend: ParticleBackend = "numba",
    ) -> None:
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("nx, ny, nz must be positive")
        if ng < 1:
            raise ValueError("ng must be at least 1 for guard cells")
        if boundary.lower() not in ("periodic", "anode", "reflecting"):
            raise ValueError(f"unsupported boundary condition: {boundary!r}")

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.ng = ng
        self.boundary = boundary.lower()
        self.eps0 = eps0
        self.anode_potential = anode_potential
        self._particle_backend = self._resolve_particle_backend(particle_backend)

        self.Lx = nx * dx
        self.Ly = ny * dy
        self.Lz = nz * dz

        shape = _shape_cell(nx, ny, nz, ng)
        self.rho = np.zeros(shape)
        self.phi = np.zeros(shape)
        self.Ex = np.zeros(shape)
        self.Ey = np.zeros(shape)
        self.Ez = np.zeros(shape)

        self._k2: NDArray[np.float64] | None = None
        self._kx: NDArray[np.float64] | None = None
        self._ky: NDArray[np.float64] | None = None
        self._kz: NDArray[np.float64] | None = None
        self._wall_solve: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
        self._wall_boundary_rhs: NDArray[np.float64] | None = None
        self._deposit_partial: NDArray[np.float64] | None = None

        if boundary == "anode":
            self._init_anode_solver()
        elif boundary == "reflecting":
            self._init_reflecting_solver()

        self.apply_boundaries()
        self._init_deposit_partial()

    def _init_deposit_partial(self) -> None:
        if self._particle_backend != "numba" or self.boundary != "periodic":
            return
        try:
            from pic_kernels import get_num_threads

            n_threads = get_num_threads()
            self._deposit_partial = np.zeros((n_threads, self.rho.size), dtype=np.float64)
        except ImportError:
            pass

    @property
    def interior_slice(self) -> tuple[slice, slice, slice]:
        ng = self.ng
        return (
            slice(ng, ng + self.nx),
            slice(ng, ng + self.ny),
            slice(ng, ng + self.nz),
        )

    @property
    def domain_lengths(self) -> tuple[float, float, float]:
        return (self.Lx, self.Ly, self.Lz)

    @property
    def particle_backend(self) -> ParticleBackend:
        return self._particle_backend

    @staticmethod
    def _resolve_particle_backend(requested: ParticleBackend) -> ParticleBackend:
        if requested != "numba":
            return "numpy"
        try:
            from pic_kernels import HAS_NUMBA, warmup_kernels

            if not HAS_NUMBA:
                return "numpy"
            warmup_kernels()
            return "numba"
        except ImportError:
            return "numpy"

    def zero_fields(self) -> None:
        for arr in (self.rho, self.phi, self.Ex, self.Ey, self.Ez):
            arr.fill(0.0)
        self.apply_boundaries()

    def zero_rho(self) -> None:
        self.rho.fill(0.0)
        self.apply_boundaries()

    def apply_boundaries(self) -> None:
        if self.boundary == "periodic":
            ng = self.ng
            for field in (self.rho, self.phi, self.Ex, self.Ey, self.Ez):
                _periodic_field(field, ng)
            return

        if self.boundary == "anode":
            self._apply_anode_boundaries()
        else:
            self._apply_reflecting_boundaries()

    def _zero_rho_guards(self) -> None:
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz
        self.rho[:ng, :, :] = 0.0
        self.rho[ng + nx :, :, :] = 0.0
        self.rho[:, :ng, :] = 0.0
        self.rho[:, ng + ny :, :] = 0.0
        self.rho[:, :, :ng] = 0.0
        self.rho[:, :, ng + nz :] = 0.0

    def _zero_e_guards(self) -> None:
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz
        for arr in (self.Ex, self.Ey, self.Ez):
            arr[:ng, :, :] = 0.0
            arr[ng + nx :, :, :] = 0.0
            arr[:, :ng, :] = 0.0
            arr[:, ng + ny :, :] = 0.0
            arr[:, :, :ng] = 0.0
            arr[:, :, ng + nz :] = 0.0

    def _apply_anode_boundaries(self) -> None:
        """Dirichlet phi on walls; zero rho guards; refresh E from phi."""
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz
        V = self.anode_potential
        ix = slice(ng, ng + nx)
        iy = slice(ng, ng + ny)
        iz = slice(ng, ng + nz)

        self._zero_rho_guards()

        for layer in range(ng):
            mirror_x_lo = ng + layer
            mirror_x_hi = ng + nx - 1 - layer
            self.phi[ng - 1 - layer, iy, iz] = 2.0 * V - self.phi[mirror_x_lo, iy, iz]
            self.phi[ng + nx + layer, iy, iz] = 2.0 * V - self.phi[mirror_x_hi, iy, iz]

            mirror_y_lo = ng + layer
            mirror_y_hi = ng + ny - 1 - layer
            self.phi[ix, ng - 1 - layer, iz] = 2.0 * V - self.phi[ix, mirror_y_lo, iz]
            self.phi[ix, ng + ny + layer, iz] = 2.0 * V - self.phi[ix, mirror_y_hi, iz]

            mirror_z_lo = ng + layer
            mirror_z_hi = ng + nz - 1 - layer
            self.phi[ix, iy, ng - 1 - layer] = 2.0 * V - self.phi[ix, iy, mirror_z_lo]
            self.phi[ix, iy, ng + nz + layer] = 2.0 * V - self.phi[ix, iy, mirror_z_hi]

        self._compute_e_field_from_phi_guards()
        self._zero_e_guards()

    def _apply_reflecting_boundaries(self) -> None:
        """Neumann phi on walls (even mirror); zero normal E on faces; zero rho guards."""
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz
        ix = slice(ng, ng + nx)
        iy = slice(ng, ng + ny)
        iz = slice(ng, ng + nz)

        self._zero_rho_guards()

        for layer in range(ng):
            self.phi[ng - 1 - layer, iy, iz] = self.phi[ng + layer, iy, iz]
            self.phi[ng + nx + layer, iy, iz] = self.phi[ng + nx - 1 - layer, iy, iz]
            self.phi[ix, ng - 1 - layer, iz] = self.phi[ix, ng + layer, iz]
            self.phi[ix, ng + ny + layer, iz] = self.phi[ix, ng + ny - 1 - layer, iz]
            self.phi[ix, iy, ng - 1 - layer] = self.phi[ix, iy, ng + layer]
            self.phi[ix, iy, ng + nz + layer] = self.phi[ix, iy, ng + nz - 1 - layer]

        self._compute_e_field_from_phi_guards()
        self._zero_normal_e_on_walls()
        self._zero_e_guards()

    def _compute_e_field_from_phi_guards(self) -> None:
        """Centered E = -grad(phi) using guard-aware stencils."""
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz
        ix = slice(ng, ng + nx)
        iy = slice(ng, ng + ny)
        iz = slice(ng, ng + nz)
        phi = self.phi

        self.Ex[ix, iy, iz] = -(phi[ng + 1 : ng + nx + 1, iy, iz] - phi[ng - 1 : ng + nx - 1, iy, iz]) / (
            2.0 * self.dx
        )
        self.Ey[ix, iy, iz] = -(phi[ix, ng + 1 : ng + ny + 1, iz] - phi[ix, ng - 1 : ng + ny - 1, iz]) / (
            2.0 * self.dy
        )
        self.Ez[ix, iy, iz] = -(phi[ix, iy, ng + 1 : ng + nz + 1] - phi[ix, iy, ng - 1 : ng + nz - 1]) / (
            2.0 * self.dz
        )

    def _zero_normal_e_on_walls(self) -> None:
        """Enforce E · n = 0 on domain faces (reflecting walls)."""
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz
        ix = slice(ng, ng + nx)
        iy = slice(ng, ng + ny)
        iz = slice(ng, ng + nz)

        self.Ex[ng, iy, iz] = 0.0
        self.Ex[ng + nx - 1, iy, iz] = 0.0
        self.Ey[ix, ng, iz] = 0.0
        self.Ey[ix, ng + ny - 1, iz] = 0.0
        self.Ez[ix, iy, ng] = 0.0
        self.Ez[ix, iy, ng + nz - 1] = 0.0

    def position_in_domain(self, pos: NDArray[np.floating]) -> NDArray[np.float64]:
        """Map a position into the physical domain according to the active boundary."""
        if self.boundary == "periodic":
            return self.wrap_position(pos, self.domain_lengths)

        return self.clamp_position(pos, self.domain_lengths, (self.dx, self.dy, self.dz))

    def position_in_domain_batch(
        self,
        positions: NDArray[np.floating],
        *,
        in_place: bool = False,
    ) -> NDArray[np.float64]:
        """Batch version of :meth:`position_in_domain` for ``(N, 3)`` positions."""
        if in_place:
            pos = np.asarray(positions, dtype=np.float64)
            if not pos.flags.c_contiguous:
                raise ValueError("in_place requires a contiguous float64 array")
        else:
            pos = np.asarray(positions, dtype=np.float64).copy()
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.boundary == "periodic":
            lengths = np.array(self.domain_lengths, dtype=np.float64)
            pos %= lengths
            return pos
        return self.clamp_position(pos, self.domain_lengths, (self.dx, self.dy, self.dz))

    @staticmethod
    def clamp_position(
        pos: NDArray[np.floating],
        lengths: tuple[float, float, float],
        cell_sizes: tuple[float, float, float],
    ) -> NDArray[np.float64]:
        clamped = np.asarray(pos, dtype=np.float64).copy()
        ds = np.array(cell_sizes)
        for axis, length in enumerate(lengths):
            upper = max(length - ds[axis], 0.0)
            clamped[axis] = np.clip(clamped[axis], 0.0, upper)
        return clamped

    @staticmethod
    def wrap_position(
        pos: NDArray[np.floating],
        lengths: tuple[float, float, float],
    ) -> NDArray[np.float64]:
        wrapped = pos.copy()
        for axis, length in enumerate(lengths):
            wrapped[axis] %= length
        return wrapped.astype(np.float64)

    @staticmethod
    def reflect_position_velocity(
        pos: NDArray[np.floating],
        vel: NDArray[np.floating],
        lengths: tuple[float, float, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Specular reflection: reverse velocity normal to any face crossed."""
        pos_out = np.asarray(pos, dtype=np.float64).copy()
        vel_out = np.asarray(vel, dtype=np.float64).copy()

        for axis, length in enumerate(lengths):
            while pos_out[axis] < 0.0:
                pos_out[axis] = -pos_out[axis]
                vel_out[axis] = -vel_out[axis]
            while pos_out[axis] >= length:
                pos_out[axis] = 2.0 * length - pos_out[axis]
                vel_out[axis] = -vel_out[axis]

        return pos_out, vel_out

    def reflect_particle(self, particle: Particle) -> None:
        """Reflect particle position/velocity at domain walls (for reflecting BC)."""
        pos, vel = self.reflect_position_velocity(
            particle.get_position(),
            particle.get_velocity(),
            self.domain_lengths,
        )
        particle.set_position(pos)
        particle.set_velocity(vel)

    def _init_anode_solver(self) -> None:
        self._wall_boundary_rhs, self._wall_solve = self._build_poisson_solver(
            dirichlet=True,
            wall_potential=self.anode_potential,
        )

    def _init_reflecting_solver(self) -> None:
        self._wall_boundary_rhs, self._wall_solve = self._build_poisson_solver(
            dirichlet=False,
            wall_potential=0.0,
        )

    def _build_poisson_solver(
        self,
        dirichlet: bool,
        wall_potential: float,
    ) -> tuple[NDArray[np.float64], Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
        nx, ny, nz = self.nx, self.ny, self.nz
        inv_dx2 = 1.0 / self.dx**2
        inv_dy2 = 1.0 / self.dy**2
        inv_dz2 = 1.0 / self.dz**2
        V = wall_potential

        n_nodes = nx * ny * nz

        def flat_index(i: int, j: int, k: int) -> int:
            return (i * ny + j) * nz + k

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        boundary_rhs = np.zeros(n_nodes, dtype=np.float64)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    row = flat_index(i, j, k)
                    diag = 0.0

                    if i == 0:
                        if dirichlet:
                            diag -= 3.0 * inv_dx2
                            boundary_rhs[row] -= 2.0 * V * inv_dx2
                        else:
                            diag -= 1.0 * inv_dx2
                        rows.append(row)
                        cols.append(flat_index(i + 1, j, k))
                        data.append(inv_dx2)
                    elif i == nx - 1:
                        if dirichlet:
                            diag -= 3.0 * inv_dx2
                            boundary_rhs[row] -= 2.0 * V * inv_dx2
                        else:
                            diag -= 1.0 * inv_dx2
                        rows.append(row)
                        cols.append(flat_index(i - 1, j, k))
                        data.append(inv_dx2)
                    else:
                        diag -= 2.0 * inv_dx2
                        rows.append(row)
                        cols.append(flat_index(i - 1, j, k))
                        data.append(inv_dx2)
                        rows.append(row)
                        cols.append(flat_index(i + 1, j, k))
                        data.append(inv_dx2)

                    if j == 0:
                        if dirichlet:
                            diag -= 3.0 * inv_dy2
                            boundary_rhs[row] -= 2.0 * V * inv_dy2
                        else:
                            diag -= 1.0 * inv_dy2
                        rows.append(row)
                        cols.append(flat_index(i, j + 1, k))
                        data.append(inv_dy2)
                    elif j == ny - 1:
                        if dirichlet:
                            diag -= 3.0 * inv_dy2
                            boundary_rhs[row] -= 2.0 * V * inv_dy2
                        else:
                            diag -= 1.0 * inv_dy2
                        rows.append(row)
                        cols.append(flat_index(i, j - 1, k))
                        data.append(inv_dy2)
                    else:
                        diag -= 2.0 * inv_dy2
                        rows.append(row)
                        cols.append(flat_index(i, j - 1, k))
                        data.append(inv_dy2)
                        rows.append(row)
                        cols.append(flat_index(i, j + 1, k))
                        data.append(inv_dy2)

                    if k == 0:
                        if dirichlet:
                            diag -= 3.0 * inv_dz2
                            boundary_rhs[row] -= 2.0 * V * inv_dz2
                        else:
                            diag -= 1.0 * inv_dz2
                        rows.append(row)
                        cols.append(flat_index(i, j, k + 1))
                        data.append(inv_dz2)
                    elif k == nz - 1:
                        if dirichlet:
                            diag -= 3.0 * inv_dz2
                            boundary_rhs[row] -= 2.0 * V * inv_dz2
                        else:
                            diag -= 1.0 * inv_dz2
                        rows.append(row)
                        cols.append(flat_index(i, j, k - 1))
                        data.append(inv_dz2)
                    else:
                        diag -= 2.0 * inv_dz2
                        rows.append(row)
                        cols.append(flat_index(i, j, k - 1))
                        data.append(inv_dz2)
                        rows.append(row)
                        cols.append(flat_index(i, j, k + 1))
                        data.append(inv_dz2)

                    rows.append(row)
                    cols.append(row)
                    data.append(diag)

        matrix = scipy.sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(n_nodes, n_nodes),
        ).tocsr()

        if not dirichlet:
            pin = flat_index(0, 0, 0)
            matrix = matrix.tolil()
            matrix[pin, :] = 0.0
            matrix[pin, pin] = 1.0
            matrix = matrix.tocsr()
            boundary_rhs[pin] = 0.0

        solve = scipy.sparse.linalg.factorized(matrix)
        return boundary_rhs, solve

    def _k_wave_grids(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if self._kx is not None and self._ky is not None and self._kz is not None:
            return self._kx, self._ky, self._kz

        self._kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        self._ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, self.dy)
        self._kz = 2.0 * np.pi * np.fft.fftfreq(self.nz, self.dz)
        return self._kx, self._ky, self._kz

    def _k2_grid(self) -> NDArray[np.float64]:
        if self._k2 is not None:
            return self._k2

        kx, ky, kz = self._k_wave_grids()
        self._k2 = (
            kx[:, np.newaxis, np.newaxis] ** 2
            + ky[np.newaxis, :, np.newaxis] ** 2
            + kz[np.newaxis, np.newaxis, :] ** 2
        )
        return self._k2

    def solve_poisson(self) -> None:
        if self.boundary == "periodic":
            self._solve_poisson_periodic()
        else:
            self._solve_poisson_wall()

    def _solve_poisson_periodic(self) -> None:
        """Solve nabla^2 phi = -rho / eps0 on the interior with periodic BC (FFT)."""
        ix, iy, iz = self.interior_slice
        rho_int = self.rho[ix, iy, iz]

        rho_k = np.fft.fftn(rho_int)
        k2 = self._k2_grid()
        phi_k = np.zeros_like(rho_k, dtype=np.complex128)
        mask = k2 > 0.0
        phi_k[mask] = -rho_k[mask] / (self.eps0 * k2[mask])

        phi_int = np.real(np.fft.ifftn(phi_k))
        self.phi[ix, iy, iz] = phi_int
        self.apply_boundaries()

    def _solve_poisson_wall(self) -> None:
        """Solve Poisson with anode (Dirichlet) or reflecting (Neumann) wall BCs."""
        if self._wall_solve is None or self._wall_boundary_rhs is None:
            raise RuntimeError("wall Poisson solver is not initialized")

        ix, iy, iz = self.interior_slice
        rho_int = self.rho[ix, iy, iz].copy()
        if self.boundary == "reflecting":
            rho_int -= np.mean(rho_int)
            self.rho[ix, iy, iz] = rho_int
        rhs = (-rho_int.ravel() / self.eps0) + self._wall_boundary_rhs
        phi_int = self._wall_solve(rhs).reshape(self.nx, self.ny, self.nz)
        self.phi[ix, iy, iz] = phi_int
        self.apply_boundaries()

    def compute_e_field(self) -> None:
        if self.boundary == "periodic":
            self._compute_e_field_periodic()
        elif self.boundary == "anode":
            self._apply_anode_boundaries()
        else:
            self._apply_reflecting_boundaries()

    def _compute_e_field_periodic(self) -> None:
        """Compute E = -grad(phi) with a spectral gradient matching the FFT Poisson solve."""
        ix, iy, iz = self.interior_slice
        phi_int = self.phi[ix, iy, iz]
        phi_k = np.fft.fftn(phi_int)
        kx, ky, kz = self._k_wave_grids()

        ex_int = np.real(np.fft.ifftn(-1j * kx[:, np.newaxis, np.newaxis] * phi_k))
        ey_int = np.real(np.fft.ifftn(-1j * ky[np.newaxis, :, np.newaxis] * phi_k))
        ez_int = np.real(np.fft.ifftn(-1j * kz[np.newaxis, np.newaxis, :] * phi_k))

        self.Ex[ix, iy, iz] = ex_int
        self.Ey[ix, iy, iz] = ey_int
        self.Ez[ix, iy, iz] = ez_int
        self.apply_boundaries()

    def solve_fields(self) -> None:
        """Solve Poisson and update E from the current rho."""
        self.solve_poisson()
        if self.boundary == "periodic":
            self.compute_e_field()

    def deposit_rho_cic(self, x: float, y: float, z: float, q: float) -> None:
        """Cloud-in-cell charge deposition onto grid-node rho."""
        pos = self.position_in_domain(np.array([x, y, z]))
        self._deposit_scalar(
            self.rho,
            pos,
            q / (self.dx * self.dy * self.dz),
            (0.0, 0.0, 0.0),
        )

    def deposit_rho_cic_batch(
        self,
        positions: NDArray[np.floating],
        charges: NDArray[np.floating],
        *,
        in_place: bool = False,
    ) -> None:
        """Vectorized CIC charge deposition for ``(N, 3)`` positions and ``(N,)`` charges."""
        pos = self.position_in_domain_batch(positions, in_place=in_place)
        charges_arr = np.asarray(charges, dtype=np.float64)
        if charges_arr.ndim != 1 or charges_arr.shape[0] != pos.shape[0]:
            raise ValueError("charges must have shape (N,) matching positions")
        cell_volume = self.dx * self.dy * self.dz
        values = charges_arr / cell_volume
        if self._particle_backend == "numba" and self.boundary == "periodic":
            from pic_kernels import deposit_cic_periodic

            if self._deposit_partial is None:
                self._init_deposit_partial()
            deposit_cic_periodic(
                self.rho,
                pos,
                values,
                self.dx,
                self.dy,
                self.dz,
                self.nx,
                self.ny,
                self.nz,
                self.ng,
                self._deposit_partial,
            )
        else:
            self._deposit_scalar_batch(
                self.rho,
                pos,
                values,
                (0.0, 0.0, 0.0),
            )

    def gather_e_cic(self, x: float, y: float, z: float) -> NDArray[np.float64]:
        """Trilinear interpolation of E at a particle position."""
        pos = self.position_in_domain(np.array([x, y, z]))
        ex = self._gather_scalar(self.Ex, pos, (0.0, 0.0, 0.0))
        ey = self._gather_scalar(self.Ey, pos, (0.0, 0.0, 0.0))
        ez = self._gather_scalar(self.Ez, pos, (0.0, 0.0, 0.0))
        return np.array([ex, ey, ez], dtype=np.float64)

    def gather_e_cic_batch(
        self,
        positions: NDArray[np.floating],
        *,
        in_place: bool = False,
    ) -> NDArray[np.float64]:
        """Trilinear interpolation of E for ``(N, 3)`` particle positions; returns ``(N, 3)``."""
        pos = self.position_in_domain_batch(positions, in_place=in_place)
        if self._particle_backend == "numba" and self.boundary == "periodic":
            from pic_kernels import gather_e_cic_periodic

            return gather_e_cic_periodic(
                self.Ex,
                self.Ey,
                self.Ez,
                pos,
                self.dx,
                self.dy,
                self.dz,
                self.nx,
                self.ny,
                self.nz,
                self.ng,
            )
        ex = self._gather_scalar_batch(self.Ex, pos, (0.0, 0.0, 0.0))
        ey = self._gather_scalar_batch(self.Ey, pos, (0.0, 0.0, 0.0))
        ez = self._gather_scalar_batch(self.Ez, pos, (0.0, 0.0, 0.0))
        return np.column_stack([ex, ey, ez])

    def gather_fields(
        self,
        x: float,
        y: float,
        z: float,
        B: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (E, B) at a particle position for particle push (B defaults to zero)."""
        E = self.gather_e_cic(x, y, z)
        if B is None:
            B_arr = np.zeros(3, dtype=np.float64)
        else:
            B_arr = np.asarray(B, dtype=np.float64)
        return E, B_arr

    def _grid_coords(
        self,
        pos: NDArray[np.floating],
        offsets: tuple[float, float, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        ds = np.array([self.dx, self.dy, self.dz])
        offs = np.array(offsets)
        g = pos / ds - offs
        i0 = np.floor(g).astype(np.int64)
        f = g - i0
        return g, i0, f

    def _grid_coords_batch(
        self,
        positions: NDArray[np.floating],
        offsets: tuple[float, float, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim == 1:
            return self._grid_coords(pos, offsets)
        ds = np.array([self.dx, self.dy, self.dz])
        offs = np.array(offsets)
        g = pos / ds - offs
        i0 = np.floor(g).astype(np.int64)
        f = g - i0
        return g, i0, f

    def _logical_index(self, logical: int, axis: int) -> int:
        n_cells = (self.nx, self.ny, self.nz)[axis]
        if self.boundary == "periodic":
            return int(logical % n_cells)
        return int(np.clip(logical, 0, n_cells - 1))

    def _logical_index_batch(self, logical: NDArray[np.int64], axis: int) -> NDArray[np.int64]:
        n_cells = (self.nx, self.ny, self.nz)[axis]
        if self.boundary == "periodic":
            return logical % n_cells
        return np.clip(logical, 0, n_cells - 1)

    def _deposit_scalar(
        self,
        field: NDArray[np.floating],
        pos: NDArray[np.floating],
        value: float,
        offsets: tuple[float, float, float],
    ) -> None:
        _, i0, f = self._grid_coords(pos, offsets)
        ng = self.ng

        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    wi = (1.0 - f[0]) if di == 0 else f[0]
                    wj = (1.0 - f[1]) if dj == 0 else f[1]
                    wk = (1.0 - f[2]) if dk == 0 else f[2]
                    ii = self._logical_index(i0[0] + di, 0) + ng
                    jj = self._logical_index(i0[1] + dj, 1) + ng
                    kk = self._logical_index(i0[2] + dk, 2) + ng
                    field[ii, jj, kk] += value * wi * wj * wk

    def _deposit_scalar_batch(
        self,
        field: NDArray[np.floating],
        positions: NDArray[np.floating],
        values: NDArray[np.floating],
        offsets: tuple[float, float, float],
    ) -> None:
        """Vectorized CIC scatter-add for many particles."""
        _, i0, f = self._grid_coords_batch(positions, offsets)
        ng = self.ng
        ny_tot = field.shape[1]
        nz_tot = field.shape[2]
        flat = field.ravel()

        for di in (0, 1):
            wi = (1.0 - f[:, 0]) if di == 0 else f[:, 0]
            ii = self._logical_index_batch(i0[:, 0] + di, 0) + ng
            for dj in (0, 1):
                wj = (1.0 - f[:, 1]) if dj == 0 else f[:, 1]
                jj = self._logical_index_batch(i0[:, 1] + dj, 1) + ng
                for dk in (0, 1):
                    wk = (1.0 - f[:, 2]) if dk == 0 else f[:, 2]
                    kk = self._logical_index_batch(i0[:, 2] + dk, 2) + ng
                    weight = values * wi * wj * wk
                    flat_idx = ii * ny_tot * nz_tot + jj * nz_tot + kk
                    np.add.at(flat, flat_idx, weight)

    def _gather_scalar(
        self,
        field: NDArray[np.floating],
        pos: NDArray[np.floating],
        offsets: tuple[float, float, float],
    ) -> float:
        _, i0, f = self._grid_coords(pos, offsets)
        ng = self.ng
        value = 0.0

        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    wi = (1.0 - f[0]) if di == 0 else f[0]
                    wj = (1.0 - f[1]) if dj == 0 else f[1]
                    wk = (1.0 - f[2]) if dk == 0 else f[2]
                    ii = self._logical_index(i0[0] + di, 0) + ng
                    jj = self._logical_index(i0[1] + dj, 1) + ng
                    kk = self._logical_index(i0[2] + dk, 2) + ng
                    value += field[ii, jj, kk] * wi * wj * wk

        return float(value)

    def _gather_scalar_batch(
        self,
        field: NDArray[np.floating],
        positions: NDArray[np.floating],
        offsets: tuple[float, float, float],
    ) -> NDArray[np.float64]:
        """Vectorized trilinear gather for many particles."""
        _, i0, f = self._grid_coords_batch(positions, offsets)
        ng = self.ng
        values = np.zeros(positions.shape[0], dtype=np.float64)

        for di in (0, 1):
            wi = (1.0 - f[:, 0]) if di == 0 else f[:, 0]
            ii = self._logical_index_batch(i0[:, 0] + di, 0) + ng
            for dj in (0, 1):
                wj = (1.0 - f[:, 1]) if dj == 0 else f[:, 1]
                jj = self._logical_index_batch(i0[:, 1] + dj, 1) + ng
                for dk in (0, 1):
                    wk = (1.0 - f[:, 2]) if dk == 0 else f[:, 2]
                    kk = self._logical_index_batch(i0[:, 2] + dk, 2) + ng
                    values += field[ii, jj, kk] * wi * wj * wk

        return values
