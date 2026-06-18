"""
Author: Nathaniel Sun
Date: 2026-06-16
Description:
    3D Yee staggered grid for electromagnetic PIC.

    WIP — not used by live electrostatic examples (``examples/``); covered only
    by unit tests until an EM-PIC driver is wired. See ``tests/test_yee_grid.py``.

    Indexing convention (ng guard cells per side):
      - Cell-centered rho, J: interior indices ng : ng+nx, etc.
      - Ex at x-faces:     ng : ng+nx+1 along x
      - Ey at y-faces:     ng : ng+ny+1 along y
      - Ez at z-faces:     ng : ng+nz+1 along z
      - Bx at y-z faces:   ng : ng+ny+1 (y), ng : ng+nz+1 (z)
      - By at x-z faces:   ng : ng+nx+1 (x), ng : ng+nz+1 (z)
      - Bz at x-y faces:   ng : ng+nx+1 (x), ng : ng+ny+1 (y)

    Boundary conditions:
      - periodic: guard cells copy opposite face (default)
      - anode: PEC walls (tangential E = 0, normal B = 0)
      - reflecting: PMC-style walls (normal E = 0, tangential B = 0) plus
        specular particle reflection via reflect_particle()

    Physical domain: x in [0, Lx), y in [0, Ly), z in [0, Lz) with L = n * d.
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from Particle import Particle

BoundaryKind = Literal["periodic", "anode", "reflecting"]

from grid_common import periodic_along_axis as _periodic_along_axis
from grid_common import periodic_field as _periodic_field

def _shape_ex(nx: int, ny: int, nz: int, ng: int) -> tuple[int, int, int]:
    return nx + 1 + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _shape_ey(nx: int, ny: int, nz: int, ng: int) -> tuple[int, int, int]:
    return nx + 2 * ng, ny + 1 + 2 * ng, nz + 2 * ng


def _shape_ez(nx: int, ny: int, nz: int, ng: int) -> tuple[int, int, int]:
    return nx + 2 * ng, ny + 2 * ng, nz + 1 + 2 * ng


def _shape_bx(nx: int, ny: int, nz: int, ng: int) -> tuple[int, int, int]:
    return nx + 2 * ng, ny + 1 + 2 * ng, nz + 1 + 2 * ng


def _shape_by(nx: int, ny: int, nz: int, ng: int) -> tuple[int, int, int]:
    return nx + 1 + 2 * ng, ny + 2 * ng, nz + 1 + 2 * ng


def _shape_bz(nx: int, ny: int, nz: int, ng: int) -> tuple[int, int, int]:
    return nx + 1 + 2 * ng, ny + 1 + 2 * ng, nz + 2 * ng


def _shape_rho(nx: int, ny: int, nz: int, ng: int) -> tuple[int, int, int]:
    return nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


class YeeGrid:
    """3D Yee staggered grid with explicit FDTD updates and selectable wall BCs."""

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
        mu0: float = 1.0,
        anode_potential: float = 0.0,
    ) -> None:
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("nx, ny, nz must be positive")
        if ng < 1:
            raise ValueError("ng must be at least 1 for guard cells")
        if boundary not in ("periodic", "anode", "reflecting"):
            raise ValueError(f"unsupported boundary condition: {boundary!r}")

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.ng = ng
        self.boundary = boundary
        self.eps0 = eps0
        self.mu0 = mu0
        self.anode_potential = anode_potential

        self.Lx = nx * dx
        self.Ly = ny * dy
        self.Lz = nz * dz

        self.Ex = np.zeros(_shape_ex(nx, ny, nz, ng))
        self.Ey = np.zeros(_shape_ey(nx, ny, nz, ng))
        self.Ez = np.zeros(_shape_ez(nx, ny, nz, ng))
        self.Bx = np.zeros(_shape_bx(nx, ny, nz, ng))
        self.By = np.zeros(_shape_by(nx, ny, nz, ng))
        self.Bz = np.zeros(_shape_bz(nx, ny, nz, ng))
        self.rho = np.zeros(_shape_rho(nx, ny, nz, ng))
        self.Jx = np.zeros(_shape_ex(nx, ny, nz, ng))
        self.Jy = np.zeros(_shape_ey(nx, ny, nz, ng))
        self.Jz = np.zeros(_shape_ez(nx, ny, nz, ng))

        self.apply_boundaries()

    @property
    def domain_lengths(self) -> tuple[float, float, float]:
        return (self.Lx, self.Ly, self.Lz)

    @property
    def c(self) -> float:
        return 1.0 / np.sqrt(self.mu0 * self.eps0)

    def cfl_dt_limit(self, safety: float = 0.99) -> float:
        """3D Courant limit for explicit Yee update (c=1 when eps0=mu0=1)."""
        inv_dx2 = 1.0 / self.dx**2
        inv_dy2 = 1.0 / self.dy**2
        inv_dz2 = 1.0 / self.dz**2
        return safety / (self.c * np.sqrt(inv_dx2 + inv_dy2 + inv_dz2))

    def zero_fields(self) -> None:
        for arr in (self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz, self.rho):
            arr.fill(0.0)
        self.zero_currents()
        self.apply_boundaries()

    def zero_currents(self) -> None:
        for arr in (self.Jx, self.Jy, self.Jz):
            arr.fill(0.0)

    def apply_boundaries(self) -> None:
        if self.boundary == "periodic":
            ng = self.ng
            for field in (self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz, self.rho,
                          self.Jx, self.Jy, self.Jz):
                _periodic_field(field, ng)
            return

        if self.boundary == "anode":
            self._apply_anode_boundaries()
        else:
            self._apply_reflecting_boundaries()

    @staticmethod
    def _zero_guards_along_axis(
        field: NDArray[np.floating],
        axis: int,
        ng: int,
        n_interior: int,
    ) -> None:
        idx = [slice(None)] * field.ndim
        lo = idx.copy()
        lo[axis] = slice(0, ng)
        hi = idx.copy()
        hi[axis] = slice(ng + n_interior, field.shape[axis])
        field[tuple(lo)] = 0.0
        field[tuple(hi)] = 0.0

    def _zero_source_guards(self) -> None:
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz

        self._zero_guards_along_axis(self.rho, 0, ng, nx)
        self._zero_guards_along_axis(self.rho, 1, ng, ny)
        self._zero_guards_along_axis(self.rho, 2, ng, nz)

        self._zero_guards_along_axis(self.Jx, 0, ng, nx + 1)
        self._zero_guards_along_axis(self.Jx, 1, ng, ny)
        self._zero_guards_along_axis(self.Jx, 2, ng, nz)

        self._zero_guards_along_axis(self.Jy, 0, ng, nx)
        self._zero_guards_along_axis(self.Jy, 1, ng, ny + 1)
        self._zero_guards_along_axis(self.Jy, 2, ng, nz)

        self._zero_guards_along_axis(self.Jz, 0, ng, nx)
        self._zero_guards_along_axis(self.Jz, 1, ng, ny)
        self._zero_guards_along_axis(self.Jz, 2, ng, nz + 1)

    def _apply_anode_boundaries(self) -> None:
        """PEC walls: tangential E = 0 and normal B = 0 in guard regions."""
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz
        self._zero_source_guards()

        # Tangential E = 0
        self._zero_guards_along_axis(self.Ey, 0, ng, nx)
        self._zero_guards_along_axis(self.Ez, 0, ng, nx)
        self._zero_guards_along_axis(self.Ex, 1, ng, ny)
        self._zero_guards_along_axis(self.Ez, 1, ng, ny)
        self._zero_guards_along_axis(self.Ex, 2, ng, nz)
        self._zero_guards_along_axis(self.Ey, 2, ng, nz)

        # Normal B = 0
        self._zero_guards_along_axis(self.Bx, 0, ng, nx)
        self._zero_guards_along_axis(self.By, 1, ng, ny)
        self._zero_guards_along_axis(self.Bz, 2, ng, nz)

    def _apply_reflecting_boundaries(self) -> None:
        """PMC-style walls: normal E = 0 and tangential B = 0 in guard regions."""
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz
        self._zero_source_guards()

        # Normal E = 0
        self._zero_guards_along_axis(self.Ex, 0, ng, nx + 1)
        self._zero_guards_along_axis(self.Ey, 1, ng, ny + 1)
        self._zero_guards_along_axis(self.Ez, 2, ng, nz + 1)

        # Tangential B = 0
        self._zero_guards_along_axis(self.By, 0, ng, nx + 1)
        self._zero_guards_along_axis(self.Bz, 0, ng, nx + 1)
        self._zero_guards_along_axis(self.Bx, 1, ng, ny + 1)
        self._zero_guards_along_axis(self.Bz, 1, ng, ny + 1)
        self._zero_guards_along_axis(self.Bx, 2, ng, nz + 1)
        self._zero_guards_along_axis(self.By, 2, ng, nz + 1)

    def position_in_domain(self, pos: NDArray[np.floating]) -> NDArray[np.float64]:
        if self.boundary == "periodic":
            return self.wrap_position(pos, self.domain_lengths)

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
        """Specular wall reflection for reflecting boundary mode."""
        pos, vel = self.reflect_position_velocity(
            particle.get_position(),
            particle.get_velocity(),
            self.domain_lengths,
        )
        particle.set_position(pos)
        particle.set_velocity(vel)

    def update_b(self, dt: float) -> None:
        """Advance B by dt using Faraday's law: dB/dt = -curl(E)."""
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz

        ib = slice(ng, ng + nx)
        jb = slice(ng, ng + ny + 1)
        kb = slice(ng, ng + nz + 1)
        self.Bx[ib, jb, kb] -= dt * (
            np.diff(self.Ez[ib, slice(ng - 1, ng + ny + 1), kb], axis=1) / self.dy
            - np.diff(self.Ey[ib, jb, slice(ng - 1, ng + nz + 1)], axis=2) / self.dz
        )

        ib = slice(ng, ng + nx + 1)
        jb = slice(ng, ng + ny)
        kb = slice(ng, ng + nz + 1)
        self.By[ib, jb, kb] -= dt * (
            np.diff(self.Ex[ib, jb, slice(ng - 1, ng + nz + 1)], axis=2) / self.dz
            - np.diff(self.Ez[slice(ng - 1, ng + nx + 1), jb, kb], axis=0) / self.dx
        )

        ib = slice(ng, ng + nx + 1)
        jb = slice(ng, ng + ny + 1)
        kb = slice(ng, ng + nz)
        self.Bz[ib, jb, kb] -= dt * (
            np.diff(self.Ey[slice(ng - 1, ng + nx + 1), jb, kb], axis=0) / self.dx
            - np.diff(self.Ex[ib, slice(ng - 1, ng + ny + 1), kb], axis=1) / self.dy
        )

        self.apply_boundaries()

    def update_e(self, dt: float) -> None:
        """Advance E by dt using Ampere's law: dE/dt = curl(B)/(mu0*eps0) - J/eps0."""
        ng = self.ng
        nx, ny, nz = self.nx, self.ny, self.nz
        curl_coeff = dt / (self.mu0 * self.eps0)

        ib = slice(ng, ng + nx + 1)
        jb = slice(ng, ng + ny)
        kb = slice(ng, ng + nz)
        self.Ex[ib, jb, kb] += curl_coeff * (
            np.diff(self.Bz[ib, slice(ng, ng + ny + 1), kb], axis=1) / self.dy
            - np.diff(self.By[ib, jb, slice(ng, ng + nz + 1)], axis=2) / self.dz
        )
        self.Ex[ib, jb, kb] -= dt * self.Jx[ib, jb, kb] / self.eps0

        ib = slice(ng, ng + nx)
        jb = slice(ng, ng + ny + 1)
        kb = slice(ng, ng + nz)
        self.Ey[ib, jb, kb] += curl_coeff * (
            np.diff(self.Bx[ib, jb, slice(ng, ng + nz + 1)], axis=2) / self.dz
            - np.diff(self.Bz[slice(ng, ng + nx + 1), jb, kb], axis=0) / self.dx
        )
        self.Ey[ib, jb, kb] -= dt * self.Jy[ib, jb, kb] / self.eps0

        ib = slice(ng, ng + nx)
        jb = slice(ng, ng + ny)
        kb = slice(ng, ng + nz + 1)
        self.Ez[ib, jb, kb] += curl_coeff * (
            np.diff(self.By[slice(ng, ng + nx + 1), jb, kb], axis=0) / self.dx
            - np.diff(self.Bx[ib, slice(ng, ng + ny + 1), kb], axis=1) / self.dy
        )
        self.Ez[ib, jb, kb] -= dt * self.Jz[ib, jb, kb] / self.eps0

        self.apply_boundaries()

    def step_fields(self, dt: float) -> None:
        """One leapfrog half-step pair: B then E."""
        self.update_b(dt)
        self.update_e(dt)

    def deposit_rho_cic(self, x: float, y: float, z: float, q: float) -> None:
        """Cloud-in-cell charge deposition onto cell-centered rho (Yee grid interior)."""
        pos = self.position_in_domain(np.array([x, y, z]))
        self._deposit_scalar(self.rho, pos, q / (self.dx * self.dy * self.dz), (0.0, 0.0, 0.0))

    def deposit_rho_cic_batch(
        self,
        positions: NDArray[np.floating],
        charges: NDArray[np.floating],
    ) -> None:
        """Batch CIC rho deposit for ``(N, 3)`` positions (same API as ElectrostaticGrid)."""
        pos = np.asarray(positions, dtype=np.float64).copy()
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.boundary == "periodic":
            lengths = np.array(self.domain_lengths, dtype=np.float64)
            pos %= lengths
        else:
            pos = self.clamp_position(pos, self.domain_lengths, (self.dx, self.dy, self.dz))

        charges_arr = np.asarray(charges, dtype=np.float64)
        if charges_arr.ndim != 1 or charges_arr.shape[0] != pos.shape[0]:
            raise ValueError("charges must have shape (N,) matching positions")
        cell_volume = self.dx * self.dy * self.dz
        values = charges_arr / cell_volume

        if self.boundary == "periodic":
            try:
                from pic_kernels import deposit_cic_periodic, get_num_threads

                partial = np.zeros((get_num_threads(), self.rho.size), dtype=np.float64)
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
                    partial,
                )
                return
            except ImportError:
                pass

        for i in range(pos.shape[0]):
            self._deposit_scalar(self.rho, pos[i], values[i], (0.0, 0.0, 0.0))

    def deposit_j_cic(
        self,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        q: float,
    ) -> None:
        """
        Cloud-in-cell current deposition onto staggered J components.

        WIP: this CIC J deposit is not charge-conserving; a full EM driver needs
        Esirkepov/Villasenor-Buneman or equivalent (see plan B2).
        """
        pos = self.position_in_domain(np.array([x, y, z]))
        jx = q * vx / (self.dy * self.dz)
        jy = q * vy / (self.dx * self.dz)
        jz = q * vz / (self.dx * self.dy)

        self._deposit_component(self.Jx, pos, jx, (0.0, 0.5, 0.5))
        self._deposit_component(self.Jy, pos, jy, (0.5, 0.0, 0.5))
        self._deposit_component(self.Jz, pos, jz, (0.5, 0.5, 0.0))

    def gather_e_cic(self, x: float, y: float, z: float) -> NDArray[np.float64]:
        """Trilinear interpolation of E at a particle position."""
        pos = self.position_in_domain(np.array([x, y, z]))
        ex = self._gather_component(self.Ex, pos, (0.0, 0.5, 0.5))
        ey = self._gather_component(self.Ey, pos, (0.5, 0.0, 0.5))
        ez = self._gather_component(self.Ez, pos, (0.5, 0.5, 0.0))
        return np.array([ex, ey, ez], dtype=np.float64)

    def gather_b_cic(self, x: float, y: float, z: float) -> NDArray[np.float64]:
        """Trilinear interpolation of B at a particle position."""
        pos = self.position_in_domain(np.array([x, y, z]))
        bx = self._gather_component(self.Bx, pos, (0.5, 0.0, 0.0))
        by = self._gather_component(self.By, pos, (0.0, 0.5, 0.0))
        bz = self._gather_component(self.Bz, pos, (0.0, 0.0, 0.5))
        return np.array([bx, by, bz], dtype=np.float64)

    def gather_e_cic_batch(self, positions: NDArray[np.floating]) -> NDArray[np.float64]:
        """Batch staggered E gather; returns ``(N, 3)`` (WIP EM driver helper)."""
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        n = pos.shape[0]
        e_out = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            e_out[i] = self.gather_e_cic(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
        return e_out

    def gather_b_cic_batch(self, positions: NDArray[np.floating]) -> NDArray[np.float64]:
        """Batch staggered B gather; returns ``(N, 3)`` (WIP EM driver helper)."""
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        n = pos.shape[0]
        b_out = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            b_out[i] = self.gather_b_cic(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
        return b_out

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
                    ii = self._logical_cell_index(i0[0] + di, 0) + ng
                    jj = self._logical_cell_index(i0[1] + dj, 1) + ng
                    kk = self._logical_cell_index(i0[2] + dk, 2) + ng
                    field[ii, jj, kk] += value * wi * wj * wk

    def _logical_cell_index(self, logical: int, axis: int) -> int:
        n_cells = (self.nx, self.ny, self.nz)[axis]
        if self.boundary == "periodic":
            return int(logical % n_cells)
        return int(np.clip(logical, 0, n_cells - 1))

    def _logical_component_index(self, logical: int, axis: int, offset: float) -> int:
        n_cells = (self.nx, self.ny, self.nz)[axis]
        n_points = n_cells + 1 if offset == 0.0 else n_cells
        if self.boundary == "periodic":
            return int(logical % n_points)
        return int(np.clip(logical, 0, n_points - 1))

    def _component_index(
        self,
        logical: NDArray[np.int64],
        offsets: tuple[float, float, float],
    ) -> tuple[int, int, int]:
        ng = self.ng
        dims = (self.nx, self.ny, self.nz)
        idx = []
        for axis, (logical_i, off, n_cells) in enumerate(zip(logical, offsets, dims, strict=True)):
            idx.append(self._logical_component_index(int(logical_i), axis, off) + ng)
        return idx[0], idx[1], idx[2]

    def _deposit_component(
        self,
        field: NDArray[np.floating],
        pos: NDArray[np.floating],
        value: float,
        offsets: tuple[float, float, float],
    ) -> None:
        _, i0, f = self._grid_coords(pos, offsets)

        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    wi = (1.0 - f[0]) if di == 0 else f[0]
                    wj = (1.0 - f[1]) if dj == 0 else f[1]
                    wk = (1.0 - f[2]) if dk == 0 else f[2]
                    logical = i0 + np.array([di, dj, dk], dtype=np.int64)
                    ii, jj, kk = self._component_index(logical, offsets)
                    field[ii, jj, kk] += value * wi * wj * wk

    def _gather_component(
        self,
        field: NDArray[np.floating],
        pos: NDArray[np.floating],
        offsets: tuple[float, float, float],
    ) -> float:
        _, i0, f = self._grid_coords(pos, offsets)
        value = 0.0

        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    wi = (1.0 - f[0]) if di == 0 else f[0]
                    wj = (1.0 - f[1]) if dj == 0 else f[1]
                    wk = (1.0 - f[2]) if dk == 0 else f[2]
                    logical = i0 + np.array([di, dj, dk], dtype=np.int64)
                    ii, jj, kk = self._component_index(logical, offsets)
                    value += field[ii, jj, kk] * wi * wj * wk

        return float(value)
