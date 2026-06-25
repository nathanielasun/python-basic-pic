"""
Author: Nathaniel Sun
Date: 2026-06-16
Description:
    3D Yee staggered grid for electromagnetic PIC.

    WIP — Esirkepov charge-conserving J deposition is available for periodic BC via
    ``deposit_j_esirkepov_cic_batch``. Live EM drivers are ``examples/em_01_*.py`` …
    ``examples/em_06_*.py`` (see ``em_common.py``). Unit tests: ``tests/test_yee_grid.py``.

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
        specular particle reflection via reflect_position_velocity()

    Physical domain: x in [0, Lx), y in [0, Ly), z in [0, Lz) with L = n * d.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import PICGridBase
from .grid_common import BoundaryKind, ParticleBackend
from .grid_common import periodic_field as _periodic_field


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


class YeeGrid(PICGridBase):
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
        particle_backend: ParticleBackend = "numba",
    ) -> None:
        super().__init__(nx, ny, nz, dx, dy, dz, ng, boundary, eps0, particle_backend)
        self.mu0 = mu0
        self.anode_potential = anode_potential

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

        self._deposit_rho_partial: NDArray[np.float64] | None = None
        self._deposit_jx_partial: NDArray[np.float64] | None = None
        self._deposit_jy_partial: NDArray[np.float64] | None = None
        self._deposit_jz_partial: NDArray[np.float64] | None = None

        self.apply_boundaries()
        self._init_deposit_partials()

    def _init_deposit_partials(self) -> None:
        if not self._use_numba:
            return
        self._deposit_rho_partial = self._alloc_thread_partials(self.rho.size)
        self._deposit_jx_partial = self._alloc_thread_partials(self.Jx.size)
        self._deposit_jy_partial = self._alloc_thread_partials(self.Jy.size)
        self._deposit_jz_partial = self._alloc_thread_partials(self.Jz.size)

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

    def _node_aligned_axes(self, field: NDArray[np.floating]) -> tuple[bool, bool, bool]:
        """Per-axis flag: True where ``field`` carries ``n+1`` (node-aligned) interior points.

        Node-aligned axes (e.g. Ex along x, Bx along y/z) must wrap with period ``n``
        and keep their redundant node plane reconciled; cell-aligned axes (and all of
        ``rho``) keep the original period-``n`` cell wrap.
        """
        n_cells = (self.nx, self.ny, self.nz)
        ng = self.ng
        return (
            (field.shape[0] - 2 * ng) == n_cells[0] + 1,
            (field.shape[1] - 2 * ng) == n_cells[1] + 1,
            (field.shape[2] - 2 * ng) == n_cells[2] + 1,
        )

    def apply_boundaries(self) -> None:
        if self.boundary == "periodic":
            ng = self.ng
            # Node-aware periodic wrap: node-aligned axes use period n with the
            # redundant node plane kept equal (idempotent copy). rho is cell-aligned
            # on every axis, so it reduces to the original wrap. The Esirkepov current
            # seam (partial contributions on the two coincident faces) is summed once
            # in deposit_j_esirkepov_cic_batch; here J only needs an idempotent guard fill.
            for field in (self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz, self.rho,
                          self.Jx, self.Jy, self.Jz):
                _periodic_field(field, ng, node_aligned=self._node_aligned_axes(field))
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
        """Cloud-in-cell charge deposition onto cell-centered rho (Yee grid interior).

        rho is cell-centered (offset 0.5) so that the discrete divergence of the
        cell-centered/face Esirkepov current lands on rho's cell, i.e. discrete charge
        continuity and Gauss's law close on the Yee staggering.
        """
        pos = self.position_in_domain(np.array([x, y, z]))
        self._deposit_scalar(self.rho, pos, q / (self.dx * self.dy * self.dz), (0.5, 0.5, 0.5))

    def deposit_rho_cic_batch(
        self,
        positions: NDArray[np.floating],
        charges: NDArray[np.floating],
    ) -> None:
        """Batch CIC rho deposit for ``(N, 3)`` positions (same API as ElectrostaticGrid)."""
        pos = self.position_in_domain_batch(positions)
        charges_arr = np.asarray(charges, dtype=np.float64)
        if charges_arr.ndim != 1 or charges_arr.shape[0] != pos.shape[0]:
            raise ValueError("charges must have shape (N,) matching positions")
        values = charges_arr / (self.dx * self.dy * self.dz)

        if self._use_numba:
            try:
                from .pic_kernels import deposit_cic_periodic

                if self._deposit_rho_partial is None:
                    self._deposit_rho_partial = self._alloc_thread_partials(self.rho.size)
                if self._deposit_rho_partial is None:
                    raise ImportError("numba deposit partial buffer unavailable")
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
                    self._deposit_rho_partial,
                    0.5,
                    0.5,
                    0.5,
                )
                return
            except ImportError:
                pass

        self._deposit_scalar_batch(self.rho, pos, values, (0.5, 0.5, 0.5))

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

        Legacy non-conserving instantaneous ``q*v`` deposit. Prefer
        :meth:`deposit_j_esirkepov_cic_batch` for charge-conserving EM PIC.
        """
        pos = self.position_in_domain(np.array([x, y, z]))
        jx = q * vx / (self.dy * self.dz)
        jy = q * vy / (self.dx * self.dz)
        jz = q * vz / (self.dx * self.dy)

        self._deposit_component(self.Jx, pos, jx, (0.0, 0.5, 0.5))
        self._deposit_component(self.Jy, pos, jy, (0.5, 0.0, 0.5))
        self._deposit_component(self.Jz, pos, jz, (0.5, 0.5, 0.0))

    def deposit_j_esirkepov_cic_batch(
        self,
        pos_old: NDArray[np.floating],
        pos_new: NDArray[np.floating],
        charges: NDArray[np.floating],
        dt: float,
    ) -> None:
        """Charge-conserving Esirkepov CIC current deposit for ``(N, 3)`` trajectories.

        ``dt`` is the step over which ``pos_old -> pos_new`` occurred; the deposited J is a
        current density (charge flux / dt), consistent with the ``dt*J`` term in
        :meth:`update_e`. The longitudinal seam face (index ``n`` along a node-aligned axis)
        is left untouched here and is reconciled to face ``0`` by :meth:`apply_boundaries`
        (an idempotent copy); ``update_e`` reads the correct seam current on face ``0``
        directly, so no separate reduction is required during the field update.
        """
        old = self.position_in_domain_batch(pos_old)
        new = self.position_in_domain_batch(pos_new)
        if old.shape != new.shape:
            raise ValueError("pos_old and pos_new must have the same shape")
        charges_arr = np.asarray(charges, dtype=np.float64)
        if charges_arr.ndim != 1 or charges_arr.shape[0] != old.shape[0]:
            raise ValueError("charges must have shape (N,) matching positions")

        if self.boundary == "periodic":
            from .grid_common import unwrap_periodic_trajectory

            new = unwrap_periodic_trajectory(old, new, self.domain_lengths)

        if self._use_numba:
            try:
                from .pic_kernels import deposit_j_esirkepov_cic_periodic

                jx_partial = self._deposit_jx_partial
                jy_partial = self._deposit_jy_partial
                jz_partial = self._deposit_jz_partial
                if jx_partial is None or jy_partial is None or jz_partial is None:
                    self._init_deposit_partials()
                    jx_partial = self._deposit_jx_partial
                    jy_partial = self._deposit_jy_partial
                    jz_partial = self._deposit_jz_partial
                if jx_partial is None or jy_partial is None or jz_partial is None:
                    raise ImportError("numba partial buffers unavailable")
                deposit_j_esirkepov_cic_periodic(
                    self.Jx,
                    self.Jy,
                    self.Jz,
                    old,
                    new,
                    charges_arr,
                    dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    self.nx,
                    self.ny,
                    self.nz,
                    self.ng,
                    jx_partial,
                    jy_partial,
                    jz_partial,
                )
                return
            except ImportError:
                pass

        for i in range(old.shape[0]):
            self._deposit_j_esirkepov_scalar(
                float(old[i, 0]), float(old[i, 1]), float(old[i, 2]),
                float(new[i, 0]), float(new[i, 1]), float(new[i, 2]),
                float(charges_arr[i]), dt,
            )

    def _deposit_j_esirkepov_scalar(
        self,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
        charge: float,
        dt: float,
    ) -> None:
        """Single-particle charge-conserving deposit via the shared Esirkepov core."""
        from .pic_kernels import _esirkepov_core

        _esirkepov_core(
            self.Jx, self.Jy, self.Jz,
            x0, y0, z0, x1, y1, z1,
            charge, dt,
            self.dx, self.dy, self.dz,
            self.nx, self.ny, self.nz, self.ng,
        )

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
        """Batch staggered E gather; returns ``(N, 3)``."""
        pos = self.position_in_domain_batch(positions)
        if self._use_numba:
            try:
                from .pic_kernels import gather_e_yee_cic_periodic

                return gather_e_yee_cic_periodic(
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
            except ImportError:
                pass
        n = pos.shape[0]
        e_out = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            e_out[i] = self.gather_e_cic(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
        return e_out

    def gather_b_cic_batch(self, positions: NDArray[np.floating]) -> NDArray[np.float64]:
        """Batch staggered B gather; returns ``(N, 3)``."""
        pos = self.position_in_domain_batch(positions)
        if self._use_numba:
            try:
                from .pic_kernels import gather_b_yee_cic_periodic

                return gather_b_yee_cic_periodic(
                    self.Bx,
                    self.By,
                    self.Bz,
                    pos,
                    self.dx,
                    self.dy,
                    self.dz,
                    self.nx,
                    self.ny,
                    self.nz,
                    self.ng,
                )
            except ImportError:
                pass
        n = pos.shape[0]
        b_out = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            b_out[i] = self.gather_b_cic(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
        return b_out

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
