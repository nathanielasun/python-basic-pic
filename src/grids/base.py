"""
Shared base class for PIC field grids.

``PICGridBase`` collects the geometry, particle-backend resolution, periodic /
wall position mapping, and cloud-in-cell scatter/gather primitives common to the
node-centered :class:`grids.ElectrostaticGrid` and the staggered
:class:`grids.YeeGrid`. Concrete grids subclass it and add their own field
arrays, boundary handling, and field solvers/updates.

Both grids index a CIC field as a periodic lattice of ``n_cells`` points per
axis with ``ng`` guard cells on each side, so the cell-centered scatter/gather
here serves the electrostatic node fields and the Yee cell-centered ``rho``
alike. Staggered (face/edge) component handling lives in ``YeeGrid``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .grid_common import (
    BoundaryKind,
    ParticleBackend,
    clamp_position,
    wrap_position,
)

Offsets = tuple[float, float, float]


class PICGridBase:
    """Geometry, particle backend, position mapping, and CIC kernels for PIC grids."""

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        dy: float,
        dz: float,
        ng: int,
        boundary: BoundaryKind,
        eps0: float,
        particle_backend: ParticleBackend,
    ) -> None:
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("nx, ny, nz must be positive")
        if ng < 1:
            raise ValueError("ng must be at least 1 for guard cells")
        boundary = boundary.lower()  # type: ignore[assignment]
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

        self.Lx = nx * dx
        self.Ly = ny * dy
        self.Lz = nz * dz

        self._particle_backend: ParticleBackend = self._resolve_particle_backend(particle_backend)

    # ------------------------------------------------------------------ geometry

    @property
    def domain_lengths(self) -> tuple[float, float, float]:
        return (self.Lx, self.Ly, self.Lz)

    @property
    def cell_sizes(self) -> tuple[float, float, float]:
        return (self.dx, self.dy, self.dz)

    @property
    def n_cells(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    # --------------------------------------------------------- particle backend

    @property
    def particle_backend(self) -> ParticleBackend:
        return self._particle_backend

    @property
    def _use_numba(self) -> bool:
        """True when periodic Numba particle kernels should be used."""
        return self._particle_backend == "numba" and self.boundary == "periodic"

    @staticmethod
    def _resolve_particle_backend(requested: ParticleBackend) -> ParticleBackend:
        """Fall back to ``numpy`` unless Numba is importable and warmed up."""
        if requested != "numba":
            return "numpy"
        try:
            from .pic_kernels import HAS_NUMBA, warmup_kernels

            if not HAS_NUMBA:
                return "numpy"
            warmup_kernels()
            return "numba"
        except ImportError:
            return "numpy"

    def _alloc_thread_partials(self, size: int) -> NDArray[np.float64] | None:
        """Per-thread scratch buffer for parallel scatter-add, or None without Numba."""
        try:
            from .pic_kernels import get_num_threads

            return np.zeros((get_num_threads(), size), dtype=np.float64)
        except ImportError:
            return None

    # --------------------------------------------------------- position mapping

    def position_in_domain(self, pos: NDArray[np.floating]) -> NDArray[np.float64]:
        """Map one position into the physical domain according to the active boundary."""
        if self.boundary == "periodic":
            return wrap_position(pos, self.domain_lengths)
        return clamp_position(pos, self.domain_lengths, self.cell_sizes)

    def position_in_domain_batch(
        self,
        positions: NDArray[np.floating],
        *,
        in_place: bool = False,
    ) -> NDArray[np.float64]:
        """Map ``(N, 3)`` positions into the physical domain for the active boundary."""
        if in_place:
            pos = np.asarray(positions, dtype=np.float64)
            if not pos.flags.c_contiguous:
                raise ValueError("in_place requires a contiguous float64 array")
        else:
            pos = np.asarray(positions, dtype=np.float64).copy()
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.boundary == "periodic":
            pos %= np.array(self.domain_lengths, dtype=np.float64)
            return pos
        return clamp_position(pos, self.domain_lengths, self.cell_sizes)

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

    # ------------------------------------------------------- CIC index helpers

    def _grid_coords(
        self,
        pos: NDArray[np.floating],
        offsets: Offsets,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        """Logical coordinate, lower node index, and fractional offset for one position."""
        g = np.asarray(pos, dtype=np.float64) / np.array(self.cell_sizes) - np.array(offsets)
        i0 = np.floor(g).astype(np.int64)
        return g, i0, g - i0

    def _grid_coords_batch(
        self,
        positions: NDArray[np.floating],
        offsets: Offsets,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        """Vectorized :meth:`_grid_coords` for ``(N, 3)`` positions (1D delegates back)."""
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim == 1:
            return self._grid_coords(pos, offsets)
        g = pos / np.array(self.cell_sizes) - np.array(offsets)
        i0 = np.floor(g).astype(np.int64)
        return g, i0, g - i0

    def _logical_index(self, logical: int, axis: int) -> int:
        """Wrap (periodic) or clamp (wall) a cell index onto ``[0, n_cells)``."""
        n_cells = self.n_cells[axis]
        if self.boundary == "periodic":
            return int(logical % n_cells)
        return int(np.clip(logical, 0, n_cells - 1))

    def _logical_index_batch(self, logical: NDArray[np.int64], axis: int) -> NDArray[np.int64]:
        n_cells = self.n_cells[axis]
        if self.boundary == "periodic":
            return logical % n_cells
        return np.clip(logical, 0, n_cells - 1)

    # ------------------------------------------------------- CIC scatter/gather

    def _deposit_scalar(
        self,
        field: NDArray[np.floating],
        pos: NDArray[np.floating],
        value: float,
        offsets: Offsets,
    ) -> None:
        """Cloud-in-cell scatter of a scalar value at one position."""
        _, i0, f = self._grid_coords(pos, offsets)
        ng = self.ng
        for di in (0, 1):
            wi = (1.0 - f[0]) if di == 0 else f[0]
            ii = self._logical_index(i0[0] + di, 0) + ng
            for dj in (0, 1):
                wj = (1.0 - f[1]) if dj == 0 else f[1]
                jj = self._logical_index(i0[1] + dj, 1) + ng
                for dk in (0, 1):
                    wk = (1.0 - f[2]) if dk == 0 else f[2]
                    kk = self._logical_index(i0[2] + dk, 2) + ng
                    field[ii, jj, kk] += value * wi * wj * wk

    def _deposit_scalar_batch(
        self,
        field: NDArray[np.floating],
        positions: NDArray[np.floating],
        values: NDArray[np.floating],
        offsets: Offsets,
    ) -> None:
        """Vectorized cloud-in-cell scatter-add for many particles."""
        _, i0, f = self._grid_coords_batch(positions, offsets)
        ng = self.ng
        ny_tot, nz_tot = field.shape[1], field.shape[2]
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
                    np.add.at(flat, ii * ny_tot * nz_tot + jj * nz_tot + kk, values * wi * wj * wk)

    def _gather_scalar(
        self,
        field: NDArray[np.floating],
        pos: NDArray[np.floating],
        offsets: Offsets,
    ) -> float:
        """Trilinear cloud-in-cell gather of a scalar field at one position."""
        _, i0, f = self._grid_coords(pos, offsets)
        ng = self.ng
        value = 0.0
        for di in (0, 1):
            wi = (1.0 - f[0]) if di == 0 else f[0]
            ii = self._logical_index(i0[0] + di, 0) + ng
            for dj in (0, 1):
                wj = (1.0 - f[1]) if dj == 0 else f[1]
                jj = self._logical_index(i0[1] + dj, 1) + ng
                for dk in (0, 1):
                    wk = (1.0 - f[2]) if dk == 0 else f[2]
                    kk = self._logical_index(i0[2] + dk, 2) + ng
                    value += field[ii, jj, kk] * wi * wj * wk
        return float(value)

    def _gather_scalar_batch(
        self,
        field: NDArray[np.floating],
        positions: NDArray[np.floating],
        offsets: Offsets,
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
