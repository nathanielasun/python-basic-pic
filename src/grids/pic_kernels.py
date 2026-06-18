"""
Numba parallel kernels for electrostatic PIC particle operations.

All kernels assume periodic boundaries and grid-node CIC interpolation
with ghost padding ``ng`` matching :class:`grids.ElectrostaticGrid`.
"""

from __future__ import annotations

import numpy as np

try:
    import numba
    from numba import get_num_threads, njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    numba = None  # type: ignore[assignment]

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args, **kwargs):  # type: ignore[misc]
        return range(*args)

    def get_num_threads() -> int:  # type: ignore[misc]
        return 1


if HAS_NUMBA:

    @njit(cache=True)
    def _logical_index_periodic(logical: int, n_cells: int) -> int:
        return int(logical % n_cells)

    @njit(cache=True)
    def _flat_index(ii: int, jj: int, kk: int, ny_tot: int, nz_tot: int) -> int:
        return ii * ny_tot * nz_tot + jj * nz_tot + kk

    @njit(cache=True)
    def _deposit_particles_to_flat(
        flat: np.ndarray,
        positions: np.ndarray,
        values: np.ndarray,
        start: int,
        end: int,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
        ny_tot: int,
        nz_tot: int,
    ) -> None:
        for p in range(start, end):
            gx = positions[p, 0] / dx
            gy = positions[p, 1] / dy
            gz = positions[p, 2] / dz
            i0x = int(np.floor(gx))
            i0y = int(np.floor(gy))
            i0z = int(np.floor(gz))
            fx = gx - i0x
            fy = gy - i0y
            fz = gz - i0z
            val = values[p]

            for di in range(2):
                wi = (1.0 - fx) if di == 0 else fx
                ii = _logical_index_periodic(i0x + di, nx) + ng
                for dj in range(2):
                    wj = (1.0 - fy) if dj == 0 else fy
                    jj = _logical_index_periodic(i0y + dj, ny) + ng
                    for dk in range(2):
                        wk = (1.0 - fz) if dk == 0 else fz
                        kk = _logical_index_periodic(i0z + dk, nz) + ng
                        idx = _flat_index(ii, jj, kk, ny_tot, nz_tot)
                        flat[idx] += val * wi * wj * wk

    @njit(parallel=True, cache=True)
    def deposit_cic_periodic(
        rho: np.ndarray,
        positions: np.ndarray,
        values: np.ndarray,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
        partial: np.ndarray,
    ) -> None:
        """Scatter-add CIC weights onto ``rho`` (in-place) for periodic boundaries."""
        ny_tot = rho.shape[1]
        nz_tot = rho.shape[2]
        flat = rho.ravel()
        n_particles = positions.shape[0]
        n_threads = partial.shape[0]
        chunk = (n_particles + n_threads - 1) // n_threads

        for t in range(n_threads):
            for i in range(flat.size):
                partial[t, i] = 0.0

        for t in prange(n_threads):
            start = t * chunk
            end = start + chunk
            if end > n_particles:
                end = n_particles
            if start < end:
                _deposit_particles_to_flat(
                    partial[t],
                    positions,
                    values,
                    start,
                    end,
                    dx,
                    dy,
                    dz,
                    nx,
                    ny,
                    nz,
                    ng,
                    ny_tot,
                    nz_tot,
                )

        for i in prange(flat.size):
            acc = 0.0
            for t in range(n_threads):
                acc += partial[t, i]
            flat[i] += acc

    @njit(parallel=True, cache=True)
    def gather_e_cic_periodic(
        ex: np.ndarray,
        ey: np.ndarray,
        ez: np.ndarray,
        positions: np.ndarray,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> np.ndarray:
        """Trilinear gather of E = (Ex, Ey, Ez) at particle positions."""
        n_particles = positions.shape[0]
        e_out = np.empty((n_particles, 3), dtype=np.float64)

        for p in prange(n_particles):
            gx = positions[p, 0] / dx
            gy = positions[p, 1] / dy
            gz = positions[p, 2] / dz
            i0x = int(np.floor(gx))
            i0y = int(np.floor(gy))
            i0z = int(np.floor(gz))
            fx = gx - i0x
            fy = gy - i0y
            fz = gz - i0z

            ex_val = 0.0
            ey_val = 0.0
            ez_val = 0.0
            for di in range(2):
                wi = (1.0 - fx) if di == 0 else fx
                ii = _logical_index_periodic(i0x + di, nx) + ng
                for dj in range(2):
                    wj = (1.0 - fy) if dj == 0 else fy
                    jj = _logical_index_periodic(i0y + dj, ny) + ng
                    for dk in range(2):
                        wk = (1.0 - fz) if dk == 0 else fz
                        kk = _logical_index_periodic(i0z + dk, nz) + ng
                        w = wi * wj * wk
                        ex_val += ex[ii, jj, kk] * w
                        ey_val += ey[ii, jj, kk] * w
                        ez_val += ez[ii, jj, kk] * w

            e_out[p, 0] = ex_val
            e_out[p, 1] = ey_val
            e_out[p, 2] = ez_val

        return e_out

    @njit(parallel=True, cache=True)
    def electric_kick_b0(
        vel: np.ndarray,
        efield: np.ndarray,
        q_over_m: float,
        dt: float,
    ) -> None:
        """Non-relativistic Boris push with B=0: v += (q/m)*dt*E (in-place)."""
        n_particles = vel.shape[0]
        for p in prange(n_particles):
            vel[p, 0] += q_over_m * dt * efield[p, 0]
            vel[p, 1] += q_over_m * dt * efield[p, 1]
            vel[p, 2] += q_over_m * dt * efield[p, 2]

    @njit(parallel=True, cache=True)
    def wrap_positions_periodic(
        pos: np.ndarray,
        lx: float,
        ly: float,
        lz: float,
    ) -> None:
        """Periodic wrap of particle positions (in-place)."""
        n_particles = pos.shape[0]
        for p in prange(n_particles):
            pos[p, 0] %= lx
            pos[p, 1] %= ly
            pos[p, 2] %= lz


def warmup_kernels() -> None:
    """JIT-compile kernels on dummy data so first timestep is not penalized."""
    if not HAS_NUMBA:
        return

    ng = 1
    nx, ny, nz = 4, 4, 4
    shape = (nx + 2 * ng, ny + 2 * ng, nz + 2 * ng)
    rho = np.zeros(shape, dtype=np.float64)
    ex = np.zeros(shape, dtype=np.float64)
    ey = np.zeros(shape, dtype=np.float64)
    ez = np.zeros(shape, dtype=np.float64)
    pos = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], dtype=np.float64)
    values = np.array([1.0, -1.0], dtype=np.float64)
    vel = np.zeros((2, 3), dtype=np.float64)
    efield = np.ones((2, 3), dtype=np.float64)
    dx = dy = dz = 1.0
    lx = ly = lz = 4.0

    partial = np.zeros((get_num_threads(), rho.size), dtype=np.float64)
    deposit_cic_periodic(rho, pos, values, dx, dy, dz, nx, ny, nz, ng, partial)
    gather_e_cic_periodic(ex, ey, ez, pos, dx, dy, dz, nx, ny, nz, ng)
    electric_kick_b0(vel, efield, -1.0, 1e-12)
    wrap_positions_periodic(pos, lx, ly, lz)
