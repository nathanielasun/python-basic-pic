"""
Numba parallel kernels for electrostatic and Yee EM PIC particle operations.

Electrostatic kernels use grid-node CIC with ghost padding ``ng``.
Yee EM kernels use staggered component offsets matching :class:`grids.YeeGrid`.
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
    def _logical_yee_index(logical: int, n_points: int) -> int:
        return int(logical % n_points)

    @njit(cache=True)
    def _yee_n_points(n_cells: int, offset_half: bool) -> int:
        return n_cells if offset_half else n_cells + 1

    @njit(cache=True)
    def _flat_index(ii: int, jj: int, kk: int, ny_tot: int, nz_tot: int) -> int:
        return ii * ny_tot * nz_tot + jj * nz_tot + kk

    @njit(cache=True)
    def _gather_yee_component(
        field: np.ndarray,
        px: float,
        py: float,
        pz: float,
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> float:
        gx = px / dx - ox
        gy = py / dy - oy
        gz = pz / dz - oz
        i0x = int(np.floor(gx))
        i0y = int(np.floor(gy))
        i0z = int(np.floor(gz))
        fx = gx - i0x
        fy = gy - i0y
        fz = gz - i0z

        npx = _yee_n_points(nx, ox == 0.5)
        npy = _yee_n_points(ny, oy == 0.5)
        npz = _yee_n_points(nz, oz == 0.5)

        value = 0.0
        for di in range(2):
            wi = (1.0 - fx) if di == 0 else fx
            ii = _logical_yee_index(i0x + di, npx) + ng
            for dj in range(2):
                wj = (1.0 - fy) if dj == 0 else fy
                jj = _logical_yee_index(i0y + dj, npy) + ng
                for dk in range(2):
                    wk = (1.0 - fz) if dk == 0 else fz
                    kk = _logical_yee_index(i0z + dk, npz) + ng
                    value += field[ii, jj, kk] * wi * wj * wk
        return value

    @njit(cache=True)
    def _deposit_yee_component(
        field: np.ndarray,
        px: float,
        py: float,
        pz: float,
        value: float,
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> None:
        gx = px / dx - ox
        gy = py / dy - oy
        gz = pz / dz - oz
        i0x = int(np.floor(gx))
        i0y = int(np.floor(gy))
        i0z = int(np.floor(gz))
        fx = gx - i0x
        fy = gy - i0y
        fz = gz - i0z

        npx = _yee_n_points(nx, ox == 0.5)
        npy = _yee_n_points(ny, oy == 0.5)
        npz = _yee_n_points(nz, oz == 0.5)

        for di in range(2):
            wi = (1.0 - fx) if di == 0 else fx
            ii = _logical_yee_index(i0x + di, npx) + ng
            for dj in range(2):
                wj = (1.0 - fy) if dj == 0 else fy
                jj = _logical_yee_index(i0y + dj, npy) + ng
                for dk in range(2):
                    wk = (1.0 - fz) if dk == 0 else fz
                    kk = _logical_yee_index(i0z + dk, npz) + ng
                    field[ii, jj, kk] += value * wi * wj * wk

    @njit(cache=True)
    def _add_jx_yz_cic(
        jx: np.ndarray,
        iface: int,
        jy0: int,
        jz0: int,
        fy: float,
        fz: float,
        val: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> None:
        ii = _logical_yee_index(iface, nx + 1) + ng
        for dj in range(2):
            wj = (1.0 - fy) if dj == 0 else fy
            jj = _logical_yee_index(jy0 + dj, ny) + ng
            for dk in range(2):
                wk = (1.0 - fz) if dk == 0 else fz
                kk = _logical_yee_index(jz0 + dk, nz) + ng
                jx[ii, jj, kk] += val * wj * wk

    @njit(cache=True)
    def _add_jy_xz_cic(
        jy: np.ndarray,
        jiface: int,
        jx0: int,
        jz0: int,
        fx: float,
        fz: float,
        val: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> None:
        jj = _logical_yee_index(jiface, ny + 1) + ng
        for di in range(2):
            wi = (1.0 - fx) if di == 0 else fx
            ii = _logical_yee_index(jx0 + di, nx) + ng
            for dk in range(2):
                wk = (1.0 - fz) if dk == 0 else fz
                kk = _logical_yee_index(jz0 + dk, nz) + ng
                jy[ii, jj, kk] += val * wi * wk

    @njit(cache=True)
    def _add_jz_xy_cic(
        jz: np.ndarray,
        kface: int,
        jx0: int,
        jy0: int,
        fx: float,
        fy: float,
        val: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> None:
        kk = _logical_yee_index(kface, nz + 1) + ng
        for di in range(2):
            wi = (1.0 - fx) if di == 0 else fx
            ii = _logical_yee_index(jx0 + di, nx) + ng
            for dj in range(2):
                wj = (1.0 - fy) if dj == 0 else fy
                jj = _logical_yee_index(jy0 + dj, ny) + ng
                jz[ii, jj, kk] += val * wi * wj

    @njit(cache=True)
    def _deposit_jx_esirkepov(
        jx: np.ndarray,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
        charge: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> None:
        ym = 0.5 * (y0 + y1)
        zm = 0.5 * (z0 + z1)
        gy = ym / dy - 0.5
        gz = zm / dz - 0.5
        jy0 = int(np.floor(gy))
        jz0 = int(np.floor(gz))
        fy = gy - jy0
        fz = gz - jz0
        inv_dyz = 1.0 / (dy * dz)

        gx0 = x0 / dx
        gx1 = x1 / dx
        if gx1 > gx0:
            remaining = gx1 - gx0
            idx = int(np.floor(gx0))
            frac = gx0 - idx
            while remaining > 1e-15:
                if frac + remaining < 1.0:
                    contrib = remaining
                    remaining = 0.0
                else:
                    contrib = 1.0 - frac
                    remaining -= contrib
                val = charge * contrib * dx * inv_dyz
                _add_jx_yz_cic(jx, idx + 1, jy0, jz0, fy, fz, val, nx, ny, nz, ng)
                idx += 1
                frac = 0.0
        elif gx1 < gx0:
            remaining = gx0 - gx1
            idx = int(np.floor(gx0))
            frac = gx0 - idx
            if frac < 1e-15:
                idx -= 1
                frac = 1.0
            while remaining > 1e-15:
                if frac - remaining > 0.0:
                    contrib = remaining
                    remaining = 0.0
                else:
                    contrib = frac
                    remaining -= contrib
                val = charge * contrib * dx * inv_dyz
                _add_jx_yz_cic(jx, idx + 1, jy0, jz0, fy, fz, val, nx, ny, nz, ng)
                idx -= 1
                frac = 1.0

    @njit(cache=True)
    def _deposit_jy_esirkepov(
        jy: np.ndarray,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
        charge: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> None:
        xm = 0.5 * (x0 + x1)
        zm = 0.5 * (z0 + z1)
        gx = xm / dx - 0.5
        gz = zm / dz - 0.5
        jx0 = int(np.floor(gx))
        jz0 = int(np.floor(gz))
        fx = gx - jx0
        fz = gz - jz0
        inv_dxz = 1.0 / (dx * dz)

        gy0 = y0 / dy
        gy1 = y1 / dy
        if gy1 > gy0:
            remaining = gy1 - gy0
            idx = int(np.floor(gy0))
            frac = gy0 - idx
            while remaining > 1e-15:
                if frac + remaining < 1.0:
                    contrib = remaining
                    remaining = 0.0
                else:
                    contrib = 1.0 - frac
                    remaining -= contrib
                val = charge * contrib * dy * inv_dxz
                _add_jy_xz_cic(jy, idx + 1, jx0, jz0, fx, fz, val, nx, ny, nz, ng)
                idx += 1
                frac = 0.0
        elif gy1 < gy0:
            remaining = gy0 - gy1
            idx = int(np.floor(gy0))
            frac = gy0 - idx
            if frac < 1e-15:
                idx -= 1
                frac = 1.0
            while remaining > 1e-15:
                if frac - remaining > 0.0:
                    contrib = remaining
                    remaining = 0.0
                else:
                    contrib = frac
                    remaining -= contrib
                val = charge * contrib * dy * inv_dxz
                _add_jy_xz_cic(jy, idx + 1, jx0, jz0, fx, fz, val, nx, ny, nz, ng)
                idx -= 1
                frac = 1.0

    @njit(cache=True)
    def _deposit_jz_esirkepov(
        jz: np.ndarray,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
        charge: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> None:
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        gx = xm / dx - 0.5
        gy = ym / dy - 0.5
        jx0 = int(np.floor(gx))
        jy0 = int(np.floor(gy))
        fx = gx - jx0
        fy = gy - jy0
        inv_dxy = 1.0 / (dx * dy)

        gz0 = z0 / dz
        gz1 = z1 / dz
        if gz1 > gz0:
            remaining = gz1 - gz0
            idx = int(np.floor(gz0))
            frac = gz0 - idx
            while remaining > 1e-15:
                if frac + remaining < 1.0:
                    contrib = remaining
                    remaining = 0.0
                else:
                    contrib = 1.0 - frac
                    remaining -= contrib
                val = charge * contrib * dz * inv_dxy
                _add_jz_xy_cic(jz, idx + 1, jx0, jy0, fx, fy, val, nx, ny, nz, ng)
                idx += 1
                frac = 0.0
        elif gz1 < gz0:
            remaining = gz0 - gz1
            idx = int(np.floor(gz0))
            frac = gz0 - idx
            if frac < 1e-15:
                idx -= 1
                frac = 1.0
            while remaining > 1e-15:
                if frac - remaining > 0.0:
                    contrib = remaining
                    remaining = 0.0
                else:
                    contrib = frac
                    remaining -= contrib
                val = charge * contrib * dz * inv_dxy
                _add_jz_xy_cic(jz, idx + 1, jx0, jy0, fx, fy, val, nx, ny, nz, ng)
                idx -= 1
                frac = 1.0

    @njit(cache=True)
    def _deposit_particles_esirkepov_to_flat(
        jx_flat: np.ndarray,
        jy_flat: np.ndarray,
        jz_flat: np.ndarray,
        jx_shape: tuple,
        jy_shape: tuple,
        jz_shape: tuple,
        pos_old: np.ndarray,
        pos_new: np.ndarray,
        charges: np.ndarray,
        start: int,
        end: int,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> None:
        jx = jx_flat.reshape(jx_shape)
        jy = jy_flat.reshape(jy_shape)
        jz = jz_flat.reshape(jz_shape)
        for p in range(start, end):
            q = charges[p]
            x0, y0, z0 = pos_old[p, 0], pos_old[p, 1], pos_old[p, 2]
            x1, y1, z1 = pos_new[p, 0], pos_new[p, 1], pos_new[p, 2]
            _deposit_jx_esirkepov(jx, x0, y0, z0, x1, y1, z1, q, dx, dy, dz, nx, ny, nz, ng)
            _deposit_jy_esirkepov(jy, x0, y0, z0, x1, y1, z1, q, dx, dy, dz, nx, ny, nz, ng)
            _deposit_jz_esirkepov(jz, x0, y0, z0, x1, y1, z1, q, dx, dy, dz, nx, ny, nz, ng)

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
    def gather_e_yee_cic_periodic(
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
        """Staggered Yee CIC gather of E at particle positions."""
        n_particles = positions.shape[0]
        e_out = np.empty((n_particles, 3), dtype=np.float64)
        for p in prange(n_particles):
            px = positions[p, 0]
            py = positions[p, 1]
            pz = positions[p, 2]
            e_out[p, 0] = _gather_yee_component(
                ex, px, py, pz, 0.0, 0.5, 0.5, dx, dy, dz, nx, ny, nz, ng
            )
            e_out[p, 1] = _gather_yee_component(
                ey, px, py, pz, 0.5, 0.0, 0.5, dx, dy, dz, nx, ny, nz, ng
            )
            e_out[p, 2] = _gather_yee_component(
                ez, px, py, pz, 0.5, 0.5, 0.0, dx, dy, dz, nx, ny, nz, ng
            )
        return e_out

    @njit(parallel=True, cache=True)
    def gather_b_yee_cic_periodic(
        bx: np.ndarray,
        by: np.ndarray,
        bz: np.ndarray,
        positions: np.ndarray,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
    ) -> np.ndarray:
        """Staggered Yee CIC gather of B at particle positions."""
        n_particles = positions.shape[0]
        b_out = np.empty((n_particles, 3), dtype=np.float64)
        for p in prange(n_particles):
            px = positions[p, 0]
            py = positions[p, 1]
            pz = positions[p, 2]
            b_out[p, 0] = _gather_yee_component(
                bx, px, py, pz, 0.5, 0.0, 0.0, dx, dy, dz, nx, ny, nz, ng
            )
            b_out[p, 1] = _gather_yee_component(
                by, px, py, pz, 0.0, 0.5, 0.0, dx, dy, dz, nx, ny, nz, ng
            )
            b_out[p, 2] = _gather_yee_component(
                bz, px, py, pz, 0.0, 0.0, 0.5, dx, dy, dz, nx, ny, nz, ng
            )
        return b_out

    @njit(parallel=True, cache=True)
    def deposit_j_esirkepov_cic_periodic(
        jx: np.ndarray,
        jy: np.ndarray,
        jz: np.ndarray,
        pos_old: np.ndarray,
        pos_new: np.ndarray,
        charges: np.ndarray,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
        ng: int,
        partial_jx: np.ndarray,
        partial_jy: np.ndarray,
        partial_jz: np.ndarray,
    ) -> None:
        """Charge-conserving Esirkepov CIC current deposit (periodic, in-place)."""
        jx_flat = jx.ravel()
        jy_flat = jy.ravel()
        jz_flat = jz.ravel()
        n_particles = pos_old.shape[0]
        n_threads = partial_jx.shape[0]
        chunk = (n_particles + n_threads - 1) // n_threads
        jx_shape = jx.shape
        jy_shape = jy.shape
        jz_shape = jz.shape

        for t in range(n_threads):
            for i in range(jx_flat.size):
                partial_jx[t, i] = 0.0
            for i in range(jy_flat.size):
                partial_jy[t, i] = 0.0
            for i in range(jz_flat.size):
                partial_jz[t, i] = 0.0

        for t in prange(n_threads):
            start = t * chunk
            end = start + chunk
            if end > n_particles:
                end = n_particles
            if start < end:
                _deposit_particles_esirkepov_to_flat(
                    partial_jx[t],
                    partial_jy[t],
                    partial_jz[t],
                    jx_shape,
                    jy_shape,
                    jz_shape,
                    pos_old,
                    pos_new,
                    charges,
                    start,
                    end,
                    dx,
                    dy,
                    dz,
                    nx,
                    ny,
                    nz,
                    ng,
                )

        for i in prange(jx_flat.size):
            acc = 0.0
            for t in range(n_threads):
                acc += partial_jx[t, i]
            jx_flat[i] += acc

        for i in prange(jy_flat.size):
            acc = 0.0
            for t in range(n_threads):
                acc += partial_jy[t, i]
            jy_flat[i] += acc

        for i in prange(jz_flat.size):
            acc = 0.0
            for t in range(n_threads):
                acc += partial_jz[t, i]
            jz_flat[i] += acc

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
    shape_rho = (nx + 2 * ng, ny + 2 * ng, nz + 2 * ng)
    rho = np.zeros(shape_rho, dtype=np.float64)
    ex_node = np.zeros(shape_rho, dtype=np.float64)
    ey_node = np.zeros(shape_rho, dtype=np.float64)
    ez_node = np.zeros(shape_rho, dtype=np.float64)
    ex = np.zeros((nx + 1 + 2 * ng, ny + 2 * ng, nz + 2 * ng), dtype=np.float64)
    ey = np.zeros((nx + 2 * ng, ny + 1 + 2 * ng, nz + 2 * ng), dtype=np.float64)
    ez = np.zeros((nx + 2 * ng, ny + 2 * ng, nz + 1 + 2 * ng), dtype=np.float64)
    bx = np.zeros((nx + 2 * ng, ny + 1 + 2 * ng, nz + 1 + 2 * ng), dtype=np.float64)
    by = np.zeros((nx + 1 + 2 * ng, ny + 2 * ng, nz + 1 + 2 * ng), dtype=np.float64)
    bz = np.zeros((nx + 1 + 2 * ng, ny + 1 + 2 * ng, nz + 2 * ng), dtype=np.float64)
    jx = np.zeros_like(ex)
    jy = np.zeros_like(ey)
    jz = np.zeros_like(ez)
    pos = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], dtype=np.float64)
    pos_new = pos + 0.1
    values = np.array([1.0, -1.0], dtype=np.float64)
    charges = np.array([1.0, -1.0], dtype=np.float64)
    vel = np.zeros((2, 3), dtype=np.float64)
    efield = np.ones((2, 3), dtype=np.float64)
    dx = dy = dz = 1.0
    lx = ly = lz = 4.0
    n_threads = get_num_threads()

    partial = np.zeros((n_threads, rho.size), dtype=np.float64)
    deposit_cic_periodic(rho, pos, values, dx, dy, dz, nx, ny, nz, ng, partial)
    gather_e_cic_periodic(ex_node, ey_node, ez_node, pos, dx, dy, dz, nx, ny, nz, ng)
    gather_e_yee_cic_periodic(ex, ey, ez, pos, dx, dy, dz, nx, ny, nz, ng)
    gather_b_yee_cic_periodic(bx, by, bz, pos, dx, dy, dz, nx, ny, nz, ng)
    partial_jx = np.zeros((n_threads, jx.size), dtype=np.float64)
    partial_jy = np.zeros((n_threads, jy.size), dtype=np.float64)
    partial_jz = np.zeros((n_threads, jz.size), dtype=np.float64)
    deposit_j_esirkepov_cic_periodic(
        jx, jy, jz, pos, pos_new, charges, dx, dy, dz, nx, ny, nz, ng,
        partial_jx, partial_jy, partial_jz,
    )
    electric_kick_b0(vel, efield, -1.0, 1e-12)
    wrap_positions_periodic(pos, lx, ly, lz)

    try:
        from Pushers import Pushers

        bfield = np.zeros((2, 3), dtype=np.float64)
        Pushers.push_batch("higuera_cary", vel, efield, bfield, 1.0, 1.0, 1e-12, c=1.0)
    except ImportError:
        pass
