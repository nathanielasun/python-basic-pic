"""
Numba parallel kernels for electrostatic and Yee EM PIC particle operations.

Electrostatic kernels use grid-node CIC with ghost padding ``ng``.
Yee EM kernels use staggered component offsets matching :class:`grids.YeeGrid`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

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


def _esirkepov_core(
    jx: np.ndarray,
    jy: np.ndarray,
    jz: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    q: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
    ng: int,
) -> None:
    """First-order Esirkepov charge-conserving current deposit for one trajectory.

    The (already minimum-image-unwrapped) move is split at integer crossings of the
    cell-centered coordinates ``u = x/dx - 0.5`` so each sub-segment stays within one
    cell on every axis. Each sub-segment deposits the exact first-order charge-conserving
    current, which satisfies the discrete continuity equation
    ``(rho^{n+1}-rho^n)/dt + div(J) = 0`` to machine precision against the cell-centered
    rho (offset 0.5). Key properties vs the previous scheme:

    - longitudinal current for cell ``I = floor(x/dx - 0.5)`` lands on x-face ``I+1``
      (cell-centered staggering), period-``n`` wrapped so the seam face maps onto face 0;
    - transverse CIC weights use each sub-segment's own midpoint (exact for linear shapes);
    - the displacement carries the sign of motion (correct for both directions);
    - the explicit ``1/dt`` makes J a current density consistent with ``update_e``'s ``dt*J``.

    Pure-Python and Numba paths share this single implementation, so they agree exactly.
    """
    qdt = q / dt
    inv_dyz = 1.0 / (dy * dz)
    inv_dxz = 1.0 / (dx * dz)
    inv_dxy = 1.0 / (dx * dy)

    ua = x0 / dx - 0.5
    va = y0 / dy - 0.5
    wa = z0 / dz - 0.5
    du = (x1 / dx - 0.5) - ua
    dv = (y1 / dy - 0.5) - va
    dw = (z1 / dz - 0.5) - wa

    eps = 1e-12
    t = 0.0
    while t < 1.0 - eps:
        # Nearest next integer crossing in u, v, or w after t (along the motion direction).
        t_next = 1.0
        if du > eps or du < -eps:
            cur = ua + du * t
            nxt = (np.floor(cur + eps) + 1.0) if du > 0.0 else (np.ceil(cur - eps) - 1.0)
            tc = (nxt - ua) / du
            if tc > t + eps and tc < t_next:
                t_next = tc
        if dv > eps or dv < -eps:
            cur = va + dv * t
            nxt = (np.floor(cur + eps) + 1.0) if dv > 0.0 else (np.ceil(cur - eps) - 1.0)
            tc = (nxt - va) / dv
            if tc > t + eps and tc < t_next:
                t_next = tc
        if dw > eps or dw < -eps:
            cur = wa + dw * t
            nxt = (np.floor(cur + eps) + 1.0) if dw > 0.0 else (np.ceil(cur - eps) - 1.0)
            tc = (nxt - wa) / dw
            if tc > t + eps and tc < t_next:
                t_next = tc
        if t_next > 1.0:
            t_next = 1.0

        ua_s = ua + du * t
        va_s = va + dv * t
        wa_s = wa + dw * t
        seg_du = du * (t_next - t)
        seg_dv = dv * (t_next - t)
        seg_dw = dw * (t_next - t)

        um = ua_s + 0.5 * seg_du
        vm = va_s + 0.5 * seg_dv
        wm = wa_s + 0.5 * seg_dw
        ic = int(np.floor(um))
        jc = int(np.floor(vm))
        kc = int(np.floor(wm))
        fu = um - ic
        fv = vm - jc
        fw = wm - kc

        # Each current component's transverse weight is a product of two linear shapes
        # that both vary along the sub-segment. The exact time-integral of that product
        # is (midpoint product) + (slope_a * slope_b)/12 -- the Esirkepov bilinear cross
        # term that the bare midpoint weighting omits. The slopes are +-Δ on the two
        # transverse cells, so the correction is sign(di)*sign(dj)*Δ_a*Δ_b/12.
        valx = qdt * seg_du * inv_dyz
        cross_x = seg_dv * seg_dw / 12.0
        ix = (ic + 1) % nx + ng
        for dj in range(2):
            wj = (1.0 - fv) if dj == 0 else fv
            sj = -1.0 if dj == 0 else 1.0
            jj = (jc + dj) % ny + ng
            for dk in range(2):
                wk = (1.0 - fw) if dk == 0 else fw
                sk = -1.0 if dk == 0 else 1.0
                kk = (kc + dk) % nz + ng
                jx[ix, jj, kk] += valx * (wj * wk + sj * sk * cross_x)

        valy = qdt * seg_dv * inv_dxz
        cross_y = seg_du * seg_dw / 12.0
        jyf = (jc + 1) % ny + ng
        for di in range(2):
            wi = (1.0 - fu) if di == 0 else fu
            si = -1.0 if di == 0 else 1.0
            ii = (ic + di) % nx + ng
            for dk in range(2):
                wk = (1.0 - fw) if dk == 0 else fw
                sk = -1.0 if dk == 0 else 1.0
                kk = (kc + dk) % nz + ng
                jy[ii, jyf, kk] += valy * (wi * wk + si * sk * cross_y)

        valz = qdt * seg_dw * inv_dxy
        cross_z = seg_du * seg_dv / 12.0
        kzf = (kc + 1) % nz + ng
        for di in range(2):
            wi = (1.0 - fu) if di == 0 else fu
            si = -1.0 if di == 0 else 1.0
            ii = (ic + di) % nx + ng
            for dj in range(2):
                wj = (1.0 - fv) if dj == 0 else fv
                sj = -1.0 if dj == 0 else 1.0
                jj = (jc + dj) % ny + ng
                jz[ii, jj, kzf] += valz * (wi * wj + si * sj * cross_z)

        t = t_next


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

    # Charge-conserving current deposit: Numba-compiled view of the shared core.
    _esirkepov_core_nb = cast("Callable[..., None]", njit(cache=True)(_esirkepov_core))

    @njit(cache=True)
    def _deposit_particles_esirkepov_to_flat(
        jx_flat: np.ndarray,
        jy_flat: np.ndarray,
        jz_flat: np.ndarray,
        jx_shape: tuple[int, ...],
        jy_shape: tuple[int, ...],
        jz_shape: tuple[int, ...],
        pos_old: np.ndarray,
        pos_new: np.ndarray,
        charges: np.ndarray,
        start: int,
        end: int,
        dt: float,
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
            _esirkepov_core_nb(
                jx, jy, jz, x0, y0, z0, x1, y1, z1, q, dt, dx, dy, dz, nx, ny, nz, ng
            )

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
        ox: float,
        oy: float,
        oz: float,
    ) -> None:
        for p in range(start, end):
            gx = positions[p, 0] / dx - ox
            gy = positions[p, 1] / dy - oy
            gz = positions[p, 2] / dz - oz
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
        ox: float = 0.0,
        oy: float = 0.0,
        oz: float = 0.0,
    ) -> None:
        """Scatter-add CIC weights onto ``rho`` (in-place) for periodic boundaries.

        ``(ox, oy, oz)`` is the grid-point offset in cells: ``0.0`` deposits on integer
        nodes (electrostatic / node-centered rho), ``0.5`` on cell centers (Yee rho).
        """
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
                    ox,
                    oy,
                    oz,
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
        dt: float,
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
                    dt,
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
        jx, jy, jz, pos, pos_new, charges, 1.0, dx, dy, dz, nx, ny, nz, ng,
        partial_jx, partial_jy, partial_jz,
    )
    electric_kick_b0(vel, efield, -1.0, 1e-12)
    wrap_positions_periodic(pos, lx, ly, lz)
