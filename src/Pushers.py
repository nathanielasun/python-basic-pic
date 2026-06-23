"""
Author: Nathaniel Sun
Date: 2026-06-17
Description:
    Explicit particle pushers for PIC velocity updates.

    All pushers share the interface::

        v_new = Pushers.boris(vel, E, B, q, m, dt)

    Inputs are SI or normalized consistent units; ``c`` defaults to 1 for
    normalized PIC. Relativistic pushers accept velocity ``v`` and internally
    use proper velocity ``u = gamma * v``.

    Batch usage for ``(N, 3)`` velocity and per-particle field arrays::

        vel = Pushers.push_batch("boris", vel, E, B, q, m, dt)

    Typical driver usage after field gather::

        E, B = grid.gather_boris_fields(x, y, z)
        E = E + efield.at(pos, t)
        B = B + bfield.at(pos, t)
        vel = Pushers.vay(vel, E, B, q, m, dt)
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class PusherKind(StrEnum):
    BORIS = "boris"
    BORIS_RELATIVISTIC = "boris_relativistic"
    VAY = "vay"
    HIGUERA_CARY = "higuera_cary"


def _as_vec3(value: NDArray[np.floating]) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("expected a 3-component vector")
    return arr


def _as_batch_fields(
    vel: NDArray[np.floating],
    E: NDArray[np.floating],
    B: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Coerce ``vel``, ``E``, and ``B`` to matching ``(N, 3)`` float64 arrays."""
    v = np.asarray(vel, dtype=np.float64)
    e = np.asarray(E, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vel must have shape (N, 3)")
    if e.shape != v.shape or b.shape != v.shape:
        raise ValueError("vel, E, B must have shape (N, 3)")
    return v, e, b


def lorentz_gamma_from_velocity(vel: NDArray[np.floating], c: float) -> float:
    inv_c2 = 1.0 / (c * c)
    beta2 = float(np.dot(vel, vel)) * inv_c2
    if beta2 >= 1.0:
        raise ValueError("velocity magnitude must be less than c")
    return 1.0 / np.sqrt(1.0 - beta2)


def lorentz_gamma_from_proper_velocity(u: NDArray[np.floating], c: float) -> float:
    inv_c2 = 1.0 / (c * c)
    return float(np.sqrt(1.0 + float(np.dot(u, u)) * inv_c2))


def velocity_to_proper_velocity(vel: NDArray[np.floating], c: float) -> NDArray[np.float64]:
    v = _as_vec3(vel)
    return v * lorentz_gamma_from_velocity(v, c)


def proper_velocity_to_velocity(u: NDArray[np.floating], c: float) -> NDArray[np.float64]:
    u_arr = _as_vec3(u)
    gamma = lorentz_gamma_from_proper_velocity(u_arr, c)
    return u_arr / gamma


def boris_push(
    vel: NDArray[np.floating],
    E: NDArray[np.floating],
    B: NDArray[np.floating],
    q: float,
    m: float,
    dt: float,
) -> NDArray[np.float64]:
    """
    Classical (non-relativistic) Boris pusher.

    Suitable when |v| << c. With B = 0 this is a centered electric kick.
    """
    qmdt = (q / m) * dt
    v_minus = _as_vec3(vel) + qmdt * _as_vec3(E) / 2.0

    t = qmdt * _as_vec3(B) / 2.0
    s = 2.0 * t / (1.0 + float(np.dot(t, t)))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    return v_plus + qmdt * _as_vec3(E) / 2.0


def _boris_push_batch(
    vel: NDArray[np.floating],
    E: NDArray[np.floating],
    B: NDArray[np.floating],
    q: float,
    m: float,
    dt: float,
) -> NDArray[np.float64]:
    """Vectorized classical Boris pusher for ``(N, 3)`` arrays."""
    qmdt = (q / m) * dt
    v, e, b = _as_batch_fields(vel, E, B)

    v_minus = v + qmdt * e / 2.0
    t_vec = qmdt * b / 2.0
    t_sq = np.sum(t_vec * t_vec, axis=1)
    s = 2.0 * t_vec / (1.0 + t_sq)[:, np.newaxis]
    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s)
    return np.asarray(v_plus + qmdt * e / 2.0, dtype=np.float64)


def boris_push_batch(
    vel: NDArray[np.floating],
    E: NDArray[np.floating],
    B: NDArray[np.floating],
    q: float,
    m: float,
    dt: float,
) -> NDArray[np.float64]:
    """Classical Boris pusher for ``(N, 3)`` velocity and field arrays."""
    return _boris_push_batch(vel, E, B, q, m, dt)


def _push_batch_scalar(
    pusher: Callable[..., NDArray[np.float64]],
    vel: NDArray[np.floating],
    E: NDArray[np.floating],
    B: NDArray[np.floating],
    q: float,
    m: float,
    dt: float,
    *,
    c: float,
    relativistic: bool,
) -> NDArray[np.float64]:
    """Apply a single-particle pusher row-wise to ``(N, 3)`` arrays."""
    v, e, b = _as_batch_fields(vel, E, B)
    out = np.empty_like(v)
    for i in range(v.shape[0]):
        if relativistic:
            out[i] = pusher(v[i], e[i], b[i], q, m, dt, c=c)
        else:
            out[i] = pusher(v[i], e[i], b[i], q, m, dt)
    return out


def boris_relativistic_push(
    vel: NDArray[np.floating],
    E: NDArray[np.floating],
    B: NDArray[np.floating],
    q: float,
    m: float,
    dt: float,
    *,
    c: float = 1.0,
) -> NDArray[np.float64]:
    """
    Relativistic Boris pusher using u = gamma*v (Birdsall & Langdon style).

    Reduces to the classical Boris pusher in the non-relativistic limit.
    """
    inv_c2 = 1.0 / (c * c)
    v = _as_vec3(vel)
    E_arr = _as_vec3(E)
    B_arr = _as_vec3(B)

    gamma = lorentz_gamma_from_velocity(v, c)
    u = v * gamma

    u_minus = u + q * dt * E_arr / (2.0 * m)
    gamma1 = float(np.sqrt(1.0 + float(np.dot(u_minus, u_minus)) * inv_c2))

    t = q * B_arr * dt / (2.0 * gamma1 * m)
    s = 2.0 * t / (1.0 + float(np.dot(t, t)))
    u_prime = u_minus + np.cross(u_minus, t)
    u_plus = u_minus + np.cross(u_prime, s)
    u_new = u_plus + q * dt * E_arr / (2.0 * m)

    gamma2 = float(np.sqrt(1.0 + float(np.dot(u_new, u_new)) * inv_c2))
    return u_new / gamma2


def _vay_push_proper(
    u: NDArray[np.float64],
    E: NDArray[np.float64],
    B: NDArray[np.float64],
    q: float,
    m: float,
    dt: float,
    c: float,
) -> NDArray[np.float64]:
    """Vay pusher in proper-velocity space (WarpX UpdateMomentumVay, full push)."""
    inv_c2 = 1.0 / (c * c)
    inv_c = 1.0 / c
    econst = q * dt / m
    bconst = 0.5 * q * dt / m

    ux, uy, uz = u
    Ex, Ey, Ez = E
    Bx, By, Bz = B

    inv_gamma = 1.0 / np.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * inv_c2)

    taux = bconst * Bx
    tauy = bconst * By
    tauz = bconst * Bz
    tausq = taux * taux + tauy * tauy + tauz * tauz

    uxpr = ux + econst * Ex + (uy * tauz - uz * tauy) * inv_gamma
    uypr = uy + econst * Ey + (uz * taux - ux * tauz) * inv_gamma
    uzpr = uz + econst * Ez + (ux * tauy - uy * taux) * inv_gamma

    gprsq = 1.0 + (uxpr * uxpr + uypr * uypr + uzpr * uzpr) * inv_c2
    ust = (uxpr * taux + uypr * tauy + uzpr * tauz) * inv_c
    sigma = gprsq - tausq
    gisq = 2.0 / (sigma + np.sqrt(sigma * sigma + 4.0 * (tausq + ust * ust)))
    bg = bconst * np.sqrt(gisq)
    tx = bg * Bx
    ty = bg * By
    tz = bg * Bz
    s = 1.0 / (1.0 + tausq * gisq)
    tu = tx * uxpr + ty * uypr + tz * uzpr

    ux_out = s * (uxpr + tx * tu + uypr * tz - uzpr * ty)
    uy_out = s * (uypr + ty * tu + uzpr * tx - uxpr * tz)
    uz_out = s * (uzpr + tz * tu + uxpr * ty - uypr * tx)
    return np.array([ux_out, uy_out, uz_out], dtype=np.float64)


def vay_push(
    vel: NDArray[np.floating],
    E: NDArray[np.floating],
    B: NDArray[np.floating],
    q: float,
    m: float,
    dt: float,
    *,
    c: float = 1.0,
) -> NDArray[np.float64]:
    """
    Vay relativistic pusher (J.-L. Vay, Phys. Plasmas 2007).

    Improves Lorentz covariance compared with Boris at high gamma.
    """
    u = velocity_to_proper_velocity(vel, c)
    u_new = _vay_push_proper(u, _as_vec3(E), _as_vec3(B), q, m, dt, c)
    return proper_velocity_to_velocity(u_new, c)


def _higuera_cary_push_proper(
    u: NDArray[np.float64],
    E: NDArray[np.float64],
    B: NDArray[np.float64],
    q: float,
    m: float,
    dt: float,
    c: float,
) -> NDArray[np.float64]:
    """Higuera-Cary pusher in proper-velocity space (WarpX UpdateMomentumHigueraCary)."""
    inv_c2 = 1.0 / (c * c)
    inv_c = 1.0 / c
    qmt = 0.5 * q * dt / m

    ux, uy, uz = u
    Ex, Ey, Ez = E
    Bx, By, Bz = B

    umx = ux + qmt * Ex
    umy = uy + qmt * Ey
    umz = uz + qmt * Ez

    gamma_sq = 1.0 + (umx * umx + umy * umy + umz * umz) * inv_c2

    betax = qmt * Bx
    betay = qmt * By
    betaz = qmt * Bz
    betam = betax * betax + betay * betay + betaz * betaz
    sigma = gamma_sq - betam

    ust = (umx * betax + umy * betay + umz * betaz) * inv_c
    gamma_inv = 1.0 / np.sqrt(0.5 * (sigma + np.sqrt(sigma * sigma + 4.0 * (betam + ust * ust))))

    tx = gamma_inv * betax
    ty = gamma_inv * betay
    tz = gamma_inv * betaz
    s = 1.0 / (1.0 + tx * tx + ty * ty + tz * tz)
    umt = umx * tx + umy * ty + umz * tz

    upx = s * (umx + umt * tx + umy * tz - umz * ty)
    upy = s * (umy + umt * ty + umz * tx - umx * tz)
    upz = s * (umz + umt * tz + umx * ty - umy * tx)

    ux_out = upx + qmt * Ex + upy * tz - upz * ty
    uy_out = upy + qmt * Ey + upz * tx - upx * tz
    uz_out = upz + qmt * Ez + upx * ty - upy * tx
    return np.array([ux_out, uy_out, uz_out], dtype=np.float64)


def higuera_cary_push(
    vel: NDArray[np.floating],
    E: NDArray[np.floating],
    B: NDArray[np.floating],
    q: float,
    m: float,
    dt: float,
    *,
    c: float = 1.0,
) -> NDArray[np.float64]:
    """
    Higuera-Cary relativistic pusher (Phys. Plasmas 24, 052104, 2017).

    Preserves phase-space volume and improves E x B drift compared with Boris.
    """
    u = velocity_to_proper_velocity(vel, c)
    u_new = _higuera_cary_push_proper(u, _as_vec3(E), _as_vec3(B), q, m, dt, c)
    return proper_velocity_to_velocity(u_new, c)


try:
    import numba
except ImportError:
    _has_numba_push = False
else:
    from numba import njit, prange as _prange

    _has_numba_push = True

    @njit(cache=True)
    def _higuera_cary_push_proper_numba(
        ux: float,
        uy: float,
        uz: float,
        Ex: float,
        Ey: float,
        Ez: float,
        Bx: float,
        By: float,
        Bz: float,
        q: float,
        m: float,
        dt: float,
        c: float,
    ) -> tuple[float, float, float]:
        inv_c2 = 1.0 / (c * c)
        qmt = 0.5 * q * dt / m

        umx = ux + qmt * Ex
        umy = uy + qmt * Ey
        umz = uz + qmt * Ez

        gamma_sq = 1.0 + (umx * umx + umy * umy + umz * umz) * inv_c2

        betax = qmt * Bx
        betay = qmt * By
        betaz = qmt * Bz
        betam = betax * betax + betay * betay + betaz * betaz
        sigma = gamma_sq - betam

        ust = (umx * betax + umy * betay + umz * betaz) * (1.0 / c)
        gamma_inv = 1.0 / np.sqrt(0.5 * (sigma + np.sqrt(sigma * sigma + 4.0 * (betam + ust * ust))))

        tx = gamma_inv * betax
        ty = gamma_inv * betay
        tz = gamma_inv * betaz
        s = 1.0 / (1.0 + tx * tx + ty * ty + tz * tz)
        umt = umx * tx + umy * ty + umz * tz

        upx = s * (umx + umt * tx + umy * tz - umz * ty)
        upy = s * (umy + umt * ty + umz * tx - umx * tz)
        upz = s * (umz + umt * tz + umx * ty - umy * tx)

        ux_out = upx + qmt * Ex + upy * tz - upz * ty
        uy_out = upy + qmt * Ey + upz * tx - upx * tz
        uz_out = upz + qmt * Ez + upx * ty - upy * tx
        return ux_out, uy_out, uz_out

    @njit(parallel=True, cache=True)
    def _higuera_cary_push_batch(
        vel: np.ndarray,
        E: np.ndarray,
        B: np.ndarray,
        q: float,
        m: float,
        dt: float,
        c: float,
    ) -> np.ndarray:
        n_particles = vel.shape[0]
        out = np.empty_like(vel)
        inv_c2 = 1.0 / (c * c)
        for p in _prange(n_particles):
            vx, vy, vz = vel[p, 0], vel[p, 1], vel[p, 2]
            beta2 = (vx * vx + vy * vy + vz * vz) * inv_c2
            gamma = 1.0 / np.sqrt(1.0 - beta2)
            ux, uy, uz = vx * gamma, vy * gamma, vz * gamma
            ux, uy, uz = _higuera_cary_push_proper_numba(
                ux, uy, uz,
                E[p, 0], E[p, 1], E[p, 2],
                B[p, 0], B[p, 1], B[p, 2],
                q, m, dt, c,
            )
            gamma_u = np.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * inv_c2)
            out[p, 0] = ux / gamma_u
            out[p, 1] = uy / gamma_u
            out[p, 2] = uz / gamma_u
        return out


def higuera_cary_push_batch(
    vel: NDArray[np.floating],
    E: NDArray[np.floating],
    B: NDArray[np.floating],
    q: float,
    m: float,
    dt: float,
    *,
    c: float = 1.0,
) -> NDArray[np.float64]:
    """Batch Higuera-Cary push for ``(N, 3)`` arrays."""
    if _has_numba_push:
        v, e, b = _as_batch_fields(vel, E, B)
        return _higuera_cary_push_batch(v, e, b, q, m, dt, c)
    return _push_batch_scalar(higuera_cary_push, vel, E, B, q, m, dt, c=c, relativistic=True)


class Pushers:
    """Registry and dispatcher for explicit PIC particle pushers."""

    _DISPATCH = {
        PusherKind.BORIS: boris_push,
        PusherKind.BORIS_RELATIVISTIC: boris_relativistic_push,
        PusherKind.VAY: vay_push,
        PusherKind.HIGUERA_CARY: higuera_cary_push,
    }

    _BATCH_DISPATCH: dict[PusherKind, Callable[..., NDArray[np.float64]]] = {
        PusherKind.BORIS: _boris_push_batch,
        PusherKind.HIGUERA_CARY: higuera_cary_push_batch,
    }

    boris = staticmethod(boris_push)
    boris_relativistic = staticmethod(boris_relativistic_push)
    vay = staticmethod(vay_push)
    higuera_cary = staticmethod(higuera_cary_push)
    boris_batch = staticmethod(boris_push_batch)

    @classmethod
    def push(
        cls,
        kind: PusherKind | str,
        vel: NDArray[np.floating],
        E: NDArray[np.floating],
        B: NDArray[np.floating],
        q: float,
        m: float,
        dt: float,
        *,
        c: float = 1.0,
    ) -> NDArray[np.float64]:
        pusher_kind = PusherKind(kind)
        pusher = cls._DISPATCH[pusher_kind]
        if pusher_kind is PusherKind.BORIS:
            return pusher(vel, E, B, q, m, dt)
        return pusher(vel, E, B, q, m, dt, c=c)

    @classmethod
    def push_batch(
        cls,
        kind: PusherKind | str,
        vel: NDArray[np.floating],
        E: NDArray[np.floating],
        B: NDArray[np.floating],
        q: float,
        m: float,
        dt: float,
        *,
        c: float = 1.0,
    ) -> NDArray[np.float64]:
        """
        Batch pusher for ``(N, 3)`` velocity and per-particle field arrays.

        Boris and Higuera-Cary use dedicated batch kernels; other kinds fall back
        to the scalar pusher applied row-wise.
        """
        pusher_kind = PusherKind(kind)
        batch_pusher = cls._BATCH_DISPATCH.get(pusher_kind)
        if batch_pusher is not None:
            if pusher_kind is PusherKind.HIGUERA_CARY:
                return batch_pusher(vel, E, B, q, m, dt, c=c)
            return batch_pusher(vel, E, B, q, m, dt)

        pusher = cls._DISPATCH[pusher_kind]
        relativistic = pusher_kind is not PusherKind.BORIS
        return _push_batch_scalar(pusher, vel, E, B, q, m, dt, c=c, relativistic=relativistic)

    @classmethod
    def available(cls) -> tuple[PusherKind, ...]:
        return tuple(cls._DISPATCH)

    @staticmethod
    def warmup() -> None:
        """JIT-compile the Numba batch pushers so the first timestep is not penalized."""
        if not _has_numba_push:
            return
        vel = np.zeros((2, 3), dtype=np.float64)
        efield = np.ones((2, 3), dtype=np.float64)
        bfield = np.zeros((2, 3), dtype=np.float64)
        _higuera_cary_push_batch(vel, efield, bfield, -1.0, 1.0, 1e-12, 1.0)
