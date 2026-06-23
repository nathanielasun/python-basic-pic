"""Shared grid geometry helpers and type aliases for electrostatic and Yee PIC grids."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Selectable wall conditions and particle-operation backend, shared by every grid.
BoundaryKind = Literal["periodic", "anode", "reflecting"]
ParticleBackend = Literal["numpy", "numba"]


def periodic_along_axis(field: NDArray[np.floating], axis: int, ng: int) -> None:
    """Copy interior face values into guard cells along one axis."""
    if ng == 0:
        return

    n = field.shape[axis]
    interior = n - 2 * ng
    idx = [slice(None)] * field.ndim

    lo = idx.copy()
    lo[axis] = slice(0, ng)
    hi = idx.copy()
    hi[axis] = slice(interior, interior + ng)
    field[tuple(lo)] = field[tuple(hi)]

    lo[axis] = slice(n - ng, n)
    hi[axis] = slice(ng, 2 * ng)
    field[tuple(lo)] = field[tuple(hi)]


def periodic_field(field: NDArray[np.floating], ng: int) -> None:
    for axis in range(field.ndim):
        periodic_along_axis(field, axis, ng)


def wrap_position(
    pos: NDArray[np.floating],
    lengths: tuple[float, float, float],
) -> NDArray[np.float64]:
    wrapped = np.asarray(pos, dtype=np.float64).copy()
    for axis, length in enumerate(lengths):
        wrapped[..., axis] %= length
    return wrapped


def minimum_image_displacement(
    pos_old: NDArray[np.floating],
    pos_new: NDArray[np.floating],
    lengths: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Shortest periodic displacement ``pos_new - pos_old`` for each particle."""
    old = np.asarray(pos_old, dtype=np.float64)
    new = np.asarray(pos_new, dtype=np.float64)
    delta = new - old
    for axis, length in enumerate(lengths):
        if length <= 0.0:
            continue
        half = 0.5 * length
        delta[:, axis] = (delta[:, axis] + half) % length - half
    return delta


def unwrap_periodic_trajectory(
    pos_old: NDArray[np.floating],
    pos_new: NDArray[np.floating],
    lengths: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Endpoint positions for Esirkepov with minimum-image segment lengths."""
    old = np.asarray(pos_old, dtype=np.float64)
    return old + minimum_image_displacement(old, pos_new, lengths)


def clamp_position(
    pos: NDArray[np.floating],
    lengths: tuple[float, float, float],
    cell_sizes: tuple[float, float, float],
) -> NDArray[np.float64]:
    clamped = np.asarray(pos, dtype=np.float64).copy()
    ds = np.array(cell_sizes)
    for axis, length in enumerate(lengths):
        upper = max(length - ds[axis], 0.0)
        clamped[..., axis] = np.clip(clamped[..., axis], 0.0, upper)
    return clamped
