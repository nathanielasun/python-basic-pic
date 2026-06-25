"""Shared grid geometry helpers and type aliases for electrostatic and Yee PIC grids."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Selectable wall conditions and particle-operation backend, shared by every grid.
BoundaryKind = Literal["periodic", "anode", "reflecting"]
ParticleBackend = Literal["numpy", "numba"]


def periodic_along_axis(
    field: NDArray[np.floating],
    axis: int,
    ng: int,
    *,
    node_aligned: bool = False,
    combine_redundant: bool = False,
) -> None:
    """Apply periodic wrapping into the guard cells along one axis.

    Two layouts are supported:

    - **Cell-aligned** (default): the ``interior = shape - 2*ng`` points are all
      independent and the physical period equals ``interior``. This is the original
      behavior and is correct for cell-centered quantities (rho, ElectrostaticGrid
      fields).
    - **Node-aligned** (``node_aligned=True``): the interior holds ``n + 1`` points
      for ``n`` cells, so logical node ``n`` is the periodic image of node ``0`` and
      the true period is ``n`` (NOT ``n + 1``). The two coincident node planes are
      reconciled first, then guards are filled with period ``n``. Reconciliation is a
      sum when ``combine_redundant`` is set (used once for deposited sources such as
      the Esirkepov current, where each seam plane holds a partial contribution),
      otherwise a copy ``node[n] := node[0]`` (idempotent; used for E/B field arrays
      whose ``apply_boundaries`` runs every step).

    Using period ``n + 1`` for a node-aligned periodic field breaks the discrete-curl
    negative-transpose property and makes the Yee leapfrog unstable, so node-aligned
    axes must use this path.
    """
    if ng == 0:
        return

    base: list[slice | int] = [slice(None)] * field.ndim
    n_tot = field.shape[axis]
    interior = n_tot - 2 * ng

    if not node_aligned:
        lo = base.copy()
        lo[axis] = slice(0, ng)
        hi = base.copy()
        hi[axis] = slice(interior, interior + ng)
        field[tuple(lo)] = field[tuple(hi)]

        lo[axis] = slice(n_tot - ng, n_tot)
        hi[axis] = slice(ng, 2 * ng)
        field[tuple(lo)] = field[tuple(hi)]
        return

    # Node-aligned: interior = n + 1 nodes at indices ng .. ng + n; period is n.
    n = interior - 1

    def at(index: int) -> tuple[slice | int, ...]:
        sel = base.copy()
        sel[axis] = index
        return tuple(sel)

    first, last = at(ng), at(ng + n)
    if combine_redundant:
        total = field[first] + field[last]
        field[first] = total
        field[last] = total
    else:
        field[last] = field[first]

    for g in range(1, ng + 1):
        field[at(ng - g)] = field[at(ng + n - g)]
        field[at(ng + n + g)] = field[at(ng + g)]


def periodic_field(
    field: NDArray[np.floating],
    ng: int,
    *,
    node_aligned: bool | tuple[bool, bool, bool] = False,
    combine_redundant: bool = False,
) -> None:
    axes: tuple[bool, ...] = (
        (node_aligned,) * field.ndim if isinstance(node_aligned, bool) else node_aligned
    )
    for axis in range(field.ndim):
        periodic_along_axis(
            field,
            axis,
            ng,
            node_aligned=axes[axis],
            combine_redundant=combine_redundant,
        )


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
