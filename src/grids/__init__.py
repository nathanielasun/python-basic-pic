"""PIC field grids, shared geometry helpers, and Numba particle kernels."""

from .base import PICGridBase
from .ElectrostaticGrid import ElectrostaticGrid
from .YeeGrid import YeeGrid
from .grid_common import (
    BoundaryKind,
    ParticleBackend,
    clamp_position,
    periodic_along_axis,
    periodic_field,
    wrap_position,
)
from .pic_kernels import HAS_NUMBA, warmup_kernels

try:
    from .pic_kernels import (
        deposit_cic_periodic,
        deposit_j_esirkepov_cic_periodic,
        electric_kick_b0,
        gather_b_yee_cic_periodic,
        gather_e_cic_periodic,
        gather_e_yee_cic_periodic,
        get_num_threads,
        wrap_positions_periodic,
    )
except ImportError:
    deposit_cic_periodic = None  # type: ignore[misc, assignment]
    deposit_j_esirkepov_cic_periodic = None  # type: ignore[misc, assignment]
    electric_kick_b0 = None  # type: ignore[misc, assignment]
    gather_b_yee_cic_periodic = None  # type: ignore[misc, assignment]
    gather_e_cic_periodic = None  # type: ignore[misc, assignment]
    gather_e_yee_cic_periodic = None  # type: ignore[misc, assignment]

    def get_num_threads() -> int:  # type: ignore[misc]
        return 1

    def wrap_positions_periodic(pos, lx, ly, lz):  # type: ignore[misc]
        pos[:, 0] %= lx
        pos[:, 1] %= ly
        pos[:, 2] %= lz

__all__ = (
    "BoundaryKind",
    "ElectrostaticGrid",
    "HAS_NUMBA",
    "PICGridBase",
    "ParticleBackend",
    "YeeGrid",
    "clamp_position",
    "deposit_cic_periodic",
    "deposit_j_esirkepov_cic_periodic",
    "electric_kick_b0",
    "gather_b_yee_cic_periodic",
    "gather_e_cic_periodic",
    "gather_e_yee_cic_periodic",
    "get_num_threads",
    "periodic_along_axis",
    "periodic_field",
    "warmup_kernels",
    "wrap_position",
    "wrap_positions_periodic",
)
