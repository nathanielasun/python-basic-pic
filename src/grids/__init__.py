"""PIC field grids, shared geometry helpers, and Numba particle kernels."""

from .ElectrostaticGrid import ElectrostaticGrid, ParticleBackend
from .YeeGrid import YeeGrid
from .grid_common import clamp_position, periodic_along_axis, periodic_field, wrap_position
from .pic_kernels import (
    HAS_NUMBA,
    deposit_cic_periodic,
    electric_kick_b0,
    gather_e_cic_periodic,
    get_num_threads,
    warmup_kernels,
    wrap_positions_periodic,
)

__all__ = (
    "ElectrostaticGrid",
    "HAS_NUMBA",
    "ParticleBackend",
    "YeeGrid",
    "clamp_position",
    "deposit_cic_periodic",
    "electric_kick_b0",
    "gather_e_cic_periodic",
    "get_num_threads",
    "periodic_along_axis",
    "periodic_field",
    "warmup_kernels",
    "wrap_position",
    "wrap_positions_periodic",
)
