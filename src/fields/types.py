"""Shared type aliases, enums, and protocols for prescribed PIC field sources."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vector3Like = Sequence[float] | NDArray[np.floating]
Position = NDArray[np.floating]
Positions = NDArray[np.floating]
FieldVector = NDArray[np.float64]
FieldBatch = NDArray[np.float64]
Vec3 = NDArray[np.float64]

Waveform = Literal["sin", "cos"]
AxisName = Literal["x", "y", "z"]
SharedModeName = Literal[
    "ZERO",
    "UNIFORM",
    "SINUSOIDAL",
    "SINUSOIDAL_LINEAR",
    "SINUSOIDAL_ELLIPTICAL",
    "PLANE_WAVE",
    "GAUSSIAN_PULSE",
    "LINEAR_RAMP",
    "FILE",
]

EnvelopeWidthLike = float | Vector3Like

AXIS_INDEX: dict[AxisName, int] = {"x": 0, "y": 1, "z": 2}


class PolarizationKind(StrEnum):
    """Transverse polarization state for sinusoidal wave factories."""

    LINEAR = "linear"
    ELLIPTICAL = "elliptical"


class FieldSource(Protocol):
    """Minimal evaluator interface for native and transformed field wrappers."""

    def at(self, pos: Position, t: float = 0.0) -> FieldVector: ...

    def at_batch(self, positions: Positions, t: float = 0.0) -> FieldBatch: ...


class PrescribedFieldSpec(Protocol):
    """Shared analytical parameters for electric and magnetic field specs."""

    amplitude: NDArray[np.float64]
    offset: NDArray[np.float64]
    wavevector: NDArray[np.float64]
    omega: float
    phase0: float
    envelope_center: NDArray[np.float64]
    envelope_width: NDArray[np.float64]
    ramp_axis: int
    ramp_rate: float
    polarization_kind: PolarizationKind
    polarization_psi: float
    polarization_delta: float

    @property
    def scalar_amplitude(self) -> float: ...


@runtime_checkable
class FieldSourceSum(FieldSource, Protocol):
    """Structural protocol for superposition containers."""

    sources: list[FieldSource]


FieldSourceFactory = Callable[[list[FieldSource]], FieldSource]

# Backward-compatible alias.
PrescribedFieldSource = FieldSource
