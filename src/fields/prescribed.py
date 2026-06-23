"""
Shared base classes for prescribed spacetime field sources.

:class:`PrescribedField` factors out everything common to the electric and
magnetic field sources — the analytical evaluators, batch evaluation, structured
mesh sampling, file loading, and superposition — so :class:`fields.ElectricFields`
and :class:`fields.MagneticFields` differ only in their amplitude attribute
(``E0`` vs ``B0``), their mode/spec types, and the few extra modes one supports
(e.g. the magnetic ``MIRROR`` profile, supplied through ``_analytical_special``).

Wave sources are authored in a fixed local Cartesian frame (propagation along
local +z). Spatial rotation into the PIC grid is applied only through
``field_frame.PolarTransformedField`` / ``WaveFrame``; time enters through the
wave phase ``omega * t`` and polarization parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
import math
from pathlib import Path
from typing import ClassVar, Generic, Literal, Protocol, Self, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from .field_frame import (
    PolarizationKind,
    PolarTransformedField,
    WaveFrame,
    elliptical_components,
    elliptical_components_cos,
    evaluate_gaussian_pulse_local,
    evaluate_polarized_wave_local,
    local_wavevector,
    normalize_envelope_width,
    resolve_k_magnitude,
)
from .field_io import (
    FieldDataset,
    FieldInterpolator,
    load_field_file,
    phase,
    phase_batch,
)

Vector3Like = Sequence[float] | NDArray[np.floating]
Position = NDArray[np.floating]
Positions = NDArray[np.floating]
FieldVector = NDArray[np.float64]
FieldBatch = NDArray[np.float64]

Waveform = Literal["sin", "cos"]
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

_AXIS_INDEX: dict[Literal["x", "y", "z"], int] = {"x": 0, "y": 1, "z": 2}

ModeT = TypeVar("ModeT", bound=StrEnum)
SpecT = TypeVar("SpecT", bound="PrescribedFieldSpec")
SumT = TypeVar("SumT", bound="PrescribedFieldSum")


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


class PrescribedFieldSource(Protocol):
    """Minimal interface for field sources combined in :class:`PrescribedFieldSum`."""

    def at(self, pos: Position, t: float = 0.0) -> FieldVector: ...

    def at_batch(self, positions: Positions, t: float = 0.0) -> FieldBatch: ...


class PrescribedField(Generic[ModeT, SpecT, SumT]):
    """Spacetime-dependent prescribed vector field F(r, t) evaluated in the source frame."""

    # Subclasses bind these to their concrete mode enum / spec dataclass / sum type.
    _MODE: ClassVar[type]
    _SPEC: ClassVar[type]
    _SUM: ClassVar[type]
    _AMP_ATTR: ClassVar[str]  # spec attribute holding the scalar local-frame amplitude
    _COMPONENTS: ClassVar[tuple[str, str, str]]
    _DEFAULT_HDF5_GROUP: ClassVar[str]
    mode: ModeT
    spec: SpecT
    dataset: FieldDataset | None
    _interpolator: FieldInterpolator | None

    def __init__(
        self,
        mode: ModeT | str | None = None,
        spec: SpecT | None = None,
        dataset: FieldDataset | None = None,
    ) -> None:
        self.mode = cast(ModeT, self._MODE(mode if mode is not None else getattr(self._MODE, "ZERO")))
        self.spec = spec if spec is not None else cast(SpecT, self._SPEC())
        self.dataset = dataset
        self._interpolator = FieldInterpolator(dataset) if dataset is not None else None

    @property
    def _amplitude(self) -> float:
        """Scalar local-frame amplitude (``E0`` / ``B0``); 0 selects the legacy vector path."""
        return self.spec.scalar_amplitude

    @classmethod
    def _build_spec(cls, **kwargs: object) -> SpecT:
        return cast(SpecT, cls._SPEC(**kwargs))

    @classmethod
    def _mode(cls, name: SharedModeName) -> ModeT:
        return cast(ModeT, getattr(cls._MODE, name))

    def _is_mode(self, name: SharedModeName) -> bool:
        return self.mode == type(self)._mode(name)

    # ----------------------------------------------------------- shared factories

    @classmethod
    def zero(cls) -> Self:
        return cls(cls._mode("ZERO"))

    @classmethod
    def uniform(cls, amplitude: Vector3Like) -> Self:
        return cls(cls._mode("UNIFORM"), cls._build_spec(amplitude=np.asarray(amplitude, dtype=np.float64)))

    @classmethod
    def sinusoidal(
        cls,
        amplitude: Vector3Like,
        omega: float,
        wavevector: Vector3Like | None = None,
        phase0: float = 0.0,
    ) -> Self:
        """Legacy lab-frame sinusoidal plane wave with a fixed amplitude vector."""
        spec = cls._build_spec(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector if wavevector is not None else [0.0, 0.0, 0.0]),
            phase0=phase0,
        )
        return cls(cls._mode("SINUSOIDAL"), spec)

    @classmethod
    def plane_wave(
        cls,
        amplitude: Vector3Like,
        omega: float,
        wavevector: Vector3Like,
        phase0: float = 0.0,
    ) -> Self:
        """Legacy lab-frame cosine plane wave with a fixed amplitude vector."""
        spec = cls._build_spec(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector, dtype=np.float64),
            phase0=phase0,
        )
        return cls(cls._mode("PLANE_WAVE"), spec)

    @classmethod
    def gaussian_pulse(
        cls,
        amplitude: Vector3Like,
        omega: float,
        wavevector: Vector3Like,
        center: Vector3Like,
        width: Vector3Like,
        phase0: float = 0.0,
    ) -> Self:
        """Legacy lab-frame Gaussian-enveloped cosine wave."""
        spec = cls._build_spec(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector, dtype=np.float64),
            envelope_center=np.asarray(center, dtype=np.float64),
            envelope_width=np.asarray(width, dtype=np.float64),
            phase0=phase0,
        )
        return cls(cls._mode("GAUSSIAN_PULSE"), spec)

    @classmethod
    def linear_ramp(
        cls,
        ramp_rate: float,
        axis: Literal["x", "y", "z"] = "z",
        offset: Vector3Like | None = None,
    ) -> Self:
        spec = cls._build_spec(
            ramp_rate=ramp_rate,
            ramp_axis=_AXIS_INDEX[axis],
            offset=np.asarray(offset if offset is not None else [0.0, 0.0, 0.0]),
        )
        return cls(cls._mode("LINEAR_RAMP"), spec)

    @classmethod
    def transform(cls, source: PrescribedFieldSource, frame: WaveFrame) -> PolarTransformedField:
        """
        Apply a static polar transform from the local source frame to the lab frame.

        Time-dependent rotation is intentionally unsupported; temporal polarization
        changes belong in ``sinusoidal_elliptical`` phase parameters.
        """
        return PolarTransformedField(source, frame)

    @classmethod
    def from_file(cls, path: str | Path, *, hdf5_group: str | None = None) -> Self:
        dataset = load_field_file(path, cls._COMPONENTS, hdf5_group=hdf5_group)
        return cls(cls._mode("FILE"), dataset=dataset)

    @classmethod
    def from_csv(cls, path: str | Path) -> Self:
        return cls.from_file(path)

    @classmethod
    def from_hdf5(cls, path: str | Path, group: str | None = None) -> Self:
        return cls.from_file(path, hdf5_group=group or cls._DEFAULT_HDF5_GROUP)

    # ------------------------------------------------- shared local-wave builders

    @classmethod
    def _local_wave(
        cls,
        amplitude: float,
        omega: float,
        *,
        mode: ModeT,
        kind: PolarizationKind,
        psi: float = 0.0,
        delta: float = 0.0,
        phase0: float = 0.0,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        wavevector: Vector3Like | None = None,
    ) -> Self:
        """Build a polarized local-frame plane wave (propagation along local +z)."""
        if wavevector is not None:
            k_mag = resolve_k_magnitude(wavevector=np.asarray(wavevector, dtype=np.float64))
        else:
            k_mag = resolve_k_magnitude(k_magnitude=k_magnitude, wavelength=wavelength)
        spec = cls._build_spec(
            omega=omega,
            wavevector=local_wavevector(k_mag),
            phase0=phase0,
            polarization_kind=kind,
            polarization_psi=psi,
            polarization_delta=delta,
            **{cls._AMP_ATTR: amplitude},
        )
        return cls(mode, spec)

    @classmethod
    def _gaussian_local(
        cls,
        amplitude: float,
        omega: float,
        *,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        center: Vector3Like | None = None,
        width: float | Vector3Like = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
    ) -> Self:
        """Build a Gaussian-enveloped cosine pulse in the local source frame."""
        k_mag = resolve_k_magnitude(k_magnitude=k_magnitude, wavelength=wavelength)
        center_arr = np.zeros(3, dtype=np.float64) if center is None else np.asarray(center, dtype=np.float64)
        if center_arr.shape != (3,):
            raise ValueError("center must have shape (3,)")
        kind = PolarizationKind.LINEAR if delta == 0.0 else PolarizationKind.ELLIPTICAL
        spec = cls._build_spec(
            omega=omega,
            wavevector=local_wavevector(k_mag),
            envelope_center=center_arr,
            envelope_width=normalize_envelope_width(width),
            phase0=phase0,
            polarization_kind=kind,
            polarization_psi=psi,
            polarization_delta=delta,
            **{cls._AMP_ATTR: amplitude},
        )
        return cls(cls._mode("GAUSSIAN_PULSE"), spec)

    # ----------------------------------------------------------- file management

    def load_from_file(self, path: str | Path, *, hdf5_group: str | None = None) -> None:
        """Replace the current field definition with data read from CSV or HDF5."""
        self.mode = type(self)._mode("FILE")
        self.dataset = load_field_file(path, self._COMPONENTS, hdf5_group=hdf5_group)
        self._interpolator = FieldInterpolator(self.dataset)

    def read_from_csv(self, path: str | Path) -> None:
        """Load field samples from a CSV file."""
        self.load_from_file(path)

    # ---------------------------------------------------------------- evaluation

    def at(self, pos: Position, t: float = 0.0) -> FieldVector:
        """Evaluate the field at a single position and time in the source frame."""
        if self._is_mode("FILE"):
            if self._interpolator is None:
                raise RuntimeError("file-backed field is not initialized")
            return self._interpolator.at(pos, t)
        return self._analytical(pos, t)

    def at_batch(self, positions: Positions, t: float = 0.0) -> FieldBatch:
        """Evaluate the field at ``(N, 3)`` positions; returns ``(N, 3)``."""
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self._is_mode("FILE"):
            if self._interpolator is None:
                raise RuntimeError("file-backed field is not initialized")
            return self._interpolator.at_batch(pos, t)
        return self._analytical_batch(pos, t)

    def on_grid(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        z: NDArray[np.floating],
        t: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the field on a structured mesh (1D coordinate arrays)."""
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        batch = self.at_batch(points, t)
        shape = xx.shape
        return batch[:, 0].reshape(shape), batch[:, 1].reshape(shape), batch[:, 2].reshape(shape)

    def __add__(self, other: PrescribedFieldSource) -> SumT:
        return cast(SumT, self._SUM([self, other]))

    def _wave_phase(self, pos: Position, t: float, spec: SpecT) -> float:
        return phase(spec.wavevector, pos, spec.omega, t) + spec.phase0

    def _polarized_wave_local(
        self,
        pos: Position,
        t: float,
        spec: SpecT,
        *,
        waveform: Waveform,
    ) -> FieldVector:
        return evaluate_polarized_wave_local(
            self._wave_phase(pos, t, spec),
            amplitude=self._amplitude,
            polarization_kind=spec.polarization_kind,
            psi=spec.polarization_psi,
            delta=spec.polarization_delta,
            waveform=waveform,
        )

    def _analytical_special(self, _r: FieldVector, _t: float) -> FieldVector | None:
        """Hook for subclass-only single-position modes (e.g. magnetic MIRROR)."""
        return None

    def _analytical(self, pos: Position, t: float) -> FieldVector:
        r = np.asarray(pos, dtype=np.float64)
        spec = self.spec

        if self._is_mode("ZERO"):
            return np.zeros(3, dtype=np.float64)
        if self._is_mode("UNIFORM"):
            return spec.amplitude.copy()
        if self._is_mode("LINEAR_RAMP"):
            value = spec.offset.copy()
            value[spec.ramp_axis] += spec.ramp_rate * t
            return value
        if self._is_mode("SINUSOIDAL"):
            return spec.amplitude * math.sin(self._wave_phase(r, t, spec))
        if self._is_mode("SINUSOIDAL_LINEAR") or self._is_mode("SINUSOIDAL_ELLIPTICAL"):
            return self._polarized_wave_local(r, t, spec, waveform="sin")
        if self._is_mode("PLANE_WAVE"):
            # Non-zero scalar E0/B0: local-frame polarized wave; zero: legacy vector amplitude.
            if self._amplitude != 0.0:
                return self._polarized_wave_local(r, t, spec, waveform="cos")
            return spec.amplitude * math.cos(self._wave_phase(r, t, spec))
        if self._is_mode("GAUSSIAN_PULSE"):
            # Non-zero scalar E0/B0: local-frame polarized pulse; zero: legacy vector amplitude.
            if self._amplitude != 0.0:
                return evaluate_gaussian_pulse_local(
                    r,
                    t,
                    amplitude=self._amplitude,
                    omega=spec.omega,
                    wavevector=spec.wavevector,
                    center=spec.envelope_center,
                    width=spec.envelope_width,
                    phase0=spec.phase0,
                    polarization_kind=spec.polarization_kind,
                    psi=spec.polarization_psi,
                    delta=spec.polarization_delta,
                )
            envelope_arg = float(-np.sum(((r - spec.envelope_center) / spec.envelope_width) ** 2))
            envelope = math.exp(envelope_arg)
            return spec.amplitude * envelope * math.cos(self._wave_phase(r, t, spec))

        special = self._analytical_special(r, t)
        if special is not None:
            return special
        raise RuntimeError(f"unsupported field mode: {self.mode!r}")

    def _polarized_wave_local_batch(
        self,
        pos: Positions,
        t: float,
        spec: SpecT,
        *,
        waveform: Waveform,
    ) -> FieldBatch:
        phi = np.asarray(phase_batch(spec.wavevector, pos, spec.omega, t) + spec.phase0, dtype=np.float64)
        n = len(pos)
        amp = self._amplitude
        if spec.polarization_kind == PolarizationKind.LINEAR:
            raw_carrier = np.sin(phi) if waveform == "sin" else np.cos(phi)
            carrier = np.asarray(raw_carrier, dtype=np.float64)
            ex = np.multiply(amp * math.cos(spec.polarization_psi), carrier)
            ey = np.multiply(amp * math.sin(spec.polarization_psi), carrier)
            return np.column_stack([ex, ey, np.zeros(n, dtype=np.float64)])

        components = elliptical_components if waveform == "sin" else elliptical_components_cos
        ex = np.empty(n, dtype=np.float64)
        ey = np.empty(n, dtype=np.float64)
        for i in range(n):
            a1, a2 = components(
                phi.item(i),
                psi=spec.polarization_psi,
                delta=spec.polarization_delta,
            )
            ex[i] = amp * a1
            ey[i] = amp * a2
        return np.column_stack([ex, ey, np.zeros(n, dtype=np.float64)])

    def _analytical_special_batch(
        self, _pos: FieldVector, _t: float
    ) -> FieldBatch | None:
        """Hook for subclass-only batched modes (e.g. magnetic MIRROR)."""
        return None

    def _analytical_batch(self, positions: Positions, t: float) -> FieldBatch:
        pos = np.asarray(positions, dtype=np.float64)
        spec = self.spec
        n = len(pos)

        if self._is_mode("ZERO"):
            return np.zeros((n, 3), dtype=np.float64)
        if self._is_mode("UNIFORM"):
            return np.broadcast_to(spec.amplitude, (n, 3)).copy()
        if self._is_mode("LINEAR_RAMP"):
            value = spec.offset.copy()
            value[spec.ramp_axis] += spec.ramp_rate * t
            return np.broadcast_to(value, (n, 3)).copy()
        if self._is_mode("SINUSOIDAL"):
            phi = phase_batch(spec.wavevector, pos, spec.omega, t) + spec.phase0
            return np.sin(phi)[:, np.newaxis] * spec.amplitude
        if self._is_mode("SINUSOIDAL_LINEAR") or self._is_mode("SINUSOIDAL_ELLIPTICAL"):
            return self._polarized_wave_local_batch(pos, t, spec, waveform="sin")
        if self._is_mode("PLANE_WAVE"):
            # Non-zero scalar E0/B0: local-frame polarized wave; zero: legacy vector amplitude.
            if self._amplitude != 0.0:
                return self._polarized_wave_local_batch(pos, t, spec, waveform="cos")
            phi = phase_batch(spec.wavevector, pos, spec.omega, t) + spec.phase0
            return np.cos(phi)[:, np.newaxis] * spec.amplitude
        if self._is_mode("GAUSSIAN_PULSE"):
            envelope_arg = np.asarray(
                -np.sum(((pos - spec.envelope_center) / spec.envelope_width) ** 2, axis=1),
                dtype=np.float64,
            )
            envelope = np.exp(envelope_arg)
            # Non-zero scalar E0/B0: local-frame polarized pulse; zero: legacy vector amplitude.
            if self._amplitude != 0.0:
                field = self._polarized_wave_local_batch(pos, t, spec, waveform="cos")
                return envelope[:, np.newaxis] * field
            phi = phase_batch(spec.wavevector, pos, spec.omega, t) + spec.phase0
            return np.cos(phi)[:, np.newaxis] * spec.amplitude * envelope[:, np.newaxis]

        special = self._analytical_special_batch(pos, t)
        if special is not None:
            return special
        raise RuntimeError(f"unsupported field mode: {self.mode!r}")


class PrescribedFieldSum:
    """Superposition of prescribed and/or polar-transformed field sources."""

    sources: list[PrescribedFieldSource]

    def __init__(self, sources: list[PrescribedFieldSource]) -> None:
        self.sources = list(sources)

    def at(self, pos: Position, t: float = 0.0) -> FieldVector:
        total = np.zeros(3, dtype=np.float64)
        for source in self.sources:
            total += source.at(pos, t)
        return total

    def at_batch(self, positions: Positions, t: float = 0.0) -> FieldBatch:
        pos = np.asarray(positions, dtype=np.float64)
        total = np.zeros((len(pos), 3), dtype=np.float64)
        for source in self.sources:
            total += source.at_batch(pos, t)
        return total

    def __add__(self, other: PrescribedFieldSource) -> Self:
        return type(self)([*self.sources, other])
