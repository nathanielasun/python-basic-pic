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
from pathlib import Path
from typing import ClassVar, Literal, Protocol, cast

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

Waveform = Literal["sin", "cos"]


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


class PrescribedFieldSource(Protocol):
    """Minimal interface for field sources combined in :class:`PrescribedFieldSum`."""

    def at(self, pos: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]: ...

    def at_batch(self, positions: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]: ...


class PrescribedField:
    """Spacetime-dependent prescribed vector field F(r, t) evaluated in the source frame."""

    # Subclasses bind these to their concrete mode enum / spec dataclass / sum type.
    _MODE: ClassVar[type]
    _SPEC: ClassVar[type]
    _SUM: ClassVar[type]
    _AMP_ATTR: ClassVar[str]  # spec attribute holding the scalar local-frame amplitude
    _COMPONENTS: ClassVar[tuple[str, str, str]]
    _DEFAULT_HDF5_GROUP: ClassVar[str]

    def __init__(
        self,
        mode: object | str | None = None,
        spec: PrescribedFieldSpec | None = None,
        dataset: FieldDataset | None = None,
    ) -> None:
        self.mode = self._MODE(mode if mode is not None else self._MODE.ZERO)
        self.spec: PrescribedFieldSpec = (
            spec if spec is not None else cast(PrescribedFieldSpec, self._SPEC())
        )
        self.dataset = dataset
        self._interpolator = FieldInterpolator(dataset) if dataset is not None else None
    @property
    def _amplitude(self) -> float:
        """Scalar local-frame amplitude (``E0`` / ``B0``); 0 selects the legacy vector path."""
        return getattr(self.spec, self._AMP_ATTR)

    # ----------------------------------------------------------- shared factories
    
    @classmethod
    def zero(cls):
        return cls(cls._MODE.ZERO)

    @classmethod
    def uniform(cls, amplitude: Sequence[float] | NDArray[np.floating]):
        return cls(cls._MODE.UNIFORM, cls._SPEC(amplitude=np.asarray(amplitude, dtype=np.float64)))

    @classmethod
    def sinusoidal(
        cls,
        amplitude: Sequence[float] | NDArray[np.floating],
        omega: float,
        wavevector: Sequence[float] | NDArray[np.floating] | None = None,
        phase0: float = 0.0,
    ):
        """Legacy lab-frame sinusoidal plane wave with a fixed amplitude vector."""
        spec = cls._SPEC(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector if wavevector is not None else [0.0, 0.0, 0.0]),
            phase0=phase0,
        )
        return cls(cls._MODE.SINUSOIDAL, spec)

    @classmethod
    def plane_wave(
        cls,
        amplitude: Sequence[float] | NDArray[np.floating],
        omega: float,
        wavevector: Sequence[float] | NDArray[np.floating],
        phase0: float = 0.0,
    ):
        """Legacy lab-frame cosine plane wave with a fixed amplitude vector."""
        spec = cls._SPEC(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector, dtype=np.float64),
            phase0=phase0,
        )
        return cls(cls._MODE.PLANE_WAVE, spec)

    @classmethod
    def gaussian_pulse(
        cls,
        amplitude: Sequence[float] | NDArray[np.floating],
        omega: float,
        wavevector: Sequence[float] | NDArray[np.floating],
        center: Sequence[float] | NDArray[np.floating],
        width: Sequence[float] | NDArray[np.floating],
        phase0: float = 0.0,
    ):
        """Legacy lab-frame Gaussian-enveloped cosine wave."""
        spec = cls._SPEC(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector, dtype=np.float64),
            envelope_center=np.asarray(center, dtype=np.float64),
            envelope_width=np.asarray(width, dtype=np.float64),
            phase0=phase0,
        )
        return cls(cls._MODE.GAUSSIAN_PULSE, spec)

    @classmethod
    def linear_ramp(
        cls,
        ramp_rate: float,
        axis: Literal["x", "y", "z"] = "z",
        offset: Sequence[float] | NDArray[np.floating] | None = None,
    ):
        spec = cls._SPEC(
            ramp_rate=ramp_rate,
            ramp_axis={"x": 0, "y": 1, "z": 2}[axis],
            offset=np.asarray(offset if offset is not None else [0.0, 0.0, 0.0]),
        )
        return cls(cls._MODE.LINEAR_RAMP, spec)

    @classmethod
    def transform(cls, source, frame: WaveFrame) -> PolarTransformedField:
        """
        Apply a static polar transform from the local source frame to the lab frame.

        Time-dependent rotation is intentionally unsupported; temporal polarization
        changes belong in ``sinusoidal_elliptical`` phase parameters.
        """
        return PolarTransformedField(source, frame)

    @classmethod
    def from_file(cls, path: str | Path, *, hdf5_group: str | None = None):
        dataset = load_field_file(path, cls._COMPONENTS, hdf5_group=hdf5_group)
        return cls(cls._MODE.FILE, dataset=dataset)

    @classmethod
    def from_csv(cls, path: str | Path):
        return cls.from_file(path)

    @classmethod
    def from_hdf5(cls, path: str | Path, group: str | None = None):
        return cls.from_file(path, hdf5_group=group or cls._DEFAULT_HDF5_GROUP)

    # ------------------------------------------------- shared local-wave builders

    @classmethod
    def _local_wave(
        cls,
        amplitude: float,
        omega: float,
        *,
        mode: object,
        kind: PolarizationKind,
        psi: float = 0.0,
        delta: float = 0.0,
        phase0: float = 0.0,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        wavevector: Sequence[float] | NDArray[np.floating] | None = None,
    ):
        """Build a polarized local-frame plane wave (propagation along local +z)."""
        if wavevector is not None:
            k_mag = resolve_k_magnitude(wavevector=np.asarray(wavevector, dtype=np.float64))
        else:
            k_mag = resolve_k_magnitude(k_magnitude=k_magnitude, wavelength=wavelength)
        spec = cls._SPEC(
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
        center: Sequence[float] | NDArray[np.floating] | None = None,
        width: float | Sequence[float] | NDArray[np.floating] = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
    ):
        """Build a Gaussian-enveloped cosine pulse in the local source frame."""
        k_mag = resolve_k_magnitude(k_magnitude=k_magnitude, wavelength=wavelength)
        center_arr = np.zeros(3, dtype=np.float64) if center is None else np.asarray(center, dtype=np.float64)
        if center_arr.shape != (3,):
            raise ValueError("center must have shape (3,)")
        kind = PolarizationKind.LINEAR if delta == 0.0 else PolarizationKind.ELLIPTICAL
        spec = cls._SPEC(
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
        return cls(cls._MODE.GAUSSIAN_PULSE, spec)

    # ----------------------------------------------------------- file management

    def load_from_file(self, path: str | Path, *, hdf5_group: str | None = None) -> None:
        """Replace the current field definition with data read from CSV or HDF5."""
        self.mode = self._MODE.FILE
        self.dataset = load_field_file(path, self._COMPONENTS, hdf5_group=hdf5_group)
        self._interpolator = FieldInterpolator(self.dataset)

    def read_from_csv(self, path: str | Path) -> None:
        """Load field samples from a CSV file."""
        self.load_from_file(path)

    # ---------------------------------------------------------------- evaluation

    def at(self, pos: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        """Evaluate the field at a single position and time in the source frame."""
        if self.mode == self._MODE.FILE:
            if self._interpolator is None:
                raise RuntimeError("file-backed field is not initialized")
            return self._interpolator.at(pos, t)
        return self._analytical(pos, t)

    def at_batch(self, positions: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        """Evaluate the field at ``(N, 3)`` positions; returns ``(N, 3)``."""
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.mode == self._MODE.FILE:
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
        fx = np.zeros_like(xx, dtype=np.float64)
        fy = np.zeros_like(xx, dtype=np.float64)
        fz = np.zeros_like(xx, dtype=np.float64)
        for index in np.ndindex(xx.shape):
            fx[index], fy[index], fz[index] = self.at(
                np.array([xx[index], yy[index], zz[index]]), t
            )
        return fx, fy, fz

    def __add__(self, other):
        return self._SUM([self, other])

    def _wave_phase(self, pos: NDArray[np.floating], t: float, spec: PrescribedFieldSpec) -> float:
        return phase(spec.wavevector, pos, spec.omega, t) + spec.phase0

    def _polarized_wave_local(
        self,
        pos: NDArray[np.floating],
        t: float,
        spec: PrescribedFieldSpec,
        *,
        waveform: Waveform,
    ) -> NDArray[np.float64]:
        return evaluate_polarized_wave_local(
            self._wave_phase(pos, t, spec),
            amplitude=self._amplitude,
            polarization_kind=spec.polarization_kind,
            psi=spec.polarization_psi,
            delta=spec.polarization_delta,
            waveform=waveform,
        )

    def _analytical_special(self, r: NDArray[np.float64], t: float) -> NDArray[np.float64] | None:
        """Hook for subclass-only single-position modes (e.g. magnetic MIRROR)."""
        return None

    def _analytical(self, pos: NDArray[np.floating], t: float) -> NDArray[np.float64]:
        r = np.asarray(pos, dtype=np.float64)
        spec = self.spec
        mode = self.mode
        m = self._MODE

        if mode == m.ZERO:
            return np.zeros(3, dtype=np.float64)
        if mode == m.UNIFORM:
            return spec.amplitude.copy()
        if mode == m.LINEAR_RAMP:
            value = spec.offset.copy()
            value[spec.ramp_axis] += spec.ramp_rate * t
            return value
        if mode == m.SINUSOIDAL:
            return spec.amplitude * np.sin(self._wave_phase(r, t, spec))
        if mode in (m.SINUSOIDAL_LINEAR, m.SINUSOIDAL_ELLIPTICAL):
            return self._polarized_wave_local(r, t, spec, waveform="sin")
        if mode == m.PLANE_WAVE:
            if self._amplitude != 0.0:
                return self._polarized_wave_local(r, t, spec, waveform="cos")
            return spec.amplitude * np.cos(self._wave_phase(r, t, spec))
        if mode == m.GAUSSIAN_PULSE:
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
            envelope = np.exp(-np.sum(((r - spec.envelope_center) / spec.envelope_width) ** 2))
            return spec.amplitude * envelope * np.cos(self._wave_phase(r, t, spec))

        special = self._analytical_special(r, t)
        if special is not None:
            return special
        raise RuntimeError(f"unsupported field mode: {self.mode!r}")

    def _polarized_wave_local_batch(
        self,
        pos: NDArray[np.floating],
        t: float,
        spec: PrescribedFieldSpec,
        *,
        waveform: Waveform,
    ) -> NDArray[np.float64]:
        phi = phase_batch(spec.wavevector, pos, spec.omega, t) + spec.phase0
        n = pos.shape[0]
        amp = self._amplitude
        if spec.polarization_kind == PolarizationKind.LINEAR:
            carrier = np.sin(phi) if waveform == "sin" else np.cos(phi)
            ex = amp * np.cos(spec.polarization_psi) * carrier
            ey = amp * np.sin(spec.polarization_psi) * carrier
            return np.column_stack([ex, ey, np.zeros(n, dtype=np.float64)])

        components = elliptical_components if waveform == "sin" else elliptical_components_cos
        ex = np.empty(n, dtype=np.float64)
        ey = np.empty(n, dtype=np.float64)
        for i, p in enumerate(phi):
            a1, a2 = components(p, psi=spec.polarization_psi, delta=spec.polarization_delta)
            ex[i] = amp * a1
            ey[i] = amp * a2
        return np.column_stack([ex, ey, np.zeros(n, dtype=np.float64)])

    def _analytical_special_batch(
        self, pos: NDArray[np.float64], t: float
    ) -> NDArray[np.float64] | None:
        """Hook for subclass-only batched modes (e.g. magnetic MIRROR)."""
        return None

    def _analytical_batch(self, positions: NDArray[np.floating], t: float) -> NDArray[np.float64]:
        pos = np.asarray(positions, dtype=np.float64)
        spec = self.spec
        mode = self.mode
        m = self._MODE
        n = pos.shape[0]

        if mode == m.ZERO:
            return np.zeros((n, 3), dtype=np.float64)
        if mode == m.UNIFORM:
            return np.broadcast_to(spec.amplitude, (n, 3)).copy()
        if mode == m.LINEAR_RAMP:
            value = spec.offset.copy()
            value[spec.ramp_axis] += spec.ramp_rate * t
            return np.broadcast_to(value, (n, 3)).copy()
        if mode == m.SINUSOIDAL:
            phi = phase_batch(spec.wavevector, pos, spec.omega, t) + spec.phase0
            return np.sin(phi)[:, np.newaxis] * spec.amplitude
        if mode in (m.SINUSOIDAL_LINEAR, m.SINUSOIDAL_ELLIPTICAL):
            return self._polarized_wave_local_batch(pos, t, spec, waveform="sin")
        if mode == m.PLANE_WAVE:
            if self._amplitude != 0.0:
                return self._polarized_wave_local_batch(pos, t, spec, waveform="cos")
            phi = phase_batch(spec.wavevector, pos, spec.omega, t) + spec.phase0
            return np.cos(phi)[:, np.newaxis] * spec.amplitude
        if mode == m.GAUSSIAN_PULSE:
            envelope = np.exp(
                -np.sum(((pos - spec.envelope_center) / spec.envelope_width) ** 2, axis=1)
            )
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

    def __init__(self, sources: list[PrescribedFieldSource]) -> None:
        self.sources = list(sources)

    def at(self, pos: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        total = np.zeros(3, dtype=np.float64)
        for source in self.sources:
            total += source.at(pos, t)
        return total

    def at_batch(self, positions: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        pos = np.asarray(positions, dtype=np.float64)
        total = np.zeros((pos.shape[0], 3), dtype=np.float64)
        for source in self.sources:
            if hasattr(source, "at_batch"):
                total += source.at_batch(pos, t)
            else:
                for i in range(pos.shape[0]):
                    total[i] += source.at(pos[i], t)
        return total

    def __add__(self, other: PrescribedFieldSource):
        return type(self)([*self.sources, other])
