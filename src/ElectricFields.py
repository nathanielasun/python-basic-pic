"""
Author: Nathaniel Sun
Date: 2026-06-17
Description:
    Prescribed spacetime-dependent electric field sources for PIC drivers.

    Wave sources are authored in a fixed local Cartesian frame (propagation
    along local +z'). Spatial rotation into the PIC grid is applied only
    through ``field_frame.PolarTransformedField`` / ``WaveFrame``.

    Time enters through wave phase ``omega * t`` and polarization parameters.
    There is no time-dependent rotation of the field basis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from field_frame import (
    PolarTransformedField,
    PolarizationKind,
    TransformedField,
    WaveFrame,
    elliptical_components,
    elliptical_components_cos,
)
from field_io import (
    FieldDataset,
    FieldInterpolator,
    load_field_file,
    phase,
)

E_COMPONENTS = ("Ex", "Ey", "Ez")


class ElectricFieldMode(StrEnum):
    ZERO = "zero"
    UNIFORM = "uniform"
    SINUSOIDAL = "sinusoidal"
    SINUSOIDAL_LINEAR = "sinusoidal_linear"
    SINUSOIDAL_ELLIPTICAL = "sinusoidal_elliptical"
    PLANE_WAVE = "plane_wave"
    GAUSSIAN_PULSE = "gaussian_pulse"
    LINEAR_RAMP = "linear_ramp"
    FILE = "file"


def _resolve_k_magnitude(
    *,
    wavevector: NDArray[np.float64] | None,
    k_magnitude: float | None,
    wavelength: float | None,
) -> float:
    if wavevector is not None:
        return float(np.linalg.norm(wavevector))
    if k_magnitude is not None:
        return k_magnitude
    if wavelength is not None:
        if wavelength <= 0.0:
            raise ValueError("wavelength must be positive")
        return 2.0 * np.pi / wavelength
    raise ValueError("one of wavevector, k_magnitude, or wavelength is required")


@dataclass
class ElectricFieldSpec:
    """Parameters for analytical electric field generators."""

    amplitude: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    offset: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    wavevector: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    omega: float = 0.0
    phase0: float = 0.0
    envelope_center: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    envelope_width: NDArray[np.float64] = field(default_factory=lambda: np.ones(3))
    ramp_axis: int = 2
    ramp_rate: float = 0.0
    E0: float = 0.0
    polarization_kind: PolarizationKind = PolarizationKind.LINEAR
    polarization_psi: float = 0.0
    polarization_delta: float = 0.0


class ElectricFields:
    """
    Spacetime-dependent prescribed electric field E(r, t).

    Wave factories build sources in a fixed local frame. Rotate into the PIC
    grid with ``ElectricFields.transform`` and a static ``WaveFrame``::

        local = ElectricFields.sinusoidal_linear(E0=1e8, omega=2e15, k_magnitude=k0)
        frame = WaveFrame.from_spherical(theta=np.deg2rad(30), phi=np.deg2rad(45))
        efield = ElectricFields.transform(local, frame)
        E_p = efield.at(particle.get_position(), t=step * dt)
    """

    def __init__(
        self,
        mode: ElectricFieldMode | str = ElectricFieldMode.ZERO,
        spec: ElectricFieldSpec | None = None,
        dataset: FieldDataset | None = None,
    ) -> None:
        self.mode = ElectricFieldMode(mode)
        self.spec = spec if spec is not None else ElectricFieldSpec()
        self.dataset = dataset
        self._interpolator = FieldInterpolator(dataset) if dataset is not None else None

    @classmethod
    def _local_wavevector(cls, k_magnitude: float) -> NDArray[np.float64]:
        return np.array([0.0, 0.0, k_magnitude], dtype=np.float64)

    @classmethod
    def zero(cls) -> ElectricFields:
        return cls(ElectricFieldMode.ZERO)

    @classmethod
    def uniform(cls, amplitude: Sequence[float] | NDArray[np.floating]) -> ElectricFields:
        spec = ElectricFieldSpec(amplitude=np.asarray(amplitude, dtype=np.float64))
        return cls(ElectricFieldMode.UNIFORM, spec)

    @classmethod
    def sinusoidal(
        cls,
        amplitude: Sequence[float] | NDArray[np.floating],
        omega: float,
        wavevector: Sequence[float] | NDArray[np.floating] | None = None,
        phase0: float = 0.0,
    ) -> ElectricFields:
        """Legacy lab-frame sinusoidal plane wave with a fixed amplitude vector."""
        spec = ElectricFieldSpec(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector if wavevector is not None else [0.0, 0.0, 0.0]),
            phase0=phase0,
        )
        return cls(ElectricFieldMode.SINUSOIDAL, spec)

    @classmethod
    def sinusoidal_linear(
        cls,
        E0: float,
        omega: float,
        *,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        wavevector: Sequence[float] | NDArray[np.floating] | None = None,
        phase0: float = 0.0,
        psi: float = 0.0,
    ) -> ElectricFields:
        """
        Linearly polarized sinusoidal plane wave in the local source frame.

        Propagation is along local +z; polarization lies in the local xy-plane
        at angle ``psi``. Rotate into the PIC grid with ``ElectricFields.transform``.
        """
        if wavevector is not None:
            k_mag = _resolve_k_magnitude(
                wavevector=np.asarray(wavevector, dtype=np.float64),
                k_magnitude=None,
                wavelength=None,
            )
        else:
            k_mag = _resolve_k_magnitude(wavevector=None, k_magnitude=k_magnitude, wavelength=wavelength)
        spec = ElectricFieldSpec(
            E0=E0,
            omega=omega,
            wavevector=cls._local_wavevector(k_mag),
            phase0=phase0,
            polarization_kind=PolarizationKind.LINEAR,
            polarization_psi=psi,
        )
        return cls(ElectricFieldMode.SINUSOIDAL_LINEAR, spec)

    @classmethod
    def sinusoidal_elliptical(
        cls,
        E0: float,
        omega: float,
        *,
        psi: float = 0.0,
        delta: float = 0.0,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        wavevector: Sequence[float] | NDArray[np.floating] | None = None,
        phase0: float = 0.0,
    ) -> ElectricFields:
        """
        Elliptically polarized sinusoidal plane wave in the local source frame.

        ``delta = 0`` (default) reduces to linear polarization. Set ``delta = pi/2``
        and ``psi = pi/4`` for circular polarization.
        """
        if wavevector is not None:
            k_mag = _resolve_k_magnitude(
                wavevector=np.asarray(wavevector, dtype=np.float64),
                k_magnitude=None,
                wavelength=None,
            )
        else:
            k_mag = _resolve_k_magnitude(wavevector=None, k_magnitude=k_magnitude, wavelength=wavelength)
        spec = ElectricFieldSpec(
            E0=E0,
            omega=omega,
            wavevector=cls._local_wavevector(k_mag),
            phase0=phase0,
            polarization_kind=PolarizationKind.ELLIPTICAL,
            polarization_psi=psi,
            polarization_delta=delta,
        )
        return cls(ElectricFieldMode.SINUSOIDAL_ELLIPTICAL, spec)

    @classmethod
    def plane_wave(
        cls,
        amplitude: Sequence[float] | NDArray[np.floating],
        omega: float,
        wavevector: Sequence[float] | NDArray[np.floating],
        phase0: float = 0.0,
    ) -> ElectricFields:
        """Legacy lab-frame cosine plane wave with a fixed amplitude vector."""
        spec = ElectricFieldSpec(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector, dtype=np.float64),
            phase0=phase0,
        )
        return cls(ElectricFieldMode.PLANE_WAVE, spec)

    @classmethod
    def plane_wave_local(
        cls,
        E0: float,
        omega: float,
        *,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        phase0: float = 0.0,
        psi: float = 0.0,
    ) -> ElectricFields:
        """Cosine plane wave in the local source frame (local +z propagation)."""
        k_mag = _resolve_k_magnitude(wavevector=None, k_magnitude=k_magnitude, wavelength=wavelength)
        spec = ElectricFieldSpec(
            E0=E0,
            omega=omega,
            wavevector=cls._local_wavevector(k_mag),
            phase0=phase0,
            polarization_kind=PolarizationKind.LINEAR,
            polarization_psi=psi,
        )
        return cls(ElectricFieldMode.PLANE_WAVE, spec)

    @classmethod
    def plane_wave_from_direction(
        cls,
        E0: float,
        omega: float,
        k_direction: Sequence[float] | NDArray[np.floating],
        *,
        polarization: Sequence[float] | NDArray[np.floating] | None = None,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        phase0: float = 0.0,
        origin: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> PolarTransformedField:
        """Cosine plane wave rotated into the lab frame via a static ``WaveFrame``."""
        local = cls.plane_wave_local(
            E0,
            omega,
            k_magnitude=k_magnitude,
            wavelength=wavelength,
            phase0=phase0,
        )
        frame = WaveFrame.from_basis(k_direction, polarization=polarization, origin=origin)
        return cls.transform(local, frame)

    @classmethod
    def plane_wave_incident(
        cls,
        E0: float,
        omega: float,
        theta: float,
        phi: float,
        *,
        pol_angle: float = 0.0,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        phase0: float = 0.0,
        origin: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> PolarTransformedField:
        """Cosine plane wave incident at spherical angles ``(theta, phi)``."""
        local = cls.plane_wave_local(
            E0,
            omega,
            k_magnitude=k_magnitude,
            wavelength=wavelength,
            phase0=phase0,
        )
        frame = WaveFrame.from_spherical(theta, phi, pol_angle=pol_angle, origin=origin)
        return cls.transform(local, frame)

    @classmethod
    def gaussian_pulse(
        cls,
        amplitude: Sequence[float] | NDArray[np.floating],
        omega: float,
        wavevector: Sequence[float] | NDArray[np.floating],
        center: Sequence[float] | NDArray[np.floating],
        width: Sequence[float] | NDArray[np.floating],
        phase0: float = 0.0,
    ) -> ElectricFields:
        spec = ElectricFieldSpec(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector, dtype=np.float64),
            envelope_center=np.asarray(center, dtype=np.float64),
            envelope_width=np.asarray(width, dtype=np.float64),
            phase0=phase0,
        )
        return cls(ElectricFieldMode.GAUSSIAN_PULSE, spec)

    @classmethod
    def linear_ramp(
        cls,
        ramp_rate: float,
        axis: Literal["x", "y", "z"] = "z",
        offset: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> ElectricFields:
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        spec = ElectricFieldSpec(
            ramp_rate=ramp_rate,
            ramp_axis=axis_index,
            offset=np.asarray(offset if offset is not None else [0.0, 0.0, 0.0]),
        )
        return cls(ElectricFieldMode.LINEAR_RAMP, spec)

    @classmethod
    def transform(
        cls,
        source: ElectricFields,
        frame: WaveFrame,
    ) -> PolarTransformedField:
        """
        Apply a static polar transform from the local source frame to the lab frame.

        Time-dependent rotation is intentionally unsupported here; temporal
        polarization changes belong in ``sinusoidal_elliptical`` phase parameters.
        """
        return PolarTransformedField(source, frame)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        hdf5_group: str | None = None,
    ) -> ElectricFields:
        dataset = load_field_file(path, E_COMPONENTS, hdf5_group=hdf5_group)
        return cls(ElectricFieldMode.FILE, dataset=dataset)

    @classmethod
    def from_csv(cls, path: str | Path) -> ElectricFields:
        return cls.from_file(path)

    @classmethod
    def from_hdf5(cls, path: str | Path, group: str = "electric") -> ElectricFields:
        return cls.from_file(path, hdf5_group=group)

    def load_from_file(self, path: str | Path, *, hdf5_group: str | None = None) -> None:
        """Replace the current field definition with data read from CSV or HDF5."""
        self.mode = ElectricFieldMode.FILE
        self.dataset = load_field_file(path, E_COMPONENTS, hdf5_group=hdf5_group)
        self._interpolator = FieldInterpolator(self.dataset)

    def read_from_csv(self, path: str | Path) -> None:
        """Load electric field samples from a CSV file."""
        self.load_from_file(path)

    def at(self, pos: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        """Evaluate E at a single position and time in the source coordinate frame."""
        if self.mode == ElectricFieldMode.FILE:
            if self._interpolator is None:
                raise RuntimeError("file-backed electric field is not initialized")
            return self._interpolator.at(pos, t)
        return self._analytical(pos, t)

    def on_grid(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        z: NDArray[np.floating],
        t: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate E on a structured mesh (1D coordinate arrays)."""
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        ex = np.zeros_like(xx, dtype=np.float64)
        ey = np.zeros_like(xx, dtype=np.float64)
        ez = np.zeros_like(xx, dtype=np.float64)

        for index in np.ndindex(xx.shape):
            field = self.at(np.array([xx[index], yy[index], zz[index]]), t)
            ex[index], ey[index], ez[index] = field

        return ex, ey, ez

    def __add__(self, other: ElectricFields | PolarTransformedField) -> ElectricFieldsSum:
        return ElectricFieldsSum([self, other])

    def _wave_phase(self, pos: NDArray[np.floating], t: float, spec: ElectricFieldSpec) -> float:
        return phase(spec.wavevector, pos, spec.omega, t) + spec.phase0

    def _polarized_wave_local(
        self,
        pos: NDArray[np.floating],
        t: float,
        spec: ElectricFieldSpec,
        *,
        waveform: Literal["sin", "cos"],
    ) -> NDArray[np.float64]:
        phi = self._wave_phase(pos, t, spec)

        if spec.polarization_kind == PolarizationKind.LINEAR:
            carrier = np.sin(phi) if waveform == "sin" else np.cos(phi)
            a1 = np.cos(spec.polarization_psi) * carrier
            a2 = np.sin(spec.polarization_psi) * carrier
            return np.array([spec.E0 * a1, spec.E0 * a2, 0.0], dtype=np.float64)

        if waveform == "sin":
            a1, a2 = elliptical_components(
                phi,
                psi=spec.polarization_psi,
                delta=spec.polarization_delta,
            )
        else:
            a1, a2 = elliptical_components_cos(
                phi,
                psi=spec.polarization_psi,
                delta=spec.polarization_delta,
            )
        return np.array([spec.E0 * a1, spec.E0 * a2, 0.0], dtype=np.float64)

    def _analytical(self, pos: NDArray[np.floating], t: float) -> NDArray[np.float64]:
        r = np.asarray(pos, dtype=np.float64)
        spec = self.spec

        if self.mode == ElectricFieldMode.ZERO:
            return np.zeros(3, dtype=np.float64)
        if self.mode == ElectricFieldMode.UNIFORM:
            return spec.amplitude.copy()
        if self.mode == ElectricFieldMode.LINEAR_RAMP:
            value = spec.offset.copy()
            value[spec.ramp_axis] += spec.ramp_rate * t
            return value
        if self.mode == ElectricFieldMode.SINUSOIDAL:
            phi = self._wave_phase(r, t, spec)
            return spec.amplitude * np.sin(phi)
        if self.mode == ElectricFieldMode.SINUSOIDAL_LINEAR:
            return self._polarized_wave_local(r, t, spec, waveform="sin")
        if self.mode == ElectricFieldMode.SINUSOIDAL_ELLIPTICAL:
            return self._polarized_wave_local(r, t, spec, waveform="sin")
        if self.mode == ElectricFieldMode.PLANE_WAVE:
            if spec.E0 != 0.0:
                return self._polarized_wave_local(r, t, spec, waveform="cos")
            phi = self._wave_phase(r, t, spec)
            return spec.amplitude * np.cos(phi)
        if self.mode == ElectricFieldMode.GAUSSIAN_PULSE:
            phi = self._wave_phase(r, t, spec)
            envelope = np.exp(-np.sum(((r - spec.envelope_center) / spec.envelope_width) ** 2))
            return spec.amplitude * envelope * np.cos(phi)

        raise RuntimeError(f"unsupported electric field mode: {self.mode!r}")


class ElectricFieldsSum:
    """Superposition of multiple prescribed electric field sources."""

    def __init__(self, sources: list[object]) -> None:
        self.sources = sources

    def at(self, pos: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        total = np.zeros(3, dtype=np.float64)
        for source in self.sources:
            total += source.at(pos, t)
        return total

    def __add__(self, other: object) -> ElectricFieldsSum:
        return ElectricFieldsSum([*self.sources, other])
