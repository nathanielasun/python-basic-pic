"""
Author: Nathaniel Sun
Date: 2026-06-17
Description:
    Prescribed spacetime-dependent magnetic field sources for PIC drivers.

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
    evaluate_gaussian_pulse_local,
    evaluate_polarized_wave_local,
    local_wavevector,
    normalize_envelope_width,
    resolve_k_magnitude,
)
from field_io import (
    FieldDataset,
    FieldInterpolator,
    load_field_file,
    phase,
)

B_COMPONENTS = ("Bx", "By", "Bz")


class MagneticFieldMode(StrEnum):
    ZERO = "zero"
    UNIFORM = "uniform"
    SINUSOIDAL = "sinusoidal"
    SINUSOIDAL_LINEAR = "sinusoidal_linear"
    SINUSOIDAL_ELLIPTICAL = "sinusoidal_elliptical"
    PLANE_WAVE = "plane_wave"
    GAUSSIAN_PULSE = "gaussian_pulse"
    LINEAR_RAMP = "linear_ramp"
    MIRROR = "mirror"
    FILE = "file"


@dataclass
class MagneticFieldSpec:
    """Parameters for analytical magnetic field generators."""

    amplitude: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    offset: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    wavevector: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    omega: float = 0.0
    phase0: float = 0.0
    envelope_center: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    envelope_width: NDArray[np.float64] = field(default_factory=lambda: np.ones(3))
    ramp_axis: int = 2
    ramp_rate: float = 0.0
    B0: float = 0.0
    polarization_kind: PolarizationKind = PolarizationKind.LINEAR
    polarization_psi: float = 0.0
    polarization_delta: float = 0.0
    mirror_scale: float = 1.0


class MagneticFields:
    """
    Spacetime-dependent prescribed magnetic field B(r, t).

    Wave factories build sources in a fixed local frame. Rotate into the PIC
    grid with ``MagneticFields.transform`` and a static ``WaveFrame``::

        local = MagneticFields.sinusoidal_linear(B0=0.5, omega=1e6, k_magnitude=k0)
        frame = WaveFrame.from_spherical(theta=np.deg2rad(30), phi=np.deg2rad(45))
        bfield = MagneticFields.transform(local, frame)
        B_p = bfield.at(particle.get_position(), t=step * dt)
    """

    def __init__(
        self,
        mode: MagneticFieldMode | str = MagneticFieldMode.ZERO,
        spec: MagneticFieldSpec | None = None,
        dataset: FieldDataset | None = None,
    ) -> None:
        self.mode = MagneticFieldMode(mode)
        self.spec = spec if spec is not None else MagneticFieldSpec()
        self.dataset = dataset
        self._interpolator = FieldInterpolator(dataset) if dataset is not None else None

    @classmethod
    def zero(cls) -> MagneticFields:
        return cls(MagneticFieldMode.ZERO)

    @classmethod
    def uniform(cls, amplitude: Sequence[float] | NDArray[np.floating]) -> MagneticFields:
        spec = MagneticFieldSpec(amplitude=np.asarray(amplitude, dtype=np.float64))
        return cls(MagneticFieldMode.UNIFORM, spec)

    @classmethod
    def sinusoidal(
        cls,
        amplitude: Sequence[float] | NDArray[np.floating],
        omega: float,
        wavevector: Sequence[float] | NDArray[np.floating] | None = None,
        phase0: float = 0.0,
    ) -> MagneticFields:
        """Legacy lab-frame sinusoidal plane wave with a fixed amplitude vector."""
        spec = MagneticFieldSpec(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector if wavevector is not None else [0.0, 0.0, 0.0]),
            phase0=phase0,
        )
        return cls(MagneticFieldMode.SINUSOIDAL, spec)

    @classmethod
    def sinusoidal_linear(
        cls,
        B0: float,
        omega: float,
        *,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        wavevector: Sequence[float] | NDArray[np.floating] | None = None,
        phase0: float = 0.0,
        psi: float = 0.0,
    ) -> MagneticFields:
        """
        Linearly polarized sinusoidal plane wave in the local source frame.

        Propagation is along local +z; polarization lies in the local xy-plane
        at angle ``psi``. Rotate into the PIC grid with ``MagneticFields.transform``.
        """
        if wavevector is not None:
            k_mag = resolve_k_magnitude(
                wavevector=np.asarray(wavevector, dtype=np.float64),
                k_magnitude=None,
                wavelength=None,
            )
        else:
            k_mag = resolve_k_magnitude(wavevector=None, k_magnitude=k_magnitude, wavelength=wavelength)
        spec = MagneticFieldSpec(
            B0=B0,
            omega=omega,
            wavevector=local_wavevector(k_mag),
            phase0=phase0,
            polarization_kind=PolarizationKind.LINEAR,
            polarization_psi=psi,
        )
        return cls(MagneticFieldMode.SINUSOIDAL_LINEAR, spec)

    @classmethod
    def sinusoidal_elliptical(
        cls,
        B0: float,
        omega: float,
        *,
        psi: float = 0.0,
        delta: float = 0.0,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        wavevector: Sequence[float] | NDArray[np.floating] | None = None,
        phase0: float = 0.0,
    ) -> MagneticFields:
        """
        Elliptically polarized sinusoidal plane wave in the local source frame.

        ``delta = 0`` (default) reduces to linear polarization. Set ``delta = pi/2``
        and ``psi = pi/4`` for circular polarization.
        """
        if wavevector is not None:
            k_mag = resolve_k_magnitude(
                wavevector=np.asarray(wavevector, dtype=np.float64),
                k_magnitude=None,
                wavelength=None,
            )
        else:
            k_mag = resolve_k_magnitude(wavevector=None, k_magnitude=k_magnitude, wavelength=wavelength)
        spec = MagneticFieldSpec(
            B0=B0,
            omega=omega,
            wavevector=local_wavevector(k_mag),
            phase0=phase0,
            polarization_kind=PolarizationKind.ELLIPTICAL,
            polarization_psi=psi,
            polarization_delta=delta,
        )
        return cls(MagneticFieldMode.SINUSOIDAL_ELLIPTICAL, spec)

    @classmethod
    def plane_wave(
        cls,
        amplitude: Sequence[float] | NDArray[np.floating],
        omega: float,
        wavevector: Sequence[float] | NDArray[np.floating],
        phase0: float = 0.0,
    ) -> MagneticFields:
        """Legacy lab-frame cosine plane wave with a fixed amplitude vector."""
        spec = MagneticFieldSpec(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector, dtype=np.float64),
            phase0=phase0,
        )
        return cls(MagneticFieldMode.PLANE_WAVE, spec)

    @classmethod
    def plane_wave_local(
        cls,
        B0: float,
        omega: float,
        *,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        phase0: float = 0.0,
        psi: float = 0.0,
    ) -> MagneticFields:
        """Cosine plane wave in the local source frame (local +z propagation)."""
        k_mag = resolve_k_magnitude(wavevector=None, k_magnitude=k_magnitude, wavelength=wavelength)
        spec = MagneticFieldSpec(
            B0=B0,
            omega=omega,
            wavevector=local_wavevector(k_mag),
            phase0=phase0,
            polarization_kind=PolarizationKind.LINEAR,
            polarization_psi=psi,
        )
        return cls(MagneticFieldMode.PLANE_WAVE, spec)

    @classmethod
    def plane_wave_from_direction(
        cls,
        B0: float,
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
            B0,
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
        B0: float,
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
            B0,
            omega,
            k_magnitude=k_magnitude,
            wavelength=wavelength,
            phase0=phase0,
        )
        frame = WaveFrame.from_spherical(theta, phi, pol_angle=pol_angle, origin=origin)
        return cls.transform(local, frame)

    @classmethod
    def gaussian_pulse_local(
        cls,
        B0: float,
        omega: float,
        *,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        center: Sequence[float] | NDArray[np.floating] | None = None,
        width: float | Sequence[float] | NDArray[np.floating] = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
    ) -> MagneticFields:
        """
        Gaussian-enveloped cosine pulse in the local source frame.

        Propagation is along local +z; the envelope is axis-aligned in local
        coordinates. Rotate into the PIC grid with ``MagneticFields.transform``.
        """
        k_mag = resolve_k_magnitude(wavevector=None, k_magnitude=k_magnitude, wavelength=wavelength)
        center_arr = (
            np.zeros(3, dtype=np.float64)
            if center is None
            else np.asarray(center, dtype=np.float64)
        )
        if center_arr.shape != (3,):
            raise ValueError("center must have shape (3,)")
        pol_kind = PolarizationKind.LINEAR if delta == 0.0 else PolarizationKind.ELLIPTICAL
        spec = MagneticFieldSpec(
            B0=B0,
            omega=omega,
            wavevector=local_wavevector(k_mag),
            envelope_center=center_arr,
            envelope_width=normalize_envelope_width(width),
            phase0=phase0,
            polarization_kind=pol_kind,
            polarization_psi=psi,
            polarization_delta=delta,
        )
        return cls(MagneticFieldMode.GAUSSIAN_PULSE, spec)

    @classmethod
    def gaussian_pulse_from_direction(
        cls,
        B0: float,
        omega: float,
        k_direction: Sequence[float] | NDArray[np.floating],
        *,
        polarization: Sequence[float] | NDArray[np.floating] | None = None,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        center: Sequence[float] | NDArray[np.floating] | None = None,
        width: float | Sequence[float] | NDArray[np.floating] = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
        origin: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> PolarTransformedField:
        """Gaussian pulse rotated into the lab frame via a static ``WaveFrame``."""
        local = cls.gaussian_pulse_local(
            B0,
            omega,
            k_magnitude=k_magnitude,
            wavelength=wavelength,
            center=center,
            width=width,
            phase0=phase0,
            psi=psi,
            delta=delta,
        )
        frame = WaveFrame.from_basis(k_direction, polarization=polarization, origin=origin)
        return cls.transform(local, frame)

    @classmethod
    def gaussian_pulse_incident(
        cls,
        B0: float,
        omega: float,
        theta: float,
        phi: float,
        *,
        pol_angle: float = 0.0,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        center: Sequence[float] | NDArray[np.floating] | None = None,
        width: float | Sequence[float] | NDArray[np.floating] = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
        origin: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> PolarTransformedField:
        """Gaussian pulse incident at spherical angles ``(theta, phi)``."""
        local = cls.gaussian_pulse_local(
            B0,
            omega,
            k_magnitude=k_magnitude,
            wavelength=wavelength,
            center=center,
            width=width,
            phase0=phase0,
            psi=psi,
            delta=delta,
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
    ) -> MagneticFields:
        """Legacy lab-frame Gaussian-enveloped cosine wave."""
        spec = MagneticFieldSpec(
            amplitude=np.asarray(amplitude, dtype=np.float64),
            omega=omega,
            wavevector=np.asarray(wavevector, dtype=np.float64),
            envelope_center=np.asarray(center, dtype=np.float64),
            envelope_width=np.asarray(width, dtype=np.float64),
            phase0=phase0,
        )
        return cls(MagneticFieldMode.GAUSSIAN_PULSE, spec)

    @classmethod
    def linear_ramp(
        cls,
        ramp_rate: float,
        axis: Literal["x", "y", "z"] = "z",
        offset: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> MagneticFields:
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        spec = MagneticFieldSpec(
            ramp_rate=ramp_rate,
            ramp_axis=axis_index,
            offset=np.asarray(offset if offset is not None else [0.0, 0.0, 0.0]),
        )
        return cls(MagneticFieldMode.LINEAR_RAMP, spec)

    @classmethod
    def mirror_local(cls, B0: float, scale_length: float = 1.0) -> MagneticFields:
        """
        Mirror-field profile along local +z: ``B_z = B0 * (1 + (z/scale)^2)``.

        Rotate into the lab frame with ``MagneticFields.transform``.
        """
        spec = MagneticFieldSpec(B0=B0, mirror_scale=scale_length)
        return cls(MagneticFieldMode.MIRROR, spec)

    @classmethod
    def transform(
        cls,
        source: MagneticFields,
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
    ) -> MagneticFields:
        dataset = load_field_file(path, B_COMPONENTS, hdf5_group=hdf5_group)
        return cls(MagneticFieldMode.FILE, dataset=dataset)

    @classmethod
    def from_csv(cls, path: str | Path) -> MagneticFields:
        return cls.from_file(path)

    @classmethod
    def from_hdf5(cls, path: str | Path, group: str = "magnetic") -> MagneticFields:
        return cls.from_file(path, hdf5_group=group)

    def load_from_file(self, path: str | Path, *, hdf5_group: str | None = None) -> None:
        """Replace the current field definition with data read from CSV or HDF5."""
        self.mode = MagneticFieldMode.FILE
        self.dataset = load_field_file(path, B_COMPONENTS, hdf5_group=hdf5_group)
        self._interpolator = FieldInterpolator(self.dataset)

    def read_from_csv(self, path: str | Path) -> None:
        """Load magnetic field samples from a CSV file."""
        self.load_from_file(path)

    def at(self, pos: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        """Evaluate B at a single position and time in the source coordinate frame."""
        if self.mode == MagneticFieldMode.FILE:
            if self._interpolator is None:
                raise RuntimeError("file-backed magnetic field is not initialized")
            return self._interpolator.at(pos, t)
        return self._analytical(pos, t)

    def on_grid(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        z: NDArray[np.floating],
        t: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate B on a structured mesh (1D coordinate arrays)."""
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        bx = np.zeros_like(xx, dtype=np.float64)
        by = np.zeros_like(xx, dtype=np.float64)
        bz = np.zeros_like(xx, dtype=np.float64)

        for index in np.ndindex(xx.shape):
            field = self.at(np.array([xx[index], yy[index], zz[index]]), t)
            bx[index], by[index], bz[index] = field

        return bx, by, bz

    def __add__(self, other: MagneticFields | PolarTransformedField) -> MagneticFieldsSum:
        return MagneticFieldsSum([self, other])

    def _wave_phase(self, pos: NDArray[np.floating], t: float, spec: MagneticFieldSpec) -> float:
        return phase(spec.wavevector, pos, spec.omega, t) + spec.phase0

    def _polarized_wave_local(
        self,
        pos: NDArray[np.floating],
        t: float,
        spec: MagneticFieldSpec,
        *,
        waveform: Literal["sin", "cos"],
    ) -> NDArray[np.float64]:
        phi = self._wave_phase(pos, t, spec)
        return evaluate_polarized_wave_local(
            phi,
            amplitude=spec.B0,
            polarization_kind=spec.polarization_kind,
            psi=spec.polarization_psi,
            delta=spec.polarization_delta,
            waveform=waveform,
        )

    def _analytical(self, pos: NDArray[np.floating], t: float) -> NDArray[np.float64]:
        r = np.asarray(pos, dtype=np.float64)
        spec = self.spec

        if self.mode == MagneticFieldMode.ZERO:
            return np.zeros(3, dtype=np.float64)
        if self.mode == MagneticFieldMode.UNIFORM:
            return spec.amplitude.copy()
        if self.mode == MagneticFieldMode.LINEAR_RAMP:
            value = spec.offset.copy()
            value[spec.ramp_axis] += spec.ramp_rate * t
            return value
        if self.mode == MagneticFieldMode.SINUSOIDAL:
            phi = self._wave_phase(r, t, spec)
            return spec.amplitude * np.sin(phi)
        if self.mode == MagneticFieldMode.SINUSOIDAL_LINEAR:
            return self._polarized_wave_local(r, t, spec, waveform="sin")
        if self.mode == MagneticFieldMode.SINUSOIDAL_ELLIPTICAL:
            return self._polarized_wave_local(r, t, spec, waveform="sin")
        if self.mode == MagneticFieldMode.PLANE_WAVE:
            if spec.B0 != 0.0:
                return self._polarized_wave_local(r, t, spec, waveform="cos")
            phi = self._wave_phase(r, t, spec)
            return spec.amplitude * np.cos(phi)
        if self.mode == MagneticFieldMode.GAUSSIAN_PULSE:
            if spec.B0 != 0.0:
                return evaluate_gaussian_pulse_local(
                    r,
                    t,
                    amplitude=spec.B0,
                    omega=spec.omega,
                    wavevector=spec.wavevector,
                    center=spec.envelope_center,
                    width=spec.envelope_width,
                    phase0=spec.phase0,
                    polarization_kind=spec.polarization_kind,
                    psi=spec.polarization_psi,
                    delta=spec.polarization_delta,
                )
            phi = self._wave_phase(r, t, spec)
            envelope = np.exp(-np.sum(((r - spec.envelope_center) / spec.envelope_width) ** 2))
            return spec.amplitude * envelope * np.cos(phi)
        if self.mode == MagneticFieldMode.MIRROR:
            z = r[2]
            scale = spec.mirror_scale
            return np.array([0.0, 0.0, spec.B0 * (1.0 + (z / scale) ** 2)], dtype=np.float64)

        raise RuntimeError(f"unsupported magnetic field mode: {self.mode!r}")


class MagneticFieldsSum:
    """Superposition of multiple prescribed magnetic field sources."""

    def __init__(self, sources: list[object]) -> None:
        self.sources = sources

    def at(self, pos: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        total = np.zeros(3, dtype=np.float64)
        for source in self.sources:
            total += source.at(pos, t)
        return total

    def __add__(self, other: object) -> MagneticFieldsSum:
        return MagneticFieldsSum([*self.sources, other])
