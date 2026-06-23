"""
Author: Nathaniel Sun
Date: 2026-06-17
Description:
    Prescribed spacetime-dependent magnetic field sources for PIC drivers.

    Wave sources are authored in a fixed local Cartesian frame (propagation
    along local +z'). Spatial rotation into the PIC grid is applied only
    through ``field_frame.PolarTransformedField`` / ``WaveFrame``.

    Time enters through wave phase ``omega * t`` and polarization parameters.
    There is no time-dependent rotation of the field basis. Evaluation, batch
    evaluation, file loading, and superposition are shared with
    :class:`fields.ElectricFields` via :class:`fields.prescribed.PrescribedField`;
    this class adds the static ``MIRROR`` axial profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, override

import numpy as np
from numpy.typing import NDArray

from .field_frame import PolarizationKind, PolarTransformedField, WaveFrame
from .prescribed import PrescribedField, PrescribedFieldSource, PrescribedFieldSum, Vector3Like

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

    @property
    def scalar_amplitude(self) -> float:
        return self.B0


class MagneticFieldsSum(PrescribedFieldSum):
    """Superposition of multiple prescribed magnetic field sources."""


class MagneticFields(PrescribedField[MagneticFieldMode, MagneticFieldSpec, MagneticFieldsSum]):
    """
    Spacetime-dependent prescribed magnetic field B(r, t).

    Wave factories build sources in a fixed local frame. Rotate into the PIC
    grid with ``MagneticFields.transform`` and a static ``WaveFrame``::

        local = MagneticFields.sinusoidal_linear(B0=0.5, omega=1e6, k_magnitude=k0)
        frame = WaveFrame.from_spherical(theta=np.deg2rad(30), phi=np.deg2rad(45))
        bfield = MagneticFields.transform(local, frame)
        B_p = bfield.at(particle.get_position(), t=step * dt)
    """

    _MODE: ClassVar[type] = MagneticFieldMode
    _SPEC: ClassVar[type] = MagneticFieldSpec
    _SUM: ClassVar[type] = MagneticFieldsSum
    _AMP_ATTR: ClassVar[str] = "B0"
    _COMPONENTS: ClassVar[tuple[str, str, str]] = B_COMPONENTS
    _DEFAULT_HDF5_GROUP: ClassVar[str] = "magnetic"

    @classmethod
    @override
    def transform(cls, source: PrescribedFieldSource, frame: WaveFrame) -> PolarTransformedField:
        return PolarTransformedField(source, frame)

    @classmethod
    def sinusoidal_linear(
        cls,
        B0: float,
        omega: float,
        *,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        wavevector: Vector3Like | None = None,
        phase0: float = 0.0,
        psi: float = 0.0,
    ) -> MagneticFields:
        """
        Linearly polarized sinusoidal plane wave in the local source frame.

        Propagation is along local +z; polarization lies in the local xy-plane
        at angle ``psi``. Rotate into the PIC grid with ``MagneticFields.transform``.
        """
        return cls._local_wave(
            B0, omega,
            mode=MagneticFieldMode.SINUSOIDAL_LINEAR,
            kind=PolarizationKind.LINEAR,
            psi=psi, phase0=phase0,
            k_magnitude=k_magnitude, wavelength=wavelength, wavevector=wavevector,
        )

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
        wavevector: Vector3Like | None = None,
        phase0: float = 0.0,
    ) -> MagneticFields:
        """
        Elliptically polarized sinusoidal plane wave in the local source frame.

        ``delta = 0`` (default) reduces to linear polarization. Set ``delta = pi/2``
        and ``psi = pi/4`` for circular polarization.
        """
        return cls._local_wave(
            B0, omega,
            mode=MagneticFieldMode.SINUSOIDAL_ELLIPTICAL,
            kind=PolarizationKind.ELLIPTICAL,
            psi=psi, delta=delta, phase0=phase0,
            k_magnitude=k_magnitude, wavelength=wavelength, wavevector=wavevector,
        )

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
        return cls._local_wave(
            B0, omega,
            mode=MagneticFieldMode.PLANE_WAVE,
            kind=PolarizationKind.LINEAR,
            psi=psi, phase0=phase0,
            k_magnitude=k_magnitude, wavelength=wavelength,
        )

    @classmethod
    def plane_wave_from_direction(
        cls,
        B0: float,
        omega: float,
        k_direction: Vector3Like,
        *,
        polarization: Vector3Like | None = None,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        phase0: float = 0.0,
        origin: Vector3Like | None = None,
    ) -> PolarTransformedField:
        """Cosine plane wave rotated into the lab frame via a static ``WaveFrame``."""
        local = cls.plane_wave_local(B0, omega, k_magnitude=k_magnitude, wavelength=wavelength, phase0=phase0)
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
        origin: Vector3Like | None = None,
    ) -> PolarTransformedField:
        """Cosine plane wave incident at spherical angles ``(theta, phi)``."""
        local = cls.plane_wave_local(B0, omega, k_magnitude=k_magnitude, wavelength=wavelength, phase0=phase0)
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
        center: Vector3Like | None = None,
        width: float | Vector3Like = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
    ) -> MagneticFields:
        """
        Gaussian-enveloped cosine pulse in the local source frame.

        Propagation is along local +z; the envelope is axis-aligned in local
        coordinates. Rotate into the PIC grid with ``MagneticFields.transform``.
        """
        return cls._gaussian_local(
            B0, omega,
            k_magnitude=k_magnitude, wavelength=wavelength,
            center=center, width=width, phase0=phase0, psi=psi, delta=delta,
        )

    @classmethod
    def gaussian_pulse_from_direction(
        cls,
        B0: float,
        omega: float,
        k_direction: Vector3Like,
        *,
        polarization: Vector3Like | None = None,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        center: Vector3Like | None = None,
        width: float | Vector3Like = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
        origin: Vector3Like | None = None,
    ) -> PolarTransformedField:
        """Gaussian pulse rotated into the lab frame via a static ``WaveFrame``."""
        local = cls.gaussian_pulse_local(
            B0, omega, k_magnitude=k_magnitude, wavelength=wavelength,
            center=center, width=width, phase0=phase0, psi=psi, delta=delta,
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
        center: Vector3Like | None = None,
        width: float | Vector3Like = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
        origin: Vector3Like | None = None,
    ) -> PolarTransformedField:
        """Gaussian pulse incident at spherical angles ``(theta, phi)``."""
        local = cls.gaussian_pulse_local(
            B0, omega, k_magnitude=k_magnitude, wavelength=wavelength,
            center=center, width=width, phase0=phase0, psi=psi, delta=delta,
        )
        frame = WaveFrame.from_spherical(theta, phi, pol_angle=pol_angle, origin=origin)
        return cls.transform(local, frame)

    @classmethod
    def mirror_local(cls, B0: float, scale_length: float = 1.0) -> MagneticFields:
        """
        Mirror-field profile along local +z: ``B_z = B0 * (1 + (z/scale)^2)``.

        Rotate into the lab frame with ``MagneticFields.transform``.
        """
        return cls(MagneticFieldMode.MIRROR, MagneticFieldSpec(B0=B0, mirror_scale=scale_length))

    @override
    def _analytical_special(self, r: NDArray[np.float64], t: float) -> NDArray[np.float64] | None:
        if self.mode == MagneticFieldMode.MIRROR:
            z = r.item(2)
            bz = self.spec.B0 * (1.0 + (z / self.spec.mirror_scale) ** 2)
            return np.array([0.0, 0.0, bz], dtype=np.float64)
        return None

    @override
    def _analytical_special_batch(
        self, pos: NDArray[np.float64], t: float
    ) -> NDArray[np.float64] | None:
        if self.mode == MagneticFieldMode.MIRROR:
            out = np.zeros((pos.shape[0], 3), dtype=np.float64)
            out[:, 2] = self.spec.B0 * (1.0 + (pos[:, 2] / self.spec.mirror_scale) ** 2)
            return out
        return None