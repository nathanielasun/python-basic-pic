"""
Author: Nathaniel Sun
Date: 2026-06-17
Description:
    Prescribed spacetime-dependent electric field sources for PIC drivers.

    Wave sources are authored in a fixed local Cartesian frame (propagation
    along local +z'). Spatial rotation into the PIC grid is applied only
    through ``field_frame.PolarTransformedField`` / ``WaveFrame``.

    Time enters through wave phase ``omega * t`` and polarization parameters.
    There is no time-dependent rotation of the field basis. Evaluation, batch
    evaluation, file loading, and superposition are shared with
    :class:`fields.MagneticFields` via :class:`fields.prescribed.PrescribedField`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .field_frame import PolarTransformedField, WaveFrame, resolve_k_magnitude
from .MagneticFields import MagneticFields
from .prescribed import PrescribedField, PrescribedFieldSum
from .types import EnvelopeWidthLike, PolarizationKind, Vector3Like

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

    @property
    def scalar_amplitude(self) -> float:
        return self.E0


class ElectricFieldsSum(PrescribedFieldSum):
    """Superposition of multiple prescribed electric field sources."""


class ElectricFields(PrescribedField[ElectricFieldMode, ElectricFieldSpec, ElectricFieldsSum]):
    """
    Spacetime-dependent prescribed electric field E(r, t).

    Wave factories build sources in a fixed local frame. Rotate into the PIC
    grid with ``ElectricFields.transform`` and a static ``WaveFrame``::

        local = ElectricFields.sinusoidal_linear(E0=1e8, omega=2e15, k_magnitude=k0)
        frame = WaveFrame.from_spherical(theta=np.deg2rad(30), phi=np.deg2rad(45))
        efield = ElectricFields.transform(local, frame)
        E_p = efield.at(particle.get_position(), t=step * dt)
    """

    _MODE: ClassVar[type] = ElectricFieldMode
    _SPEC: ClassVar[type] = ElectricFieldSpec
    _SUM: ClassVar[type] = ElectricFieldsSum
    _AMP_ATTR: ClassVar[str] = "E0"
    _COMPONENTS: ClassVar[tuple[str, str, str]] = E_COMPONENTS
    _DEFAULT_HDF5_GROUP: ClassVar[str] = "electric"

    @classmethod
    def sinusoidal_linear(
        cls,
        E0: float,
        omega: float,
        *,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        wavevector: Vector3Like | None = None,
        phase0: float = 0.0,
        psi: float = 0.0,
    ) -> ElectricFields:
        """
        Linearly polarized sinusoidal plane wave in the local source frame.

        Propagation is along local +z; polarization lies in the local xy-plane
        at angle ``psi``. Rotate into the PIC grid with ``ElectricFields.transform``.
        """
        return cls._local_wave(
            E0, omega,
            mode=ElectricFieldMode.SINUSOIDAL_LINEAR,
            kind=PolarizationKind.LINEAR,
            psi=psi, phase0=phase0,
            k_magnitude=k_magnitude, wavelength=wavelength, wavevector=wavevector,
        )

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
        wavevector: Vector3Like | None = None,
        phase0: float = 0.0,
    ) -> ElectricFields:
        """
        Elliptically polarized sinusoidal plane wave in the local source frame.

        ``delta = 0`` (default) reduces to linear polarization. Set ``delta = pi/2``
        and ``psi = pi/4`` for circular polarization.
        """
        return cls._local_wave(
            E0, omega,
            mode=ElectricFieldMode.SINUSOIDAL_ELLIPTICAL,
            kind=PolarizationKind.ELLIPTICAL,
            psi=psi, delta=delta, phase0=phase0,
            k_magnitude=k_magnitude, wavelength=wavelength, wavevector=wavevector,
        )

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
        return cls._local_wave(
            E0, omega,
            mode=ElectricFieldMode.PLANE_WAVE,
            kind=PolarizationKind.LINEAR,
            psi=psi, phase0=phase0,
            k_magnitude=k_magnitude, wavelength=wavelength,
        )

    @classmethod
    def plane_wave_from_direction(
        cls,
        E0: float,
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
        local = cls.plane_wave_local(E0, omega, k_magnitude=k_magnitude, wavelength=wavelength, phase0=phase0)
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
        origin: Vector3Like | None = None,
    ) -> PolarTransformedField:
        """Cosine plane wave incident at spherical angles ``(theta, phi)``."""
        local = cls.plane_wave_local(E0, omega, k_magnitude=k_magnitude, wavelength=wavelength, phase0=phase0)
        frame = WaveFrame.from_spherical(theta, phi, pol_angle=pol_angle, origin=origin)
        return cls.transform(local, frame)

    @classmethod
    def gaussian_pulse_local(
        cls,
        E0: float,
        omega: float,
        *,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        center: Vector3Like | None = None,
        width: EnvelopeWidthLike = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
    ) -> ElectricFields:
        """
        Gaussian-enveloped cosine pulse in the local source frame.

        Propagation is along local +z; the envelope is axis-aligned in local
        coordinates. Rotate into the PIC grid with ``ElectricFields.transform``.
        """
        return cls._gaussian_local(
            E0, omega,
            k_magnitude=k_magnitude, wavelength=wavelength,
            center=center, width=width, phase0=phase0, psi=psi, delta=delta,
        )

    @classmethod
    def gaussian_pulse_from_direction(
        cls,
        E0: float,
        omega: float,
        k_direction: Vector3Like,
        *,
        polarization: Vector3Like | None = None,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        center: Vector3Like | None = None,
        width: EnvelopeWidthLike = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
        origin: Vector3Like | None = None,
    ) -> PolarTransformedField:
        """Gaussian pulse rotated into the lab frame via a static ``WaveFrame``."""
        local = cls.gaussian_pulse_local(
            E0, omega, k_magnitude=k_magnitude, wavelength=wavelength,
            center=center, width=width, phase0=phase0, psi=psi, delta=delta,
        )
        frame = WaveFrame.from_basis(k_direction, polarization=polarization, origin=origin)
        return cls.transform(local, frame)

    @classmethod
    def gaussian_pulse_incident(
        cls,
        E0: float,
        omega: float,
        theta: float,
        phi: float,
        *,
        pol_angle: float = 0.0,
        k_magnitude: float | None = None,
        wavelength: float | None = None,
        center: Vector3Like | None = None,
        width: EnvelopeWidthLike = 1.0,
        phase0: float = 0.0,
        psi: float = 0.0,
        delta: float = 0.0,
        origin: Vector3Like | None = None,
    ) -> PolarTransformedField:
        """Gaussian pulse incident at spherical angles ``(theta, phi)``."""
        local = cls.gaussian_pulse_local(
            E0, omega, k_magnitude=k_magnitude, wavelength=wavelength,
            center=center, width=width, phase0=phase0, psi=psi, delta=delta,
        )
        frame = WaveFrame.from_spherical(theta, phi, pol_angle=pol_angle, origin=origin)
        return cls.transform(local, frame)


def plane_wave_em_pair(
    E0: float,
    omega: float,
    *,
    k_magnitude: float | None = None,
    wavelength: float | None = None,
    phase0: float = 0.0,
    psi: float = 0.0,
    c: float | None = None,
) -> tuple[ElectricFields, MagneticFields]:
    """
    Maxwell-consistent vacuum plane-wave pair in the local source frame (|E| = c|B|).

    Returns ``(ElectricFields, MagneticFields)`` for EM-PIC drivers.
    """
    k_mag = resolve_k_magnitude(k_magnitude=k_magnitude, wavelength=wavelength)
    if c is None:
        c = 299792458.0
    efield = ElectricFields.plane_wave_local(E0, omega, k_magnitude=k_mag, phase0=phase0, psi=psi)
    # B = (k_hat x E)/c must be perpendicular to E, not parallel: rotate the transverse
    # polarization vector by +pi/2 within the wave frame so |B| = |E|/c, in phase, and the
    # Poynting vector E x B points along +k. (Linear polarization only, which is all this
    # helper emits; the magnitude factor E0/c is unchanged.)
    bfield = MagneticFields.plane_wave_local(
        E0 / c, omega, k_magnitude=k_mag, phase0=phase0, psi=psi + np.pi / 2.0
    )
    return efield, bfield
