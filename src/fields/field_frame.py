"""
Wave-frame geometry for prescribed PIC fields.

A ``WaveFrame`` is a **static** right-handed local Cartesian basis attached to
the simulation (lab) frame. Orientation does not vary with time; temporal
evolution of wave sources belongs in their phase ``omega * t`` term, not in
frame rotation.

All spatial rotation / polar incidence into the PIC cube is applied through
``PolarTransformedField`` (alias ``TransformedField``). Field sources such as
``ElectricFields.sinusoidal_linear`` are authored in a fixed local frame with
propagation along local +z'.

Typical workflow::

    local = ElectricFields.sinusoidal_linear(E0=1e8, omega=2e15, k_magnitude=k0)
    frame = WaveFrame.from_spherical(theta=np.deg2rad(30), phi=np.deg2rad(45))
    efield = PolarTransformedField(local, frame)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class PolarizationKind(StrEnum):
    """Transverse polarization state for sinusoidal wave factories."""

    LINEAR = "linear"
    ELLIPTICAL = "elliptical"


def normalize_vector(
    vector: Sequence[float] | NDArray[np.floating],
    *,
    name: str = "vector",
) -> NDArray[np.float64]:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return arr / norm


def direction_from_spherical(theta: float, phi: float) -> NDArray[np.float64]:
    """
    Unit propagation direction from spherical angles.

    ``theta`` is polar angle from +z (colatitude), ``phi`` is azimuth in the xy-plane.
    """
    sin_theta = np.sin(theta)
    return np.array(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)],
        dtype=np.float64,
    )


def transverse_basis(
    k_hat: NDArray[np.float64],
    polarization: Sequence[float] | NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build an orthonormal transverse pair (e1, e2) with e1 x e2 = k_hat.

    If ``polarization`` is given it defines the preferred e1 direction (projected
    orthogonal to ``k_hat``).
    """
    if polarization is None:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(reference, k_hat))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        e1 = reference - np.dot(reference, k_hat) * k_hat
    else:
        pol = np.asarray(polarization, dtype=np.float64)
        e1 = pol - np.dot(pol, k_hat) * k_hat

    e1_norm = float(np.linalg.norm(e1))
    if e1_norm <= 1e-30:
        raise ValueError("polarization must not be parallel to propagation direction")
    e1 /= e1_norm
    e2 = np.cross(k_hat, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1e-30)
    return e1, e2


def elliptical_components(
    phi: float,
    *,
    psi: float = 0.0,
    delta: float = 0.0,
) -> tuple[float, float]:
    """
    Return transverse amplitudes (a1, a2) for a sine carrier at scalar phase ``phi``.

    With ``delta = 0`` both components stay in phase and describe linear polarization
    tilted by ``psi`` in the (e1, e2) plane. With ``delta = pi/2`` and ``psi = pi/4``
    the components are ``sin(phi)`` and ``cos(phi)``, giving circular polarization
    with constant magnitude ``E0``.
    """
    if delta == 0.0:
        scalar = np.sin(phi)
        return np.cos(psi) * scalar, np.sin(psi) * scalar
    ratio = np.tan(psi)
    return np.sin(phi), ratio * np.sin(phi + delta)


def elliptical_components_cos(
    phi: float,
    *,
    psi: float = 0.0,
    delta: float = 0.0,
) -> tuple[float, float]:
    """Cosine-carrier counterpart to :func:`elliptical_components`."""
    if delta == 0.0:
        scalar = np.cos(phi)
        return np.cos(psi) * scalar, np.sin(psi) * scalar
    ratio = np.tan(psi)
    return np.cos(phi), ratio * np.cos(phi + delta)


def resolve_k_magnitude(
    *,
    wavevector: NDArray[np.float64] | None = None,
    k_magnitude: float | None = None,
    wavelength: float | None = None,
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


def local_wavevector(k_magnitude: float) -> NDArray[np.float64]:
    return np.array([0.0, 0.0, k_magnitude], dtype=np.float64)


def evaluate_polarized_wave_local(
    phi: float,
    *,
    amplitude: float,
    polarization_kind: PolarizationKind,
    psi: float = 0.0,
    delta: float = 0.0,
    waveform: Literal["sin", "cos"] = "sin",
) -> NDArray[np.float64]:
    """Return transverse wave components [Fx, Fy, 0] in the local source frame."""
    if polarization_kind == PolarizationKind.LINEAR:
        carrier = np.sin(phi) if waveform == "sin" else np.cos(phi)
        a1 = np.cos(psi) * carrier
        a2 = np.sin(psi) * carrier
        return np.array([amplitude * a1, amplitude * a2, 0.0], dtype=np.float64)

    if waveform == "sin":
        a1, a2 = elliptical_components(phi, psi=psi, delta=delta)
    else:
        a1, a2 = elliptical_components_cos(phi, psi=psi, delta=delta)
    return np.array([amplitude * a1, amplitude * a2, 0.0], dtype=np.float64)


def normalize_envelope_width(
    width: float | Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float64]:
    """Broadcast a scalar or length-3 width to a positive ``(3,)`` envelope scale."""
    if isinstance(width, (int, float, np.floating)):
        scale = float(width)
        if scale <= 0.0:
            raise ValueError("width must be positive")
        return np.array([scale, scale, scale], dtype=np.float64)

    arr = np.asarray(width, dtype=np.float64)
    if arr.shape == ():
        scale = float(arr)
        if scale <= 0.0:
            raise ValueError("width must be positive")
        return np.array([scale, scale, scale], dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("width must be a scalar or length-3 sequence")
    if np.any(arr <= 0.0):
        raise ValueError("width components must be positive")
    return arr


def gaussian_envelope_local(
    r_local: Sequence[float] | NDArray[np.floating],
    center: NDArray[np.float64],
    width: NDArray[np.float64],
) -> float:
    """Axis-aligned Gaussian envelope evaluated in the local source frame."""
    r = np.asarray(r_local, dtype=np.float64)
    return float(np.exp(-np.sum(((r - center) / width) ** 2)))


def evaluate_gaussian_pulse_local(
    r: Sequence[float] | NDArray[np.floating],
    t: float,
    *,
    amplitude: float,
    omega: float,
    wavevector: NDArray[np.float64],
    center: NDArray[np.float64],
    width: NDArray[np.float64],
    phase0: float = 0.0,
    polarization_kind: PolarizationKind = PolarizationKind.LINEAR,
    psi: float = 0.0,
    delta: float = 0.0,
) -> NDArray[np.float64]:
    """
    Gaussian-enveloped cosine pulse in the local source frame (+z propagation).

    The envelope is axis-aligned in local coordinates; rotate into the lab frame
    with ``PolarTransformedField``.
    """
    from .field_io import phase

    r_arr = np.asarray(r, dtype=np.float64)
    phi = phase(wavevector, r_arr, omega, t) + phase0
    envelope = gaussian_envelope_local(r_arr, center, width)
    field = evaluate_polarized_wave_local(
        phi,
        amplitude=amplitude,
        polarization_kind=polarization_kind,
        psi=psi,
        delta=delta,
        waveform="cos",
    )
    return envelope * field


@dataclass(frozen=True)
class WaveFrame:
    """
    Static local Cartesian frame embedded in the PIC lab frame.

    Columns of ``lab_from_local`` are the local basis vectors expressed in lab
    coordinates: local x -> e1, local y -> e2, local z -> k_hat (propagation).

    The frame is time-independent. Do not use this type for time-varying
    rotations; use elliptical polarization parameters on the source instead.
    """

    origin: NDArray[np.float64]
    lab_from_local: NDArray[np.float64]

    @property
    def e1(self) -> NDArray[np.float64]:
        """First transverse basis vector (local x / polarization reference) in lab coords."""
        return self.lab_from_local[:, 0].copy()

    @property
    def e2(self) -> NDArray[np.float64]:
        """Second transverse basis vector (local y) in lab coords; e1 x e2 = k_hat."""
        return self.lab_from_local[:, 1].copy()

    @property
    def k_hat(self) -> NDArray[np.float64]:
        """Unit propagation direction (local +z) expressed in lab coordinates."""
        return self.lab_from_local[:, 2].copy()

    @classmethod
    def from_basis(
        cls,
        k_direction: Sequence[float] | NDArray[np.floating],
        *,
        polarization: Sequence[float] | NDArray[np.floating] | None = None,
        origin: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> WaveFrame:
        """
        Build a frame from an explicit propagation direction in lab coordinates.

        ``k_direction`` need not be unit length; it defines local +z after
        normalization. Optional ``polarization`` pins the local x-axis (e1) by
        projecting a lab-frame vector orthogonal to ``k_hat``. Use this when
        incidence is specified as a Cartesian vector rather than spherical angles.
        """
        k_hat = normalize_vector(k_direction, name="k_direction")
        e1, e2 = transverse_basis(k_hat, polarization)
        origin_arr = (
            np.zeros(3, dtype=np.float64)
            if origin is None
            else np.asarray(origin, dtype=np.float64)
        )
        if origin_arr.shape != (3,):
            raise ValueError("origin must have shape (3,)")
        lab_from_local = np.column_stack([e1, e2, k_hat])
        return cls(origin=origin_arr, lab_from_local=lab_from_local)

    @classmethod
    def from_spherical(
        cls,
        theta: float,
        phi: float,
        *,
        pol_angle: float = 0.0,
        origin: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> WaveFrame:
        """
        Build a static frame from spherical incidence angles into the PIC cube.

        ``theta`` and ``phi`` define propagation direction ``k_hat``. ``pol_angle``
        is a fixed rotation of the local x-axis (e1) about ``k_hat``.
        """
        k_hat = direction_from_spherical(theta, phi)
        e1, e2 = transverse_basis(k_hat, polarization=None)
        if pol_angle != 0.0:
            e1_rot = np.cos(pol_angle) * e1 + np.sin(pol_angle) * e2
            e2_rot = -np.sin(pol_angle) * e1 + np.cos(pol_angle) * e2
            e1, e2 = e1_rot, e2_rot
        origin_arr = (
            np.zeros(3, dtype=np.float64)
            if origin is None
            else np.asarray(origin, dtype=np.float64)
        )
        return cls(origin=origin_arr, lab_from_local=np.column_stack([e1, e2, k_hat]))

    @classmethod
    def identity(cls, origin: Sequence[float] | NDArray[np.floating] | None = None) -> WaveFrame:
        """
        Lab-aligned frame: local axes coincide with simulation x, y, z.

        Use when a source is already defined in lab coordinates or when no
        rotation is needed before wrapping with ``PolarTransformedField``.
        """
        origin_arr = (
            np.zeros(3, dtype=np.float64)
            if origin is None
            else np.asarray(origin, dtype=np.float64)
        )
        return cls(origin=origin_arr, lab_from_local=np.eye(3, dtype=np.float64))

    def position_to_local(self, position_lab: Sequence[float] | NDArray[np.floating]) -> NDArray[np.float64]:
        """Map a lab-frame position to local coordinates: r_local = R^T (r_lab - origin)."""
        r_lab = np.asarray(position_lab, dtype=np.float64)
        return self.lab_from_local.T @ (r_lab - self.origin)

    def position_to_lab(self, position_local: Sequence[float] | NDArray[np.floating]) -> NDArray[np.float64]:
        """Map a local position back to the lab frame: r_lab = origin + R r_local."""
        r_local = np.asarray(position_local, dtype=np.float64)
        return self.origin + self.lab_from_local @ r_local

    def vector_to_lab(self, vector_local: Sequence[float] | NDArray[np.floating]) -> NDArray[np.float64]:
        """Rotate a vector from local to lab components (ignores ``origin``)."""
        v_local = np.asarray(vector_local, dtype=np.float64)
        return self.lab_from_local @ v_local

    def vector_to_local(self, vector_lab: Sequence[float] | NDArray[np.floating]) -> NDArray[np.float64]:
        """Rotate a vector from lab to local components (ignores ``origin``)."""
        v_lab = np.asarray(vector_lab, dtype=np.float64)
        return self.lab_from_local.T @ v_lab

    def wavevector_lab(self, k_magnitude: float) -> NDArray[np.float64]:
        """Return the lab-frame wavevector k = |k| k_hat for a local +z source."""
        return k_magnitude * self.k_hat


class PolarTransformedField:
    """
    Polar transform wrapper: map a locally-authored field into the PIC lab frame.

    The wrapped ``source`` is evaluated at local coordinates
    ``r_local = R^T (r_lab - origin)`` and its vector value is mapped back with
    ``F_lab = R F_local``. Works for electric or magnetic sources. The rotation
    ``R`` is fixed for the life of the wrapper; it does not depend on time.
    """

    def __init__(self, source: object, frame: WaveFrame) -> None:
        """
        Wrap a locally-authored ``ElectricFields`` / ``MagneticFields`` source.

        ``source`` is evaluated in the wave frame; results are rotated into the
        PIC lab frame for particle gather or diagnostics.
        """
        self.source = source
        self.frame = frame

    def at(self, pos: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        """Evaluate the rotated field at one lab-frame position and time."""
        r_local = self.frame.position_to_local(pos)
        field_local = self.source.at(r_local, t)
        return self.frame.vector_to_lab(field_local)

    def at_batch(self, positions: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        """
        Vectorized lab-frame evaluation for ``(N, 3)`` particle positions.

        Positions are mapped to local coordinates in batch, the source is
        evaluated (via ``at_batch`` when available), and field vectors are
        rotated back to lab components.
        """
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        r_local = (pos - self.frame.origin) @ self.frame.lab_from_local
        if hasattr(self.source, "at_batch"):
            field_local = self.source.at_batch(r_local, t)
        else:
            n = pos.shape[0]
            field_local = np.empty((n, 3), dtype=np.float64)
            for i in range(n):
                field_local[i] = self.source.at(r_local[i], t)
        return field_local @ self.frame.lab_from_local.T

    def __add__(self, other: object) -> object:
        """Superpose with another prescribed field; defers to typed sum helpers."""
        from .ElectricFields import ElectricFields, ElectricFieldsSum
        from .MagneticFields import MagneticFields, MagneticFieldsSum

        if isinstance(other, (ElectricFields, ElectricFieldsSum)):
            return ElectricFieldsSum([self, other])
        if isinstance(other, (MagneticFields, MagneticFieldsSum)):
            return MagneticFieldsSum([self, other])
        return TransformedFieldSum([self, other])


class TransformedFieldSum:
    """Superposition of polar-transformed and/or native prescribed field sources."""

    def __init__(self, sources: list[object]) -> None:
        """Collect one or more native or transformed field sources."""
        self.sources = sources

    def at(self, pos: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        """Sum ``source.at(pos, t)`` over all wrapped sources."""
        total = np.zeros(3, dtype=np.float64)
        for source in self.sources:
            total += source.at(pos, t)
        return total

    def at_batch(self, positions: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        """Sum batched evaluations over all wrapped sources; returns ``(N, 3)``."""
        pos = np.asarray(positions, dtype=np.float64)
        total = np.zeros((pos.shape[0], 3), dtype=np.float64)
        for source in self.sources:
            if hasattr(source, "at_batch"):
                total += source.at_batch(pos, t)
            else:
                for i in range(pos.shape[0]):
                    total[i] += source.at(pos[i], t)
        return total

    def __add__(self, other: object) -> TransformedFieldSum:
        """Append another source and return a new sum."""
        return TransformedFieldSum([*self.sources, other])


# Backward-compatible alias used by older imports and docs.
TransformedField = PolarTransformedField
