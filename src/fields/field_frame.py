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

from dataclasses import dataclass
import math
from typing import cast, final

import numpy as np
from numpy.typing import NDArray

from .types import (
    EnvelopeWidthLike,
    FieldBatch,
    FieldSource,
    FieldSourceFactory,
    FieldSourceSum,
    FieldVector,
    PolarizationKind,
    Position,
    Positions,
    Vec3,
    Vector3Like,
    Waveform,
)


def normalize_vector(
    vector: Vector3Like,
    *,
    name: str = "vector",
) -> Vec3:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return arr / norm


def _dot3(a: Vec3, b: Vec3) -> float:
    return a.item(0) * b.item(0) + a.item(1) * b.item(1) + a.item(2) * b.item(2)


def _reject_parallel(v: Vec3, axis: Vec3) -> Vec3:
    """Return the component of ``v`` orthogonal to ``axis``."""
    scale = _dot3(v, axis)
    return np.array(
        [
            v.item(0) - scale * axis.item(0),
            v.item(1) - scale * axis.item(1),
            v.item(2) - scale * axis.item(2),
        ],
        dtype=np.float64,
    )


def _combine3(
    a: Vec3,
    b: Vec3,
    scale_a: float,
    scale_b: float,
) -> Vec3:
    """Return ``scale_a * a + scale_b * b`` for fixed 3-vectors."""
    return np.array(
        [
            scale_a * a.item(0) + scale_b * b.item(0),
            scale_a * a.item(1) + scale_b * b.item(1),
            scale_a * a.item(2) + scale_b * b.item(2),
        ],
        dtype=np.float64,
    )


def direction_from_spherical(theta: float, phi: float) -> Vec3:
    """
    Unit propagation direction from spherical angles.

    ``theta`` is polar angle from +z (colatitude), ``phi`` is azimuth in the xy-plane.
    """
    sin_theta = math.sin(theta)
    return np.array(
        [sin_theta * math.cos(phi), sin_theta * math.sin(phi), math.cos(theta)],
        dtype=np.float64,
    )


def transverse_basis(
    k_hat: Vec3,
    polarization: Vector3Like | None = None,
) -> tuple[Vec3, Vec3]:
    """
    Build an orthonormal transverse pair (e1, e2) with e1 x e2 = k_hat.

    If ``polarization`` is given it defines the preferred e1 direction (projected
    orthogonal to ``k_hat``).
    """
    if polarization is None:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(_dot3(reference, k_hat)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        e1 = _reject_parallel(reference, k_hat)
    else:
        pol = np.asarray(polarization, dtype=np.float64)
        e1 = _reject_parallel(pol, k_hat)

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
        scalar = math.sin(phi)
        return math.cos(psi) * scalar, math.sin(psi) * scalar
    return math.sin(phi), math.tan(psi) * math.sin(phi + delta)


def elliptical_components_cos(
    phi: float,
    *,
    psi: float = 0.0,
    delta: float = 0.0,
) -> tuple[float, float]:
    """Cosine-carrier counterpart to :func:`elliptical_components`."""
    if delta == 0.0:
        scalar = math.cos(phi)
        return math.cos(psi) * scalar, math.sin(psi) * scalar
    return math.cos(phi), math.tan(psi) * math.cos(phi + delta)


def resolve_k_magnitude(
    *,
    wavevector: Vec3 | None = None,
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


def local_wavevector(k_magnitude: float) -> Vec3:
    return np.array([0.0, 0.0, k_magnitude], dtype=np.float64)


def evaluate_polarized_wave_local(
    phi: float,
    *,
    amplitude: float,
    polarization_kind: PolarizationKind,
    psi: float = 0.0,
    delta: float = 0.0,
    waveform: Waveform = "sin",
) -> FieldVector:
    """Return transverse wave components [Fx, Fy, 0] in the local source frame."""
    if polarization_kind == PolarizationKind.LINEAR:
        carrier = math.sin(phi) if waveform == "sin" else math.cos(phi)
        a1 = math.cos(psi) * carrier
        a2 = math.sin(psi) * carrier
        return np.array([amplitude * a1, amplitude * a2, 0.0], dtype=np.float64)

    if waveform == "sin":
        a1, a2 = elliptical_components(phi, psi=psi, delta=delta)
    else:
        a1, a2 = elliptical_components_cos(phi, psi=psi, delta=delta)
    return np.array([amplitude * a1, amplitude * a2, 0.0], dtype=np.float64)


def normalize_envelope_width(
    width: EnvelopeWidthLike,
) -> Vec3:
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
    r_local: Vector3Like,
    center: Vec3,
    width: Vec3,
) -> float:
    """Axis-aligned Gaussian envelope evaluated in the local source frame."""
    r = np.asarray(r_local, dtype=np.float64)
    envelope_arg = -sum(
        ((r.item(i) - center.item(i)) / width.item(i)) ** 2 for i in range(3)
    )
    return math.exp(envelope_arg)


def evaluate_gaussian_pulse_local(
    r: Vector3Like,
    t: float,
    *,
    amplitude: float,
    omega: float,
    wavevector: Vec3,
    center: Vec3,
    width: Vec3,
    phase0: float = 0.0,
    polarization_kind: PolarizationKind = PolarizationKind.LINEAR,
    psi: float = 0.0,
    delta: float = 0.0,
) -> FieldVector:
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

    origin: Vec3
    lab_from_local: NDArray[np.float64]

    @property
    def e1(self) -> Vec3:
        """First transverse basis vector (local x / polarization reference) in lab coords."""
        return self.lab_from_local[:, 0].copy()

    @property
    def e2(self) -> Vec3:
        """Second transverse basis vector (local y) in lab coords; e1 x e2 = k_hat."""
        return self.lab_from_local[:, 1].copy()

    @property
    def k_hat(self) -> Vec3:
        """Unit propagation direction (local +z) expressed in lab coordinates."""
        return self.lab_from_local[:, 2].copy()

    @classmethod
    def from_basis(
        cls,
        k_direction: Vector3Like,
        *,
        polarization: Vector3Like | None = None,
        origin: Vector3Like | None = None,
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
        origin: Vector3Like | None = None,
    ) -> WaveFrame:
        """
        Build a static frame from spherical incidence angles into the PIC cube.

        ``theta`` and ``phi`` define propagation direction ``k_hat``. ``pol_angle``
        is a fixed rotation of the local x-axis (e1) about ``k_hat``.
        """
        k_hat = direction_from_spherical(theta, phi)
        e1_basis, e2_basis = transverse_basis(k_hat, polarization=None)
        e1: Vec3
        e2: Vec3
        if pol_angle != 0.0:
            c = math.cos(pol_angle)
            s = math.sin(pol_angle)
            e1 = _combine3(e1_basis, e2_basis, c, s)
            e2 = _combine3(e1_basis, e2_basis, -s, c)
        else:
            e1, e2 = e1_basis, e2_basis
        origin_arr = (
            np.zeros(3, dtype=np.float64)
            if origin is None
            else np.asarray(origin, dtype=np.float64)
        )
        return cls(origin=origin_arr, lab_from_local=np.column_stack([e1, e2, k_hat]))

    @classmethod
    def identity(cls, origin: Vector3Like | None = None) -> WaveFrame:
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

    def position_to_local(self, position_lab: Vector3Like) -> Vec3:
        """Map a lab-frame position to local coordinates: r_local = R^T (r_lab - origin)."""
        r_lab = np.asarray(position_lab, dtype=np.float64)
        return np.asarray(self.lab_from_local.T @ (r_lab - self.origin), dtype=np.float64)

    def position_to_lab(self, position_local: Vector3Like) -> Vec3:
        """Map a local position back to the lab frame: r_lab = origin + R r_local."""
        r_local = np.asarray(position_local, dtype=np.float64)
        return self.origin + self.lab_from_local @ r_local

    def vector_to_lab(self, vector_local: Vector3Like) -> Vec3:
        """Rotate a vector from local to lab components (ignores ``origin``)."""
        v_local = np.asarray(vector_local, dtype=np.float64)
        return self.lab_from_local @ v_local

    def vector_to_local(self, vector_lab: Vector3Like) -> Vec3:
        """Rotate a vector from lab to local components (ignores ``origin``)."""
        v_lab = np.asarray(vector_lab, dtype=np.float64)
        return self.lab_from_local.T @ v_lab

    def wavevector_lab(self, k_magnitude: float) -> Vec3:
        """Return the lab-frame wavevector k = |k| k_hat for a local +z source."""
        return k_magnitude * self.k_hat


@final
@dataclass
class PolarTransformedField:
    """
    Polar transform wrapper: map a locally-authored field into the PIC lab frame.

    The wrapped ``source`` is evaluated at local coordinates
    ``r_local = R^T (r_lab - origin)`` and its vector value is mapped back with
    ``F_lab = R F_local``. Works for electric or magnetic sources. The rotation
    ``R`` is fixed for the life of the wrapper; it does not depend on time.
    """

    source: FieldSource
    frame: WaveFrame

    def at(self, pos: Position, t: float = 0.0) -> FieldVector:
        """Evaluate the rotated field at one lab-frame position and time."""
        r_local = self.frame.position_to_local(pos)
        field_local: FieldVector = self.source.at(r_local, t)
        return self.frame.vector_to_lab(field_local)

    def at_batch(self, positions: Positions, t: float = 0.0) -> FieldBatch:
        """
        Vectorized lab-frame evaluation for ``(N, 3)`` particle positions.

        Positions are mapped to local coordinates in batch, the source is
        evaluated (via ``at_batch`` when available), and field vectors are
        rotated back to lab components.
        """
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        r_local = np.asarray((pos - self.frame.origin) @ self.frame.lab_from_local, dtype=np.float64)
        field_local: FieldBatch = self.source.at_batch(r_local, t)
        return np.asarray(field_local @ self.frame.lab_from_local.T, dtype=np.float64)

    def __add__(self, other: FieldSource) -> FieldSource:
        """Superpose with another prescribed field; defers to typed sum helpers."""
        sum_type = getattr(type(other), "_SUM", None)
        if sum_type is not None:
            factory = cast(FieldSourceFactory, sum_type)
            return factory([self, other])
        if isinstance(other, FieldSourceSum):
            constructor = cast(FieldSourceFactory, type(other))
            return constructor([self, other])
        return TransformedFieldSum([self, other])


@final
@dataclass
class TransformedFieldSum:
    """Superposition of polar-transformed and/or native prescribed field sources."""

    sources: list[FieldSource]

    def __post_init__(self) -> None:
        self.sources = [source for source in self.sources]

    def at(self, pos: Position, t: float = 0.0) -> FieldVector:
        """Sum ``source.at(pos, t)`` over all wrapped sources."""
        total: FieldVector = np.zeros(3, dtype=np.float64)
        for source in self.sources:
            field: FieldVector = source.at(pos, t)
            total += field
        return total

    def at_batch(self, positions: Positions, t: float = 0.0) -> FieldBatch:
        """Sum batched evaluations over all wrapped sources; returns ``(N, 3)``."""
        pos = np.asarray(positions, dtype=np.float64)
        total: FieldBatch = np.zeros((pos.shape[0], 3), dtype=np.float64)
        for source in self.sources:
            field: FieldBatch = source.at_batch(pos, t)
            total += field
        return total

    def __add__(self, other: FieldSource) -> TransformedFieldSum:
        """Append another source and return a new sum."""
        return TransformedFieldSum([*self.sources, other])


# Backward-compatible alias used by older imports and docs.
TransformedField = PolarTransformedField
