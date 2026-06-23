"""Prescribed E/B field sources, wave-frame geometry, and field file I/O."""

from .ElectricFields import (
    ElectricFieldMode,
    ElectricFields,
    ElectricFieldSpec,
    ElectricFieldsSum,
    plane_wave_em_pair,
)
from .MagneticFields import MagneticFields, MagneticFieldsSum, MagneticFieldMode, MagneticFieldSpec
from .prescribed import PrescribedField, PrescribedFieldSum
from .field_frame import (
    PolarTransformedField,
    TransformedField,
    WaveFrame,
    evaluate_gaussian_pulse_local,
    evaluate_polarized_wave_local,
    gaussian_envelope_local,
    local_wavevector,
    normalize_envelope_width,
    resolve_k_magnitude,
)
from .field_io import (
    FieldDataset,
    FieldInterpolator,
    load_field_csv,
    load_field_file,
    load_field_hdf5,
    phase,
    phase_batch,
    wave_vector,
)
from .types import (
    FieldSource,
    FieldVector,
    PolarizationKind,
    PrescribedFieldSource,
    Vector3Like,
)

__all__ = (
    "ElectricFieldMode",
    "ElectricFieldSpec",
    "ElectricFields",
    "ElectricFieldsSum",
    "FieldDataset",
    "FieldInterpolator",
    "FieldSource",
    "FieldVector",
    "MagneticFieldMode",
    "MagneticFieldSpec",
    "MagneticFields",
    "MagneticFieldsSum",
    "PolarizationKind",
    "PolarTransformedField",
    "PrescribedField",
    "PrescribedFieldSource",
    "PrescribedFieldSum",
    "TransformedField",
    "Vector3Like",
    "WaveFrame",
    "evaluate_gaussian_pulse_local",
    "plane_wave_em_pair",
    "evaluate_polarized_wave_local",
    "gaussian_envelope_local",
    "load_field_csv",
    "load_field_file",
    "load_field_hdf5",
    "local_wavevector",
    "normalize_envelope_width",
    "phase",
    "phase_batch",
    "resolve_k_magnitude",
    "wave_vector",
)
