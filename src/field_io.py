"""
Shared field dataset loading and spacetime interpolation for PIC drivers.

Supported file formats:
  - CSV with optional ``# key=value`` metadata header rows
  - HDF5 (requires ``h5py``) with a documented group layout

Recommended production format: HDF5 (chunked, compressed, multi-time in one file).
CSV remains supported for small prototypes and hand-edited inputs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

try:
    from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
except ImportError as exc:  # pragma: no cover
    raise ImportError("field_io requires scipy (already used by ElectrostaticGrid)") from exc


ComponentAxis = Literal["x", "y", "z"]
Metadata = dict[str, str]


class FieldFileFormat(StrEnum):
    CSV = "csv"
    HDF5 = "hdf5"


@dataclass
class FieldDataset:
    """
    Structured or unstructured samples of a vector field in space-time.

    For structured grids, ``x``, ``y``, ``z`` are 1D coordinate axes and each
    entry in ``components`` has shape ``(nt, nz, ny, nx)`` or ``(nz, ny, nx)``.
    For unstructured CSV inputs, axes are empty and component arrays are flat.
    """

    component_names: tuple[str, str, str]
    x: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    y: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    z: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    times: NDArray[np.float64] | None = None
    components: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def is_structured(self) -> bool:
        return self.x.size > 1 and self.y.size > 1 and self.z.size > 1

    @property
    def is_time_dependent(self) -> bool:
        return self.times is not None and self.times.size > 1


def _parse_metadata_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    body = stripped.lstrip("#").strip()
    if "=" not in body:
        return None
    key, value = body.split("=", 1)
    return key.strip().lower(), value.strip()


def _detect_file_format(path: Path) -> FieldFileFormat:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return FieldFileFormat.CSV
    if suffix in {".h5", ".hdf5", ".he5"}:
        return FieldFileFormat.HDF5
    raise ValueError(f"unsupported field file extension: {path.suffix!r}")


def _structured_sort(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    components: dict[str, NDArray[np.float64]],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, NDArray[np.float64]],
]:
    nx, ny, nz = np.unique(x).size, np.unique(y).size, np.unique(z).size
    if nx * ny * nz != x.size:
        return x, y, z, components

    order = np.lexsort((x, y, z))
    x_axis = np.unique(x)
    y_axis = np.unique(y)
    z_axis = np.unique(z)
    reshaped = {
        name: values[order].reshape(nz, ny, nx)
        for name, values in components.items()
    }
    return x_axis, y_axis, z_axis, reshaped


def load_field_csv(
    path: str | Path,
    component_names: tuple[str, str, str],
) -> FieldDataset:
    """Load a vector field from CSV with optional ``# key=value`` metadata."""
    file_path = Path(path)
    metadata: Metadata = {}
    header: list[str] = []

    with file_path.open(newline="", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                parsed = _parse_metadata_line(line)
                if parsed is not None:
                    metadata[parsed[0]] = parsed[1]
                continue
            header = [part.strip() for part in line.strip().split(",")]
            break

        if not header:
            raise ValueError(f"CSV field file has no header row: {file_path}")

        required = {"x", "y", "z", *component_names}
        missing = required.difference(header)
        if missing:
            raise ValueError(f"CSV field file missing columns {sorted(missing)}: {file_path}")

        reader = csv.DictReader(handle, fieldnames=header)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV field file contains no data rows: {file_path}")

    def col(name: str) -> NDArray[np.float64]:
        return np.array([float(row[name]) for row in rows], dtype=np.float64)

    x = col("x")
    y = col("y")
    z = col("z")
    components = {name: col(name) for name in component_names}
    times = col("t") if "t" in header else None

    if times is None and "time" in metadata:
        times = np.array([float(metadata["time"])], dtype=np.float64)

    x_axis, y_axis, z_axis, components = _structured_sort(x, y, z, components)
    if not (x_axis.size > 1 and y_axis.size > 1 and z_axis.size > 1):
        x_axis, y_axis, z_axis = x, y, z

    return FieldDataset(
        component_names=component_names,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        times=times,
        components=components,
        metadata=metadata,
        source_path=file_path,
    )


def _unique_times(times: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.unique(np.round(times, decimals=12))


def load_field_hdf5(
    path: str | Path,
    group: str,
    component_names: tuple[str, str, str],
) -> FieldDataset:
    """
    Load a vector field group from HDF5.

    Expected layout::

        /<group>/x              (nx,)
        /<group>/y              (ny,)
        /<group>/z              (nz,)
        /<group>/times           (nt,)            optional
        /<group>/Ex, Ey, Ez     (nt,nz,ny,nx) or (nz,ny,nx)
        /<group>/attrs...        metadata
    """
    if h5py is None:
        raise ImportError("reading HDF5 field files requires the optional 'h5py' package")

    file_path = Path(path)
    with h5py.File(file_path, "r") as handle:
        if group not in handle:
            raise KeyError(f"HDF5 group {group!r} not found in {file_path}")
        node = handle[group]
        x = np.asarray(node["x"], dtype=np.float64)
        y = np.asarray(node["y"], dtype=np.float64)
        z = np.asarray(node["z"], dtype=np.float64)
        times = np.asarray(node["times"], dtype=np.float64) if "times" in node else None
        components = {name: np.asarray(node[name], dtype=np.float64) for name in component_names}
        metadata = {str(key).lower(): str(value) for key, value in node.attrs.items()}

    return FieldDataset(
        component_names=component_names,
        x=x,
        y=y,
        z=z,
        times=times,
        components=components,
        metadata=metadata,
        source_path=file_path,
    )


def load_field_file(
    path: str | Path,
    component_names: tuple[str, str, str],
    *,
    hdf5_group: str | None = None,
) -> FieldDataset:
    file_path = Path(path)
    fmt = _detect_file_format(file_path)
    if fmt == FieldFileFormat.CSV:
        return load_field_csv(file_path, component_names)
    if hdf5_group is None:
        raise ValueError("hdf5_group is required when loading HDF5 field files")
    return load_field_hdf5(file_path, hdf5_group, component_names)


class FieldInterpolator:
    """Evaluate a loaded dataset at arbitrary points and times."""

    def __init__(self, dataset: FieldDataset) -> None:
        self.dataset = dataset
        self._spatial: dict[str, RegularGridInterpolator | LinearNDInterpolator] = {}
        self._time_values = (
            _unique_times(dataset.times)
            if dataset.times is not None and dataset.times.size
            else np.array([0.0], dtype=np.float64)
        )
        self._build_interpolators(float(self._time_values[0]))

    def _build_interpolators(self, time_value: float) -> None:
        ds = self.dataset
        self._spatial.clear()

        if ds.is_structured:
            time_index = 0
            if ds.times is not None and ds.times.size > 1:
                time_index = int(np.argmin(np.abs(self._time_values - time_value)))
            for name in ds.component_names:
                values = ds.components[name]
                slice_ = values[time_index] if values.ndim == 4 else values
                self._spatial[name] = RegularGridInterpolator(
                    (ds.z, ds.y, ds.x),
                    slice_,
                    bounds_error=False,
                    fill_value=0.0,
                )
            return

        mask = np.ones(ds.x.shape[0], dtype=bool)
        if ds.times is not None and ds.times.size == ds.x.size:
            nearest = float(self._time_values[int(np.argmin(np.abs(self._time_values - time_value)))])
            mask = np.isclose(ds.times, nearest)

        points = np.column_stack([ds.z[mask], ds.y[mask], ds.x[mask]])
        for name in ds.component_names:
            self._spatial[name] = LinearNDInterpolator(
                points,
                ds.components[name][mask],
                fill_value=0.0,
            )

    def at(self, pos: NDArray[np.floating], t: float = 0.0) -> NDArray[np.float64]:
        nearest_time = float(self._time_values[int(np.argmin(np.abs(self._time_values - t)))])
        self._build_interpolators(nearest_time)

        point = np.asarray(pos, dtype=np.float64)
        if self.dataset.is_structured:
            sample = np.array([point[2], point[1], point[0]])
            return np.array(
                [float(np.asarray(self._spatial[name](sample)).reshape(-1)[0]) for name in self.dataset.component_names],
                dtype=np.float64,
            )

        return np.array(
            [float(np.asarray(self._spatial[name](point)).reshape(-1)[0]) for name in self.dataset.component_names],
            dtype=np.float64,
        )


def wave_vector(kx: float, ky: float, kz: float) -> NDArray[np.float64]:
    return np.array([kx, ky, kz], dtype=np.float64)


def phase(k: NDArray[np.floating], r: NDArray[np.floating], omega: float, t: float) -> float:
    return float(np.dot(k, r) - omega * t)
