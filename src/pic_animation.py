"""HDF5 frame I/O and matplotlib animation export for PIC drivers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d.art3d import Path3DCollection
    from mpl_toolkits.mplot3d.axes3d import Axes3D

ANIMATIONS_DIR_NAME = "animations"


def animations_dir(project_root: Path) -> Path:
    """Return the temporary animation output directory (created on demand)."""
    return project_root / ANIMATIONS_DIR_NAME


@dataclass(frozen=True)
class FrameRecord:
    step: int
    time_ps: float
    electrons_um: np.ndarray
    ions_um: np.ndarray


class FrameStreamWriter:
    """Append subsampled particle frames to an HDF5 file (constant RAM)."""

    def __init__(
        self,
        path: Path,
        *,
        n_e: int,
        n_i: int,
        subsample: int,
        seed: int = 0,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sub_e = min(subsample, n_e)
        self._sub_i = min(subsample, n_i)
        rng = np.random.default_rng(seed)
        self._idx_e = np.sort(rng.choice(n_e, size=self._sub_e, replace=False))
        self._idx_i = np.sort(rng.choice(n_i, size=self._sub_i, replace=False))

        self._file = h5py.File(self.path, "w")
        self._steps = self._file.create_dataset(
            "steps", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(64,)
        )
        self._times_ps = self._file.create_dataset(
            "times_ps", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(64,)
        )
        self._electrons = self._file.create_dataset(
            "electrons_um",
            shape=(0, self._sub_e, 3),
            maxshape=(None, self._sub_e, 3),
            dtype=np.float32,
            chunks=(1, self._sub_e, 3),
        )
        self._ions = self._file.create_dataset(
            "ions_um",
            shape=(0, self._sub_i, 3),
            maxshape=(None, self._sub_i, 3),
            dtype=np.float32,
            chunks=(1, self._sub_i, 3),
        )
        self._file.create_dataset("electron_indices", data=self._idx_e)
        self._file.create_dataset("ion_indices", data=self._idx_i)
        self._file.attrs["subsample"] = subsample
        self._file.attrs["seed"] = seed
        self._file.attrs["n_e_total"] = n_e
        self._file.attrs["n_i_total"] = n_i
        self._n_frames = 0

    def __enter__(self) -> FrameStreamWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def append(self, step: int, time_ps: float, pos_e: np.ndarray, pos_i: np.ndarray) -> None:
        um = np.float32(1e6)
        e_um = (pos_e[self._idx_e] * um).astype(np.float32, copy=False)
        i_um = (pos_i[self._idx_i] * um).astype(np.float32, copy=False)

        n = self._n_frames + 1
        self._steps.resize((n,))
        self._times_ps.resize((n,))
        self._electrons.resize((n, self._sub_e, 3))
        self._ions.resize((n, self._sub_i, 3))

        idx = self._n_frames
        self._steps[idx] = step
        self._times_ps[idx] = time_ps
        self._electrons[idx] = e_um
        self._ions[idx] = i_um
        self._n_frames = n

    def close(self) -> None:
        if self._file.id.valid:
            self._file.flush()
            self._file.close()

    @property
    def n_frames(self) -> int:
        return self._n_frames


def _require_dataset(file: h5py.File, name: str) -> h5py.Dataset:
    obj = file[name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"expected HDF5 dataset {name!r}, got {type(obj).__name__}")
    return obj


def _read_dataset_int(dataset: h5py.Dataset, index: int) -> int:
    value = np.empty((), dtype=np.int64)
    dataset.read_direct(value, np.s_[index])
    return int(value.item())


def _read_dataset_float(dataset: h5py.Dataset, index: int) -> float:
    value = np.empty((), dtype=np.float64)
    dataset.read_direct(value, np.s_[index])
    return float(value.item())


def _read_dataset_row(dataset: h5py.Dataset, index: int) -> np.ndarray:
    n_sub = int(dataset.shape[1])
    n_dim = int(dataset.shape[2])
    row = np.empty((n_sub, n_dim), dtype=np.float64)
    dataset.read_direct(row, np.s_[index, :, :])
    return row


class FrameStreamReader:
    """Read animation frames one at a time from HDF5."""

    _file: h5py.File
    _steps: h5py.Dataset
    _times_ps: h5py.Dataset
    _electrons: h5py.Dataset
    _ions: h5py.Dataset

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = h5py.File(self.path, "r")
        self._steps = _require_dataset(self._file, "steps")
        self._times_ps = _require_dataset(self._file, "times_ps")
        self._electrons = _require_dataset(self._file, "electrons_um")
        self._ions = _require_dataset(self._file, "ions_um")

    def __enter__(self) -> FrameStreamReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._file.id.valid:
            self._file.close()

    def __len__(self) -> int:
        return int(self._steps.shape[0])

    def read_frame(self, index: int) -> FrameRecord:
        return FrameRecord(
            step=_read_dataset_int(self._steps, index),
            time_ps=_read_dataset_float(self._times_ps, index),
            electrons_um=_read_dataset_row(self._electrons, index),
            ions_um=_read_dataset_row(self._ions, index),
        )


def _scatter3d(ax: Axes3D, positions: np.ndarray, **kwargs: Any) -> Path3DCollection:
    coords = np.asarray(positions, dtype=np.float64)
    return cast(
        Any,
        ax.scatter(
            cast(Any, coords[:, 0]),
            cast(Any, coords[:, 1]),
            cast(Any, coords[:, 2]),
            **kwargs,
        ),
    )


def _update_scatter3d(scatter: Path3DCollection, positions: np.ndarray) -> None:
    coords = np.asarray(positions, dtype=np.float64)
    scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])


def _draw_domain_box(ax: Axes3D, length_um: float) -> None:
    L = length_um
    edges = [
        ([0, L], [0, 0], [0, 0]),
        ([0, L], [L, L], [0, 0]),
        ([0, 0], [0, L], [0, 0]),
        ([L, L], [0, L], [0, 0]),
        ([0, L], [0, 0], [L, L]),
        ([0, L], [L, L], [L, L]),
        ([0, 0], [0, L], [L, L]),
        ([L, L], [0, L], [L, L]),
        ([0, 0], [0, 0], [0, L]),
        ([L, L], [0, 0], [0, L]),
        ([0, 0], [L, L], [0, L]),
        ([L, L], [L, L], [0, L]),
    ]
    for xs, ys, zs in edges:
        ax.plot(xs, ys, zs, color="0.5", linewidth=0.5, alpha=0.6)


def build_animation(
    frames: FrameStreamReader | Path,
    *,
    domain_length_um: float,
    ion_label: str = "Kr+",
    interval_ms: int = 50,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    owned_reader = isinstance(frames, Path)
    reader = FrameStreamReader(frames) if owned_reader else frames

    frame0 = reader.read_frame(0)
    fig = plt.figure(figsize=(8, 7))
    ax_candidate = fig.add_subplot(111, projection="3d")
    if not isinstance(ax_candidate, Axes3D):
        raise TypeError("expected 3D axes")
    ax = ax_candidate

    sc_e = _scatter3d(
        ax,
        frame0.electrons_um,
        c="tab:blue",
        s=4,
        alpha=0.5,
        label="e-",
    )
    sc_i = _scatter3d(
        ax,
        frame0.ions_um,
        c="tab:red",
        s=6,
        alpha=0.6,
        label=ion_label,
    )
    title = ax.set_title("")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    ax.set_xlim(0, domain_length_um)
    ax.set_ylim(0, domain_length_um)
    ax.set_zlim(0, domain_length_um)
    ax.legend(loc="upper right")
    _draw_domain_box(ax, domain_length_um)

    def update(frame: int) -> tuple:
        record = reader.read_frame(frame)
        _update_scatter3d(sc_e, record.electrons_um)
        _update_scatter3d(sc_i, record.ions_um)
        title.set_text(f"step {record.step}  t = {record.time_ps:.4f} ps")
        return sc_e, sc_i, title

    anim = FuncAnimation(
        fig,
        update,
        frames=len(reader),
        interval=interval_ms,
        blit=False,
    )
    anim._frame_reader = reader  # type: ignore[attr-defined]
    anim._owned_reader = owned_reader  # type: ignore[attr-defined]
    return anim


def save_animation(anim, output_dir: Path, *, video_stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if shutil.which("ffmpeg"):
            out = output_dir / f"{video_stem}.mp4"
            anim.save(out, writer="ffmpeg", dpi=120)
            return out
        out = output_dir / f"{video_stem}.gif"
        anim.save(out, writer="pillow", dpi=80)
        return out
    finally:
        if getattr(anim, "_owned_reader", False):
            anim._frame_reader.close()  # type: ignore[attr-defined]


def export_frames_to_video(
    frames_path: Path,
    *,
    output_dir: Path,
    video_stem: str,
    domain_length_um: float,
    ion_label: str = "Kr+",
    delete_frames: bool = True,
) -> Path:
    """Build a movie from streamed HDF5 frames and optionally remove the frame file."""
    with FrameStreamReader(frames_path) as reader:
        anim = build_animation(
            reader,
            domain_length_um=domain_length_um,
            ion_label=ion_label,
        )
        video_path = save_animation(anim, output_dir, video_stem=video_stem)

    if delete_frames:
        frames_path.unlink(missing_ok=True)

    return video_path
