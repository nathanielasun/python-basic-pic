"""HDF5 frame I/O and matplotlib animation export for PIC drivers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import h5py
import numpy as np

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


class FrameStreamReader:
    """Read animation frames one at a time from HDF5."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = h5py.File(self.path, "r")
        self._steps = self._file["steps"]
        self._times_ps = self._file["times_ps"]
        self._electrons = self._file["electrons_um"]
        self._ions = self._file["ions_um"]

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
            step=int(self._steps[index]),
            time_ps=float(self._times_ps[index]),
            electrons_um=np.asarray(self._electrons[index], dtype=np.float64),
            ions_um=np.asarray(self._ions[index], dtype=np.float64),
        )


def _draw_domain_box(ax: object, length_um: float) -> None:
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

    owned_reader = isinstance(frames, Path)
    reader = FrameStreamReader(frames) if owned_reader else frames

    frame0 = reader.read_frame(0)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc_e = ax.scatter(
        frame0.electrons_um[:, 0],
        frame0.electrons_um[:, 1],
        frame0.electrons_um[:, 2],
        c="tab:blue",
        s=4,
        alpha=0.5,
        label="e-",
    )
    sc_i = ax.scatter(
        frame0.ions_um[:, 0],
        frame0.ions_um[:, 1],
        frame0.ions_um[:, 2],
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
        sc_e._offsets3d = (
            record.electrons_um[:, 0],
            record.electrons_um[:, 1],
            record.electrons_um[:, 2],
        )
        sc_i._offsets3d = (
            record.ions_um[:, 0],
            record.ions_um[:, 1],
            record.ions_um[:, 2],
        )
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
