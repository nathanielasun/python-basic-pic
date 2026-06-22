"""Shared electrostatic PIC driver utilities for ``examples/`` simulations."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grids import ElectrostaticGrid, ParticleBackend
from pic_animation import FrameStreamWriter, animations_dir, export_frames_to_video

try:
    from grids import HAS_NUMBA, electric_kick_b0, wrap_positions_periodic
except ImportError:
    HAS_NUMBA = False

# --- SI constants ---
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
M_U = 1.66053906660e-27
M_H = 1.00784 * M_U
M_KR = 83.798 * M_U
K_B = 1.380649e-23
EPS0 = 8.8541878128e-12
EV_TO_J = E_CHARGE

DEFAULT_SUBSAMPLE = 2000
ANIMATIONS_DIR = animations_dir(_ROOT)

# Percent milestones treated as progress "frames" (1% … 100%).
_PROGRESS_TOTAL_FRAMES = 100


def _format_eta(seconds: float) -> str:
    """Human-readable remaining wall time."""
    if seconds < 1.0:
        return "<1s"
    total_sec = int(seconds + 0.5)
    minutes, sec = divmod(total_sec, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {sec:02d}s"


@dataclass
class PercentProgressTracker:
    """1% milestone reporting with ETA from running mean frame wall time."""

    loop_t0: float
    last_reported_pct: int = 0
    total_frames: int = _PROGRESS_TOTAL_FRAMES
    use_animation_frames: bool = False
    _frames_recorded: int = 0

    def note_animation_frame(self) -> None:
        """Count animation frames written (used for ETA when exporting video)."""
        self._frames_recorded += 1

    def _completed_frames(self) -> int:
        if self.use_animation_frames:
            return self._frames_recorded
        return self.last_reported_pct

    def report(self, step: int, n_steps: int, *, dt: float, mean_vel_e: np.ndarray) -> int:
        now = time.perf_counter()
        elapsed = now - self.loop_t0
        pct = (step * 100) // n_steps
        while self.last_reported_pct < pct:
            self.last_reported_pct += 1
            mean_ve = float(np.mean(np.linalg.norm(mean_vel_e, axis=1)))
            eta_suffix = ""
            frames_done = self._completed_frames()
            if frames_done > 0:
                avg_frame_time = elapsed / frames_done
                eta_seconds = avg_frame_time * self.total_frames - elapsed
                eta_suffix = f"  ETA {_format_eta(max(0.0, eta_seconds))}"
            print(
                f"  {self.last_reported_pct:3d}% complete (step {step}/{n_steps})  "
                f"t={step * dt * 1e12:.3f} ps  mean|v_e|={mean_ve:.3e} m/s{eta_suffix}"
            )
        return self.last_reported_pct


def simulation_progress_frames(n_steps: int, frame_interval: int) -> int:
    """Animation frames over a run (initial frame at t=0 plus periodic snapshots)."""
    return 1 + n_steps // frame_interval


def make_progress_tracker(
    n_steps: int,
    frame_interval: int,
    *,
    animating: bool,
    loop_t0: float,
) -> PercentProgressTracker:
    """Build a tracker using animation frame counts when exporting video."""
    if animating:
        return PercentProgressTracker(
            loop_t0=loop_t0,
            total_frames=simulation_progress_frames(n_steps, frame_interval),
            use_animation_frames=True,
        )
    return PercentProgressTracker(loop_t0=loop_t0)


class ExternalField(Protocol):
    def at(self, pos: np.ndarray, t: float = 0.0) -> np.ndarray: ...

    def at_batch(self, positions: np.ndarray, t: float = 0.0) -> np.ndarray: ...


@dataclass(frozen=True)
class IonSpecies:
    name: str
    mass: float
    charge_number: int = 1


@dataclass
class ExampleConfig:
    """Parameters for a quasi-neutral electron–ion electrostatic PIC example."""

    name: str
    domain_length: float = 1.5e-6
    n_cells: int = 24
    n_density: float = 2e20
    macros_per_cell: int = 2
    dt: float = 1e-14
    n_steps: int = 20_000
    frame_interval: int = 1000
    ion: IonSpecies = IonSpecies("Kr+", M_KR, 1)
    electron_energy_ev: tuple[float, float] = (2.0, 5.0)
    ion_energy_ev: tuple[float, float] = (2.0, 5.0)
    spatial_external_field: bool = False


def _configure_threading(backend: ParticleBackend, threads: int | None) -> None:
    if backend != "numba":
        return
    if os.environ.get("OPENBLAS_NUM_THREADS") is None:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if threads is not None:
        import numba

        numba.set_num_threads(threads)


def _electric_kick(vel: np.ndarray, efield: np.ndarray, q_over_m: float, dt: float) -> np.ndarray:
    return vel + q_over_m * dt * efield


def _wrap_positions(pos: np.ndarray, lengths: tuple[float, float, float]) -> np.ndarray:
    wrapped = pos.copy()
    for axis, length in enumerate(lengths):
        wrapped[:, axis] %= length
    return wrapped


def external_field_at_particles(
    efield: ExternalField,
    positions: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evaluate a prescribed external field at each particle position."""
    if hasattr(efield, "at_batch"):
        return np.asarray(efield.at_batch(positions, t), dtype=np.float64)
    n = positions.shape[0]
    e_out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        e_out[i] = efield.at(positions[i], t)
    return e_out


def external_field_uniform(efield: ExternalField, t: float) -> np.ndarray:
    return np.asarray(efield.at(np.zeros(3, dtype=np.float64), t), dtype=np.float64)


class ElectrostaticPICSimulation:
    """Periodic electrostatic PIC with prescribed external E and self-consistent Poisson E."""

    def __init__(
        self,
        config: ExampleConfig,
        efield: ExternalField,
        *,
        seed: int = 42,
        particle_backend: ParticleBackend = "numba",
        frame_stream: FrameStreamWriter | None = None,
    ) -> None:
        self.config = config
        self.efield = efield
        self.rng = np.random.default_rng(seed)
        self.particle_backend = particle_backend
        self.frame_stream = frame_stream
        self._progress: PercentProgressTracker | None = None

        c = config
        self.domain_length = c.domain_length
        self.n_cells = c.n_cells
        self.n_density = c.n_density
        self.dt = c.dt
        self.n_steps = c.n_steps
        self.frame_interval = c.frame_interval
        self.spatial_external_field = c.spatial_external_field

        self.dx = c.domain_length / c.n_cells
        self.n_macro = c.macros_per_cell * c.n_cells**3
        volume = c.domain_length**3
        self.macro_charge = c.n_density * E_CHARGE * volume / self.n_macro

        self.grid = ElectrostaticGrid(
            c.n_cells,
            c.n_cells,
            c.n_cells,
            dx=self.dx,
            dy=self.dx,
            dz=self.dx,
            boundary="periodic",
            eps0=EPS0,
            particle_backend=particle_backend,
        )

        self.ion = c.ion
        self.pos_e: np.ndarray
        self.vel_e: np.ndarray
        self.pos_i: np.ndarray
        self.vel_i: np.ndarray
        self.q_e: float
        self.q_i: float
        self._charges: np.ndarray
        self._pos_all: np.ndarray
        self._e_ext_e: np.ndarray
        self._e_ext_i: np.ndarray
        self._init_particles()
        self._init_buffers()
        self._init_leapfrog()

    def _maxwellian_velocities(self, energy_j: float, mass: float, count: int) -> np.ndarray:
        sigma = np.sqrt(2.0 * energy_j / (3.0 * mass))
        return self.rng.normal(0.0, sigma, size=(count, 3))

    def _sample_energy_j(self, ev_range: tuple[float, float]) -> float:
        return float(self.rng.uniform(ev_range[0], ev_range[1])) * EV_TO_J

    def _init_particles(self) -> None:
        c = self.config
        L = self.domain_length
        self.q_e = -self.macro_charge
        self.q_i = self.ion.charge_number * self.macro_charge

        e_energy = self._sample_energy_j(c.electron_energy_ev)
        i_energy = self._sample_energy_j(c.ion_energy_ev)

        self.pos_e = self.rng.uniform(0.0, L, size=(self.n_macro, 3))
        self.vel_e = self._maxwellian_velocities(e_energy, M_E, self.n_macro)
        self.pos_i = self.rng.uniform(0.0, L, size=(self.n_macro, 3))
        self.vel_i = self._maxwellian_velocities(i_energy, self.ion.mass, self.n_macro)

    def _init_buffers(self) -> None:
        n_total = 2 * self.n_macro
        self._charges = np.concatenate(
            [np.full(self.n_macro, self.q_e), np.full(self.n_macro, self.q_i)]
        )
        self._pos_all = np.empty((n_total, 3), dtype=np.float64)
        self._e_ext_e = np.empty((self.n_macro, 3), dtype=np.float64)
        self._e_ext_i = np.empty((self.n_macro, 3), dtype=np.float64)

    def _sync_stacked_positions(self) -> None:
        self._pos_all[: self.n_macro] = self.pos_e
        self._pos_all[self.n_macro :] = self.pos_i

    def _deposit_both(self) -> None:
        self._sync_stacked_positions()
        self.grid.deposit_rho_cic_batch(self._pos_all, self._charges, in_place=True)

    def _gather_e_both(self) -> tuple[np.ndarray, np.ndarray]:
        self._sync_stacked_positions()
        e_all = self.grid.gather_e_cic_batch(self._pos_all, in_place=True)
        return e_all[: self.n_macro], e_all[self.n_macro :]

    def _external_e(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if self.spatial_external_field:
            self._e_ext_e[:] = external_field_at_particles(self.efield, self.pos_e, t)
            self._e_ext_i[:] = external_field_at_particles(self.efield, self.pos_i, t)
            return self._e_ext_e, self._e_ext_i
        e_uniform = external_field_uniform(self.efield, t)
        return e_uniform, e_uniform

    def _init_leapfrog(self) -> None:
        """Backward half-kick so the main loop is staggered leapfrog (kick + drift)."""
        self.grid.zero_rho()
        self._deposit_both()
        self.grid.solve_fields()
        e_ext_e, e_ext_i = self._external_e(0.0)
        e_self_e, e_self_i = self._gather_e_both()
        e_total_e = e_self_e + e_ext_e
        e_total_i = e_self_i + e_ext_i
        half_dt = -0.5 * self.dt
        if self.grid.particle_backend == "numba":
            electric_kick_b0(self.vel_e, e_total_e, self.q_e / M_E, half_dt)
            electric_kick_b0(self.vel_i, e_total_i, self.q_i / self.ion.mass, half_dt)
        else:
            self.vel_e = _electric_kick(self.vel_e, e_total_e, self.q_e / M_E, half_dt)
            self.vel_i = _electric_kick(self.vel_i, e_total_i, self.q_i / self.ion.mass, half_dt)

    def _record_frame(self, step: int, t: float) -> None:
        if self.frame_stream is not None:
            self.frame_stream.append(step, t * 1e12, self.pos_e, self.pos_i)
            if self._progress is not None:
                self._progress.note_animation_frame()

    def step(self, step_index: int) -> None:
        # E_self and E_ext both evaluated at the position time t = (n-1)*dt.
        t_kick = (step_index - 1) * self.dt
        self.grid.zero_rho()
        self._deposit_both()
        self.grid.solve_fields()

        e_ext_e, e_ext_i = self._external_e(t_kick)
        e_self_e, e_self_i = self._gather_e_both()
        e_total_e = e_self_e + e_ext_e
        e_total_i = e_self_i + e_ext_i

        if self.grid.particle_backend == "numba":
            electric_kick_b0(self.vel_e, e_total_e, self.q_e / M_E, self.dt)
            electric_kick_b0(self.vel_i, e_total_i, self.q_i / self.ion.mass, self.dt)
            self.pos_e += self.vel_e * self.dt
            self.pos_i += self.vel_i * self.dt
            lx, ly, lz = self.grid.domain_lengths
            wrap_positions_periodic(self.pos_e, lx, ly, lz)
            wrap_positions_periodic(self.pos_i, lx, ly, lz)
        else:
            self.vel_e = _electric_kick(self.vel_e, e_total_e, self.q_e / M_E, self.dt)
            self.vel_i = _electric_kick(self.vel_i, e_total_i, self.q_i / self.ion.mass, self.dt)
            self.pos_e = _wrap_positions(self.pos_e + self.vel_e * self.dt, self.grid.domain_lengths)
            self.pos_i = _wrap_positions(self.pos_i + self.vel_i * self.dt, self.grid.domain_lengths)

        if step_index % self.frame_interval == 0:
            self._record_frame(step_index, step_index * self.dt)

    def run(self, *, verbose: bool = True) -> None:
        c = self.config
        if verbose:
            print(f"{c.name}")
            print(
                f"  {self.n_macro} e- + {self.n_macro} {self.ion.name} macros, "
                f"{self.n_cells}^3 grid, n={self.n_density:.2e} m^-3, "
                f"dt={self.dt:.2e} s, {self.n_steps} steps, backend={self.grid.particle_backend}"
            )

        t0 = time.perf_counter()
        self._progress = make_progress_tracker(
            self.n_steps,
            self.frame_interval,
            animating=self.frame_stream is not None,
            loop_t0=t0,
        )
        if verbose:
            self._record_frame(0, 0.0)

        for step in range(1, self.n_steps + 1):
            self.step(step)
            if verbose:
                self._progress.report(
                    step,
                    self.n_steps,
                    dt=self.dt,
                    mean_vel_e=self.vel_e,
                )

        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"  done in {elapsed:.1f} s")
            if self.frame_stream is not None:
                self.frame_stream.close()
                print(
                    f"  streamed {self.frame_stream.n_frames} frames to {self.frame_stream.path} "
                    f"(temporary)"
                )


def make_example_parser(doc: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=doc,
    )
    add_example_cli(parser)
    return parser


def add_example_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("numba", "numpy"), default="numba")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-animate", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANIMATIONS_DIR,
        help="Directory for temporary frame HDF5 and exported animation (default: animations/)",
    )
    parser.add_argument("--frame-subsample", type=int, default=DEFAULT_SUBSAMPLE)


def run_example(
    config: ExampleConfig,
    efield: ExternalField,
    args: argparse.Namespace,
    *,
    script_stem: str,
) -> ElectrostaticPICSimulation | Path:
    if args.steps is not None:
        config.n_steps = args.steps

    backend: ParticleBackend = args.backend
    if backend == "numba" and not HAS_NUMBA:
        raise SystemExit("Numba backend requested but numba is not installed.")

    _configure_threading(backend, args.threads)

    frame_stream = None
    if not args.no_animate:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_stream = FrameStreamWriter(
            out_dir / f"{script_stem}_frames.h5",
            n_e=config.macros_per_cell * config.n_cells**3,
            n_i=config.macros_per_cell * config.n_cells**3,
            subsample=args.frame_subsample,
            seed=args.seed,
        )

    sim = ElectrostaticPICSimulation(
        config,
        efield,
        seed=args.seed,
        particle_backend=backend,
        frame_stream=frame_stream,
    )
    if backend == "numba" and sim.grid.particle_backend != "numba":
        raise SystemExit("Numba backend requested but could not be initialized.")
    sim.run()

    if frame_stream is not None:
        video_path = export_frames_to_video(
            frame_stream.path,
            output_dir=args.output_dir,
            video_stem=script_stem,
            domain_length_um=config.domain_length * 1e6,
            ion_label=config.ion.name,
            delete_frames=True,
        )
        print(f"  animation: {video_path}")
        return video_path

    return sim
