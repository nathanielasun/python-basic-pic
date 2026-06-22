"""Shared electromagnetic (Yee-grid) PIC driver utilities for ``examples/em_*.py``."""

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

from Pushers import Pushers
from common import (  # noqa: E402
    ANIMATIONS_DIR,
    DEFAULT_SUBSAMPLE,
    E_CHARGE,
    EPS0,
    EV_TO_J,
    IonSpecies,
    M_E,
    M_KR,
    PercentProgressTracker,
    _configure_threading,
    external_field_at_particles,
    external_field_uniform,
    make_progress_tracker,
)
from grids import ParticleBackend, YeeGrid, wrap_positions_periodic
from pic_animation import FrameStreamWriter, export_frames_to_video

try:
    from grids import HAS_NUMBA
except ImportError:
    HAS_NUMBA = False

C_VAC = 299_792_458.0
MU0 = 4.0e-7 * np.pi


class ExternalField(Protocol):
    def at(self, pos: np.ndarray, t: float = 0.0) -> np.ndarray: ...

    def at_batch(self, positions: np.ndarray, t: float = 0.0) -> np.ndarray: ...


@dataclass
class EMExampleConfig:
    """Parameters for a high-density relativistic EM PIC example on ``YeeGrid``."""

    name: str
    domain_length: float = 1.0e-6
    n_cells: int = 16
    n_density: float = 1e26
    macros_per_cell: int = 2
    dt: float | None = None
    n_steps: int = 10_000
    frame_interval: int = 500
    ion: IonSpecies = IonSpecies("Kr+", M_KR)
    electron_energy_ev: tuple[float, float] = (100.0, 1000.0)
    ion_energy_ev: tuple[float, float] = (1.0, 10.0)
    spatial_external_field: bool = False
    cfl_safety: float = 0.95


def _zero_field_batch(n: int) -> np.ndarray:
    return np.zeros((n, 3), dtype=np.float64)


def _push_higuera_cary_batch(
    vel: np.ndarray,
    efield: np.ndarray,
    bfield: np.ndarray,
    q: float,
    m: float,
    dt: float,
) -> np.ndarray:
    return Pushers.push_batch("higuera_cary", vel, efield, bfield, q, m, dt, c=C_VAC)


class EMPICSimulation:
    """Periodic EM PIC on ``YeeGrid`` with Higuera–Cary relativistic particle push."""

    def __init__(
        self,
        config: EMExampleConfig,
        efield: ExternalField,
        bfield: ExternalField | None = None,
        *,
        seed: int = 42,
        particle_backend: ParticleBackend = "numba",
        frame_stream: FrameStreamWriter | None = None,
    ) -> None:
        self.config = config
        self.efield = efield
        self.bfield = bfield
        self.particle_backend = particle_backend
        self.rng = np.random.default_rng(seed)
        self.frame_stream = frame_stream
        self._progress: PercentProgressTracker | None = None

        c = config
        self.domain_length = c.domain_length
        self.n_cells = c.n_cells
        self.n_density = c.n_density
        self.n_steps = c.n_steps
        self.frame_interval = c.frame_interval
        self.spatial_external_field = c.spatial_external_field

        self.dx = c.domain_length / c.n_cells
        self.n_macro = c.macros_per_cell * c.n_cells**3
        volume = c.domain_length**3
        self.macro_charge = c.n_density * E_CHARGE * volume / self.n_macro

        self.grid = YeeGrid(
            c.n_cells,
            c.n_cells,
            c.n_cells,
            dx=self.dx,
            dy=self.dx,
            dz=self.dx,
            boundary="periodic",
            eps0=EPS0,
            mu0=MU0,
            particle_backend=particle_backend,
        )

        if c.dt is None:
            self.dt = c.cfl_safety * self.grid.cfl_dt_limit()
        else:
            self.dt = c.dt

        self.ion = c.ion
        self.pos_e: np.ndarray
        self.vel_e: np.ndarray
        self.pos_i: np.ndarray
        self.vel_i: np.ndarray
        self.q_e: float
        self.q_i: float
        self._pos_all: np.ndarray
        self._charges: np.ndarray
        self._e_ext_e = np.empty((self.n_macro, 3), dtype=np.float64)
        self._e_ext_i = np.empty((self.n_macro, 3), dtype=np.float64)
        self._b_ext_e = np.empty((self.n_macro, 3), dtype=np.float64)
        self._b_ext_i = np.empty((self.n_macro, 3), dtype=np.float64)

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
        self._pos_e_old = np.empty((self.n_macro, 3), dtype=np.float64)
        self._pos_i_old = np.empty((self.n_macro, 3), dtype=np.float64)

    def _deposit_j_esirkepov_both(self) -> None:
        q_e = np.full(self.n_macro, self.q_e, dtype=np.float64)
        q_i = np.full(self.n_macro, self.q_i, dtype=np.float64)
        self.grid.deposit_j_esirkepov_cic_batch(self._pos_e_old, self.pos_e, q_e)
        self.grid.deposit_j_esirkepov_cic_batch(self._pos_i_old, self.pos_i, q_i)

    def _sync_stacked_positions(self) -> None:
        self._pos_all[: self.n_macro] = self.pos_e
        self._pos_all[self.n_macro :] = self.pos_i

    def _deposit_rho_both(self) -> None:
        self._sync_stacked_positions()
        self.grid.deposit_rho_cic_batch(self._pos_all, self._charges)

    def _gather_fields_both(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._sync_stacked_positions()
        e_all = self.grid.gather_e_cic_batch(self._pos_all)
        b_all = self.grid.gather_b_cic_batch(self._pos_all)
        return (
            e_all[: self.n_macro],
            e_all[self.n_macro :],
            b_all[: self.n_macro],
            b_all[self.n_macro :],
        )

    def _external_fields(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.spatial_external_field:
            e_ext_e = external_field_at_particles(self.efield, self.pos_e, t)
            e_ext_i = external_field_at_particles(self.efield, self.pos_i, t)
            if self.bfield is not None:
                b_ext_e = external_field_at_particles(self.bfield, self.pos_e, t)
                b_ext_i = external_field_at_particles(self.bfield, self.pos_i, t)
            else:
                b_ext_e = _zero_field_batch(self.n_macro)
                b_ext_i = _zero_field_batch(self.n_macro)
            return e_ext_e, e_ext_i, b_ext_e, b_ext_i

        e_uniform = external_field_uniform(self.efield, t)
        if self.bfield is not None:
            b_uniform = external_field_uniform(self.bfield, t)
        else:
            b_uniform = np.zeros(3, dtype=np.float64)
        return e_uniform, e_uniform, b_uniform, b_uniform

    def _init_leapfrog(self) -> None:
        """Backward half-kick with Higuera–Cary for staggered EM leapfrog."""
        self.grid.rho.fill(0.0)
        self._deposit_rho_both()

        e_se, e_si, b_se, b_si = self._gather_fields_both()
        e_ee, e_ei, b_ee, b_ei = self._external_fields(0.0)
        half_dt = -0.5 * self.dt
        self.vel_e = _push_higuera_cary_batch(
            self.vel_e, e_se + e_ee, b_se + b_ee, self.q_e, M_E, half_dt
        )
        self.vel_i = _push_higuera_cary_batch(
            self.vel_i, e_si + e_ei, b_si + b_ei, self.q_i, self.ion.mass, half_dt
        )

    def _record_frame(self, step: int, t: float) -> None:
        if self.frame_stream is not None:
            self.frame_stream.append(step, t * 1e12, self.pos_e, self.pos_i)
            if self._progress is not None:
                self._progress.note_animation_frame()

    def step(self, step_index: int) -> None:
        t_kick = (step_index - 1) * self.dt

        self.grid.rho.fill(0.0)
        self._deposit_rho_both()

        self.grid.update_b(self.dt)

        e_se, e_si, b_se, b_si = self._gather_fields_both()
        e_ee, e_ei, b_ee, b_ei = self._external_fields(t_kick)
        self.vel_e = _push_higuera_cary_batch(
            self.vel_e,
            e_se + e_ee,
            b_se + b_ee,
            self.q_e,
            M_E,
            self.dt,
        )
        self.vel_i = _push_higuera_cary_batch(
            self.vel_i,
            e_si + e_ei,
            b_si + b_ei,
            self.q_i,
            self.ion.mass,
            self.dt,
        )

        np.copyto(self._pos_e_old, self.pos_e)
        np.copyto(self._pos_i_old, self.pos_i)
        self.pos_e += self.vel_e * self.dt
        self.pos_i += self.vel_i * self.dt
        lx, ly, lz = self.grid.domain_lengths
        wrap_positions_periodic(self.pos_e, lx, ly, lz)
        wrap_positions_periodic(self.pos_i, lx, ly, lz)

        self.grid.zero_currents()
        self._deposit_j_esirkepov_both()
        self.grid.update_e(self.dt)

        if step_index % self.frame_interval == 0:
            self._record_frame(step_index, step_index * self.dt)

    def run(self, *, verbose: bool = True) -> None:
        c = self.config
        if verbose:
            mean_gamma_e = float(
                np.mean(1.0 / np.sqrt(1.0 - np.minimum(np.sum(self.vel_e**2, axis=1) / C_VAC**2, 0.999999)))
            )
            print(f"{c.name}")
            print(
                f"  {self.n_macro} e- + {self.n_macro} {self.ion.name} macros, "
                f"{self.n_cells}^3 Yee grid, n={self.n_density:.2e} m^-3, "
                f"dt={self.dt:.2e} s (CFL limit {self.grid.cfl_dt_limit():.2e}), "
                f"pusher=higuera_cary, backend={self.grid.particle_backend}, "
                f"mean gamma_e~{mean_gamma_e:.2f}"
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


def make_em_example_parser(doc: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=doc,
    )
    add_em_example_cli(parser)
    return parser


def add_em_example_cli(parser: argparse.ArgumentParser) -> None:
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


def run_em_example(
    config: EMExampleConfig,
    efield: ExternalField,
    args: argparse.Namespace,
    *,
    script_stem: str,
    bfield: ExternalField | None = None,
) -> EMPICSimulation | Path:
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

    sim = EMPICSimulation(
        config,
        efield,
        bfield,
        seed=args.seed,
        particle_backend=backend,
        frame_stream=frame_stream,
    )
    sim.run()

    if backend == "numba" and sim.grid.particle_backend != "numba":
        raise SystemExit("Numba backend requested but could not be initialized.")

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
