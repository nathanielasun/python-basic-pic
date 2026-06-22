"""Benchmark EM step timing: particle kernels should dominate with Numba backend."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_SRC = Path(__file__).resolve().parent.parent / "src"
for path in (_EXAMPLES, _SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from em_common import EMExampleConfig, EMPICSimulation  # noqa: E402
from fields import ElectricFields  # noqa: E402

try:
    from grids import HAS_NUMBA
except ImportError:
    HAS_NUMBA = False


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestEMBenchmark(unittest.TestCase):
    def test_numba_step_particle_kernels_dominate(self) -> None:
        config = EMExampleConfig(
            name="benchmark",
            n_cells=8,
            macros_per_cell=2,
            n_steps=1,
            frame_interval=10_000,
        )
        sim = EMPICSimulation(
            config,
            ElectricFields.zero(),
            seed=0,
            particle_backend="numba",
        )
        self.assertEqual(sim.grid.particle_backend, "numba")

        n = 5
        t_rho = t_gather = t_push = t_j = t_b = t_e = 0.0
        for _ in range(n):
            sim.grid.rho.fill(0.0)
            t0 = time.perf_counter()
            sim._deposit_rho_both()
            t_rho += time.perf_counter() - t0

            t0 = time.perf_counter()
            sim.grid.update_b(sim.dt)
            t_b += time.perf_counter() - t0

            t0 = time.perf_counter()
            e_se, e_si, b_se, b_si = sim._gather_fields_both()
            t_gather += time.perf_counter() - t0

            t0 = time.perf_counter()
            sim.vel_e = sim.vel_e  # placeholder
            from em_common import _push_higuera_cary_batch
            from common import M_E

            sim.vel_e = _push_higuera_cary_batch(
                sim.vel_e, e_se, b_se, sim.q_e, M_E, sim.dt
            )
            sim.vel_i = _push_higuera_cary_batch(
                sim.vel_i, e_si, b_si, sim.q_i, sim.ion.mass, sim.dt
            )
            t_push += time.perf_counter() - t0

            import numpy as np

            np.copyto(sim._pos_e_old, sim.pos_e)
            np.copyto(sim._pos_i_old, sim.pos_i)
            sim.pos_e += sim.vel_e * sim.dt
            sim.pos_i += sim.vel_i * sim.dt
            from grids.pic_kernels import wrap_positions_periodic

            lx, ly, lz = sim.grid.domain_lengths
            wrap_positions_periodic(sim.pos_e, lx, ly, lz)
            wrap_positions_periodic(sim.pos_i, lx, ly, lz)

            t0 = time.perf_counter()
            sim.grid.zero_currents()
            sim._deposit_j_esirkepov_both()
            t_j += time.perf_counter() - t0

            t0 = time.perf_counter()
            sim.grid.update_e(sim.dt)
            t_e += time.perf_counter() - t0

        particle_frac = (t_rho + t_gather + t_push + t_j) / (
            t_rho + t_gather + t_push + t_j + t_b + t_e + 1e-30
        )
        self.assertGreater(particle_frac, 0.80)

        total = (t_rho + t_gather + t_push + t_j + t_b + t_e) / n
        self.assertLess(total, 2.0, "expected sub-second mean step on 8^3 / 2 ppc")


if __name__ == "__main__":
    unittest.main()
