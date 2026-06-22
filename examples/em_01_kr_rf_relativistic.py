#!/usr/bin/env python3
"""
Physics:
    High-density (10^26 m^-3) quasi-neutral Kr+ / e- plasma in a periodic cube
    driven by a uniform 10 GHz RF electric field. Electrons are initialized with
  100–500 eV Maxwellian speeds (relativistic). Full Maxwell advance on a Yee
    grid with self-consistent rho/J deposition; particles are pushed with the
    Higuera–Cary relativistic integrator (no classical Boris).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from em_common import EMExampleConfig, make_em_example_parser, run_em_example  # noqa: E402
from common import IonSpecies, M_KR  # noqa: E402

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fields import ElectricFields, MagneticFields  # noqa: E402

RF_FREQ = 10e9
E0_PEAK = 5e7


def main() -> None:
    parser = make_em_example_parser(__doc__)
    args = parser.parse_args()

    config = EMExampleConfig(
        name="Relativistic Kr+ / e- RF plasma (Yee EM)",
        domain_length=1.5e-6,
        n_cells=16,
        n_density=1e26,
        macros_per_cell=2,
        n_steps=5_000,
        frame_interval=500,
        ion=IonSpecies("Kr+", M_KR),
        electron_energy_ev=(100.0, 500.0),
        ion_energy_ev=(1.0, 5.0),
        spatial_external_field=False,
    )

    omega = 2.0 * math.pi * RF_FREQ
    efield = ElectricFields.sinusoidal(
        amplitude=[0.0, 0.0, E0_PEAK],
        omega=omega,
        wavevector=[0.0, 0.0, 0.0],
    )
    bfield = MagneticFields.zero()

    run_em_example(config, efield, args, script_stem="em_01_kr_rf_relativistic", bfield=bfield)


if __name__ == "__main__":
    main()
