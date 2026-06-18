#!/usr/bin/env python3
"""
Physics:
    Quasi-neutral Kr+ / e- plasma in a periodic cube driven by a uniform, spatially
    constant RF electric field (10 GHz sinusoid along z). Macro-particles sample a
    low-density Maxwellian (2–5 eV). Self-consistent electrostatic fields from
    charge deposition + Poisson FFT are added to the prescribed RF; particles are
    pushed with Boris in the B=0 limit. Models a weakly coupled RF plasma slab /
    bounded plasma oscillation testbed at ~2×10^14 cm^-3 in a 1.5 µm periodic cube.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from common import (  # noqa: E402
    ExampleConfig,
    IonSpecies,
    M_KR,
    make_example_parser,
    run_example,
)

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fields import ElectricFields  # noqa: E402

RF_FREQ = 10e9
E0_PEAK = 1e4


def main() -> None:
    parser = make_example_parser(__doc__)
    args = parser.parse_args()

    config = ExampleConfig(
        name="Kr+ / e- RF plasma",
        domain_length=1.5e-6,
        n_cells=24,
        n_density=2e20,
        macros_per_cell=2,
        dt=1e-14,
        n_steps=20_000,
        frame_interval=1000,
        ion=IonSpecies("Kr+", M_KR),
        spatial_external_field=False,
    )

    omega = 2.0 * 3.141592653589793 * RF_FREQ
    efield = ElectricFields.sinusoidal(
        amplitude=[0.0, 0.0, E0_PEAK],
        omega=omega,
        wavevector=[0.0, 0.0, 0.0],
    )

    run_example(config, efield, args, script_stem="01_kr_rf_plasma")


if __name__ == "__main__":
    main()
