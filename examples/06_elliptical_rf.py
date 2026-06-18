#!/usr/bin/env python3
"""
Physics:
    Kr+ / e- plasma in a uniform-volume, spatially constant RF field with
    elliptical polarization (linear + pi/4 phase offset between transverse
    components). Same 10 GHz drive as the baseline RF case in a 1.5 µm cube at
    ~2×10^14 cm^-3; exercises elliptical polarization and vector RF heating of
    electrons vs ions in the electrostatic PIC limit (prescribed E only, B=0).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from common import ExampleConfig, IonSpecies, M_KR, make_example_parser, run_example  # noqa: E402

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fields import ElectricFields  # noqa: E402

RF_FREQ = 10e9
E0 = 8e3


def main() -> None:
    parser = make_example_parser(__doc__)
    args = parser.parse_args()

    config = ExampleConfig(
        name="Elliptically polarized RF on Kr+ / e- plasma",
        domain_length=1.5e-6,
        n_cells=24,
        n_density=2e20,
        macros_per_cell=2,
        dt=1e-14,
        n_steps=30_000,
        frame_interval=1500,
        ion=IonSpecies("Kr+", M_KR),
        spatial_external_field=False,
    )

    omega = 2.0 * math.pi * RF_FREQ
    efield = ElectricFields.sinusoidal_elliptical(
        E0=E0,
        omega=omega,
        wavevector=[0.0, 0.0, 0.0],
        psi=0.0,
        delta=math.pi / 4,
    )

    run_example(config, efield, args, script_stem="06_elliptical_rf")


if __name__ == "__main__":
    main()
