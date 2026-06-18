#!/usr/bin/env python3
"""
Physics:
    Thermal hydrogen (H+ / e-) plasma with no external field. Equal macro-particle
    counts and quasi-neutral initialization at ~2×10^14 cm^-3 in a periodic 1.5 µm cube.
    Electrons and ions receive independent 2–5 eV Maxwellian velocities. Only
    self-fields from Poisson solve accelerate particles—useful for observing
    plasma oscillations, Landau damping at modest grid resolution, and net charge
    neutrality without an imposed drive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from common import ExampleConfig, IonSpecies, M_H, make_example_parser, run_example  # noqa: E402

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fields import ElectricFields  # noqa: E402


def main() -> None:
    parser = make_example_parser(__doc__)
    args = parser.parse_args()

    config = ExampleConfig(
        name="Thermal H+ / e- plasma (no drive)",
        domain_length=1.5e-6,
        n_cells=24,
        n_density=2e20,
        macros_per_cell=2,
        dt=1e-14,
        n_steps=40_000,
        frame_interval=2000,
        ion=IonSpecies("H+", M_H),
        spatial_external_field=False,
    )

    efield = ElectricFields.zero()
    run_example(config, efield, args, script_stem="02_hydrogen_thermal")


if __name__ == "__main__":
    main()
