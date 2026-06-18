#!/usr/bin/env python3
"""
Physics:
    Kr+ / e- plasma driven by a Gaussian-enveloped laser pulse (800 nm carrier,
    ~200 fs FWHM-scale envelope) propagating along +z through a 4.5 µm periodic
    domain. The pulse is defined in a local wave frame and mapped into the lab
    grid. Combines finite-duration EM drive with self-consistent electrostatic
    response—representative of short-pulse laser–plasma interaction in the
    electrostatic (non-relativistic, B=0) limit.
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

WAVELENGTH = 800e-9
E0 = 1e8
OMEGA = 2.0 * math.pi * 3.75e14
PULSE_WIDTH = 600e-9


def main() -> None:
    parser = make_example_parser(__doc__)
    args = parser.parse_args()

    domain = 4.5e-6
    config = ExampleConfig(
        name="Gaussian laser pulse on Kr+ / e- plasma",
        domain_length=domain,
        n_cells=24,
        n_density=2e20,
        macros_per_cell=2,
        dt=2e-16,
        n_steps=40_000,
        frame_interval=2000,
        ion=IonSpecies("Kr+", M_KR),
        spatial_external_field=True,
    )

    efield = ElectricFields.gaussian_pulse_incident(
        E0=E0,
        omega=OMEGA,
        theta=0.0,
        phi=0.0,
        wavelength=WAVELENGTH,
        width=PULSE_WIDTH,
        center=[0.0, 0.0, 0.5 * domain],
    )

    run_example(config, efield, args, script_stem="04_gaussian_laser_pulse")


if __name__ == "__main__":
    main()
