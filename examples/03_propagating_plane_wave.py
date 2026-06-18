#!/usr/bin/env python3
"""
Physics:
    Low-density Kr+ / e- plasma illuminated by a propagating linearly polarized
    plane wave (sinusoidal E, finite k) incident along +x into a periodic
    3.2 µm cube (4 × 800 nm wavelengths). External E(r,t) varies in space and
    time; self-consistent Poisson fields are superposed. Exercises spatially
    varying prescribed fields (gathered per particle) and wave–plasma coupling
    at ~800 nm vacuum wavelength (~10^14 cm^-3).
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
E0 = 5e7
OMEGA = 2.0 * math.pi * 3.75e14


def main() -> None:
    parser = make_example_parser(__doc__)
    args = parser.parse_args()

    config = ExampleConfig(
        name="Propagating plane wave on Kr+ / e- plasma",
        domain_length=4 * WAVELENGTH,
        n_cells=24,
        n_density=1e20,
        macros_per_cell=2,
        dt=5e-16,
        n_steps=30_000,
        frame_interval=1500,
        ion=IonSpecies("Kr+", M_KR),
        spatial_external_field=True,
    )

    efield = ElectricFields.plane_wave_incident(
        E0=E0,
        omega=OMEGA,
        theta=math.pi / 2,
        phi=0.0,
        wavelength=WAVELENGTH,
    )

    run_example(config, efield, args, script_stem="03_propagating_plane_wave")


if __name__ == "__main__":
    main()
