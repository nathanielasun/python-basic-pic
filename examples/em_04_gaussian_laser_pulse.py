#!/usr/bin/env python3
"""
Physics:
    High-density Kr+ / e- plasma driven by a Gaussian-enveloped relativistic-
    scale laser pulse (800 nm carrier, ~600 nm envelope width) propagating along
    +z with paired vacuum B. Full Yee FDTD for self-fields plus prescribed
    oblique-capable pulse geometry via ``gaussian_pulse_incident``; Higuera–Cary
    particle advance for hot (100–1000 eV) electrons at 10^26 m^-3.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from em_common import EMExampleConfig, C_VAC, make_em_example_parser, run_em_example  # noqa: E402
from common import IonSpecies, M_KR  # noqa: E402

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fields import ElectricFields, MagneticFields  # noqa: E402

WAVELENGTH = 800e-9
E0 = 8e9
OMEGA = 2.0 * math.pi * C_VAC / WAVELENGTH
PULSE_WIDTH = 600e-9


def main() -> None:
    parser = make_em_example_parser(__doc__)
    args = parser.parse_args()

    domain = 4.5e-6
    config = EMExampleConfig(
        name="Relativistic Gaussian laser pulse (Yee EM)",
        domain_length=domain,
        n_cells=20,
        n_density=1e26,
        macros_per_cell=2,
        n_steps=8_000,
        frame_interval=800,
        ion=IonSpecies("Kr+", M_KR),
        electron_energy_ev=(100.0, 1000.0),
        ion_energy_ev=(1.0, 10.0),
        spatial_external_field=True,
    )

    b0 = E0 / C_VAC
    efield = ElectricFields.gaussian_pulse_incident(
        E0=E0,
        omega=OMEGA,
        theta=0.0,
        phi=0.0,
        wavelength=WAVELENGTH,
        width=PULSE_WIDTH,
        center=[0.0, 0.0, 0.5 * domain],
    )
    bfield = MagneticFields.gaussian_pulse_incident(
        B0=b0,
        omega=OMEGA,
        theta=0.0,
        phi=0.0,
        wavelength=WAVELENGTH,
        width=PULSE_WIDTH,
        center=[0.0, 0.0, 0.5 * domain],
        pol_angle=math.pi / 2,
    )

    run_em_example(
        config,
        efield,
        args,
        script_stem="em_04_gaussian_laser_pulse",
        bfield=bfield,
    )


if __name__ == "__main__":
    main()
