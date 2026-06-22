#!/usr/bin/env python3
"""
Physics:
    High-density Kr+ / e- plasma (10^26 m^-3) illuminated by a relativistic-
    intensity-scale propagating plane wave (800 nm, linear polarization along y,
    propagation +x) with a matched transverse B wave (|B| = |E|/c). Exercises
    staggered Yee gather of E and B, spatially varying external fields, and
    Higuera–Cary pushes in a 3.2 µm periodic domain.
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
E0 = 5e9
OMEGA = 2.0 * math.pi * C_VAC / WAVELENGTH


def main() -> None:
    parser = make_em_example_parser(__doc__)
    args = parser.parse_args()

    config = EMExampleConfig(
        name="Relativistic plane wave on Kr+ / e- (Yee EM)",
        domain_length=4 * WAVELENGTH,
        n_cells=20,
        n_density=1e26,
        macros_per_cell=2,
        n_steps=6_000,
        frame_interval=600,
        ion=IonSpecies("Kr+", M_KR),
        electron_energy_ev=(200.0, 800.0),
        ion_energy_ev=(2.0, 8.0),
        spatial_external_field=True,
    )

    b0 = E0 / C_VAC
    efield = ElectricFields.plane_wave_incident(
        E0=E0,
        omega=OMEGA,
        theta=math.pi / 2,
        phi=0.0,
        wavelength=WAVELENGTH,
    )
    bfield = MagneticFields.plane_wave_incident(
        B0=b0,
        omega=OMEGA,
        theta=math.pi / 2,
        phi=0.0,
        wavelength=WAVELENGTH,
        pol_angle=math.pi / 2,
    )

    run_em_example(
        config,
        efield,
        args,
        script_stem="em_03_propagating_plane_wave",
        bfield=bfield,
    )


if __name__ == "__main__":
    main()
