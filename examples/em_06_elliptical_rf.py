#!/usr/bin/env python3
"""
Physics:
    High-density (10^26 m^-3) Kr+ / e- plasma in a uniform-volume elliptically
    polarized 10 GHz RF field (delta = pi/4) with a weak uniform B_x guide.
    Relativistic electrons (150–600 eV) are advanced with Higuera–Cary on the
    Yee grid; exercises vector RF heating with non-zero B in the full Maxwell
    PIC limit (prescribed E + B superposed on self-consistent grid fields).
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
E0 = 5e7
B_GUIDE = 0.2


def main() -> None:
    parser = make_em_example_parser(__doc__)
    args = parser.parse_args()

    config = EMExampleConfig(
        name="Elliptical RF + guide B on relativistic Kr+ / e- (Yee EM)",
        domain_length=1.5e-6,
        n_cells=16,
        n_density=1e26,
        macros_per_cell=2,
        n_steps=60_000,
        frame_interval=100,
        ion=IonSpecies("Kr+", M_KR),
        electron_energy_ev=(150.0, 600.0),
        ion_energy_ev=(1.0, 8.0),
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
    bfield = MagneticFields.uniform([B_GUIDE, 0.0, 0.0])

    run_em_example(config, efield, args, script_stem="em_06_elliptical_rf", bfield=bfield)


if __name__ == "__main__":
    main()
