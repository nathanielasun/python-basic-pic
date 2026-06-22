#!/usr/bin/env python3
"""
Physics:
    High-density (10^26 m^-3) thermal H+ / e- plasma with no external drive.
    Hot electrons (50–200 eV) and warm ions (0.5–2 eV) in a 1.5 µm periodic
    cube. Exercises Yee-grid vacuum Maxwell modes plus self-consistent J/rho
    coupling and relativistic Higuera–Cary pushes without imposed RF or laser
    fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from em_common import EMExampleConfig, make_em_example_parser, run_em_example  # noqa: E402
from common import IonSpecies, M_H  # noqa: E402

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fields import ElectricFields  # noqa: E402


def main() -> None:
    parser = make_em_example_parser(__doc__)
    args = parser.parse_args()

    config = EMExampleConfig(
        name="Relativistic thermal H+ / e- (Yee EM, no drive)",
        domain_length=1.5e-6,
        n_cells=16,
        n_density=1e26,
        macros_per_cell=2,
        n_steps=8_000,
        frame_interval=800,
        ion=IonSpecies("H+", M_H),
        electron_energy_ev=(50.0, 200.0),
        ion_energy_ev=(0.5, 2.0),
        spatial_external_field=False,
    )

    run_em_example(config, ElectricFields.zero(), args, script_stem="em_02_hydrogen_relativistic_thermal")


if __name__ == "__main__":
    main()
