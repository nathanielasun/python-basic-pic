#!/usr/bin/env python3
"""
Physics:
    Ultra-high-density (10^27 m^-3) Kr+ / e- plasma with a uniform DC electric
    bias (+z) and a static axial guide field B_z. Relativistic hot electrons
    (200–800 eV) sample strong self-fields on the Yee grid while Higuera–Cary
    pushes retain accuracy at gamma > 1. Tests rho/J deposition and Ampere
    coupling under combined external E and B at solid-density-scale n_e.
"""

from __future__ import annotations

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

DC_FIELD = 1e8
B_GUIDE = 0.5


def main() -> None:
    parser = make_em_example_parser(__doc__)
    args = parser.parse_args()

    config = EMExampleConfig(
        name="Ultra-high-density Kr+ / e- with DC + guide B (Yee EM)",
        domain_length=1.5e-6,
        n_cells=16,
        n_density=1e27,
        macros_per_cell=12,
        n_steps=6_000,
        frame_interval=100,
        ion=IonSpecies("Kr+", M_KR),
        electron_energy_ev=(200.0, 800.0),
        ion_energy_ev=(2.0, 10.0),
        spatial_external_field=False,
    )

    efield = ElectricFields.uniform([0.0, 0.0, DC_FIELD])
    bfield = MagneticFields.uniform([0.0, 0.0, B_GUIDE])

    run_em_example(
        config,
        efield,
        args,
        script_stem="em_05_high_density_dc_bias",
        bfield=bfield,
    )


if __name__ == "__main__":
    main()
