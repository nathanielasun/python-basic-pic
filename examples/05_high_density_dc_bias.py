#!/usr/bin/env python3
"""
Physics:
    High-density (2×10^15 cm^-3) Kr+ / e- gas in a 1.5 µm periodic cube with a
    uniform DC electric bias (+z). Stronger self-fields from discrete particle
    noise compete with the applied bias—useful for testing charge neutrality,
    field quality, and electrostatic heating under combined external +
    space-charge fields (20× longer runtime than the original demo).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from common import ExampleConfig, IonSpecies, M_KR, make_example_parser, run_example  # noqa: E402

if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fields import ElectricFields  # noqa: E402

DC_FIELD = 5e5


def main() -> None:
    parser = make_example_parser(__doc__)
    args = parser.parse_args()

    config = ExampleConfig(
        name="High-density Kr+ / e- with DC bias",
        domain_length=1.5e-6,
        n_cells=24,
        n_density=2e21,
        macros_per_cell=4,
        dt=5e-15,
        n_steps=20_000,
        frame_interval=1000,
        ion=IonSpecies("Kr+", M_KR),
        electron_energy_ev=(0.5, 1.0),
        ion_energy_ev=(0.05, 0.1),
        spatial_external_field=False,
    )

    efield = ElectricFields.uniform([0.0, 0.0, DC_FIELD])
    run_example(config, efield, args, script_stem="05_high_density_dc_bias")


if __name__ == "__main__":
    main()
