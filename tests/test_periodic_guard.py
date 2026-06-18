"""Periodic guard-cell copy regression tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grids import periodic_field


class TestPeriodicGuard(unittest.TestCase):
    def test_ng2_copies_interior_face_not_guard(self) -> None:
        ng = 2
        nx = ny = nz = 8
        shape = (nx + 2 * ng, ny + 2 * ng, nz + 2 * ng)
        field = np.zeros(shape, dtype=np.float64)

        # Last ng interior x-faces before the high guard (indices ng+nx-ng .. ng+nx-1)
        field[ng + nx - ng, ng, ng] = 100.0
        field[ng + nx - 1, ng, ng] = 200.0
        field[0, ng, ng] = -1.0
        field[1, ng, ng] = -1.0

        periodic_field(field, ng)

        self.assertEqual(float(field[0, ng, ng]), 100.0)
        self.assertEqual(float(field[1, ng, ng]), 200.0)

        field[ng, ng, ng] = 3.0
        field[ng + 1, ng, ng] = 4.0
        field[-1, ng, ng] = -1.0
        field[-2, ng, ng] = -1.0
        periodic_field(field, ng)
        self.assertEqual(float(field[-2, ng, ng]), 3.0)
        self.assertEqual(float(field[-1, ng, ng]), 4.0)


if __name__ == "__main__":
    unittest.main()
