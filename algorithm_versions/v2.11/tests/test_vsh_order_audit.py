from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "vsh_order_audit.py"
SPEC = importlib.util.spec_from_file_location("vsh_order_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VshBasisTests(unittest.TestCase):
    def test_basis_and_single_mode_recovery(self) -> None:
        result = MODULE.run_basis_self_tests()
        self.assertLess(result["sphere_weight_error"], 5e-4)
        self.assertLess(result["selected_mode_max_norm_error"], 2e-3)
        self.assertLess(result["selected_mode_max_x_rx_inner_product"], 2e-10)
        self.assertLess(result["synthetic_mode_max_coefficient_error"], 2e-3)


if __name__ == "__main__":
    unittest.main()
