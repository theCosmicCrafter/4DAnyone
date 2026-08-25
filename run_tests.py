"""Test runner for 4DAnyone test suite without external plugin dependencies."""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import all test modules
import tests.test_views as t_views
import tests.test_routing as t_routing
import tests.test_streaming as t_streaming
import tests.test_quantization as t_quant
import tests.test_deformable_gs as t_deform
import tests.test_exporter_4d as t_exporter

TEST_MODULES = [
    t_views,
    t_routing,
    t_streaming,
    t_quant,
    t_deform,
    t_exporter,
]


def run_all_tests():
    total_passed = 0
    total_failed = 0
    start_time = time.monotonic()

    print("=" * 60)
    print(" 4DAnyone Test Suite Verification")
    print("=" * 60)

    for mod in TEST_MODULES:
        mod_name = mod.__name__
        print(f"\n[Running Module] {mod_name}")
        functions = [
            (name, func)
            for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]
        for name, func in functions:
            try:
                func()
                print(f"  [PASS] {name}")
                total_passed += 1
            except Exception as exc:
                print(f"  [FAIL] {name}: {exc}")
                total_failed += 1

    duration = time.monotonic() - start_time
    print("\n" + "=" * 60)
    print(f" Summary: {total_passed} passed, {total_failed} failed in {duration:.3f}s")
    print("=" * 60)

    if total_failed > 0:
        sys.exit(1)
    print("\nALL 4DANYONE TESTS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    run_all_tests()
