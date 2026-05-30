"""Milestone 1: run all correctness tests (CPU math + GPU flex/parity).

  python scripts/milestone1_parity.py
"""

import _bootstrap  # noqa: F401

import traceback

from tests.test_gate_grad import (
    test_factorized_equals_dense_reference_gate_ones,
    test_gate_zeroes_contribution_without_renorm,
    test_ste_forward_is_hard_and_grad_reaches_all_tokens,
)
from tests.test_flex_mask import test_flex_mask_matches_dense
from tests.test_parity import test_factorized_flex_equals_reference


def main():
    tests = [
        test_ste_forward_is_hard_and_grad_reaches_all_tokens,
        test_gate_zeroes_contribution_without_renorm,
        test_factorized_equals_dense_reference_gate_ones,
        test_flex_mask_matches_dense,
        test_factorized_flex_equals_reference,
    ]
    ok = True
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            ok = False
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print("MILESTONE 1: ALL PASS" if ok else "MILESTONE 1: FAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
