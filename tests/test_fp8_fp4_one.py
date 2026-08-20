"""Same functionality as test_fp8_fp4.py, but cases can be run one by one.

Only the enumerate functions (from generators.py, used by test_fp8_fp4.py)
are patched: an index number is mapped to the selected case, and the
original test body runs it.

Usage:
    python tests/test_fp8_fp4_one.py                      # show help
    python tests/test_fp8_fp4_one.py <suite>              # dry run: list all cases of a suite
    python tests/test_fp8_fp4_one.py <suite> dry          # same as above
    python tests/test_fp8_fp4_one.py <suite> 0            # run case 0
    python tests/test_fp8_fp4_one.py <suite> 0 2 5        # run cases 0, 2 and 5
    python tests/test_fp8_fp4_one.py <suite> all          # run all cases (same as test_fp8_fp4.py)

Suites:
    gemm          (test_gemm)
    m_contiguous  (test_m_grouped_gemm_contiguous)
    m_masked      (test_m_grouped_gemm_masked)
    k_contiguous  (test_k_grouped_gemm_contiguous)
"""

import random
import sys
import torch

import test_fp8_fp4

SUITES = {
    'test_gemm':         ('enumerate_normal',                 test_fp8_fp4.test_gemm),
    'test_m_grouped_gemm_contiguous': ('enumerate_m_grouped_contiguous',   test_fp8_fp4.test_m_grouped_gemm_contiguous),
    'test_m_grouped_gemm_masked':     ('enumerate_m_grouped_masked',       test_fp8_fp4.test_m_grouped_gemm_masked),
    'test_k_grouped_gemm_contiguous': ('enumerate_k_grouped_contiguous',   test_fp8_fp4.test_k_grouped_gemm_contiguous),
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in SUITES:
        print(__doc__.strip())
        return

    name = args[0]
    enum_name, test_fn = SUITES[name]
    enum_fn = getattr(test_fp8_fp4, enum_name)

    # Seed before enumerating: enumerate_k_grouped_contiguous draws random ks,
    # and a fixed seed keeps the case list (and thus indices) stable across runs
    torch.manual_seed(0)
    random.seed(0)
    cases = list(enum_fn(torch.float8_e4m3fn))

    # No case index given: list all cases
    if len(args) == 1:
        print(f'Suite "{name}": {len(cases)} cases')
        for i, case in enumerate(cases):
            print(f' [{i:4}] {case}')
        return

    indices = list(range(len(cases))) if args[1] == 'all' else [int(x) for x in args[1:]]
    for i in indices:
        assert 0 <= i < len(cases), f'Case index {i} out of range [0, {len(cases)})'

    # Map index number -> selected cases, then run the original test body
    selected = [cases[i] for i in indices]
    setattr(test_fp8_fp4, enum_name, lambda *a, **kw: iter(selected))
    for i, case in zip(indices, selected):
        print(f'[test_fp8_fp4 {name} #{i}] {case}')
    test_fn()
    print()


if __name__ == '__main__':
    main()
