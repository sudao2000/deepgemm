"""Same functionality as test_bf16.py, but cases can be run one by one.

Only the enumerate functions (from generators.py, used by test_bf16.py) are
patched: an index number is mapped to the selected case, and the original
test body runs it.

Usage:
    python tests/test_bf16_one.py                      # show help
    python tests/test_bf16_one.py <suite>              # dry run: list all cases of a suite
    python tests/test_bf16_one.py <suite> dry          # same as above
    python tests/test_bf16_one.py <suite> 0            # run case 0
    python tests/test_bf16_one.py <suite> 0 2 5        # run cases 0, 2 and 5
    python tests/test_bf16_one.py <suite> all          # run all cases (same as test_bf16.py)

Suites:
    gemm          (test_gemm)
    m_contiguous  (test_m_grouped_gemm_contiguous)
    m_masked      (test_m_grouped_gemm_masked)
    k_contiguous  (test_k_grouped_gemm_contiguous)
    cublaslt      (test_cublaslt_gemm)
"""

import random
import sys
import torch

import test_bf16

def map_index_to_test_alias(index) -> bool:
    operands = (True, False)
    return operands[index % 2]

SUITES = {
    'test_gemm':         ('enumerate_normal',                 test_bf16.test_gemm, map_index_to_test_alias),
    'test_m_grouped_gemm_contiguous': ('enumerate_m_grouped_contiguous',   test_bf16.test_m_grouped_gemm_contiguous, map_index_to_test_alias),
    'test_m_grouped_gemm_masked':     ('enumerate_m_grouped_masked',       test_bf16.test_m_grouped_gemm_masked),
    'test_k_grouped_gemm_contiguous': ('enumerate_k_grouped_contiguous_with_variants',   test_bf16.test_k_grouped_gemm_contiguous),
    'test_cublaslt_gemm':     ('enumerate_normal',                 test_bf16.test_cublaslt_gemm),
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in SUITES:
        print(__doc__.strip())
        return

    name = args[0]
    enum_name, test_fn, map_test_alias = SUITES[name]
    enum_fn = getattr(test_bf16, enum_name)
    cases = list(enum_fn(dtype=torch.bfloat16) if enum_name == 'enumerate_normal' else enum_fn(torch.bfloat16))

    # No case index given: list all cases
    if len(args) == 1:
        print(f'Suite "{name}": {len(cases)} cases')
        for i, case in enumerate(cases):
            print(f' [{i:4}] {case}')
        return

    indices = list(range(len(cases))) if args[1] == 'all' else [int(x) for x in args[1:]]
    for i in indices:
        assert 0 <= i < len(cases), f'Case index {i} out of range [0, {len(cases)})'

    torch.manual_seed(0)
    random.seed(0)

    # Map index number -> selected cases, then run the original test body
    selected = [cases[i] for i in indices]
    setattr(test_bf16, enum_name, lambda *a, **kw: iter(selected))
    for i, case in zip(indices, selected):
        print(f'[test_bf16 {name} #{i}] {case}')
    if map_test_alias is not None:
        test_fn(map_test_alias(int(args[1])))
    else:
        test_fn()
    print()


if __name__ == '__main__':
    main()
