"""Same functionality as test_legacy.py, but cases can be run one by one.

Only the enumerate functions (imported by test_legacy.py from generators) are
patched: an index number is mapped to the selected case, and the original test
body runs it.

Usage:
    python tests/test_legacy_one.py                      # show help
    python tests/test_legacy_one.py <suite>              # dry run: list all cases of a suite
    python tests/test_legacy_one.py <suite> dry          # same as above
    python tests/test_legacy_one.py <suite> 0            # run case 0
    python tests/test_legacy_one.py <suite> 0 2 5        # run cases 0, 2 and 5
    python tests/test_legacy_one.py <suite> all          # run all cases (same as test_legacy.py)

Suites:
    m_grouped (test_m_grouped_gemm_contiguous_tl)
    k_grouped (test_k_grouped_gemm_contiguous_tl)

Note: in the k_grouped suite, cases skipped by the original test body
(use_psum_layout or k_alignment != 128) are still listed but do nothing.
"""

import random
import sys
import torch

import test_legacy

SUITES = {
    'test_m_grouped_gemm_contiguous_tl': ('enumerate_m_grouped_contiguous', test_legacy.test_m_grouped_gemm_contiguous_tl),
    'test_k_grouped_gemm_contiguous_tl': ('enumerate_k_grouped_contiguous', test_legacy.test_k_grouped_gemm_contiguous_tl),
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in SUITES:
        print(__doc__.strip())
        return

    name = args[0]
    enum_name, test_fn = SUITES[name]
    cases = list(getattr(test_legacy, enum_name)(torch.bfloat16))

    # No case index given: list all cases
    if len(args) == 1 or args[1] == 'dry':
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
    setattr(test_legacy, enum_name, lambda *a, **kw: iter(selected))
    for i, case in zip(indices, selected):
        print(f'[test_legacy {name} #{i}] {case}')
    test_fn()
    print()


if __name__ == '__main__':
    main()
