"""Same functionality as test_einsum.py, but cases can be run one by one.

Only the enumerate functions (defined in test_einsum.py) are patched: an
index number is mapped to the selected case, and the original test body
runs it.

Usage:
    python tests/test_einsum_one.py                      # show help
    python tests/test_einsum_one.py <suite>              # dry run: list all cases of a suite
    python tests/test_einsum_one.py <suite> dry          # same as above
    python tests/test_einsum_one.py <suite> 0            # run case 0
    python tests/test_einsum_one.py <suite> 0 2 5        # run cases 0, 2 and 5
    python tests/test_einsum_one.py <suite> all          # run all cases (same as test_einsum.py)

Suites:
    bmk_bnk_mn      (test_bmk_bnk_mn)
    bhr_hdr_bhd     (test_bhr_hdr_bhd)
    bhd_hdr_bhr     (test_bhd_hdr_bhr)
    fp8_bhr_hdr_bhd (test_fp8_bhr_hdr_bhd)
"""

import random
import sys
import torch

import test_einsum

SUITES = {
    'test_bmk_bnk_mn':      ('enumerate_bmk_bnk_mn', test_einsum.test_bmk_bnk_mn),
    'test_bhr_hdr_bhd':     ('enumerate_hrd_b',      test_einsum.test_bhr_hdr_bhd),
    'test_bhd_hdr_bhr':     ('enumerate_hrd_b',      test_einsum.test_bhd_hdr_bhr),
    'test_fp8_bhr_hdr_bhd': ('enumerate_fp8_hrd_b',  test_einsum.test_fp8_bhr_hdr_bhd),
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in SUITES:
        print(__doc__.strip())
        return

    name = args[0]
    enum_name, test_fn = SUITES[name]
    cases = list(getattr(test_einsum, enum_name)())

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
    setattr(test_einsum, enum_name, lambda *a, **kw: iter(selected))
    for i, case in zip(indices, selected):
        print(f'[test_einsum {name} #{i}] {case}')
    test_fn()
    print()


if __name__ == '__main__':
    main()
