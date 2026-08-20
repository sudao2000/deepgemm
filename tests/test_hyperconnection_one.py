"""Same functionality as test_hyperconnection.py, but cases can be run one by one.

Only the enumerate function of test_hyperconnection.py is reused/patched: an index
number is mapped to the selected case, and the original test body runs it.

Usage:
    python tests/test_hyperconnection_one.py            # show help
    python tests/test_hyperconnection_one.py dry        # list all cases
    python tests/test_hyperconnection_one.py 0            # run case 0
    python tests/test_hyperconnection_one.py 0 2 5      # run cases 0, 2 and 5
    python tests/test_hyperconnection_one.py all        # run all cases (same as test_hyperconnection.py)
"""

import random
import sys
import torch

import test_hyperconnection


def main() -> None:
    args = sys.argv[1:]
    cases = list(test_hyperconnection.enumerate_hc_prenorm_gemm())

    if not args or args[0] == 'dry':
        print(f'{len(cases)} cases')
        for i, (m, n, k, num_splits) in enumerate(cases):
            print(f' [{i:4}] m={m}, n={n}, k={k}, num_splits={num_splits}')
        return

    indices = list(range(len(cases))) if args[0] == 'all' else [int(x) for x in args]
    for i in indices:
        assert 0 <= i < len(cases), f'Case index {i} out of range [0, {len(cases)})'

    torch.manual_seed(0)
    random.seed(0)

    # Map index number -> selected cases, then run the original test body
    selected = [cases[i] for i in indices]
    test_hyperconnection.enumerate_hc_prenorm_gemm = lambda: iter(selected)
    for i, (m, n, k, num_splits) in zip(indices, selected):
        print(f'[test_hyperconnection test_hc_prenorm_gemm #{i}] m={m}, n={n}, k={k}, num_splits={num_splits}')
    test_hyperconnection.test_hc_prenorm_gemm()


if __name__ == '__main__':
    main()
