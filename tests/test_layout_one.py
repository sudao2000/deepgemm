"""Same functionality as test_layout.py, but cases can be run one by one.

Only the enumerate functions (from generators.py, used by test_layout.py) are
patched: an index number is mapped to the selected case, and the original
test body runs it.

Usage:
    python tests/test_layout_one.py                      # show help
    python tests/test_layout_one.py <suite>              # dry run: list all cases of a suite
    python tests/test_layout_one.py <suite> dry          # same as above
    python tests/test_layout_one.py <suite> 0            # run case 0
    python tests/test_layout_one.py <suite> 0 2 5        # run cases 0, 2 and 5
    python tests/test_layout_one.py <suite> all          # run all cases (same as test_layout.py)

Suites:
    sf          (test_sf_layout_kernels)
    k_grouped   (test_k_grouped_sf_layout_kernels)
    psum        (test_k_grouped_psum_sf_layout_kernels)
"""

import random
import sys
import torch

import test_layout

SUITES = {
    'sf':         ('enumerate_sf_layout',                test_layout.test_sf_layout_kernels),
    'k_grouped':  ('enumerate_k_grouped_sf_layout',      test_layout.test_k_grouped_sf_layout_kernels),
    'psum':       ('enumerate_k_grouped_psum_sf_layout', test_layout.test_k_grouped_psum_sf_layout_kernels),
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in SUITES:
        print(__doc__.strip())
        return

    name = args[0]
    enum_name, test_fn = SUITES[name]
    enum_fn = getattr(test_layout, enum_name)
    cases = list(enum_fn())

    # No case index given: list all cases
    if len(args) == 1 or args[1] == 'dry':
        print(f'Suite "{name}": {len(cases)} cases')
        for i, case in enumerate(cases):
            print(f' [{i:4}] {case}')
        return

    indices = list(range(len(cases))) if args[1] == 'all' else [int(x) for x in args[1:]]
    for i in indices:
        assert 0 <= i < len(cases), f'Case index {i} out of range [0, {len(cases)})'

    torch.manual_seed(1)
    random.seed(1)

    # Map index number -> selected cases, then run the original test body
    selected = [cases[i] for i in indices]
    setattr(test_layout, enum_name, lambda *a, **kw: iter(selected))
    for i, case in zip(indices, selected):
        print(f'[test_layout {name} #{i}] {case}')
    test_fn()
    print()


if __name__ == '__main__':
    main()
