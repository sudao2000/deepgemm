"""Same functionality as test_attention.py, but cases can be run one by one.

Only the enumerate functions of test_attention.py are reused/patched: an index
number is mapped to the selected case, and the original test body runs it.

Usage:
    python tests/test_attention_one.py                      # show help
    python tests/test_attention_one.py <suite>              # list all cases of a suite
    python tests/test_attention_one.py <suite> 0            # run case 0
    python tests/test_attention_one.py <suite> 0 2 5        # run cases 0, 2 and 5
    python tests/test_attention_one.py <suite> all          # run all cases (same as test_attention.py)

Suites: gemm (test_gemm_skip_head_mid), mqa (test_mqa_logits), paged (test_paged_mqa_logits)
"""

import random
import sys
import torch

import test_attention

SUITES = {
    'gemm':  ('enumerate_gemm_skip_head_mid', test_attention.test_gemm_skip_head_mid),
    'mqa':   ('enumerate_mqa_logits',         test_attention.test_mqa_logits),
    'paged': ('enumerate_paged_mqa_logits',   test_attention.test_paged_mqa_logits),
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in SUITES:
        print(__doc__.strip())
        return

    name = args[0]
    enum_name, test_fn = SUITES[name]
    cases = list(getattr(test_attention, enum_name)())

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
    setattr(test_attention, enum_name, lambda: iter(selected))
    for i, case in zip(indices, selected):
        print(f'[{name} #{i}] {case}')
    test_fn()
    print()


if __name__ == '__main__':
    main()
