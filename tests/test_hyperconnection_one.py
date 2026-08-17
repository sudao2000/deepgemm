"""Same functionality as test_hyperconnection.py, but cases can be run one by one.

The case list is identical to the nested loops in
test_hyperconnection.test_hc_prenorm_gemm(); an index number selects which
case(s) to run, with the same per-case logic as the original test body.

Usage:
    python tests/test_hyperconnection_one.py            # show help
    python tests/test_hyperconnection_one.py dry        # list all cases
    python tests/test_hyperconnection_one.py 0          # run case 0
    python tests/test_hyperconnection_one.py 0 2 5      # run cases 0, 2 and 5
    python tests/test_hyperconnection_one.py all        # run all cases (same as test_hyperconnection.py)
"""

import random
import sys
import torch

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes


def enumerate_cases():
    for m in (13, 137, 4096, 8192):
        for n, k in [(24, 28672), (24, 7680), (24, 7168)]:
            for num_splits in [None, 16]:
                yield m, n, k, num_splits


def run_case(m: int, n: int, k: int, num_splits) -> None:
    # Needs TF32 precision for PyTorch GEMMs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((n, k), dtype=torch.float, device='cuda')
    d = torch.empty((m, n), dtype=torch.float, device='cuda') if num_splits is None else \
            torch.empty((num_splits, m, n), dtype=torch.float, device='cuda')
    s = torch.empty((m, ), dtype=torch.float, device='cuda') if num_splits is None else \
            torch.empty((num_splits, m), dtype=torch.float, device='cuda')
    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
    final_d = d if num_splits is None else d.sum(0)
    final_s = s if num_splits is None else s.sum(0)

    ref_d = a.float() @ b.T
    ref_s = a.float().square().sum(-1)

    diff = max(calc_diff(final_d, ref_d), calc_diff(final_s, ref_s))
    assert diff < 1e-8, f'{m=}, {n=}, {k=}, {diff:.10f}'

    t = bench_kineto(lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits), 'tf32_hc_prenorm_gemm', suppress_kineto_output=True)
    print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, num_splits={(num_splits or 0):2}): '
          f'{t * 1e6:4.0f} us | '
          f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
          f'{count_bytes(a, b, d, s) / 1e9 / t:4.0f} GB/s')


def main() -> None:
    args = sys.argv[1:]
    cases = list(enumerate_cases())

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

    for i in indices:
        m, n, k, num_splits = cases[i]
        print(f'test_hyperconnection [#{i}] m={m}, n={n}, k={k}, num_splits={num_splits}')
        run_case(m, n, k, num_splits)
    print()


if __name__ == '__main__':
    main()
