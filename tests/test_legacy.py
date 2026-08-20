import os
import torch
import random

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes
)
from generators import (
    enumerate_m_grouped_contiguous, enumerate_k_grouped_contiguous,
    generate_m_grouped_contiguous, generate_k_grouped_contiguous,
    print_kernel_io,
)
from deep_gemm.testing.bench import _CudaClosureContext

def test_m_grouped_gemm_contiguous_tl() -> None:    
    print('Testing m-grouped contiguous Triton GEMM:')
    for _, _, num_groups, expected_m_per_group, n, k, major_a, major_b, _, _ in enumerate_m_grouped_contiguous(torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'

        if os.getenv('CORRECTNESS'):
            for expand in (False, True):
                for test_alias in (False, True):
                    m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_bf16=True)
                    func_name = f"{'a_fused_' if expand else ''}m_grouped_bf16_gemm_{major_opt.lower() if test_alias else 'nt'}_contiguous_tl"
                    if test_alias:
                        assert major_a.is_k_major()
                        b = b if major_b.is_k_major() else b.mT
                        assert a[0].is_contiguous() and b[0].is_contiguous()
                    if expand:
                        m_row_indices = torch.arange(0, m, dtype=torch.int32, device='cpu')
                        print_kernel_io(func_name, dict(a=a, b=b, grouped_indices=(m_indices, m_row_indices)), dict(d=d))
                        kernel = lambda: getattr(deep_gemm.legacy, func_name)(a, b, d, (m_indices, m_row_indices))
                        with _CudaClosureContext(kernel, tensor_vars=('a', 'b', 'd', 'm_indices', 'm_row_indices')):
                            kernel()
                        print_kernel_io(func_name, {}, dict(d=d))
                    else:
                        print_kernel_io(func_name, dict(a=a, b=b, grouped_indices=m_indices), dict(d=d))
                        kernel = lambda: getattr(deep_gemm.legacy, func_name)(a, b, d, m_indices)
                        with _CudaClosureContext(kernel, tensor_vars=('a', 'b', 'd', 'm_indices')):
                            kernel()
                        print_kernel_io(func_name, {}, dict(d=d))
                    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
                    diff = calc_diff(d, ref_d)
                    assert diff < 0.001, f'{m=}, {n=}, {k=}, {major_opt}, {diff:.5f}, alias={test_alias}'
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_bf16=True)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.legacy.m_grouped_bf16_gemm_nt_contiguous_tl(a, b, d, m_indices)

        t = bench_kineto(test_func, 'm_grouped_bf16_gemm_nt_contiguous_tl',
                         tensor_vars=('a', 'b', 'd', 'm_indices'),
                         input_vars=('a', 'b', 'm_indices'), output_vars=('d',), suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous_tl() -> None:    
    print('Testing k-grouped contiguous Triton GEMM:')
    for num_groups, m, n, major_a, major_b, _, aligned_ks_cpu, expected_k_per_group, _, k_alignment, use_psum_layout in enumerate_k_grouped_contiguous(torch.bfloat16):
        # Legacy Triton kernels do not mask per-group K tails inside the BLOCK_SIZE_K loop.
        if use_psum_layout or k_alignment != 128:
            continue

        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'

        if os.getenv('CORRECTNESS'):
            for fused_operand in ('a', 'b'):
                k, a, b, c, d, ref_d, _, _ = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, aligned_ks_cpu, use_ue8m0=False, use_bf16=True)
                func_name = f"{fused_operand}_fused_k_grouped_bf16_gemm_{major_opt.lower()}_contiguous_tl"
                k_indices = torch.arange(0, k, dtype=torch.int32, device='cpu')
                k_start = torch.empty(len(aligned_ks_cpu), dtype=torch.int32, device='cpu')
                k_end = torch.empty(len(aligned_ks_cpu), dtype=torch.int32, device='cpu')
                for i, group_k in enumerate(aligned_ks_cpu):
                    k_start[i] = k_end[i-1] if i > 0 else 0
                    k_end[i] = k_start[i] + group_k
                print_kernel_io(func_name, dict(a=a, b=b, c=c, grouped_indices=(k_indices, k_start, k_end), accumulate=True), dict(c=c))
                kernel = lambda: getattr(deep_gemm.legacy, func_name)(a, b, c, (k_indices, k_start, k_end), True)
                with _CudaClosureContext(kernel, tensor_vars=('a', 'b', 'c', 'k_indices', 'k_start', 'k_end')):
                    kernel()
                print_kernel_io(func_name, {}, dict(c=c))
                diff = calc_diff(c, ref_d)
                assert diff < 0.001, f'{m=}, {n=}, {k=}, {major_opt}, {diff:.5f}'
        k, a, b, c, d, ref_d, _, _ = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, aligned_ks_cpu, use_ue8m0=False, use_bf16=True)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.legacy.b_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a, b, c, (k_indices, k_start, k_end), True)

        t = bench_kineto(test_func, 'b_fused_k_grouped_bf16_gemm_tn_contiguous_tl',
                         tensor_vars=('a', 'b', 'c', 'k_indices', 'k_start', 'k_end'),
                         input_vars=('a', 'b', 'k_indices', 'k_start', 'k_end'),
                         output_vars=('c',), suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_m_grouped_gemm_contiguous_tl()
    test_k_grouped_gemm_contiguous_tl()
