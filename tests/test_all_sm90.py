import torch
import random

import test_attention
import test_bf16
import test_einsum
import test_fp8_fp4
import test_hyperconnection
import test_layout
import test_legacy

if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    test_attention.test_gemm_skip_head_mid()
    test_attention.test_mqa_logits()
    test_attention.test_paged_mqa_logits()

    test_bf16.test_gemm()
    test_bf16.test_m_grouped_gemm_contiguous()
    test_bf16.test_m_grouped_gemm_masked()
    test_bf16.test_k_grouped_gemm_contiguous()
    test_bf16.test_cublaslt_gemm()

    test_einsum.test_bmk_bnk_mn()
    test_einsum.test_bhr_hdr_bhd()
    test_einsum.test_bhd_hdr_bhr()
    test_einsum.test_fp8_bhr_hdr_bhd()

    test_fp8_fp4.test_gemm()
    test_fp8_fp4.test_m_grouped_gemm_contiguous()
    test_fp8_fp4.test_m_grouped_gemm_masked()
    test_fp8_fp4.test_k_grouped_gemm_contiguous()

    test_hyperconnection.test_hc_prenorm_gemm()

    test_layout.test_sf_layout_kernels()
    test_layout.test_k_grouped_sf_layout_kernels()
    test_layout.test_k_grouped_psum_sf_layout_kernels()

    test_legacy.test_m_grouped_gemm_contiguous_tl()
    test_legacy.test_k_grouped_gemm_contiguous_tl()