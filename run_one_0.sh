PYTHONPATH=. python tests/test_attention_one.py test_gemm_skip_head_mid 0
# PYTHONPATH=. python tests/test_attention_one.py test_gemm_skip_head_mid 1
PYTHONPATH=. python tests/test_attention_one.py test_mqa_logits 0
# PYTHONPATH=. python tests/test_attention_one.py test_mqa_logits 1
PYTHONPATH=. python tests/test_attention_one.py test_paged_mqa_logits 0
# PYTHONPATH=. python tests/test_attention_one.py test_paged_mqa_logits 1

PYTHONPATH=. python tests/test_bf16_one.py test_gemm 0
# PYTHONPATH=. python tests/test_bf16_one.py test_gemm 1
PYTHONPATH=. python tests/test_bf16_one.py test_m_grouped_gemm_contiguous 0
# PYTHONPATH=. python tests/test_bf16_one.py test_m_grouped_gemm_contiguous 1
PYTHONPATH=. python tests/test_bf16_one.py test_m_grouped_gemm_masked 0
# PYTHONPATH=. python tests/test_bf16_one.py test_m_grouped_gemm_masked 1
PYTHONPATH=. python tests/test_bf16_one.py test_k_grouped_gemm_contiguous 0
# PYTHONPATH=. python tests/test_bf16_one.py test_k_grouped_gemm_contiguous 1
PYTHONPATH=. python tests/test_bf16_one.py test_cublaslt_gemm 0
# PYTHONPATH=. python tests/test_bf16_one.py test_cublaslt_gemm 1


PYTHONPATH=. python tests/test_einsum_one.py test_bmk_bnk_mn 0
# PYTHONPATH=. python tests/test_einsum_one.py test_bmk_bnk_mn 1
PYTHONPATH=. python tests/test_einsum_one.py test_bhr_hdr_bhd 0
# PYTHONPATH=. python tests/test_einsum_one.py test_bhr_hdr_bhd 1
PYTHONPATH=. python tests/test_einsum_one.py test_bhd_hdr_bhr 0
# PYTHONPATH=. python tests/test_einsum_one.py test_bhd_hdr_bhr 1
PYTHONPATH=. python tests/test_einsum_one.py test_fp8_bhr_hdr_bhd 0
# PYTHONPATH=. python tests/test_einsum_one.py test_fp8_bhr_hdr_bhd 1

PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 0
# PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 1
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_contiguous 0
# PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_contiguous 1
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_masked 0
# PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_masked 1
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_k_grouped_gemm_contiguous 0
# PYTHONPATH=. python tests/test_fp8_fp4_one.py test_k_grouped_gemm_contiguous 1

PYTHONPATH=. python tests/test_hyperconnection_one.py  0

PYTHONPATH=. python tests/test_layout_one.py test_sf_layout_kernels 0
# PYTHONPATH=. python tests/test_layout_one.py test_sf_layout_kernels 1
PYTHONPATH=. python tests/test_layout_one.py test_k_grouped_sf_layout_kernels 0
# PYTHONPATH=. python tests/test_layout_one.py test_k_grouped_sf_layout_kernels 1
PYTHONPATH=. python tests/test_layout_one.py test_k_grouped_psum_sf_layout_kernels 0
# PYTHONPATH=. python tests/test_layout_one.py test_k_grouped_psum_sf_layout_kernels 1

PYTHONPATH=. python tests/test_legacy_one.py test_m_grouped_gemm_contiguous_tl 0
# PYTHONPATH=. python tests/test_legacy_one.py test_m_grouped_gemm_contiguous_tl 1
PYTHONPATH=. python tests/test_legacy_one.py test_k_grouped_gemm_contiguous_tl 0
# PYTHONPATH=. python tests/test_legacy_one.py test_k_grouped_gemm_contiguous_tl 1
