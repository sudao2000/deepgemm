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
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm_llm_layer_shapes 0
# PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm_llm_layer_shapes 1
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

//////
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 7
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 9
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 10
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 11
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 12
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 13

PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 14
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 16
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 17
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 18
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 19
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 20

PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 21
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 23
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 24
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 25
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 26
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_gemm 27

PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_contiguous 8
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_contiguous 10
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_contiguous 20
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_contiguous 22

PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_masked 70
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_masked 71
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_masked 76
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_masked 77
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_masked 82
PYTHONPATH=. python tests/test_fp8_fp4_one.py test_m_grouped_gemm_masked 83
