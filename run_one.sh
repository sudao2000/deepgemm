PYTHONPATH=. python tests/test_attention_one.py gemm 0
PYTHONPATH=. python tests/test_attention_one.py gemm 1
PYTHONPATH=. python tests/test_attention_one.py mqa 0
PYTHONPATH=. python tests/test_attention_one.py maq 1
PYTHONPATH=. python tests/test_attention_one.py paged 0
PYTHONPATH=. python tests/test_attention_one.py paged 1

PYTHONPATH=. python tests/test_bf16_one.py gemm 0
PYTHONPATH=. python tests/test_bf16_one.py gemm 1
PYTHONPATH=. python tests/test_bf16_one.py m_contiguous 0
PYTHONPATH=. python tests/test_bf16_one.py m_contiguous 1
PYTHONPATH=. python tests/test_bf16_one.py m_masked 0
PYTHONPATH=. python tests/test_bf16_one.py m_masked 1
PYTHONPATH=. python tests/test_bf16_one.py k_contiguous 0
PYTHONPATH=. python tests/test_bf16_one.py k_contiguous 1
PYTHONPATH=. python tests/test_bf16_one.py cublaslt 0
PYTHONPATH=. python tests/test_bf16_one.py cublaslt 1


PYTHONPATH=. python tests/test_einsum_one.py bmk_bnk_mn 0
PYTHONPATH=. python tests/test_einsum_one.py bmk_bnk_mn 1
PYTHONPATH=. python tests/test_einsum_one.py bhr_hdr_bhd 0
PYTHONPATH=. python tests/test_einsum_one.py bhr_hdr_bhd 1
PYTHONPATH=. python tests/test_einsum_one.py bhd_hdr_bhr 0
PYTHONPATH=. python tests/test_einsum_one.py bhd_hdr_bhr 1
PYTHONPATH=. python tests/test_einsum_one.py fp8_bhr_hdr_bhd 0
PYTHONPATH=. python tests/test_einsum_one.py fp8_bhr_hdr_bhd 1

PYTHONPATH=. python tests/test_fp8_fp4_one.py gemm 0
PYTHONPATH=. python tests/test_fp8_fp4_one.py gemm 1
PYTHONPATH=. python tests/test_fp8_fp4_one.py m_contiguous 0
PYTHONPATH=. python tests/test_fp8_fp4_one.py m_contiguous 1
PYTHONPATH=. python tests/test_fp8_fp4_one.py m_masked 0
PYTHONPATH=. python tests/test_fp8_fp4_one.py m_masked 1
PYTHONPATH=. python tests/test_fp8_fp4_one.py k_contiguous 0
PYTHONPATH=. python tests/test_fp8_fp4_one.py k_contiguous 1

PYTHONPATH=. python tests/test_hyperconnection_one.py  0

PYTHONPATH=. python tests/test_layout_one.py sf 0
PYTHONPATH=. python tests/test_layout_one.py sf 1
PYTHONPATH=. python tests/test_layout_one.py k_grouped 0
PYTHONPATH=. python tests/test_layout_one.py k_grouped 1
PYTHONPATH=. python tests/test_layout_one.py psum 0
PYTHONPATH=. python tests/test_layout_one.py psum 1

PYTHONPATH=. python tests/test_legacy_one.py m_grouped 0
PYTHONPATH=. python tests/test_legacy_one.py m_grouped 1
PYTHONPATH=. python tests/test_legacy_one.py k_grouped 0
PYTHONPATH=. python tests/test_legacy_one.py k_grouped 1
