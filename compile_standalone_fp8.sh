#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NVCC="${NVCC:-nvcc}"

$NVCC -std=c++17 -O3 \
    -gencode arch=compute_90a,code=sm_90a \
    -I third-party/cutlass/include \
    -I third-party/cutlass/include/cute \
    standalone_sm90_fp8_gemm_1d1d.cu \
    -o standalone_sm90_fp8_gemm_1d1d \
    -lcuda -lcudart

echo "Build succeeded: ./standalone_sm90_fp8_gemm_1d1d"
