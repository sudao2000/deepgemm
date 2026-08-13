# 指到phycucc所在路径
export DG_JIT_NVCC_COMPILER=/user/local/cuda/bin/nvcc
# export DG_JIT_PRINT_COMPILER_COMMAND=1

PYTHONPATH=. python tests/test_all_sm90.py