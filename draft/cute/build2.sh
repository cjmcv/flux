#!/bin/bash
# naive_gemm/gemm_multistage.cu
# streamk_gemm/gemm_streamk.cu  
# gemm_simple.cu
rm a.out
nvcc -gencode arch=compute_89,code=sm_89 \
     -I../../3rdparty/cutlass/include \
     -I../../3rdparty/cutlass/examples/common \
     -I../../3rdparty/cutlass/tools/util/include \
     -I../../include    \
     -Iinclude -lcublas \
     gemm/gemm_simple.cu

echo "Done!"