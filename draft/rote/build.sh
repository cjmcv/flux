#!/bin/bash
# naive_gemm/gemm_multistage.cu
# streamk_gemm/gemm_streamk.cu  
# gemm_simple.cu
rm a.out
nvcc -gencode arch=compute_89,code=sm_89 elementwise_add.cu

echo "Done!"