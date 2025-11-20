/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 #include <cublasLt.h>

 #include "xop/lt_coll/common_cublaslt.h"
 #include "sample_cublasLt_LtBlk128x128Fp8Matmul.h"
 
 /// Sample wrapper executing mxfp8 matmul with cublasLtMatmul, with addition of per-tensor scaling, and
 /// the workspace to support split-K algorithms.
 ///
 /// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
 /// attribute matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to
 /// change this configure appropriate attribute in the preference handle
 void LtBlk128x128Fp8Matmul(cublasLtHandle_t ltHandle,
                  cublasOperation_t transa,
                  cublasOperation_t transb,
                  int m,
                  int n,
                  int k,
                  const float *alpha, /* host pointer */
                  const float *a_scale, /* device pointer */
                  const __nv_fp8_e4m3 *A,
                  int lda,
                  const float *b_scale, /* device pointer */
                  const __nv_fp8_e4m3 *B,
                  int ldb,
                  const float *beta, /* host pointer */
                  __nv_bfloat16 *C,
                  int ldc,
                  __nv_bfloat16 *D,
                  int ldd,
                  void *workspace,
                  size_t workspaceSize,
                  cublasLtMatmulMatrixScale_t AScaleMode,
                  cublasLtMatmulMatrixScale_t BScaleMode) {
     cublasLtMatmulDesc_t operationDesc = NULL;
     cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
     cublasLtMatmulPreference_t preference = NULL;
 
     int returnedResults                             = 0;
     cublasLtMatmulHeuristicResult_t heuristicResult = {};
 
     // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
     // set the transforms for A and B
     CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
     CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
     CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
 
 
     // set block scaling mode
     CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &AScaleMode, sizeof(AScaleMode)));
     CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &BScaleMode, sizeof(BScaleMode)));
 
     // set scaling factors
     CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
     CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
 
     // create matrix descriptors, we are good with the details here so no need to set any extra attributes
     // table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
     CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
     CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
     CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));
     CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, m, n, ldd));
 
     // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
     // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
     // directly come from cudaMalloc)
     CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
     CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
 
     // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
     // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
     CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));
 
     if (returnedResults == 0) {
         CUBLASLT_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
     }
 
     CUBLASLT_CHECK(cublasLtMatmul(ltHandle,
                                      operationDesc,
                                      alpha,
                                      A,
                                      Adesc,
                                      B,
                                      Bdesc,
                                      &beta,
                                      C,
                                      Cdesc,
                                      D,
                                      Ddesc,
                                      &heuristicResult.algo,
                                      workspace,
                                      workspaceSize,
                                      0));
 
     // descriptors are no longer needed as all GPU work was already enqueued
     if (preference) CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
     if (Ddesc) CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Ddesc));
     if (Cdesc) CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
     if (Bdesc) CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
     if (Adesc) CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
     if (operationDesc) CUBLASLT_CHECK(cublasLtMatmulDescDestroy(operationDesc));
 }

  cublasLtMatmulMatrixScale_t AScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
  cublasLtMatmulMatrixScale_t BScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
  cublasLtMatmulMatrixScale_t CScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  cublasLtMatmulMatrixScale_t DScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  cublasLtMatmulMatrixScale_t DOutScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;


 TestBench<__nv_fp8_e4m3, __nv_bfloat16, float, float, float, __nv_bfloat16> props(
  CUBLAS_OP_T, CUBLAS_OP_N,
  64, 128, 256, 2.0f, 1.0f, 32ULL * 1024 * 1024, 1,
  CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F, CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);

  TestBench(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    ComputeType alpha, ComputeType beta,
    size_t workspaceSize, int N,
    cublasLtMatmulMatrixScale_t AScaleMode,
    cublasLtMatmulMatrixScale_t BScaleMode,
    cublasLtMatmulMatrixScale_t CScaleMode,
    cublasLtMatmulMatrixScale_t DScaleMode,
    cublasLtMatmulMatrixScale_t DOutScaleMode):

props.run([&props] {
  LtBlk128x128Fp8Matmul(props.ltHandle,
              props.transa,
              props.transb,
              props.m,
              props.n,
              props.k,
              &props.alpha,
              props.AscaleDev,
              props.Adev,
              props.lda,
              props.BscaleDev,
              props.Bdev,
              props.ldb,
              &props.beta,
              props.Cdev,
              props.ldc,
              props.Ddev,
              props.ldd,
              props.workspace,
              props.workspaceSize,
              props.AScaleMode,
              props.BScaleMode);