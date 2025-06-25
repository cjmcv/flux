/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Example of running an Ada FP8 GEMM.

    In addition to using FP8 Tensor Core instructions, the Ada FP8 GEMM uses a distinct epilogue
    that enables additional scaling of operands/outputs, storing a pre-activation-function output
    tensor (called the "auxiliary" output), and computing the absolute maximum value of the
    outputs.

    Pseudocode for this epilogue is as follows:

    Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias
    D = activation(Aux)

    if Aux is fp8 type:
        abs_max_output = max( abs(aux) | (for every aux in Aux))
        Aux = scale_aux * Aux
    endif

    if D is fp8 type:
        abs_max_output = max( abs(d) | (for every d in D))
        D = scale_d * D
    endif

    Parameter Aux is optionally stored to global memory
*/
// #include "cutlass/../../test/unit/common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

// #include "cutlass/../../test/unit/reduction/thread/testbed.h"
#include "testbed.h"

template <typename T>
void cuda_check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA_CHECK error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(val) cuda_check((val), #val, __FILE__, __LINE__)


struct GpuTimer {
  GpuTimer() {
      CUDA_CHECK(cudaEventCreate(&start_));
      CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
      CUDA_CHECK(cudaEventDestroy(start_));
      CUDA_CHECK(cudaEventDestroy(stop_));
  }
  void Start() {
      CUDA_CHECK(cudaEventRecord(start_, NULL));
  }
  void Stop() {
      CUDA_CHECK(cudaEventRecord(stop_, NULL));
  }
  float ElapsedMillis() {
      float elapsed;
      CUDA_CHECK(cudaEventSynchronize(stop_));
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
      return elapsed;
  }

  cudaEvent_t start_;
  cudaEvent_t stop_;
};

int main() {
  {
    // #if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
    // TEST(SM89_Device_Gemm_fe4m3t_fe4m3n_f16t_tensor_op_f16, 128x256x64_64x64x64)
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static int const kStages = 3;

    using Gemm = cutlass::gemm::device::Gemm<
      ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
      cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

    GpuTimer gpu_timer;
    float total_ms = 0;
    for (int i=0; i<5; i++) {

      gpu_timer.Start();
      bool res = test::gemm::device::TestAllGemm<Gemm>();
      gpu_timer.Stop();
      if (i!=0)
        total_ms += gpu_timer.ElapsedMillis();

      if (res == false) {
        printf("e");
      }
    }
    printf("acc fp16 total_ms: %f.\n", total_ms);    
  }
  {
    // TEST(SM89_Device_Gemm_fe4m3t_fe4m3n_f32t_tensor_op_f32, 128x256x64_64x64x64) {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementOutput = cutlass::half_t; //float;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static int const kStages = 3;
  
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
        ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;
  
    GpuTimer gpu_timer;
    float total_ms = 0;
    for (int i=0; i<5; i++) {

      gpu_timer.Start();
      bool res = test::gemm::device::TestAllGemm<Gemm>();
      gpu_timer.Stop();
      if (i!=0)
        total_ms += gpu_timer.ElapsedMillis();

      if (res == false) {
        printf("e");
      }
    }
    printf("acc fp32 total_ms: %f.\n", total_ms);   
    // }
  }

  
  return 0;
}

