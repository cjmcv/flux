/* Copyright 2025 Mirage Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "tasks/common/common_header.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/arch/memory.h"

namespace kernel {

template <typename T,
          int THREAD_NUM,
          int TILE_DIM_X, 
          int TILE_DIM_Y, 
          int TILE_DIM_Z,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int I_STRIDE,
          int O_STRIDE>
__device__ __forceinline__ void silu_mul_kernel(const int bx, const int by, const int bz,
                                                   void const *input_ptr,
                                                   void *output_ptr,
                                                   int num_active_tokens) {
  // if (threadIdx.x == 0) {
  //   printf("[%d-(%d,%d,%d)]-tile(%d,%d,%d)(%d) silu_mul input_ptr: %lld, output_ptr: %lld: %d.\n", blockIdx.x, bx, by, bz, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, THREAD_NUM, input_ptr, output_ptr, OUTPUT_SIZE);
  // }  
  // printf("silu_mul_kernel: (%d, %d, %d)-(%d, %d, %d).\n", bx, by, bz, blockIdx.x, blockIdx.y, blockIdx.z);
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr) + bx * TILE_DIM_X; // by * 2*O_STRIDE + // by == 0
  T const *__restrict__ d_mul = d_input + O_STRIDE;
  T *__restrict__ d_output = static_cast<T *>(output_ptr) + bx * TILE_DIM_X; // by * O_STRIDE + 

#pragma unroll
  for (int i=0; i<num_active_tokens; i++) {
    int step1 = i*OUTPUT_SIZE;
    int step2 = i*OUTPUT_SIZE*2;
    float input_val = float(d_input[step2 + threadIdx.x]);
    T mul_val = d_mul[step2 + threadIdx.x];
    d_output[step1 + threadIdx.x] = T(input_val / (1.0f + expf(-input_val))) * mul_val;    
  }
}

} // namespace kernel
