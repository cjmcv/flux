
#pragma once

#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

namespace kernel {

template <typename T, int THREAD_NUM>
__device__ __forceinline__ void copy_kernel(const int bx, const int by, const int bz,
                                            const int layer_id, const int onelayer_size, 
                                            const int step, const int onestep_size,
                                            const void* __restrict__ input_ptr1, 
                                            const void* __restrict__ input_ptr2, 
                                            void* __restrict__ output_ptr1, 
                                            void* __restrict__ output_ptr2) {
  // (bx=2, by=1, bz=1)
  // if (threadIdx.x == 0) {
  //   printf("layer_id: (%d, %d), (%d, %d)\n", layer_id, onelayer_size, step, onestep_size);
  // }
  const bfloat16_t* __restrict__ input1 = static_cast<const bfloat16_t*>(input_ptr1);
  const bfloat16_t* __restrict__ input2 = static_cast<const bfloat16_t*>(input_ptr2);
  bfloat16_t* __restrict__ output1 = static_cast<bfloat16_t*>(output_ptr1) + layer_id * onelayer_size + step * onestep_size;
  bfloat16_t* __restrict__ output2 = static_cast<bfloat16_t*>(output_ptr2) + layer_id * onelayer_size + step * onestep_size;
  // if (threadIdx.x == 0) {
  //   printf("copy_kernel: (%d,%d,%d): %d, %d, %d, %d, %d\n", bx, by, bz, layer_id, onelayer_size, step, onestep_size, THREAD_NUM);   
  //   printf("copy_kernel2: (%lld, %lld), (%lld, %lld), %f, %f\n", output1, output2, input1, input2, static_cast<float>(input1[0]), static_cast<float>(input1[1]));    
  // }
  if (bx == 0) {
    // for (int i = threadIdx.x; i < onestep_size; i += THREAD_NUM) {
    //   output1[i] = input1[i];
    //   // printf("(%d, %f, %f), ", bx, static_cast<float>(output1[i]), static_cast<float>(input1[i]));
    // }
  }
  else if (bx == 1) {
    for (int i = threadIdx.x; i < onestep_size; i += THREAD_NUM) {
      output2[i] = input2[i];
      // printf("(%d, %f, %f), ", bx, static_cast<float>(output2[i]), static_cast<float>(input2[i]));
    }
  }
}

} // namespace kernel
