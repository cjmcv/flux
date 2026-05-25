
#include "gemm_normal.h"
#include "xop/ops_impl/global_resource.h"
#include "xop/common_cuda.h"
#include "xop/common_torch.h"
#include "xop/common_strategy.h"

#include "xop/lt_coll/gemm_lt.h"

#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <utility>

/////////////////////////////
#include "xop/xop.h"
#include "xop/dsl/linear_gemm_tl_1_6144_1024_top0.cuh"
//////////////////////////////

namespace xop {


// extern "C" int gemm_dsl_init(void* __restrict__ A, void* __restrict__ B, void* __restrict__ C) {
//   return create_linear_gemm_tl_1_6144_1024((cutlass::bfloat16_t *)A, (cutlass::bfloat16_t *)B, (cutlass::bfloat16_t *)C);
// }

// template <typename T,
//   int THREAD_NUM,
//   int TILE_DIM_X, 
//   int TILE_DIM_Y, 
//   int TILE_DIM_Z,
//   int M,
//   int N,
//   int K>
//   __global__ __forceinline__ void gemm_dsl(const void* __restrict__ input_ptr, const void* __restrict__ weight_ptr, const void* __restrict__ residual_ptr, void* __restrict__ output_ptr) {
//   kernel::linear_gemm_tl_1_6144_1024<T,THREAD_NUM,TILE_DIM_X,TILE_DIM_Y,TILE_DIM_Z,M,N,K>(blockIdx.x, blockIdx.y, blockIdx.z, input_ptr, weight_ptr, nullptr, output_ptr);
// }

// void gemm_dsl(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> output_buf, cudaStream_t stream) {
//   gemm_dsl_init(input.data_ptr(), weight.data_ptr(), output_buf.value().data_ptr());

//   gemm_dsl<bfloat16_t,128,64,16,32,1,6144,1024><<<LAUNCH_INFO_linear_gemm_tl_1_6144_1024>>>(input.data_ptr(), weight.data_ptr(), nullptr, output_buf.value().data_ptr());
// }

}  // namespace xop