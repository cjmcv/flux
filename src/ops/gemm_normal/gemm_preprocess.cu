
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
#if XOP_CUDA_ARCHS>=90
#include "xop/ops_impl/gemm_normal/gemm_w4a16_sm90_impl.h"
#endif
//////////////////////////////

namespace xop {

void gemm_w4a16_sm90_reorder_weight(torch::Tensor weight) {
  XOP_CHECK_INPUT(weight, c10::ScalarType::Char); // 2 x int4 => 1 x int8
#if XOP_CUDA_ARCHS>=90
  int32_t n = weight.size(0);
  int32_t k = weight.size(1) * 2;
  int32_t l = 1;
  printf("gemm_w4a16_sm90_reorder_weight n: %d, k:%d\n", n,k);
  
  using GemmW4A16Sm90 = GemmW4A16Sm90Impl<
    /*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::arch::Sm90,
    /*hparam*/
    cute::Shape<cute::_128,cute::_128,cute::_128>,
    cute::Shape<cute::_1,cute::_1,cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative>;

  using StrideB = typename GemmW4A16Sm90::StrideB;
  using ElementB = typename GemmW4A16Sm90::ElementB;
  using LayoutB_Reordered = typename GemmW4A16Sm90::LayoutB_Reordered;
  using LayoutAtomQuant = typename GemmW4A16Sm90::LayoutAtomQuant;
  
  auto shape_B = cute::make_shape(n, k, l);
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  auto layout_B = make_layout(shape_B, stride_B);

  LayoutB_Reordered layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
  // Shuffle, Repeat the reorder layout atom to tile the whole tensor shape 
  cutlass::reorder_tensor((ElementB *)weight.data_ptr(), layout_B, layout_B_reordered);
#else
  printf("The function (gemm_w4a16_sm90_reorder_weight) is valid only when condition (XOP_CUDA_ARCHS>=90) is satisfied.\n");
#endif
}

}  // namespace xop