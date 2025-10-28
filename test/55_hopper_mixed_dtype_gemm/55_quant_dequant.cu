
#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/mixed_dtype_utils.hpp"

// #include "helper.h"
#include "xop/common_cuda.h"
#include "xop/ops_impl/common_cutlass.h"
#include "mixed_dtype_utils.hpp"

#include "symmetric_quantize.cuh"

using namespace cutlass;
using namespace cute;

using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::int4b_t;


// B matrix configuration
using         ElementA    = MmaType; 
using         ElementB    = QuantType;                                      // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
// constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

// Define the CuTe layout for reoredered quantized tensor B
// LayoutAtomQuant places values that will be read by the same thread in contiguous locations in global memory.
// It specifies the reordering within a single warp's fragment
//using ValueShuffle = Layout<_1>;                          // no value reordering
using ValueShuffle = Layout<Shape<_2,_4>, Stride<_4,_1>>; // order [0,2,4,6,1,3,5,7]
int constexpr NumShuffleAtoms = 1;
using MmaAtomShape = Layout<Shape<_1,Int<NumShuffleAtoms>>>;
using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape, ValueShuffle>());
using LayoutB_Reordered = decltype(cute::tile_to_shape(LayoutAtomQuant{}, Layout<Shape<int,int,int>, StrideB>{}));

using ElementScale = MmaType;
using ElementZero = ElementScale;
using LayoutScale = cutlass::layout::RowMajor;


StrideB stride_B;
LayoutB_Reordered layout_B_reordered;


cutlass::DeviceAllocation<ElementB> block_B;
cutlass::DeviceAllocation<ElementA> block_B_dq;
cutlass::DeviceAllocation<ElementScale> block_scale;
cutlass::DeviceAllocation<ElementZero> block_zero;


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Element>
bool initialize_scale_m(
  cutlass::DeviceAllocation<Element>& block, 
  uint64_t seed = 2023) {
  
  // If no scales, initialize with 1 so we can use the same kernel to dequantize the data
  float scope_max = 1.0f, scope_min = 1.0f;
  // if (options.mode != MixedDtypeGemmMode::ConvertOnly) {
  //   float elt_max_f = float(cutlass::platform::numeric_limits<Element>::max());
  //   scope_max = 2.f;
  //   scope_min = 0.1f;
  // }
  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}

template <class Element>
bool initialize_zero_m(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed = 2023) {
  
  // If no bias, initialize with 0 so we can use the same kernel to dequantize the data
  float scope_max = 0.0f, scope_min = 0.0f;
  // if (options.mode == MixedDtypeGemmMode::ScaleWithZeroPoint) {
  //   scope_max = 2.0f;
  //   scope_min = -2.0f;
  // }
  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}

template <typename T>
void print_device_allocation(const cutlass::DeviceAllocation<T>& d, const char* name = "") {
    std::vector<T> h(d.size());
    cudaMemcpy(h.data(), d.get(), d.bytes(), cudaMemcpyDeviceToHost);
    std::cout << name << " = ";
    for (size_t i = 0; i < h.size(); ++i) {
        std::cout << float(h[i]) << " ";
        if ((i + 1) % 32 == 0) std::cout << "\n      ";
    }
    std::cout << std::endl;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize() {
  int m = 128;
  int n = 128;
  int k = 128;
  int l = 1;
  int g = 32;

  auto shape_B = cute::make_shape(n, k, l);
  int const scale_k = cutlass::ceil_div(k, g);
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  
  auto layout_B = make_layout(shape_B, stride_B);

  auto a_coord = cutlass::make_Coord(m * l, k);
  auto b_coord = cutlass::make_Coord(k, n * l);
  auto c_coord = cutlass::make_Coord(m * l, n);


  block_B.reset(b_coord.product());
  block_B_dq.reset(b_coord.product());

  block_scale.reset(scale_k * l * n);
  block_zero.reset(scale_k * l * n);


  initialize_tensor(block_B_dq, 2021);
  initialize_scale_m(block_scale);
  initialize_zero_m(block_zero);

  using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;

  auto shape_scale_zero = cute::make_shape(n, scale_k, l);
  // stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, l));
  StrideS_ref stride_S_ref = cutlass::make_cute_packed_stride(StrideS_ref{}, cute::make_shape(n, scale_k, l));
  auto layout_scale_zero = cute::make_layout(shape_scale_zero, stride_S_ref);

  print_device_allocation(block_B_dq, "block_B_dq_before");

  cudaStream_t stream = cudaStreamDefault;
  xop::symmetric_quantize<__nv_bfloat16, __nv_bfloat16>((int8_t*)block_B.get(), nullptr, (__nv_bfloat16 *)block_scale.get(), (__nv_bfloat16 *)block_B_dq.get(),
    {1, (unsigned long)n, (unsigned long)k}, xop::QuantType::W4_A16, false);
  // symmetric_quantize<__nv_bfloat16, __nv_bfloat16>(processed_quantized_weight_ptr,
  //   unprocessed_quantized_weight_ptr, get_ptr<__nv_bfloat16>(scales), get_ptr<__nv_bfloat16 const>(weight),
  //   {num_experts, num_rows, num_cols}, ft_quant_type, force_interleave);
    
  // quantize_pack((uint8_t*)block_B.get(), block_B_dq.get(), layout_B, block_scale.get(), block_zero.get(), g, stream);
  print_device_allocation(block_B, "block_B0");
  print_device_allocation(block_scale, "block_scale0");
  print_device_allocation(block_zero, "block_zero0");

  cutlass::dequantize(block_B_dq.get(), block_B.get(), layout_B, block_scale.get(), block_zero.get(), layout_scale_zero, g, stream);
  print_device_allocation(block_B_dq, "block_B_dq after");
  print_device_allocation(block_scale, "block_scale after");
  print_device_allocation(block_zero, "block_zero after");

  if (true) {
    // Repeat the reorder layout atom to tile the whole tensor shape 
    layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
    cutlass::reorder_tensor(block_B.get(), layout_B, layout_B_reordered);

    print("Quantized tensor layout: ");
    print(layout_B_reordered);
    print("\n");
  }
}



///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);

  initialize();
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
