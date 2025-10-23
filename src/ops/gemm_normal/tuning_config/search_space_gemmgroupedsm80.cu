// clang-format off
#if XOP_CUDA_ARCHS==80 || XOP_CUDA_ARCHS==86 || XOP_CUDA_ARCHS==89
#include "xop/ops_impl/gemm_normal/gemm_grouped_sm80_impl.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int search_space_gemmsm80 = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int16_t)ME::GemmGrouped,(int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm80}, /*op*/[]() { return new GemmGroupedSm80Impl</*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,4>();});
  return 0;
}();
}
#endif // #if XOP_CUDA_ARCHSXOP_CUDA_ARCHS==80 || XOP_CUDA_ARCHS==86 || XOP_CUDA_ARCHS==89
// clang-format on