// clang-format off
#include "xop/ops_impl/gemm_comm/gemm_ar_v2_impl.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int search_space_gemmcomm_ar = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int16_t)ME::GemmCommAr,(int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm80}, /*op*/[]() { return new GemmArV2Impl</*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,1,-1>();});
  return 0;
}();
}// clang-format on