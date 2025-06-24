// clang-format off
#include "xop/ops_impl/gemm_normal/gemm_v2_simt_impl.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int search_space_gemmnormalsimt = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int16_t)ME::GemmNormalSimt,(int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm80}, /*op*/[]() { return new GemmPureV2SimtImpl</*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<64,64,4>,cutlass::gemm::GemmShape<32,16,4>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,1>();});
  ins.add({1,(int16_t)ME::GemmNormalSimt,(int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm80}, /*op*/[]() { return new GemmPureV2SimtImpl</*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<64,64,4>,cutlass::gemm::GemmShape<32,16,4>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,2>();});
  return 0;
}();
}// clang-format on