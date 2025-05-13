// clang-format off
#include "ctlop/ops_impl/gemm_normal/gemm_v2_impl.h"

namespace ctlop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int search_space_gemmnormal = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int16_t)ME::GemmNormal,(int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm80}, /*op*/[]() { return new GemmPureV2Impl</*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,256,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,1,-1>();});
  ins.add({1,(int16_t)ME::GemmNormal,(int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm80}, /*op*/[]() { return new GemmPureV2Impl</*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,256,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,2,-1>();});
  return 0;
}();
}// clang-format on