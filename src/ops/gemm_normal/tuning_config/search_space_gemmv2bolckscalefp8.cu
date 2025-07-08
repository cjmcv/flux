// clang-format off
#include "xop/ops_impl/gemm_normal/gemm_v2_blockscale_fp8_impl.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int search_space_gemmv2bolckscalefp8 = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int16_t)ME::GemmBolckScaleFp8,(int16_t)ME::E4M3, (int16_t)ME::E4M3, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm89}, /*op*/[]() { return new GemmV2BlockScaleFp8Impl</*meta*/cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm89/*hparam*/>();});
  return 0;
}();
}// clang-format on