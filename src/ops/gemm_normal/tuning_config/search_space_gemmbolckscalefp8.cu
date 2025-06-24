// clang-format off
#include "xop/ops_impl/gemm_normal/gemm_v3_blockscale_fp8_impl.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int search_space_gemmbolckscalefp8 = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int16_t)ME::GemmBolckScaleFp8,(int16_t)ME::E4M3, (int16_t)ME::E4M3, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm90}, /*op*/[]() { return new GemmBlockScaleFp8Impl</*meta*/cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm90,/*hparam*/cutlass::gemm::PersistentScheduler,cute::Shape<cute::_128,cute::_128,cute::_128>,cute::Shape<cute::_1,cute::_2,cute::_1>,gemm::kernel::detail::RasterOrderOptions::Heuristic,2>();});
  ins.add({1,(int16_t)ME::GemmBolckScaleFp8,(int16_t)ME::E4M3, (int16_t)ME::E4M3, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm90}, /*op*/[]() { return new GemmBlockScaleFp8Impl</*meta*/cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm90,/*hparam*/cutlass::gemm::PersistentScheduler,cute::Shape<cute::_128,cute::_128,cute::_128>,cute::Shape<cute::_1,cute::_2,cute::_1>,gemm::kernel::detail::RasterOrderOptions::Heuristic,4>();});
  return 0;
}();
}// clang-format on