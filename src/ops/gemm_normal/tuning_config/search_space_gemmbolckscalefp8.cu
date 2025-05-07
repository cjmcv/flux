// clang-format off
#include "ctlop/ops_impl/gemm_normal/gemm_v3_blockscale_fp8_impl.h"

namespace ctlop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int search_space_gemmbolckscalefp8 = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int8_t)ME::GemmBolckScaleFp8,(int8_t)ME::E4M3, (int8_t)ME::E4M3, (int8_t)ME::BF16, (int8_t)ME::RCR, (int8_t)ME::Sm90}, /*op*/[]() { return new GemmBlockScaleFp8Impl</*meta*/cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm90,/*hparam*/cute::Shape<cute::_1,cute::_2,cute::_1>,gemm::kernel::detail::RasterOrderOptions::Heuristic,2>();});
  ins.add({1,(int8_t)ME::GemmBolckScaleFp8,(int8_t)ME::E4M3, (int8_t)ME::E4M3, (int8_t)ME::BF16, (int8_t)ME::RCR, (int8_t)ME::Sm90}, /*op*/[]() { return new GemmBlockScaleFp8Impl</*meta*/cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm90,/*hparam*/cute::Shape<cute::_1,cute::_2,cute::_1>,gemm::kernel::detail::RasterOrderOptions::AlongM,2>();});
  ins.add({2,(int8_t)ME::GemmBolckScaleFp8,(int8_t)ME::E4M3, (int8_t)ME::E4M3, (int8_t)ME::BF16, (int8_t)ME::RCR, (int8_t)ME::Sm90}, /*op*/[]() { return new GemmBlockScaleFp8Impl</*meta*/cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm90,/*hparam*/cute::Shape<cute::_1,cute::_2,cute::_1>,gemm::kernel::detail::RasterOrderOptions::AlongN,2>();});
  ins.add({3,(int8_t)ME::GemmBolckScaleFp8,(int8_t)ME::E4M3, (int8_t)ME::E4M3, (int8_t)ME::BF16, (int8_t)ME::RCR, (int8_t)ME::Sm90}, /*op*/[]() { return new GemmBlockScaleFp8Impl</*meta*/cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm90,/*hparam*/cute::Shape<cute::_2,cute::_1,cute::_1>,gemm::kernel::detail::RasterOrderOptions::Heuristic,2>();});
  ins.add({4,(int8_t)ME::GemmBolckScaleFp8,(int8_t)ME::E4M3, (int8_t)ME::E4M3, (int8_t)ME::BF16, (int8_t)ME::RCR, (int8_t)ME::Sm90}, /*op*/[]() { return new GemmBlockScaleFp8Impl</*meta*/cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm90,/*hparam*/cute::Shape<cute::_2,cute::_1,cute::_1>,gemm::kernel::detail::RasterOrderOptions::AlongM,2>();});
  ins.add({5,(int8_t)ME::GemmBolckScaleFp8,(int8_t)ME::E4M3, (int8_t)ME::E4M3, (int8_t)ME::BF16, (int8_t)ME::RCR, (int8_t)ME::Sm90}, /*op*/[]() { return new GemmBlockScaleFp8Impl</*meta*/cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm90,/*hparam*/cute::Shape<cute::_2,cute::_1,cute::_1>,gemm::kernel::detail::RasterOrderOptions::AlongN,2>();});
  return 0;
}();
}// clang-format on