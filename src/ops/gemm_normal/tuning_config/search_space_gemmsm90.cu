// clang-format off
#if XOP_CUDA_ARCHS<90
#include "xop/ops_impl/gemm_normal/gemm_sm90_impl.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int search_space_gemmsm90 = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int16_t)ME::GemmNormal,(int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm90}, /*op*/[]() { return new GemmSm90Impl</*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm90,/*hparam*/cutlass::gemm::PersistentScheduler,cute::Shape<cute::_128,cute::_128,cute::_128>,cute::Shape<cute::_1,cute::_2,cute::_1>,xop::RasterOrderOptions::Heuristic,2>();});
  ins.add({1,(int16_t)ME::GemmNormal,(int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm90}, /*op*/[]() { return new GemmSm90Impl</*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, arch::Sm90,/*hparam*/cutlass::gemm::PersistentScheduler,cute::Shape<cute::_128,cute::_128,cute::_128>,cute::Shape<cute::_1,cute::_2,cute::_1>,xop::RasterOrderOptions::Heuristic,4>();});
  return 0;
}();
}
#endif // #if XOP_CUDA_ARCHS==90
// clang-format on