// clang-format off
#if XOP_CUDA_ARCHS>=89
#include "xop/ops_impl/gemm_normal/gemm_w4a16_sm90_impl.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int search_space_gemmw4a16sm90 = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int16_t)ME::GemmW4A16,(int16_t)ME::BF16, (int16_t)ME::S8, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm90}, /*op*/[]() { return new GemmW4A16Sm90Impl</*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, arch::Sm90,/*hparam*/cute::Shape<cute::_128,cute::_128>,cute::Shape<cute::_1,cute::_1,cute::_1>,cutlass::gemm::KernelTmaWarpSpecializedCooperative,cutlass::epilogue::TmaWarpSpecializedCooperative>();});
  return 0;
}();
}
#endif // #if XOP_CUDA_ARCHS==90
// clang-format on