// clang-format off
#if XOP_CUDA_ARCHS==80 || XOP_CUDA_ARCHS==86 || XOP_CUDA_ARCHS==89 || XOP_CUDA_ARCHS==90 
#include "xop/ops_impl/gemm_normal/gemv_simt_impl.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int search_space_gemvsimt = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int16_t)ME::GemvSimt,(int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::BF16, (int16_t)ME::FP32, (int16_t)ME::RCR, (int16_t)ME::Sm80}, /*op*/[]() { return new GemvSimtImpl</*meta*/cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, layout::RowMajor, arch::Sm80,/*hparam*/2,8,128,128>();});
  return 0;
}();
}
#endif // #if XOP_CUDA_ARCHSXOP_CUDA_ARCHS==80 || XOP_CUDA_ARCHS==86 || XOP_CUDA_ARCHS==89 || XOP_CUDA_ARCHS==90 
// clang-format on