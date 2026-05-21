#include "xop/xop.h"
#include "xop/ops_impl/global_resource.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmsm80 = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.02 ms vs 0.629 tflops
  // tins.add({1,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{103, (int16_t)ME::GemmNormal}); // 0.024 ms vs 0.524 tflops
  // tins.add({1,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{90, (int16_t)ME::GemmNormal}); // 0.025 ms vs 0.503 tflops
  return 0;
}();
}// clang-format on