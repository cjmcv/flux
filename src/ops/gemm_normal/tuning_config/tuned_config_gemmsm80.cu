#include "xop/xop.h"
#include "xop/ops_impl/global_resource.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmsm80 = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.181 tflops
  // tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.181 tflops
  // tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{107, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.181 tflops
  tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{90, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.363 tflops
  // tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{96, (int16_t)ME::GemmNormal}); // 0.186 ms vs 0.361 tflops
  // tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{59, (int16_t)ME::GemmNormal}); // 0.187 ms vs 0.359 tflops
  tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.187 ms vs 0.718 tflops
  // tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{90, (int16_t)ME::GemmNormal}); // 0.187 ms vs 0.718 tflops
  // tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{77, (int16_t)ME::GemmNormal}); // 0.187 ms vs 0.718 tflops
  tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{92, (int16_t)ME::GemmNormal}); // 0.186 ms vs 1.443 tflops
  // tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.186 ms vs 1.443 tflops
  // tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{103, (int16_t)ME::GemmNormal}); // 0.187 ms vs 1.435 tflops
  tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.187 ms vs 2.871 tflops
  // tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{55, (int16_t)ME::GemmNormal}); // 0.188 ms vs 2.856 tflops
  // tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{100, (int16_t)ME::GemmNormal}); // 0.188 ms vs 2.856 tflops
  return 0;
}();
}// clang-format on