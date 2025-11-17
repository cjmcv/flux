#include "xop/xop.h"
#include "xop/ops_impl/global_resource.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmsm80 = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{90, (int16_t)ME::GemmNormal}); // 0.184 ms vs 0.182 tflops
  // tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{68, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.181 tflops
  // tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.181 tflops
  tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.363 tflops
  // tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{107, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.363 tflops
  // tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.186 ms vs 0.361 tflops
  tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.726 tflops
  // tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{92, (int16_t)ME::GemmNormal}); // 0.186 ms vs 0.722 tflops
  // tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.186 ms vs 0.722 tflops
  tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{90, (int16_t)ME::GemmNormal}); // 0.185 ms vs 1.451 tflops
  // tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.185 ms vs 1.451 tflops
  // tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.186 ms vs 1.443 tflops
  tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{90, (int16_t)ME::GemmNormal}); // 0.186 ms vs 2.886 tflops
  // tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{107, (int16_t)ME::GemmNormal}); // 0.187 ms vs 2.871 tflops
  // tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{100, (int16_t)ME::GemmNormal}); // 0.187 ms vs 2.871 tflops
  tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{55, (int16_t)ME::GemmNormal}); // 0.189 ms vs 5.681 tflops
  // tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{57, (int16_t)ME::GemmNormal}); // 0.189 ms vs 5.681 tflops
  // tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{63, (int16_t)ME::GemmNormal}); // 0.19 ms vs 5.651 tflops
  tins.add({1,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{69, (int16_t)ME::GemmNormal}); // 0.037 ms vs 0.028 tflops
  // tins.add({1,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{49, (int16_t)ME::GemmNormal}); // 0.041 ms vs 0.026 tflops
  // tins.add({1,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{59, (int16_t)ME::GemmNormal}); // 0.049 ms vs 0.021 tflops
  tins.add({2,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{92, (int16_t)ME::GemmNormal}); // 0.057 ms vs 0.037 tflops
  // tins.add({2,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{66, (int16_t)ME::GemmNormal}); // 0.061 ms vs 0.034 tflops
  // tins.add({2,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{90, (int16_t)ME::GemmNormal}); // 0.062 ms vs 0.034 tflops
  tins.add({4,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{55, (int16_t)ME::GemmNormal}); // 0.063 ms vs 0.067 tflops
  // tins.add({4,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{68, (int16_t)ME::GemmNormal}); // 0.064 ms vs 0.066 tflops
  // tins.add({4,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{90, (int16_t)ME::GemmNormal}); // 0.064 ms vs 0.066 tflops
  tins.add({8,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{87, (int16_t)ME::GemmNormal}); // 0.024 ms vs 0.35 tflops
  // tins.add({8,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{92, (int16_t)ME::GemmNormal}); // 0.028 ms vs 0.3 tflops
  // tins.add({8,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{54, (int16_t)ME::GemmNormal}); // 0.041 ms vs 0.205 tflops
  tins.add({16,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{96, (int16_t)ME::GemmNormal}); // 0.031 ms vs 0.541 tflops
  // tins.add({16,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{7, (int16_t)ME::GemmNormal}); // 0.056 ms vs 0.3 tflops
  // tins.add({16,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{50, (int16_t)ME::GemmNormal}); // 0.063 ms vs 0.266 tflops
  tins.add({32,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{64, (int16_t)ME::GemmNormal}); // 0.024 ms vs 1.398 tflops
  // tins.add({32,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{103, (int16_t)ME::GemmNormal}); // 0.036 ms vs 0.932 tflops
  // tins.add({32,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{54, (int16_t)ME::GemmNormal}); // 0.043 ms vs 0.78 tflops
  return 0;
}();
}// clang-format on