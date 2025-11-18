#include "xop/xop.h"
#include "xop/ops_impl/global_resource.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmsm80 = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmLt, 13,0,0,0,0,0,1,0,0,0,0,0,75,0,0,0,1,0,-10268,1,14,0,14,0,14,14,68,0,0,0,0,0}); // 0.019 ms vs 0.055 tflops
  // tins.add({1,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{94, (int16_t)ME::GemmNormal}); // 0.02 ms vs 0.052 tflops
  // tins.add({1,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{3, (int16_t)ME::GemmLt, 13,0,0,0,0,0,1,0,0,0,0,0,75,0,0,0,1,0,-10268,1,14,0,14,0,14,14,68,0,0,0,0,0}); // 0.02 ms vs 0.052 tflops
  tins.add({2,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.02 ms vs 0.105 tflops
  // tins.add({2,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{93, (int16_t)ME::GemmNormal}); // 0.02 ms vs 0.105 tflops
  // tins.add({2,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{80, (int16_t)ME::GemmNormal}); // 0.021 ms vs 0.1 tflops
  tins.add({4,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmLt, 21,0,5,0,20,0,1,0,0,0,0,0,0,0,0,0,1,0,-10268,1,14,0,14,0,14,14,68,0,0,0,0,0}); // 0.019 ms vs 0.221 tflops
  // tins.add({4,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.02 ms vs 0.21 tflops
  // tins.add({4,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{97, (int16_t)ME::GemmNormal}); // 0.02 ms vs 0.21 tflops
  tins.add({8,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmLt, 21,0,5,0,20,0,1,0,0,0,0,0,0,0,0,0,1,0,-10268,1,14,0,14,0,14,14,68,0,0,0,0,0}); // 0.02 ms vs 0.419 tflops
  // tins.add({8,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.02 ms vs 0.419 tflops
  // tins.add({8,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmLt, 21,0,5,0,20,0,1,0,0,0,0,0,0,0,0,0,1,0,-10268,1,14,0,14,0,14,14,68,0,0,0,0,0}); // 0.02 ms vs 0.419 tflops
  tins.add({16,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmLt, 21,0,5,0,20,0,1,0,0,0,0,0,0,0,0,0,1,0,-10268,1,14,0,14,0,14,14,68,0,0,0,0,0}); // 0.019 ms vs 0.883 tflops
  // tins.add({16,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmLt, 21,0,5,0,20,0,1,0,0,0,0,0,0,0,0,0,1,0,-10268,1,14,0,14,0,14,14,68,0,0,0,0,0}); // 0.019 ms vs 0.883 tflops
  // tins.add({16,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.02 ms vs 0.839 tflops
  tins.add({32,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{105, (int16_t)ME::GemmNormal}); // 0.02 ms vs 1.678 tflops
  // tins.add({32,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{62, (int16_t)ME::GemmNormal}); // 0.021 ms vs 1.598 tflops
  // tins.add({32,128,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{58, (int16_t)ME::GemmNormal}); // 0.021 ms vs 1.598 tflops
  tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.181 tflops
  // tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{107, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.181 tflops
  // tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{66, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.181 tflops
  tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.363 tflops
  // tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.186 ms vs 0.361 tflops
  // tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{92, (int16_t)ME::GemmNormal}); // 0.187 ms vs 0.359 tflops
  tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{90, (int16_t)ME::GemmNormal}); // 0.184 ms vs 0.729 tflops
  // tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.185 ms vs 0.726 tflops
  // tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{107, (int16_t)ME::GemmNormal}); // 0.186 ms vs 0.722 tflops
  tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmLt, 21,0,5,0,20,0,1,0,0,0,0,0,0,0,0,0,1,0,-10268,1,14,0,14,0,14,14,68,0,0,0,0,0}); // 0.185 ms vs 1.451 tflops
  // tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.185 ms vs 1.451 tflops
  // tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.186 ms vs 1.443 tflops
  tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.185 ms vs 2.902 tflops
  // tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.187 ms vs 2.871 tflops
  // tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{92, (int16_t)ME::GemmNormal}); // 0.187 ms vs 2.871 tflops
  tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{66, (int16_t)ME::GemmNormal}); // 0.19 ms vs 5.651 tflops
  // tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{59, (int16_t)ME::GemmNormal}); // 0.19 ms vs 5.651 tflops
  // tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{74, (int16_t)ME::GemmNormal}); // 0.192 ms vs 5.592 tflops
  return 0;
}();
}// clang-format on