#include "xop/xop.h"
#include "xop/ops_impl/global_resource.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmsm80 = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{79, (int16_t)ME::GemmNormal}); // 0.013 ms vs 0.968 tflops
  // tins.add({1,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{55, (int16_t)ME::GemmNormal}); // 0.013 ms vs 0.968 tflops
  // tins.add({1,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{77, (int16_t)ME::GemmNormal}); // 0.013 ms vs 0.968 tflops
  tins.add({2,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.013 ms vs 1.936 tflops
  // tins.add({2,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{90, (int16_t)ME::GemmNormal}); // 0.013 ms vs 1.936 tflops
  // tins.add({2,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{57, (int16_t)ME::GemmNormal}); // 0.013 ms vs 1.936 tflops
  tins.add({4,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.013 ms vs 3.872 tflops
  // tins.add({4,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.013 ms vs 3.872 tflops
  // tins.add({4,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{77, (int16_t)ME::GemmNormal}); // 0.013 ms vs 3.872 tflops
  tins.add({8,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{79, (int16_t)ME::GemmNormal}); // 0.013 ms vs 7.743 tflops
  // tins.add({8,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{66, (int16_t)ME::GemmNormal}); // 0.013 ms vs 7.743 tflops
  // tins.add({8,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{57, (int16_t)ME::GemmNormal}); // 0.013 ms vs 7.743 tflops
  tins.add({16,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.013 ms vs 15.487 tflops
  // tins.add({16,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{68, (int16_t)ME::GemmNormal}); // 0.013 ms vs 15.487 tflops
  // tins.add({16,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.013 ms vs 15.487 tflops
  tins.add({32,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{66, (int16_t)ME::GemmNormal}); // 0.013 ms vs 30.973 tflops
  // tins.add({32,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.013 ms vs 30.973 tflops
  // tins.add({32,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{79, (int16_t)ME::GemmNormal}); // 0.013 ms vs 30.973 tflops
  tins.add({64,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{68, (int16_t)ME::GemmNormal}); // 0.013 ms vs 61.947 tflops
  // tins.add({64,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{55, (int16_t)ME::GemmNormal}); // 0.013 ms vs 61.947 tflops
  // tins.add({64,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{57, (int16_t)ME::GemmNormal}); // 0.013 ms vs 61.947 tflops
  tins.add({128,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{44, (int16_t)ME::GemmNormal}); // 0.015 ms vs 107.374 tflops
  // tins.add({128,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{46, (int16_t)ME::GemmNormal}); // 0.015 ms vs 107.374 tflops
  // tins.add({128,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{58, (int16_t)ME::GemmNormal}); // 0.016 ms vs 100.663 tflops
  tins.add({256,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{47, (int16_t)ME::GemmNormal}); // 0.022 ms vs 146.419 tflops
  // tins.add({256,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{51, (int16_t)ME::GemmNormal}); // 0.022 ms vs 146.419 tflops
  // tins.add({256,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{3, (int16_t)ME::GemmNormal}); // 0.023 ms vs 140.053 tflops
  tins.add({512,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{47, (int16_t)ME::GemmNormal}); // 0.034 ms vs 189.484 tflops
  // tins.add({512,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{51, (int16_t)ME::GemmNormal}); // 0.035 ms vs 184.07 tflops
  // tins.add({512,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{45, (int16_t)ME::GemmNormal}); // 0.037 ms vs 174.12 tflops
  tins.add({1024,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{3, (int16_t)ME::GemmNormal}); // 0.061 ms vs 211.228 tflops
  // tins.add({1024,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{47, (int16_t)ME::GemmNormal}); // 0.064 ms vs 201.327 tflops
  // tins.add({1024,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{51, (int16_t)ME::GemmNormal}); // 0.065 ms vs 198.229 tflops
  tins.add({2048,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{3, (int16_t)ME::GemmNormal}); // 0.115 ms vs 224.085 tflops
  // tins.add({2048,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{47, (int16_t)ME::GemmNormal}); // 0.122 ms vs 211.228 tflops
  // tins.add({2048,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{51, (int16_t)ME::GemmNormal}); // 0.122 ms vs 211.228 tflops
  tins.add({4096,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{3, (int16_t)ME::GemmNormal}); // 0.23 ms vs 224.085 tflops
  // tins.add({4096,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{47, (int16_t)ME::GemmNormal}); // 0.242 ms vs 212.974 tflops
  // tins.add({4096,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormal}); // 0.242 ms vs 212.974 tflops
  tins.add({8192,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{3, (int16_t)ME::GemmNormal}); // 0.454 ms vs 227.047 tflops
  // tins.add({8192,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormal}); // 0.465 ms vs 221.676 tflops
  // tins.add({8192,6144,1024,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{4, (int16_t)ME::GemmNormal}); // 0.468 ms vs 220.255 tflops
  return 0;
}();
}// clang-format on