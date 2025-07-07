#include "xop/xop.h"
#include "xop/ops_impl/global_resource.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmnormal = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{98, (int16_t)ME::GemmNormal}); // 0.187ms
  // tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{117, (int16_t)ME::GemmNormal}); // 0.189ms
  // tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{101, (int16_t)ME::GemmNormal}); // 0.189ms
  tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 0.189ms
  // tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{74, (int16_t)ME::GemmNormal}); // 0.189ms
  // tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{117, (int16_t)ME::GemmNormal}); // 0.189ms
  tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{74, (int16_t)ME::GemmNormal}); // 0.19ms
  // tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.19ms
  // tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{98, (int16_t)ME::GemmNormal}); // 0.19ms
  tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.19ms
  // tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 0.191ms
  // tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{86, (int16_t)ME::GemmNormal}); // 0.191ms
  tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{111, (int16_t)ME::GemmNormal}); // 0.191ms
  // tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{63, (int16_t)ME::GemmNormal}); // 0.192ms
  // tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{105, (int16_t)ME::GemmNormal}); // 0.192ms
  tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{75, (int16_t)ME::GemmNormal}); // 0.192ms
  // tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{62, (int16_t)ME::GemmNormal}); // 0.193ms
  // tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.193ms
  tins.add({64,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{63, (int16_t)ME::GemmNormal}); // 0.195ms
  // tins.add({64,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{61, (int16_t)ME::GemmNormal}); // 0.201ms
  // tins.add({64,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{27, (int16_t)ME::GemmNormal}); // 0.202ms
  tins.add({128,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{32, (int16_t)ME::GemmNormal}); // 0.221ms
  // tins.add({128,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{5, (int16_t)ME::GemmNormal}); // 0.221ms
  // tins.add({128,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormal}); // 0.223ms
  tins.add({256,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{4, (int16_t)ME::GemmNormal}); // 0.337ms
  // tins.add({256,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{8, (int16_t)ME::GemmNormal}); // 0.349ms
  // tins.add({256,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{16, (int16_t)ME::GemmNormal}); // 0.35ms
  tins.add({512,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormal}); // 0.675ms
  // tins.add({512,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{16, (int16_t)ME::GemmNormal}); // 0.683ms
  // tins.add({512,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{48, (int16_t)ME::GemmNormal}); // 0.687ms
  tins.add({1024,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormal}); // 1.323ms
  // tins.add({1024,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{48, (int16_t)ME::GemmNormal}); // 1.325ms
  // tins.add({1024,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{5, (int16_t)ME::GemmNormal}); // 1.326ms
  tins.add({2048,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormal}); // 2.607ms
  // tins.add({2048,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormal}); // 2.612ms
  // tins.add({2048,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{4, (int16_t)ME::GemmNormal}); // 2.619ms
  tins.add({4096,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{4, (int16_t)ME::GemmNormal}); // 5.196ms
  // tins.add({4096,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{57, (int16_t)ME::GemmNormal}); // 5.21ms
  // tins.add({4096,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{52, (int16_t)ME::GemmNormal}); // 5.23ms
  tins.add({8192,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{53, (int16_t)ME::GemmNormal}); // 10.307ms
  // tins.add({8192,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{56, (int16_t)ME::GemmNormal}); // 10.317ms
  // tins.add({8192,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{57, (int16_t)ME::GemmNormal}); // 10.32ms
  return 0;
}();
}// clang-format on