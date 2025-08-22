#include "xop/xop.h"
#include "xop/ops_impl/global_resource.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmnormal = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{92, (int16_t)ME::GemmNormal}); // 0.187ms
  // tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{57, (int16_t)ME::GemmNormal}); // 0.188ms
  // tins.add({1,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.188ms
  tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{99, (int16_t)ME::GemmNormal}); // 0.191ms
  // tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{97, (int16_t)ME::GemmNormal}); // 0.192ms
  // tins.add({2,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{70, (int16_t)ME::GemmNormal}); // 0.193ms
  tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{88, (int16_t)ME::GemmNormal}); // 0.19ms
  // tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{74, (int16_t)ME::GemmNormal}); // 0.19ms
  // tins.add({4,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{94, (int16_t)ME::GemmNormal}); // 0.191ms
  tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{103, (int16_t)ME::GemmNormal}); // 0.191ms
  // tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{96, (int16_t)ME::GemmNormal}); // 0.191ms
  // tins.add({8,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{70, (int16_t)ME::GemmNormal}); // 0.191ms
  tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{92, (int16_t)ME::GemmNormal}); // 0.189ms
  // tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{96, (int16_t)ME::GemmNormal}); // 0.19ms
  // tins.add({16,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{79, (int16_t)ME::GemmNormal}); // 0.191ms
  tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{74, (int16_t)ME::GemmNormal}); // 0.197ms
  // tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{72, (int16_t)ME::GemmNormal}); // 0.198ms
  // tins.add({32,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{24, (int16_t)ME::GemmNormal}); // 0.198ms
  tins.add({64,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{62, (int16_t)ME::GemmNormal}); // 0.204ms
  // tins.add({64,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{19, (int16_t)ME::GemmNormal}); // 0.205ms
  // tins.add({64,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{56, (int16_t)ME::GemmNormal}); // 0.205ms
  tins.add({128,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{29, (int16_t)ME::GemmNormal}); // 0.226ms
  // tins.add({128,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{4, (int16_t)ME::GemmNormal}); // 0.23ms
  // tins.add({128,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{62, (int16_t)ME::GemmNormal}); // 0.231ms
  tins.add({256,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{3, (int16_t)ME::GemmNormal}); // 0.358ms
  // tins.add({256,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{7, (int16_t)ME::GemmNormal}); // 0.365ms
  // tins.add({256,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{45, (int16_t)ME::GemmNormal}); // 0.37ms
  tins.add({512,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormal}); // 0.671ms
  // tins.add({512,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{7, (int16_t)ME::GemmNormal}); // 0.694ms
  // tins.add({512,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{14, (int16_t)ME::GemmNormal}); // 0.697ms
  tins.add({1024,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{4, (int16_t)ME::GemmNormal}); // 1.325ms
  // tins.add({1024,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormal}); // 1.326ms
  // tins.add({1024,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormal}); // 1.329ms
  tins.add({2048,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormal}); // 2.636ms
  // tins.add({2048,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{44, (int16_t)ME::GemmNormal}); // 2.641ms
  // tins.add({2048,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{51, (int16_t)ME::GemmNormal}); // 2.644ms
  tins.add({4096,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{47, (int16_t)ME::GemmNormal}); // 5.181ms
  // tins.add({4096,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{48, (int16_t)ME::GemmNormal}); // 5.184ms
  // tins.add({4096,4096,4096,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{51, (int16_t)ME::GemmNormal}); // 5.19ms
  return 0;
}();
}// clang-format on