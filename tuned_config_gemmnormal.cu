#include "xop/xop.h"
#include "xop/ops_impl/global_resource.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmnormal = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({2048,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormal}); // 1.192ms
  // tins.add({2048,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormal}); // 1.455ms
  // tins.add({2048,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormalSimt}); // 3.481ms
  tins.add({4096,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormal}); // 1.782ms
  // tins.add({4096,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormal}); // 1.814ms
  // tins.add({4096,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormalSimt}); // 7.032ms
  tins.add({2048,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormal}); // 0.912ms
  // tins.add({2048,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormal}); // 1.084ms
  // tins.add({2048,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormalSimt}); // 3.437ms
  tins.add({4096,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormal}); // 1.767ms
  // tins.add({4096,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{1, (int16_t)ME::GemmNormal}); // 1.804ms
  // tins.add({4096,576,7168,1,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{0, (int16_t)ME::GemmNormalSimt}); // 7.001ms
  return 0;
}();
}// clang-format on