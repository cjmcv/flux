// clang-format off
#include "flux/op_registry.h"
namespace bytedance::flux {
using namespace cute;

static int config_single_gemm_sm89 = []() {
  auto &inst = TuningConfigRegistry::instance();
  // inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}(),_FP32{}(),_FP32{}()),_Sm89{}(),_RCR{}(),_GemmV2{}(),make_gemm_v2_meta(0)),make_runtime_config(1,27648,5120),make_gemm_hparams(make_gemm_v2_hparams(cute::make_tuple(64l,64l,32l),cute::make_tuple(16l,8l,16l),_StreamkSK{}()),cute::make_tuple(128l,128l,32l),_GemmStreamK{}(),4,_RasterAlongN{}()));
  return 0;
}();
}
// clang-format on
