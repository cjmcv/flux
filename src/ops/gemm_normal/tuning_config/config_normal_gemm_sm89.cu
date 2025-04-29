// clang-format off
#include "flux/ops_impl/normal/gemm_v2_impl.h"
#include "flux/ops_impl/normal/gemm_v2_simt_impl.h"

namespace xop {
using namespace cute;

static int config_normal_gemm_sm89 = []() {
  using         ElementA    = cutlass::half_t;
  using         LayoutA     = cutlass::layout::RowMajor;
  using         ElementB    = cutlass::half_t;
  using         LayoutB     = cutlass::layout::ColumnMajor;
  using         ElementC    = cutlass::half_t;
  using         LayoutC     = cutlass::layout::RowMajor;
  using ElementAccumulator  = cutlass::half_t;
 
  // TODO: 1. 使用脚本，按meta和hparam组合成搜索空间，生成注册代码，一份meta会对应多个由不同hparam组成的op。
  //          如 meta:   _bf16_bf16_void_bf16_fp32_fp32_sm89_rcr_gemmv2_0,
  //             hparam: _64x64x32_16x8x16_streamksk_128x128x32_gemmstreamk_4_rasteralongn
  //             得到的op_name是二者叠加：_bf16_bf16_void_bf16_fp32_fp32_sm89_rcr_gemmv2_0___64x64x32_16x8x16_streamksk_128x128x32_gemmstreamk_4_rasteralongn
  //             注册时：ins.add("op_name", []() { return new op_name(); });
  //             std::map<std::string, vector<string>> tuning_map;
  //             vector.push_back(op_name)
  //             tuning_map[meta_name] = vector
  //       2. profile时对每个输入，生成其对应的meta_name，不管shape遍历其对应所有op，找到top1，重新生成注册表。
  //          注册表中会将shape合并到meta中
  //          std::map<std::string, string> running_map;
  //          running_map[shape+meta_name] = op_name;
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  printf("init GemmConfigRegister.\n");
  using GemmSimt = GemmPureV2SimtDevice<ElementA, ElementB, ElementC, ElementAccumulator, LayoutA, LayoutB, LayoutC, cutlass::arch::Sm89, 1, cutlass::gemm::GemmShape<64, 64, 4>, cutlass::gemm::GemmShape<32, 16, 4>>;
  using GemmBasicSk1 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, 1, -1>;
  using GemmBasicSk2 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, 2, -1>;
  using GemmStreamKSk1Sm0 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4, 1, -1>;
  using GemmStreamKSk1Sm1 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4, 1, 1>;
  using GemmStreamKSk2Sm0 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4, 2, 1>;

  ins.add("GemmSimt", []() { return new GemmSimt(); });
  ins.add("GemmBasicSk1", []() { return new GemmBasicSk1(); });
  ins.add("GemmBasicSk2", []() { return new GemmBasicSk2(); });
  ins.add("GemmStreamKSk1Sm0", []() { return new GemmStreamKSk1Sm0(); });
  ins.add("GemmStreamKSk1Sm1", []() { return new GemmStreamKSk1Sm1(); });
  ins.add("GemmStreamKSk2Sm0", []() { return new GemmStreamKSk2Sm0(); });
  
  return 0;
}();

}
// clang-format on
