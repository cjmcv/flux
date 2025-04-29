// clang-format off
#include "flux/ops_impl/normal/gemm_v2_impl.h"
#include "flux/ops_impl/normal/gemm_v2_simt_impl.h"

namespace xop {
using namespace cute;

static int config_normal_gemm_sm89 = []() {
  printf("init GemmConfigRegister.\n");

  using         ElementA    = cutlass::half_t;
  using         LayoutA     = cutlass::layout::RowMajor;
  using         ElementB    = cutlass::half_t;
  using         LayoutB     = cutlass::layout::ColumnMajor;
  using         ElementC    = cutlass::half_t;
  using         LayoutC     = cutlass::layout::RowMajor;
  using ElementAccumulator  = float;
 
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  using GemmSimt = GemmPureV2SimtDevice<ElementA, ElementB, ElementC, ElementAccumulator, LayoutA, LayoutB, LayoutC, cutlass::arch::Sm89, 1, cutlass::gemm::GemmShape<64, 64, 4>, cutlass::gemm::GemmShape<32, 16, 4>>;
  using GemmBasicSk1 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, 1, -1>;
  using GemmBasicSk2 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, 2, -1>;
  using GemmStreamKSk1Sm0 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4, 1, -1>;
  using GemmStreamKSk1Sm1 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4, 1, 1>;
  using GemmStreamKSk2Sm0 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4, 2, 1>;

  // cute::make_tuple(_BF16{}, _Sm89{}, _RCR{});
  // ins.add("GemmSimt", []() { return new GemmSimt(); });
  // ins.add("GemmBasicSk1", []() { return new GemmBasicSk1(); });
  // ins.add("GemmBasicSk2", []() { return new GemmBasicSk2(); });
  // ins.add("GemmStreamKSk1Sm0", []() { return new GemmStreamKSk1Sm0(); });
  // ins.add("GemmStreamKSk1Sm1", []() { return new GemmStreamKSk1Sm1(); });
  // ins.add("GemmStreamKSk2Sm0", []() { return new GemmStreamKSk2Sm0(); });

  // 基础meta: shape(m,n,k), layout, arch, type
  // hparam: stges,shape..
  // 1）1个problem会固定layout/arch/type，没有其他选择, 统称为meta.
  // 2）1个meta，会配置多个hparam。同一个meta与不同hparam组合构建op，key的0号位会设置为id号，后面接meta。
  // 3）tuning时输入某个shape，针对其输入类型构建meta，并拼接从0开始的序列，逐一访问对应的op。
  //    结束后得到top1的key，拆分得到id+meta，前面拼接shape+meta作为key，id+meta作为value，以value为key获取op.
  using ME = UnifiedMetaEnum;
  ins.add({1, (int8_t)ME::Sm80, (int8_t)ME::RCR, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::Void, (int8_t)ME::FP16, (int8_t)ME::FP32}, []() { return new GemmSimt(); });
  ins.add({2, (int8_t)ME::Sm80, (int8_t)ME::RCR, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::Void, (int8_t)ME::FP16, (int8_t)ME::FP32}, []() { return new GemmBasicSk1(); });
  ins.add({3, (int8_t)ME::Sm80, (int8_t)ME::RCR, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::Void, (int8_t)ME::FP16, (int8_t)ME::FP32}, []() { return new GemmBasicSk2(); });
  ins.add({4, (int8_t)ME::Sm80, (int8_t)ME::RCR, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::Void, (int8_t)ME::FP16, (int8_t)ME::FP32}, []() { return new GemmStreamKSk1Sm0(); });
  ins.add({5, (int8_t)ME::Sm80, (int8_t)ME::RCR, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::Void, (int8_t)ME::FP16, (int8_t)ME::FP32}, []() { return new GemmStreamKSk1Sm1(); });
  ins.add({6, (int8_t)ME::Sm80, (int8_t)ME::RCR, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::Void, (int8_t)ME::FP16, (int8_t)ME::FP32}, []() { return new GemmStreamKSk2Sm0(); });
  
  return 0;
}();

}
// clang-format on
