// clang-format off
#include "flux/ops_impl/normal/gemm_v2_impl.h"
#include "flux/ops_impl/normal/gemm_v2_simt_impl.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int config_normal_sm89 = []() {
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  ins.add({0,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,1,1>();});

  ins.add({1,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,2,1>();});

  ins.add({2,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,4,1,1>();});

  ins.add({3,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,4,2,1>();});

  ins.add({4,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,3,1,-1>();});

  ins.add({5,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,3,1,1>();});

  ins.add({6,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,3,2,-1>();});

  ins.add({7,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,3,2,1>();});

  ins.add({8,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,4,1,-1>();});

  ins.add({9,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,4,1,1>();});

  ins.add({10,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,4,2,-1>();});

  ins.add({11,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,4,2,1>();});

  ins.add({12,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,1,1>();});

  ins.add({13,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,2,1>();});

  ins.add({14,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,4,1,1>();});

  ins.add({15,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,4,2,1>();});

  ins.add({16,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,3,1,-1>();});

  ins.add({17,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,3,1,1>();});

  ins.add({18,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,3,2,-1>();});

  ins.add({19,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,3,2,1>();});

  ins.add({20,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,4,1,-1>();});

  ins.add({21,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,4,1,1>();});

  ins.add({22,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,4,2,-1>();});

  ins.add({23,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,4,2,1>();});

  ins.add({0,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,1,1>();});

  ins.add({1,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,2,1>();});

  ins.add({2,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,4,1,1>();});

  ins.add({3,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,4,2,1>();});

  ins.add({4,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,3,1,-1>();});

  ins.add({5,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,3,1,1>();});

  ins.add({6,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,3,2,-1>();});

  ins.add({7,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,3,2,1>();});

  ins.add({8,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,4,1,-1>();});

  ins.add({9,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,4,1,1>();});

  ins.add({10,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,4,2,-1>();});

  ins.add({11,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,16>,gemm::threadblock::ThreadblockSwizzleStreamK,4,2,1>();});

  ins.add({12,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,1,1>();});

  ins.add({13,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,2,1>();});

  ins.add({14,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,4,1,1>();});

  ins.add({15,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::GemmIdentityThreadblockSwizzle<>,4,2,1>();});

  ins.add({16,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,3,1,-1>();});

  ins.add({17,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,3,1,1>();});

  ins.add({18,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,3,2,-1>();});

  ins.add({19,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,3,2,1>();});

  ins.add({20,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,4,1,-1>();});

  ins.add({21,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,4,1,1>();});

  ins.add({22,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,4,2,-1>();});

  ins.add({23,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RRR, (int8_t)ME::Sm80}, 
    []() { return new GemmPureV2Impl<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor, cutlass::arch::Sm80,/*hparam*/cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,gemm::threadblock::ThreadblockSwizzleStreamK,4,2,1>();});

return 0;
}();
}// clang-format on