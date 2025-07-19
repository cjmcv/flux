
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/tensor_view_io.h"
#include "xop/ops_impl/debug_util.h"

using namespace cute;

int main() {
  {
    printf("case 1:\n");
    using MMA_Atom = MMA_Atom<cute::SM89_16x8x32_F32E4M3E4M3F32_TN>;
    using AtomLayoutMNK = Layout<Shape<_2, _2, _1>>;
    using PermutationMNK = Tile<Int<32>, Int<32>, Int<32>>;
    using TiledMma = TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK>;

    // 输入对应 MNK => 32,128,128 
    TiledMma mma;
    auto mma_shape_A = cute::partition_shape_A(mma, cute::make_shape(cute::Int<32>{}, cute::Int<128>{})); // M, K
    auto tCrA = cute::make_tensor<cutlass::float_e4m3_t>(mma_shape_A);
    
    auto mma_shape_B = cute::partition_shape_B(mma, cute::make_shape(cute::Int<128>{}, cute::Int<128>{})); // N, K
    auto tCrB = cute::make_tensor<cutlass::float_e4m3_t>(mma_shape_B);

    auto mma_shape_B2 = cute::partition_shape_B(mma, cute::make_shape(cute::Int<32>{}, cute::Int<128>{}, cute::Int<4>{})); // N/4, K, 4
    auto tCrB2 = cute::make_tensor<cutlass::float_e4m3_t>(mma_shape_B2);

    xop::print_tensor("tCrA", tCrA);
    xop::print_tensor("tCrB", tCrB);
    xop::print_tensor("tCrB2", tCrB2);
    
    // tCrA: ptr[8b](0x7fffff40c8e0) o ((_4,_2,_2),_1,_4):((_1,_4,_8),_0,_16):
    // tCrB: ptr[8b](0x7fff2b6bb170) o ((_4,_2),_8,_4):((_1,_4),_8,_64):
    // tCrB2: ptr[8b](0x7ffff26c5880) o ((_4,_2),_2,_4,_4):((_1,_4),_8,_16,_64)
    //   
  }

  {
    printf("case 2:\n");
    using MMA_Atom = MMA_Atom<cute::SM89_16x8x32_F32E4M3E4M3F32_TN>;
    using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;
    using PermutationMNK = Tile<Int<1>, Int<1>, Int<1>>;
    using TiledMma = TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK>;

    // 输入对应 MNK => 32,128,128 
    TiledMma mma;
    auto mma_shape_A = cute::partition_shape_A(mma, cute::make_shape(cute::Int<32>{}, cute::Int<128>{})); // M, K
    auto tCrA = cute::make_tensor<cutlass::float_e4m3_t>(mma_shape_A);
    
    auto mma_shape_B = cute::partition_shape_B(mma, cute::make_shape(cute::Int<128>{}, cute::Int<128>{})); // N, K
    auto tCrB = cute::make_tensor<cutlass::float_e4m3_t>(mma_shape_B);

    auto mma_shape_B2 = cute::partition_shape_B(mma, cute::make_shape(cute::Int<32>{}, cute::Int<128>{}, cute::Int<4>{})); // N/4, K, 4
    auto tCrB2 = cute::make_tensor<cutlass::float_e4m3_t>(mma_shape_B2);

    xop::print_tensor("tCrA", tCrA);
    xop::print_tensor("tCrB", tCrB);
    xop::print_tensor("tCrB2", tCrB2);

    // rank1是指令shape，rank2和rank3是两个方向上的mma次数。
    // tCrA: ptr[8b](0x7ffcb383d0d0) o ((_4,_2,_2),_2,_4):((_1,_4,_8),_16,_32):
    // tCrB: ptr[8b](0x7ffcb383d150) o ((_4,_2),_16,_4):((_1,_4),_8,_128):
    // tCrB2: ptr[8b](0x7ffcb383d350) o ((_4,_2),_4,_4,_4):((_1,_4),_8,_32,_128)
  }

  {
    printf("case 2:\n");
    using MMA_Atom = MMA_Atom<cute::SM89_16x8x32_F32E4M3E4M3F32_TN>;
    using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;
    using PermutationMNK = Tile<Int<1>, Int<1>, Int<1>>;
    using TiledMma = TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK>;

    // 输入对应 MNK => 32,128,128 
    TiledMma mma;
    auto mma_shape_A = cute::partition_shape_A(mma, cute::make_shape(cute::Int<32>{}, cute::Int<128>{})); // M, K
    auto tCrA = cute::make_tensor<cutlass::float_e4m3_t>(mma_shape_A);
    
    auto mma_shape_B = cute::partition_shape_B(mma, cute::make_shape(cute::Int<128>{}, cute::Int<128>{})); // N, K
    auto tCrB = cute::make_tensor<cutlass::float_e4m3_t>(mma_shape_B);

    auto mma_shape_B2 = cute::partition_shape_B(mma, cute::make_shape(cute::Int<32>{}, cute::Int<128>{}, cute::Int<4>{})); // N/4, K, 4
    auto tCrB2 = cute::make_tensor<cutlass::float_e4m3_t>(mma_shape_B2);

    xop::print_tensor("tCrA", tCrA);
    xop::print_tensor("tCrB", tCrB);
    xop::print_tensor("tCrB2", tCrB2);

    // tCrA: ptr[8b](0x7ffcb383d0d0) o ((_4,_2,_2),_2,_4):((_1,_4,_8),_16,_32):
    // tCrB: ptr[8b](0x7ffcb383d150) o ((_4,_2),_16,_4):((_1,_4),_8,_128):
    // tCrB2: ptr[8b](0x7ffcb383d350) o ((_4,_2),_4,_4,_4):((_1,_4),_8,_32,_128)
  }
  return 0;
}


