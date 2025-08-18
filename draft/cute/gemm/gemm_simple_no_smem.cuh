// reference: https://github.com/reed-lau/cute-gemm
#pragma once
#include "common.h"
#include "cute/tensor.hpp"
#include "xop/ops_impl/debug_util.h"
#include "common_traits.cuh"
namespace gemm_no_smem {

using namespace cute;

template <typename ElementType_, typename OutElementType_, typename AccumElementType_, class TileShape_>
struct KernelTraits {
  using ElementInput = ElementType_;
  using ElementOutput = OutElementType_;
  using ElementAccumulator = AccumElementType_;

  using MMA_Atom_SM80 = typename xop::traits::MMA_Atom_Selector<80, ElementInput, ElementAccumulator>::MMA_Atom_SMSP;

  using TiledMma = decltype(make_tiled_mma(MMA_Atom_SM80{}, 
                                          make_layout(Shape<_2, _2, _1>{}), 
                                          make_layout(Shape<_1, _2, _1>{})));

  static constexpr int kTileM = size<0>(TileShape_{});
  static constexpr int kTileN = size<1>(TileShape_{});
  static constexpr int kTileK = size<2>(TileShape_{});
};

template <typename KT>
__global__ void GemmSimpleKernel(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k) {

  using ElementInput = typename KT::ElementInput;
  using ElementOutput = typename KT::ElementOutput;
  using ElementAccumulator = typename KT::ElementAccumulator;

  constexpr int kTileM = KT::kTileM;
  constexpr int kTileN = KT::kTileN;
  constexpr int kTileK = KT::kTileK;

  // 基于gmem，构建Tensor
  Tensor A = make_tensor(make_gmem_ptr<ElementInput>(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr<ElementInput>(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr<ElementOutput>(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  // 取block分块，对应的还是gmem的数据，make_tile的大小就是block的大小，make_coord是block索引与tile的对应关系。
  // make_coord(iy, _)表示用iy去取行，列用“_”表示不拆分全部取出，
  // 所以对于block(iy,ix)会取出A中以[kTileM, kTileK]为单位的第iy行的所有tile。
  // 即gA[kTileM, kTileK, num_tile_k]
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(ix, _));  // (kTileM, kTileK, num_tile_k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(iy, _));  // (kTileN, kTileK, num_tile_k)
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(ix, iy)); // (kTileM, kTileN) 

  // 基于线程，拿到warp级别的mma分块
  typename KT::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  // 因为gA/gB是三维，对应的partition tAgA/tBgB多了一个线程对应的MMA维度，变成了四维，最后一维同样是num_tile_k，
  // 基于partition创建寄存器fragment，因为后面用num_tile_k的for循环来计算，fragment只需要取一份即可，所以取了0.
  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
  clear(tCrC);
  
  // 手动k循环，从gmem中取出对应tile，拷贝到reg。
  // 然后基于reg fragment进行计算。
  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    // 对应数据类型应对应mma指令，如果
    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }
  // 结果放回到tCgC
  // tCrC取自mma atom，对应的就是mma指令的输出类型。
  // tCgC取自外面提供的输出类型数据，不一定与mma执行的输出类型一致。
  // 如fp16*fp16=fp16, accum采用fp32, 则tCrC时fp32的，而tCgC时fp16的。
  // 即使类型不一致，仍可以直接使用cute::copy，会采用比较低效的拷贝方式完成类型转换。
  // xop::print_tensor("btCrC", tCrC);
  cute::copy(tCrC, tCgC); 
  // xop::print_tensor("atCgC", tCgC);
}

} // namespace gemm_no_smem