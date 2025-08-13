// reference: https://github.com/reed-lau/cute-gemm
#pragma once
#include "common.h"
#include "cute/tensor.hpp"

namespace gemm_no_smem {

using namespace cute;


template <typename ElementType, typename OutElementType, typename AccumElementType, class CTA_tile>
struct KernelTraits {
  using MMA_Atom_SM80 = std::conditional_t<
    std::is_same_v<ElementType, cutlass::half_t>,
    MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
    MMA_Atom<SM80_16x8x4_F32TF32TF32F32_TN>
  >;
  using TiledMma = decltype(make_tiled_mma(MMA_Atom_SM80{}, 
                                          make_layout(Shape<_2, _2, _1>{}), 
                                          make_layout(Shape<_1, _2, _1>{})));

  static constexpr int kTileM = size<0>(CTA_tile{});
  static constexpr int kTileN = size<1>(CTA_tile{});
  static constexpr int kTileK = size<2>(CTA_tile{});

  using ElementInput = ElementType;
  using ElementOutput = OutElementType;
  using ElementAccumulator = AccumElementType;
};

template <typename KT>
__global__ void GemmSimpleKernel(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k) {

  using T = typename KT::ElementInput;

  constexpr int kTileM = KT::kTileM;
  constexpr int kTileN = KT::kTileN;
  constexpr int kTileK = KT::kTileK;

  // 基于gmem，构建Tensor
  Tensor A = make_tensor(make_gmem_ptr<T>(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr<T>(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr<T>(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  // 取block分块，对应的还是gmem的数据，make_tile的大小就是block的大小，make_coord是block索引与tile的对应关系。
  // make_coord(iy, _)表示用iy去取行，列用“_”表示不拆分全部取出，
  // 所以对于block(iy,ix)会取出A中以[kTileM, kTileK]为单位的第iy行的所有tile。
  // 即gA[kTileM, kTileK, num_tile_k]
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));  // (kTileM, kTileK, num_tile_k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));  // (kTileN, kTileK, num_tile_k)
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix)); // (kTileM, kTileN) 

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

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }
  // 结果放回到tCgC
  cute::copy(tCrC, tCgC); 
}

} // namespace gemm_no_smem