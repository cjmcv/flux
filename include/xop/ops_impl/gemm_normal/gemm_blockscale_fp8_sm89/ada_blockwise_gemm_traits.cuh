
#pragma once
#include "cute/atom/mma_atom.hpp"
#include <cuda_runtime.h>
#include <cute/arch/mma.hpp>
#include <cute/config.hpp>
#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>
#include <cutlass/arch/mma.h>
#include <cutlass/cutlass.h>

#include "xop/ops_impl/debug_util.h"

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
#define CUTE_ARCH_MMA_F16_SM89_ENABLED
#endif

namespace cute
{

// MMA 16x8x32 TN
// ref: SM80_16x8x16_F16F16F16F16_TN vs SM80_16x8x16_F32F16F16F32_TN
// struct SM89_16x8x32_F16E4M3E4M3F16_TN
// {
//   using DRegisters = uint32_t[2];
//   using ARegisters = uint32_t[4];
//   using BRegisters = uint32_t[2];
//   using CRegisters = uint32_t[2];

//   CUTE_HOST_DEVICE static void 
//   fma(uint32_t    & d0, uint32_t    & d1,
//     uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
//     uint32_t const& b0, uint32_t const& b1,
//     uint32_t const& c0, uint32_t const& c1)
//   {
// #if defined(CUTE_ARCH_MMA_F16_SM89_ENABLED)
//     asm volatile(
//       "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
//       "{%0,  %1},"
//       "{%2,  %3,  %4,  %5},"
//       "{%6,  %7},"
//       "{%8, %9};\n"
//       : "=r"(d0), "=r"(d1)
//       : "r"(a0), "r"(a1), "r"(a2), "r"(a3), 
//         "r"(b0), "r"(b1), 
//         "r"(c0), "r"(c1));
// #else
//     CUTE_INVALID_CONTROL_PATH(
//       "Attempting to use SM89_16x8x32_F16F4M3FE4M3F16_TN without "
//       "CUTE_ARCH_MMA_F16_SM89_ENABLED");
// #endif
//   }
// };

// // ref: MMA_Traits<SM80_16x8x16_F16F16F16F16_TN> vs MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>
// template <>
// struct MMA_Traits<SM89_16x8x32_F16E4M3E4M3F16_TN> {
//   using ValTypeD = half_t;
//   using ValTypeA = float_e4m3_t;
//   using ValTypeB = float_e4m3_t;
//   using ValTypeC = half_t;

//   using Shape_MNK = Shape<_16,_8,_32>;
//   using ThrID   = Layout<_32>;
//   using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
//                          Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
//   using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_4,  _2>>,
//                          Stride<Stride<_32,_1>,Stride<_8,_128>>>;
//   using CLayout = SM80_16x8_Row;
// };

} // namespace cute

using namespace cute;

namespace xop {
namespace traits {

template <typename ElementType, typename OutElementType, typename AccumElementType, typename BlockScaleElementType,
          typename TileShape_, typename PermShape_, int Stages_>
struct AdaBlockwiseGemmTraits {
  using ElementInput = ElementType;
  using ElementOutput = OutElementType;
  using ElementAccumulator = AccumElementType;
  using ElementBlockScale = BlockScaleElementType;
  using TileShape = TileShape_;

  static_assert(size<0>(TileShape{}) % 16 == 0);
  static_assert(size<1>(TileShape{}) % 32 == 0);
  static_assert(size<2>(TileShape{}) % 32 == 0);
  static_assert(Stages_ >= 2);

  static constexpr int Stages = Stages_;
  static constexpr int kTileM = size<0>(TileShape{});
  static constexpr int kTileN = size<1>(TileShape{});
  static constexpr int kTileK = size<2>(TileShape{});

  static constexpr int kWarpsCount = 4;
  static constexpr int kThreadCount = kWarpsCount * 32;

  static constexpr int ScaleGranularityM = 1;
  static constexpr int ScaleGranularityN = 128;
  static constexpr int ScaleGranularityK = 128;

  static constexpr int ScaleMsPerTile = (kTileM + ScaleGranularityM - 1) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = (kTileN + ScaleGranularityN - 1) / ScaleGranularityN;
  static constexpr int ScaleKsPerTile = (kTileK + ScaleGranularityK - 1) / ScaleGranularityK;

  using ScalePerTileShape = Shape<Int<ScaleMsPerTile>, Int<ScaleNsPerTile>, Int<ScaleKsPerTile>>;
  
  /////////
  // mma 
  static constexpr int kMmaPermM = size<0>(PermShape_{});
  static constexpr int kMmaPermN = size<1>(PermShape_{});
  static constexpr int kMmaPermK = size<2>(PermShape_{});
  constexpr static int NUM_GROUP_M = kTileM / kMmaPermM;
  constexpr static int NUM_GROUP_N = kTileN / kMmaPermN;
  constexpr static int NUM_GROUP_K = kTileK / kMmaPermK;

  using MMA_Atom_SM89 = std::conditional_t<
    std::is_same_v<AccumElementType, cutlass::half_t>,
    MMA_Atom<SM89_16x8x32_F16E4M3E4M3F16_TN>,
    MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>
  >;
  using TiledMma = decltype(make_tiled_mma(MMA_Atom_SM89{}, 
                                           Layout<Shape<_2, _2, _1>>{},
                                           Tile<Int<kMmaPermM>, Int<kMmaPermN>, Int<kMmaPermK>>{}));

  ///////////////////////
  // load gmem -> smem 

  // ====== A/B ======
  using G2STiledCopy = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementInput>{},
                                                Layout<Shape<_16, _8>, Stride<_8, _1>>{}, 
                                                Layout<Shape<_1, _16>>{}));
  // ====== scale A/B ======
  using GmemCopyAtomScale = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;
  using G2STileCopySFA = decltype(make_tiled_copy_impl(GmemCopyAtomScale{}, 
                                                       Layout<Shape<Shape<Int<ScaleMsPerTile>, Int<kThreadCount / ScaleMsPerTile>>, Shape<_1, _1>>, 
                                                              Stride<Stride<_1, _0>, Stride<_1, _1>>>{}, 
                                                       Shape<Int<ScaleMsPerTile>, Int<ScaleKsPerTile>>{}));
  using G2STileCopySFB = decltype(make_tiled_copy_impl(GmemCopyAtomScale{}, 
                                                       Layout<Shape<Shape<_32, _4>, Shape<_1, _1>>, 
                                                              Stride<Stride<_0, _0>, Stride<_1, _1>>>{}, 
                                                       Shape<Int<ScaleNsPerTile>, Int<ScaleKsPerTile>>{}));

  /////////////////////
  // load smem -> rf 

  // ====== A/B =====
  // using SmemAtomLayoutLoad = decltype(composition(Swizzle<2, 4, 3>{}, Layout<Shape<_8, _128>, Stride<_128, _1>>{}));
  using SmemAtomLayoutLoad = decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<_16, _128>, Stride<_128, _1>>{}));
  using SmemLayoutA = decltype(tile_to_shape(SmemAtomLayoutLoad{}, Shape<Int<kTileM>, Int<kTileK>, Int<Stages>>{}));
  using SmemLayoutB = decltype(tile_to_shape(SmemAtomLayoutLoad{}, Shape<Int<kTileN>, Int<kTileK>, Int<Stages>>{}));
  using S2RCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  // ====== scale A/B ======
  using SmemCopyAtomScale = Copy_Atom<UniversalCopy<ElementBlockScale>, ElementBlockScale>;
  using S2RLayoutTVSFA = Layout<Shape<Shape<_4, _8, _2, _2>, Shape<_2>>, Stride<Stride<_0, _1, _16, _0>, Stride<_8, _0>>>;
  using S2RShapeSFA = Shape<Int<kMmaPermM>, _1>;
  using S2RCopySFA = decltype(make_tiled_copy_impl(SmemCopyAtomScale{}, S2RLayoutTVSFA{}, S2RShapeSFA{}));
  using SmemLayoutSFA = decltype(tile_to_shape(make_layout(S2RShapeSFA{}),
                                               make_shape(shape<0>(ScalePerTileShape{}), 
                                                          shape<2>(ScalePerTileShape{}), 
                                                          Int<Stages>{}))); // BLK_M, BLK_K, Stages

  using SmemLayoutTVSFB = Layout<Shape<Shape<_4, _8, _2, _2>, Shape<_1>>, Stride<Stride<_0, _0, _0, _0>, Stride<_0, _0>>>;
  using S2RShapeSFB = Shape<_1, _1>;
  using S2RCopySFB = decltype(make_tiled_copy_impl(SmemCopyAtomScale{}, SmemLayoutTVSFB{}, S2RShapeSFB{}));
  using SmemLayoutSFB = decltype(tile_to_shape(make_layout(S2RShapeSFB{}),
                                               make_shape(shape<1>(ScalePerTileShape{}), 
                                                          shape<2>(ScalePerTileShape{}), 
                                                          Int<Stages>{}))); // BLK_N, BLK_K, Stages

  //////////////////////
  // store rf -> smem 

  using SmemAtomLayoutStore = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{})); //  8x64
  using SmemLayoutO = decltype(tile_to_shape(SmemAtomLayoutStore{}, Shape<Int<kTileM>, Int<kTileN>>{}));
  using SmemCopyAtomR2S = Copy_Atom<AutoVectorizingCopy, ElementOutput>;

  ////////////////////////
  // store smem -> gmem 

  using SmemCopyAtomS2R = Copy_Atom<UniversalCopy<uint128_t>, ElementOutput>;
  using GmemCopyAtomR2G = SmemCopyAtomS2R;
  using TiledCopyS2G = decltype(make_tiled_copy(SmemCopyAtomS2R{}, Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{})); // 16x64

  ///////////////////////////
  // shared memory storage

  struct SharedStorageLoad : aligned_struct<128> {
    array_aligned<ElementInput, cosize_v<SmemLayoutA>> smem_a;
    array_aligned<ElementInput, cosize_v<SmemLayoutB>> smem_b;
    array_aligned<float, cosize_v<SmemLayoutSFA>> smem_sfa;
    array_aligned<float, cosize_v<SmemLayoutSFB>> smem_sfb;
  };

  struct SharedStorageStore : aligned_struct<128> {
    array_aligned<ElementOutput, cosize_v<SmemLayoutO>> smem_o;
  };

  union SharedStorage {
    SharedStorageLoad load;
    SharedStorageStore store;
  };

  static constexpr int kSmemSize = static_cast<int>(sizeof(SharedStorage));
};

} // namespace traits
} // namespace xop
