
#pragma once

#include "cute/tensor.hpp"

namespace xop {
namespace traits {

using namespace cute;


template <int SM, typename ElementInput, typename ElementAccumulator>
struct MMA_Atom_Selector;

// ½«SM80Æ«ÌØ»¯
template <typename ElementInput, typename ElementAccumulator>
struct MMA_Atom_Selector<80, ElementInput, ElementAccumulator> {
  // fp16: SM80_16x8x16_F16F16F16F16_TN + SM80_16x8x16_F32F16F16F32_TN
  // bf16: SM80_16x8x16_F32BF16BF16F32_TN
  // fp32: SM80_16x8x4_F32TF32TF32F32_TN

  // using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  // using mma_traits = MMA_Traits<mma_op>;
  // using MMA_Atom_SMSP = MMA_Atom<mma_traits>;

  using MMA_Atom_HalfIn_SM80 = std::conditional_t<
    std::is_same_v<ElementAccumulator, cutlass::half_t>,
    MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>
  >;
  using MMA_Atom_SMSP = std::conditional_t<
    std::is_same_v<ElementInput, float>,  
    MMA_Atom<SM80_16x8x4_F32TF32TF32F32_TN>,
    std::conditional_t<
      std::is_same_v<ElementInput, cutlass::half_t>,  
      MMA_Atom_HalfIn_SM80,
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >
  >;
};

} // namespace traits
} // namespace xop