/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// #include "tensorrt_llm/common/cudaUtils.h"
// #include "tensorrt_llm/kernels/fusedActivationQuant.h"
// #include "tensorrt_llm/kernels/quantization.cuh"
// #include "tensorrt_llm/kernels/quantization.h"

#include <cstdint>
#include <optional>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace xop {

constexpr int kEltsPerThread = 8;

__device__ __forceinline__ float relu2_f32(float x)
{
    float r = fmaxf(0.0f, x);
    return r * r;
}

inline __device__ float reciprocal_approximate_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}


// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
    return val;
#else
    // static_assert(false, "not supported.");
    return 0;
#endif
}

// template <typename T_OUT, typename T_IN>
// __device__ inline T_OUT cuda_cast(T_IN val)
// {
//     return val;
// }
// Binary maximum: compute the max of two values.
template <typename T>
__device__ inline T cuda_max(T val1, T val2)
{
    return (val1 > val2) ? val1 : val2;
}

template <typename T>
__device__ inline T cuda_abs(T val)
{
    assert(false);
    return {};
}

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ inline __nv_bfloat16 cuda_abs(__nv_bfloat16 val)
{
    return __habs(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 val)
{
    return __habs2(val);
}
#endif

inline __host__ __device__ int64_t get_sf_out_offset_128x4(
    std::optional<int> batchIdx, int mIdx, int kIdx, std::optional<int> numRows, int numColVecs)
{
    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    // batched tensor
    // SF layout [numBTiles, numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [bTileIdx, mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4 * innerKStride; // 4

    // M tile layout [32, 4] is column-major.
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * innerMStride; // 16

    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * outerMStride; // 512

    // SF vector size 16 or 32. We round the "numCols" up to a multiple of 64 or 128.
    // It is the same as rounding the "numColVecs" up to a multiple of 4.
    int32_t numKTiles = (numColVecs + 4 - 1) / 4;
    int32_t mTileIdx = mIdx / (32 * 4);
    int64_t mTileStride = numKTiles * kTileStride;

    // Each SF block has 128 rows so pad rows to the multiple of 128.
    int32_t numMTiles = (numRows.value_or(0) + 128 - 1) / 128;
    int64_t bTileStride = numMTiles * mTileStride;

    // Compute the global offset.
    int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride + kTileIdx * kTileStride
        + outerMIdx * outerMStride + innerMIdx * innerMStride + innerKIdx * innerKStride;

    return SFOffset;
}

enum class QuantizationSFLayout
{
    // Block scale factors are stored in swizzled layout for cutlass FP4 kernel. Scale factor
    // blocks are organized in 512-byte blocks in global memory, with each block having 128x4 FP8 values.
    // The SF matrix dimensions are therefore padded - rows to the nearest multiple of 128 and columns to
    // the nearest multiple of 4.
    //
    // The scale factor block rows map to data block rows in an interleaved pattern:
    // For a scale factor row 'i', it maps to data block row: (i % 4) * 32 + (i / 4)
    // Column 'j' in the scale factor block corresponds to scaling the j-th block in the data tensor.
    //
    // Please refer to https://nvbugs/4165523 for more details about the swizzled layout.
    SWIZZLED,
    // Block scale factors are stored in linear layout (row-major). This is used in some trtllm-gen kernels standard.
    LINEAR
};


template <class SFType, int CVT_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_get_sf_out_offset(std::optional<int> batchIdx, int rowIdx, int colVecIdx,
    std::optional<int> numRows, int numColVecs, SFType* SFout, QuantizationSFLayout layout)
{
#if XOP_CUDA_ARCHS==120
    // Each thread holds one vector.
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2 || CVT_NUM_THREADS_PER_SF == 4);

    // One pair of threads write one SF to global memory.
    // TODO: stage through smem for packed STG.32
    // is it better than STG.8 from 4 threads ?
    if (threadIdx.x % CVT_NUM_THREADS_PER_SF == 0)
    {
        if (layout == QuantizationSFLayout::SWIZZLED)
        {
            // SF vector index (16 elements share one SF in the K dimension).
            // numRows and numCols are unpadded.
            int32_t kIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;
            int32_t mIdx = rowIdx;

            auto SFOffset = get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numColVecs);
            return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
        }
        else if (layout == QuantizationSFLayout::LINEAR)
        {
            // Linear row-major layout, no padding required.
            int32_t KTileIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;

            int32_t numKTiles = numColVecs;
            int64_t mTileStride = numKTiles;

            int64_t BTileStride = numRows.value_or(0) * mTileStride;

            int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
            return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
        }
        else
        {
            return nullptr;
        }
    }
#endif
    return nullptr;
}

// Fused relu2 + NVFP4 quantization kernel.
//
// To match the unfused path (PyTorch relu2 -> cvt_warp_fp16_to_fp4), relu2 is
// computed in f32 then rounded back to native precision (bf16/fp16) before
// quantization. Absmax and scale-factor math follow cvt_warp_fp16_to_fp4 exactly.
// Column padding to a multiple of (4 * kSfVecSize) matches quantize_with_block_size
// for the swizzled SF layout.
template <typename T>
__global__ void fusedRelu2QuantizeKernel(T const* __restrict__ input, float const* __restrict__ sfScale,
    uint32_t* __restrict__ outputFp4, uint32_t* __restrict__ outputSf, int m, int n)
{
#if XOP_CUDA_ARCHS==120
    constexpr int kSfVecSize = 16;
    constexpr int kNumThreadsPerSf = kSfVecSize / kEltsPerThread;
    constexpr int kPackedPerThread = kEltsPerThread / 2;

    using PackedType = std::conditional_t<std::is_same_v<T, half>, __half2, __nv_bfloat162>;

    float const SFScaleVal = sfScale[0];
    int const numColThreads = n / kEltsPerThread;
    int const numColVecs = n / kSfVecSize;
    int const numColThreadsPadded = ((n + 4 * kSfVecSize - 1) / (4 * kSfVecSize)) * (4 * kSfVecSize) / kEltsPerThread;
    int const rowIdx = blockIdx.x;

    if (rowIdx >= m)
        return;

    for (int colIdx = threadIdx.x; colIdx < numColThreadsPadded; colIdx += blockDim.x)
    {
        bool const isValidCol = colIdx < numColThreads;
        PackedType packedVals[kPackedPerThread];

        if (isValidCol)
        {
            int const inputOffset = rowIdx * n + colIdx * kEltsPerThread;
#pragma unroll
            for (int i = 0; i < kPackedPerThread; i++)
            {
                float f0 = relu2_f32(static_cast<float>(input[inputOffset + i * 2]));
                float f1 = relu2_f32(static_cast<float>(input[inputOffset + i * 2 + 1]));
                if constexpr (std::is_same_v<T, half>)
                {
                    packedVals[i] = __floats2half2_rn(f0, f1);
                }
                else
                {
                    packedVals[i] = __floats2bfloat162_rn(f0, f1);
                }
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < kPackedPerThread; i++)
            {
                if constexpr (std::is_same_v<T, half>)
                {
                    packedVals[i] = __float2half2_rn(0.0f);
                }
                else
                {
                    packedVals[i] = __float2bfloat162_rn(0.0f);
                }
            }
        }

        // Absmax in native precision, then reduce across the SF group (2 threads).
        auto localMax = cuda_abs(packedVals[0]);
#pragma unroll
        for (int i = 1; i < kPackedPerThread; i++)
        {
            localMax = cuda_max(localMax, cuda_abs(packedVals[i]));
        }
        localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
        float vecMax = float(cuda_max(localMax.x, localMax.y));

        // Scale-factor computation (identical to cvt_warp_fp16_to_fp4).
        float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
        __nv_fp8_e4m3 fp8SF = __nv_fp8_e4m3(SFValue);
        uint8_t fp8SFVal = fp8SF.__x;
        SFValue = static_cast<float>(fp8SF);

        float outputScale
            = vecMax != 0.0f ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

        if (colIdx % kNumThreadsPerSf == 0)
        {
            auto sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, kNumThreadsPerSf>(std::nullopt, rowIdx, colIdx,
                std::optional<int>(m), numColVecs, outputSf, QuantizationSFLayout::SWIZZLED);
            if (sfOutPtr != nullptr)
            {
                *sfOutPtr = fp8SFVal;
            }
        }

        if (isValidCol)
        {
            float2 fp2Vals[kPackedPerThread];
#pragma unroll
            for (int i = 0; i < kPackedPerThread; i++)
            {
                if constexpr (std::is_same_v<T, half>)
                {
                    fp2Vals[i] = __half22float2(packedVals[i]);
                }
                else
                {
                    fp2Vals[i] = __bfloat1622float2(packedVals[i]);
                }
                fp2Vals[i].x *= outputScale;
                fp2Vals[i].y *= outputScale;
            }

            outputFp4[rowIdx * numColThreads + colIdx] = fp32_vec_to_e2m1(fp2Vals);
        }
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("FP4 quantization requires SM100 (Blackwell) or later!\n");
    }
#endif
}

template <typename T>
void invokeFusedRelu2Quantize(T const* input, float const* sfScale, std::uint8_t* outputFp4, std::uint8_t* outputSf,
    int m, int n, int sfVecSize, cudaStream_t stream)
{
    constexpr int kSfVecSize = 16;
    int const numColThreadsPadded = ((n + 4 * kSfVecSize - 1) / (4 * kSfVecSize)) * (4 * kSfVecSize) / kEltsPerThread;
    int threadsPerBlock = min(512, numColThreadsPadded);
    threadsPerBlock = max(32, ((threadsPerBlock + 31) / 32) * 32);

    fusedRelu2QuantizeKernel<T><<<m, threadsPerBlock, 0, stream>>>(
        input, sfScale, reinterpret_cast<uint32_t*>(outputFp4), reinterpret_cast<uint32_t*>(outputSf), m, n);
}

// template void invokeFusedRelu2Quantize<half>(
//     half const*, float const*, std::uint8_t*, std::uint8_t*, int, int, int, cudaStream_t);

// #ifdef ENABLE_BF16
template void invokeFusedRelu2Quantize<__nv_bfloat16>(
    __nv_bfloat16 const*, float const*, std::uint8_t*, std::uint8_t*, int, int, int, cudaStream_t);
// #endif

} // namespace xop