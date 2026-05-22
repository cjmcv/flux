
// #define ENABLE_QWEN3_06B
// #define ENABLE_QWEN3_4B

// #define TL_ENABLE_L2_PREFETCH 1
#include "linear.cuh"
#include "silu_mul.cuh"
#include "rmsnorm.cuh"
#include "gqa_decode.cuh"
#include "rope.cuh"
#include "copy.cuh"
#include "prefetch.cuh"