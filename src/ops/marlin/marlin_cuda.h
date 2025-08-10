
#pragma once

#include <torch/all.h>
  
namespace xop {

void helloABCM(int a);

// Marlin FP16xINT4 multiply; can be used within `torch.compile`.
// @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
// @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
// @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
// @s: `torch.half` scales of shape `(m / groupsize, n)`
// @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
// @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
// @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
// @sms: number of SMs to use for the kernel (can usually be left as auto -1)
// @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
void marlin_fp16xint4_matmul(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& s,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
);

}  // namespace xop
