//===- comm_none.h ------------------------------------------------ C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "cute/int_tuple.hpp"

namespace bytedance::flux {

struct GemmOnlyArguments {
  int m;
  int n;
  int k;
  float alpha;
  float beta;
  void const *input;
  void const *weight;
  void const *bias;
  void *output;
};

// GEMM with dequantization
// Dequant[i, j] = scale_a[i] * scale_b[j] * accumulator[i, j]
// D = Dequant + bias
// Accumulator dtype can be different from scale, Dequant calculation is based on scale dtype,
// D/bias dtype can be different from Dequant.
// For example:
//    Dequant and scale_a/scale_b: fp32
//    accumulator: s32
//    D and bias: bf16
struct S8GemmDequantArguments {
  int m;
  int n;
  int k;
  float alpha;
  float beta;
  void const *A;        // m * k
  void const *B;        // k * n
  void const *bias;     // bias, 1 * n
  void const *scale_A;  // m * 1
  void const *scale_B;  // 1 * n
  void *D;              // output: m * n
};

// FP8 GEMM
// Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias
// D = activation(Aux)
// if Aux is fp8:
//   abs_max_output = max( abs(aux) | (for every aux in Aux) )
//   Aux = scale_aux * Aux
// if D is fp8 type:
//   abs_max_output = max( abs(d) | (for every d in D) )
//   D = scale_d * D
struct GemmFP8Arguments {
  int m;
  int n;
  int k;
  float alpha;
  float beta;
  void const *A;  // m * k
  void const *B;  // k * n
  void const *C;  // m * n
  void *Aux;      // m * n
  void *D;        // output: m * n
  void *Vector;   // bias: 1 * n
  float *abs_max_Aux;
  float *abs_max_D;
  // scaling tensors
  float const *scaleA;
  float const *scaleB;
  float const *scaleC;
  float const *scaleD;    // require if D is fp8
  float const *scaleAux;  // require if Aux is fp8
};

}  // namespace bytedance::flux
