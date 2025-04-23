//===- test_single_gemm.cc ------------------------------------------ C++ ---===//
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

#include <cassert>
#include <exception>
#include "flux/common_cuda.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/profiler/device_allocation.h"

#include "flux/gemm_args.h"
namespace bytedance::flux {

// 这里临时注册配置进行手动调优，前提是对应shape在其他地方没有被注册
// 同时，各种参数的组合必须是 registers 搜索空间里能找到的
// using namespace cute;
// static int config_single_gemm_sm89_2 = []() {
//   auto &inst = TuningConfigRegistry::instance();
//   inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}(),_FP32{}(),_FP32{}()),_Sm89{}(),_RCR{}(),_GemmV2{}(),make_gemm_v2_meta(false),None{}),make_runtime_config(1,27648,5120),make_gemm_hparams(make_gemm_v2_hparams(cute::make_tuple(16l,64l,64l),cute::make_tuple(16l,8l,16l),_StreamkDP{}()),None{},cute::make_tuple(32l,128l,64l),_GemmStreamK{}(),3,_RasterHeuristic{}()));
// return 0;
// }();

template <class DType>
void
run_single_gemm(int m, int n, int k) {
  cudaSetDevice(0);
  auto arch = get_arch();

  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(DType{}));
  auto meta = make_gemm_meta(
      dt_conf,
      arch,
      _RCR{},
      ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}());

  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  using ElementD = decltype(to_cutlass_element(dt_conf.d()));
  using ElementC_ = decltype(to_cutlass_element(dt_conf.c()));
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>, ElementD, ElementC_>;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementD> block_D;
  cutlass::DeviceAllocation<uint8_t> workspace;
  block_A.reset(m * k);
  block_B.reset(k * n);
  block_C.reset(m * n);
  block_D.reset(m * n);

  auto rt_conf = make_runtime_config(m, n, k);

  auto hparams = OpRegistry::instance().get_hparams(meta, rt_conf);
  

  // hparams: "impl_spec", "comm_spec", "tile_shape", "gemm_kind", "mainloop_stage", "raster_order"
  // impl_spec: "warp_shape", "instruction_shape", "streamk_mode"
  auto unified_hparams = to_gemm_hparams(hparams);
  // hparams.impl_spec(): std::variant<bytedance::flux::None, bytedance::flux::GemmV2HParams<cute::tuple<long int, long int, long int>, cute::tuple<long int, long int, long int>, bytedance::flux::GemmStreamkModeEnum> >&
  auto impl_spec = std::get<1>(hparams.impl_spec());
  auto gemm_v2_hparams = to_gemm_v2_hparams(impl_spec);

  // FLUX_CHECK_EQ(gemm_v2_hparams.warp_shape(), cute::make_tuple(16l,64l,64l));
  // FLUX_CHECK_EQ(gemm_v2_hparams.instruction_shape(), cute::make_tuple(16l,8l,16l));
  // FLUX_CHECK_EQ(gemm_v2_hparams.streamk_mode(), _StreamkDP{}());

  // FLUX_CHECK_EQ(unified_hparams.comm_spec(), None{});
  // FLUX_CHECK_EQ(unified_hparams.tile_shape(), cute::make_tuple(32l,128l,64l));
  // FLUX_CHECK_EQ(unified_hparams.gemm_kind(), _GemmStreamK{}());
  // FLUX_CHECK_EQ(unified_hparams.mainloop_stage(), 4);
  // FLUX_CHECK_EQ(unified_hparams.raster_order(), _RasterAlongM{}());

  //
  auto gemm_op = OpRegistry::instance().get_op(meta, hparams);
  auto stream = nullptr;
  const SingleGemmArguments args{
      m, n, k, 1.0, 0.0, block_A.get(), block_B.get(), block_C.get(), block_D.get()};
  int64_t workspace_size = gemm_op->get_workspace_size(args);
  workspace.reset(workspace_size);
  constexpr int warm_iters = 200;
  constexpr int iters = 50;
  GpuTimer timer;
  for (int i = 0; i < warm_iters + iters; ++i) {
    if (i == warm_iters) {
      timer.start(stream);
    }
    gemm_op->run(args, workspace.get(), stream);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  printf("op time elapsed: %.3f ms\n", timer.elapsed_millis() / iters);
}

}  // namespace bytedance::flux

int
main(int argc, char *argv[]) {
  if ((argc != 4) and (argc != 5)) {
    std::cerr << "Usage: " << argv[0] << " <m> <n> <k> [dtype]\n";
    return 1;
  }
  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);

  std::string dtype = "BF16";
  if (argc == 5) {
    dtype = argv[4];
  }

  try {
    using namespace bytedance::flux;
    if (dtype == "FP8" or dtype == "fp8") {
      assert((int)get_arch() > (int)_Sm80{}());
      bytedance::flux::run_single_gemm<decltype(make_gemm_dtype_config(
          _E4M3{}, _E4M3{}, _BF16{}, _BF16{}))>(m, n, k);
    } else if (dtype == "FP16" or dtype == "fp16") {
      bytedance::flux::run_single_gemm<decltype(make_gemm_dtype_config(
          _FP16{}, _FP16{}, _Void{}, _FP16{}))>(m, n, k);
    } else if (dtype == "BF16" or dtype == "bf16") {
      bytedance::flux::run_single_gemm<decltype(make_gemm_dtype_config(
          _BF16{}, _BF16{}, _Void{}, _BF16{}))>(m, n, k);
    } else {
      FLUX_CHECK(false) << "unsupported dtype: " << dtype;
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
  }
}
