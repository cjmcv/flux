/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// <NT> Stream-k，主要是为了优化wave quantization问题而设计
// https://zhuanlan.zhihu.com/p/716352563
// https://zhuanlan.zhihu.com/p/7259261717
// 
// wave quantization定义：每个CTA(thread block)会被分配到特定的SM硬件单元中，所有的SM可以并行执行的CTA称为一个wave。
//
/// 如有a矩阵：0  1  2  3  4  5
//            6  7  8  9 10 11
//           12 13 14 15 16 17
//           18 19 20 21 22 23
//           24 25 26 27 28 29
// 1.在DP-gemm方案中，沿着无依赖关系的MN维度进行划分
//   如有4个sm，每个sm一时刻负责一个block线程，一个block负责一行6个tile数据。
//   则有wave0(sm0(第0行)，sm1(第1行), sm2(第二行), sm3(第三行))，wave1时只处理最后一行，wave1未占满sm，wave1浪费4/3资源；
// 2. 在DP-gemm的基础上加上split-k，即a矩阵列方向切分，假设对半切，
//    则wave0=(sm0(0,1,2), sm1(3,4,5), sm2(6,7,8), sm3(9,10,11)), wave2(sm0(25,26,27), sm1(27,28,29)),wave2浪费1/2资源。
//    另外k方向的切分是需要汇总的，即同一行负责不同列的两个sm需要做同步归约。如012和345由两个sm负责，二者结果需要规约，
// 3. Stream-K算法，将直接将 MNK 三个维度融合了之后统一考虑并行切分，如wave0(sm0(0-7), sm1(8-15), sm2(16-23), sm3(24-29)),
//    其中sm3是负责6个tile，即浪费了2/8资源，是三个方案中浪费最少的。
// 注意split-k和stream-k都需要做跨sm的数据同步（block间同步）。
//
// 扩展笔记：线程块内同步__syncthreads();          设备同步：cudaDeviceSynchronize();
//          事件同步：cudaEventSynchronize();     流同步：cudaStreamSynchronize();
//          block之间没有提供同步机制，一般的同步方式是：原子操作与全局内存标记，各自结果叠加到目的地；分多个kernel，在外部规约；
//
// 下面例子也展示了dp-gemm，split-k，stream-k的对比试验。

/***************************************************************************************************
 Example contrasting the Stream-K parallel decomposition for GEMM threadblocks versus the
 "classic data-parallel" and "Split-K" decompositions.

 For more details regarding the Stream-K method, see "Stream-K: Work-centric Parallel Decomposition
 for Dense Matrix-Matrix Multiplication on the GPU" (https://arxiv.org/abs/2301.03598)

 Requires NVIDIA Ampere or newer device (SM80+).

 - To lock persistence mode, power (400W), clocks (1005MHz) for evaluation (assumes device 0 and A100)

     cutlass$ sudo nvidia-smi -pm 1 -i 0

     cutlass$ sudo nvidia-smi -i 0 -pl 400

     cutlass$ sudo nvidia-smi -i 0 -lgc 1005

 - Build and run:

     cutlass$ mkdir build

     cutlass$ cd build

     cutlass/build$ cmake .. -DCUTLASS_NVCC_ARCHS=80

     cutlass/build$ make 47_ampere_gemm_universal_streamk

     cutlass/build$ ./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk

        10000 timing iterations of 2048 x 2048 x 2048 matrix-matrix multiply

        Basic data-parallel GEMM
          Disposition: Passed
          Avg runtime: 0.112633 ms
          GFLOPs: 152530

        StreamK GEMM with default load-balancing
          Disposition: Passed
          Avg runtime: 0.0941929 ms
          GFLOPs: 182390
          Speedup vs Basic-DP: 1.196

        StreamK emulating basic data-parallel GEMM
          Disposition: Passed
          Avg runtime: 0.113119 ms
          GFLOPs: 151875
          Speedup vs Basic-DP: 0.996

        Basic split-K GEMM with tile-splitting factor 2
          Disposition: Passed
          Avg runtime: 0.104772 ms
          GFLOPs: 163973

        StreamK emulating Split-K GEMM with tile-splitting factor 2
          Disposition: Passed
          Avg runtime: 0.105379 ms
          GFLOPs: 163029
          Speedup vs Basic-SplitK: 0.994

 **************************************************************************************************/

 #include <iostream>
 #include <string>
 #include <any>

 #include "cutlass/cutlass.h"
 #include "cutlass/gemm/device/gemm_universal.h"
 
 #include "cutlass/util/command_line.h"
 #include "cutlass/util/host_tensor.h"
 #include "cutlass/util/reference/device/gemm.h"
 #include "cutlass/util/reference/host/tensor_compare.h"
 #include "cutlass/util/reference/host/tensor_copy.h"
 #include "cutlass/util/reference/host/tensor_fill.h"
 #include "cutlass/util/tensor_view_io.h"
 
//  #include "helper.h"
#include <map>
#include "flux/common_cuda.h"
#include "named_tuple.h"
using namespace xop;

/// Command line options parsing
struct Options {
  cutlass::gemm::GemmCoord problem_size;
  float alpha;
  float beta;

  void *ptr_A;
  void *ptr_B;
  void *ptr_C;
  void *ptr_D;

  int stride_a;
  int stride_b;
  int stride_c;
  int stride_d;
  Options() : problem_size({2048, 2048, 2048}), alpha(1.0f), beta(0.0f) {}
};


class GemmBase {
public:
  virtual void initialize(Options &options) = 0;
  virtual void run() = 0;
};

template <class LayoutA, class LayoutB, class LayoutC>
class ImplHelper {
public:
  ImplHelper(int m, int n, int k): m_(m), n_(n), k_(k) {};

  int get_stride_a() const {
    if constexpr (cute::is_same_v<LayoutA, cutlass::layout::RowMajor>) {
      return k_;
    } else {
      static_assert(cute::is_same_v<LayoutA, cutlass::layout::ColumnMajor>, "requires ColumnMajor.");
      return m_;
    }
  }
  int get_stride_b() const {
    if constexpr (cute::is_same_v<LayoutB, cutlass::layout::RowMajor>) {
      return n_;
    } else {
      static_assert(cute::is_same_v<LayoutB, cutlass::layout::ColumnMajor>, "requires ColumnMajor.");
      return k_;
    }
  }
  int get_stride_c() const {
    if constexpr (cute::is_same_v<LayoutC, cutlass::layout::RowMajor>) {
      return n_;
    } else {
      static_assert(cute::is_same_v<LayoutC, cutlass::layout::ColumnMajor>, "requires ColumnMajor.");
      return m_;
    }
  }

private:
  int m_;
  int n_;
  int k_;
};

// 定义工厂函数类型
using GemmFactory = std::function<GemmBase*()>;

// 单例类来管理 gemmMap
class GemmMapManager {
private:
    std::map<std::string, GemmFactory> gemmMap;
    std::map<std::string, GemmBase*> createdInstances;

    // 私有构造函数，防止外部实例化
    GemmMapManager() = default;

    // 防止拷贝构造和赋值操作
    GemmMapManager(const GemmMapManager&) = delete;
    GemmMapManager& operator=(const GemmMapManager&) = delete;

public:
    // 获取单例实例
    static GemmMapManager& getInstance() {
        static GemmMapManager instance;
        return instance;
    }

    // 注册函数
    void registerGemm(const std::string& name, GemmFactory factory) {
        gemmMap[name] = factory;
    }
    GemmBase* createGemm(const std::string& name) {
      auto it = gemmMap.find(name);
      if (it != gemmMap.end()) {
          GemmBase* instance = it->second();
          createdInstances[name] = instance;
          return instance;
      }
      throw std::runtime_error("Gemm type not found.");
    }
    // 获取 Gemm 实例，如果已存在则直接返回，不存在则创建
    GemmBase* getGemm(const std::string& name) {
        auto it = createdInstances.find(name);
        if (it != createdInstances.end()) {
            return it->second;
        }
        return createGemm(name);
    }

    // 析构时释放所有创建的实例
    ~GemmMapManager() {
        for (auto& pair : createdInstances) {
            delete pair.second;
        }
    }
};

//////////////////////////////////////////////////////////
template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC, 
          class ArchTag, int SplitKFactor,
          class ThreadblockShape, class WarpShape>
class GemmPureV2SimtDevice : public GemmBase {

  using EpilogueOpSimt = cutlass::epilogue::thread::LinearCombination<
      ElementC,               // Element type for C and D matrix operands
      1,                      // Memory access granularity of C and D matrix in units of elements
      ElementAccumulator,     // Element type from internal accumaccumulation
      ElementAccumulator>;    // Data type used to compute linear combination

  using DeviceGemmSimt = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt, //OperatorClass,
    ArchTag, // ArchTag,cutlass::arch::Sm89
    ThreadblockShape, //cutlass::gemm::GemmShape<64, 64, 4>,
    WarpShape, //cutlass::gemm::GemmShape<32, 16, 4>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueOpSimt>;

public:
  typename DeviceGemmSimt::Arguments args_from_options(const Options &options) {

    return typename DeviceGemmSimt::Arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
      options.problem_size,                     // problem_size
      SplitKFactor,                             // batch count / splitk slices
      {                                         // epilogue parameters
        ElementAccumulator(options.alpha),
        ElementAccumulator(options.beta)
      },
      options.ptr_A,                   // ptr_A
      options.ptr_B,                   // ptr_B
      options.ptr_C,                   // ptr_C
      options.ptr_D,                   // ptr_D
      options.problem_size.mk().product(),      // batch_stride_A
      options.problem_size.nk().product(),      // batch_stride_B
      options.problem_size.mn().product(),      // batch_stride_C
      options.problem_size.mn().product(),      // batch_stride_D
      options.stride_a,              // stride_a
      options.stride_b,              // stride_b
      options.stride_c,              // stride_c
      options.stride_d);             // stride_d
  }

  void initialize(Options &options) {
    gemm_dev_ = DeviceGemmSimt();

    ImplHelper<LayoutA, LayoutB, LayoutC> helper(options.problem_size.m(), options.problem_size.n(), options.problem_size.k());
    options.stride_a = helper.get_stride_a();
    options.stride_b = helper.get_stride_b();
    options.stride_c = helper.get_stride_c();
    options.stride_d = helper.get_stride_c();

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    auto arguments = args_from_options(options);
    size_t workspace_size = DeviceGemmSimt::get_workspace_size(arguments);
  
    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  
    // Check the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace.get()));
  }

  void run() {
    CUTLASS_CHECK(gemm_dev_());
  }

private:
  DeviceGemmSimt gemm_dev_;
};

//////////////////////////////////////////////////////////
template <class ElementA, class LayoutA,
          class ElementB, class LayoutB,
          class ElementC, class LayoutC,
          class ElementAccumulator,
          class ArchTag, 
          class ThreadblockShape, class WarpShape, class InstructionShape,
          class ThreadBlockSwizzle, int NumStages, int SplitKFactor, int AvailSms>
class GemmPureV2Impl : public GemmBase  {

  // Epilogue output operator
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,               // Element type for C and D matrix operands
      128 / cutlass::sizeof_bits<ElementC>::value, // Memory access granularity of C and D matrix in units of elements
      ElementAccumulator,     // Element type from internal accumaccumulation
      ElementAccumulator>;    // Data type used to compute linear combination

  // Classic data-parallel device GEMM implementation type
  using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      ThreadBlockSwizzle, // cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> / cutlass::gemm::threadblock::ThreadblockSwizzleStreamK
      NumStages,
      128 / cutlass::sizeof_bits<ElementA>::value,  // AlignmentA, Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
      128 / cutlass::sizeof_bits<ElementB>::value>; // AlignmentB

public:
  void initialize(Options &options) {
    gemm_dev_ = DeviceGemmBasic();
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    ImplHelper<LayoutA, LayoutB, LayoutC> helper(options.problem_size.m(), options.problem_size.n(), options.problem_size.k());
    options.stride_a = helper.get_stride_a();
    options.stride_b = helper.get_stride_b();
    options.stride_c = helper.get_stride_c();
    options.stride_d = helper.get_stride_c();
    auto arguments = args_from_options(options);
    size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);
  
    // Allocate workspace memory
    if (workspace_.size() < workspace_size)
      workspace_.reallocate(workspace_size);
  
    // Check the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace_.get()));
  }

  void run() {
    CUTLASS_CHECK(gemm_dev_());
  }

private:
  // avail_sms: Number of device SMs to use is unlimited
  //         1: Set loadbalancing width to 1 SM (no load balancing)
  //        -1: Reset loadbalancing width to unspecified SMs (i.e., the number of device SMs)
  typename DeviceGemmBasic::Arguments args_from_options(const Options &options) {
    if constexpr (cute::is_same_v<ThreadBlockSwizzle, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>) {
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        options.problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
        {                                         // epilogue parameters
          ElementAccumulator(options.alpha),
          ElementAccumulator(options.beta)
        },
        options.ptr_A,                   // ptr_A
        options.ptr_B,                   // ptr_B
        options.ptr_C,                   // ptr_C
        options.ptr_D,                   // ptr_D
        options.problem_size.mk().product(),      // batch_stride_A
        options.problem_size.nk().product(),      // batch_stride_B
        options.problem_size.mn().product(),      // batch_stride_C
        options.problem_size.mn().product(),      // batch_stride_D
        options.stride_a,              // stride_a
        options.stride_b,              // stride_b
        options.stride_c,              // stride_c
        options.stride_d);             // stride_d    
    }
    else {
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        options.problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
        {                                         // epilogue parameters
          ElementAccumulator(options.alpha),
          ElementAccumulator(options.beta)
        },
        options.ptr_A,                   // ptr_A
        options.ptr_B,                   // ptr_B
        options.ptr_C,                   // ptr_C
        options.ptr_D,                   // ptr_D
        options.problem_size.mk().product(),      // batch_stride_A
        options.problem_size.nk().product(),      // batch_stride_B
        options.problem_size.mn().product(),      // batch_stride_C
        options.problem_size.mn().product(),      // batch_stride_D
        options.stride_a,              // stride_a
        options.stride_b,              // stride_b
        options.stride_c,              // stride_c
        options.stride_d,              // stride_d
        AvailSms);                                // avail_sms
    }
  }

private:
  DeviceGemmBasic gemm_dev_;
  cutlass::device_memory::allocation<uint8_t> workspace_;
};

 /////////////////////////////////////////////////////////////////////////////////////////////////
 /// Testbed utility types
 /////////////////////////////////////////////////////////////////////////////////////////////////
 
/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(true)
  {}

};
 
 
 
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Execute a given example GEMM computation
template <class ElementC, class LayoutC>
Result run(GemmBase *gemm, std::string description, Options &options, int iterations, 
           cutlass::HostTensor<ElementC, LayoutC> &tensor_d,
           cutlass::HostTensor<ElementC, LayoutC> &tensor_ref_d)
{
  // Display test description
  std::cout << std::endl << description << std::endl;

  // Zero-initialize test output matrix D
  cutlass::reference::host::TensorFill(tensor_d.host_view());
  tensor_d.sync_device();

  // Create a structure of gemm kernel arguments suitable for invoking an instance of DeviceGemmT
  gemm->initialize(options);
  gemm->run();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  // Run profiling loop
  if (iterations > 0)
  {
    bytedance::flux::GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < iterations; ++iter) {
    gemm->run();
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(iterations);
    result.gflops = 2.0 * double(options.problem_size.product()) / double(1.0e9) / (result.avg_runtime_ms * 1000.0);

    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPs: " << result.gflops << std::endl;
  }

  if (!result.passed) {
    exit(-1);
  }

  return result;
}
 
/// Program entrypoint
int main(int argc, const char **argv)
{
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;

    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  // Current device must must have compute capability at least 80
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  if (!((props.major * 10 + props.minor) >= 80))
  {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;

    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  // A matrix configuration
  using         ElementA    = cutlass::half_t;                                // Element type for A matrix operand
  using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  // B matrix configuration
  using         ElementB    = cutlass::half_t;                                // Element type for B matrix operand
  using         LayoutB     = cutlass::layout::ColumnMajor;                      // Layout type for B matrix operand
  // C/D matrix configuration
  using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
  using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
  // Multiply-accumulate blocking/pipelining details
  using ElementAccumulator  = cutlass::half_t;                          // Element type for internal accumulation

  cutlass::HostTensor<ElementA, LayoutA> tensor_a;
  cutlass::HostTensor<ElementB, LayoutB> tensor_b;
  cutlass::HostTensor<ElementC, LayoutC> tensor_c;
  cutlass::HostTensor<ElementC, LayoutC> tensor_d;
  cutlass::HostTensor<ElementC, LayoutC> tensor_ref_d;

  Options options;
  tensor_a.resize(options.problem_size.mk());       // <- Create matrix A with dimensions M x K
  tensor_b.resize(options.problem_size.kn());       // <- Create matrix B with dimensions K x N
  tensor_c.resize(options.problem_size.mn());       // <- Create matrix C with dimensions M x N
  tensor_d.resize(options.problem_size.mn());       // <- Create matrix D with dimensions M x N used to store output from CUTLASS kernel
  tensor_ref_d.resize(options.problem_size.mn());   // <- Create matrix D with dimensions M x N used to store output from reference kernel

  // Fill matrix A on host with uniform-random data [-2, 2]
  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 1, ElementA(2), ElementA(-2), 0);
  // Fill matrix B on host with uniform-random data [-2, 2]
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 1, ElementB(2), ElementB(-2), 0);
  // Fill matrix C on host with uniform-random data [-2, 2]
  cutlass::reference::host::TensorFillRandomUniform(tensor_c.host_view(), 1, ElementC(2), ElementC(-2), 0);


  //
  // Compute reference output
  //

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();

  options.ptr_A = tensor_a.device_data();
  options.ptr_B = tensor_b.device_data();
  options.ptr_C = tensor_c.device_data();
  options.ptr_D = tensor_d.device_data();

  // Zero-initialize reference output matrix D
  cutlass::reference::host::TensorFill(tensor_ref_d.host_view());
  tensor_ref_d.sync_device();


// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      ElementAccumulator>;

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;

  // Launch device reference gemm kernel
  gemm_reference(
    options.problem_size,
    ElementAccumulator(options.alpha),
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    ElementAccumulator(options.beta),
    tensor_c.device_ref(),
    tensor_ref_d.device_ref());

  // Wait for kernels to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy output data from reference kernel to host for comparison
  tensor_ref_d.sync_host();
 
 
  //
  // Evaluate CUTLASS kernels
  GemmMapManager& manager = GemmMapManager::getInstance();
  using GemmSimt = GemmPureV2SimtDevice<ElementA, ElementB, ElementC, ElementAccumulator, LayoutA, LayoutB, LayoutC, cutlass::arch::Sm89, 1, cutlass::gemm::GemmShape<64, 64, 4>, cutlass::gemm::GemmShape<32, 16, 4>>;
  using GemmBasicSk1 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, 1, -1>;
  using GemmBasicSk2 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, 2, -1>;
  using GemmStreamKSk1Sm0 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4, 1, -1>;
  using GemmStreamKSk1Sm1 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4, 1, 1>;
  using GemmStreamKSk2Sm0 = GemmPureV2Impl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 4, 2, 1>;

  manager.registerGemm("GemmSimt", []() { return new GemmSimt(); });
  manager.registerGemm("GemmBasicSk1", []() { return new GemmBasicSk1(); });
  manager.registerGemm("GemmBasicSk2", []() { return new GemmBasicSk2(); });
  manager.registerGemm("GemmStreamKSk1Sm0", []() { return new GemmStreamKSk1Sm0(); });
  manager.registerGemm("GemmStreamKSk1Sm1", []() { return new GemmStreamKSk1Sm1(); });
  manager.registerGemm("GemmStreamKSk2Sm0", []() { return new GemmStreamKSk2Sm0(); });

  int iterations = 2000;
  // xop::GemmBase* gemm_simt = manager.createGemm("GemmSimt");
  // printf("%p, %p, %p\n", manager.getGemm("GemmSimt"),manager.getGemm("GemmBasic"), manager.getGemm("GemmSimt"));
  Result basic_simt       = run(manager.getGemm("GemmSimt"), "Basic simt GEMM", options, iterations, tensor_d, tensor_ref_d);
  Result basic_dp         = run(manager.getGemm("GemmBasicSk1"), "Basic data-parallel GEMM", options, iterations, tensor_d, tensor_ref_d);
  Result streamk_default  = run(manager.getGemm("GemmStreamKSk1Sm0"), "StreamK GEMM with default load-balancing", options, iterations, tensor_d, tensor_ref_d);

  printf("  Speedup vs Basic-DP: %.3f\n", (basic_dp.avg_runtime_ms / streamk_default.avg_runtime_ms));

  Result streamk_dp       = run(manager.getGemm("GemmStreamKSk1Sm1"), "StreamK emulating basic data-parallel GEMM", options, iterations, tensor_d, tensor_ref_d);
  printf("  Speedup vs Basic-DP: %.3f\n", (basic_dp.avg_runtime_ms / streamk_dp.avg_runtime_ms));
 
  // Show that StreamK can emulate "Split-K" with a tile-splitting factor
  Result basic_splitk = run(manager.getGemm("GemmBasicSk2"), 
    std::string("Basic split-K GEMM with tile-splitting factor ") + std::to_string(2),
    options, iterations, tensor_d, tensor_ref_d);

  Result streamk_splitk = run(manager.getGemm("GemmStreamKSk2Sm0"), 
    std::string("StreamK emulating Split-K GEMM with tile-splitting factor ") + std::to_string(2),
    options, iterations, tensor_d, tensor_ref_d);

  printf("  Speedup vs Basic-SplitK: %.3f\n", (basic_splitk.avg_runtime_ms / streamk_splitk.avg_runtime_ms));
 
  return 0;
}
 