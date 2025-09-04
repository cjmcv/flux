#pragma once
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/common_cutlass.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
// #include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "gemm_ar_v2/default_gemm_universal_with_visitor_rs.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "gemm_ar_v2/visitor_store_rs.hpp"
// #include "gemm_ar_v2/gemm_universal_rs.h"

////////////////////////////////
// #include "xop/../../src/ops/allreduce_normal/custom_all_reduce.h"
#include "xop/../../src/ops/allreduce_normal/custom_all_reduce.cuh"
///////////////////////////////

// 实现cuda kernel，填充4096*4096的矩阵，输入int* 类型的flag / result, 以及int类型的flag_len ：
// 1）kernel内一个block有128个线程，每个block一次负责一个128*128的tile的数值填充为1；
// 2）每个线程一次需要连续填充8个元素。
// 3）输入还有一个一维数组flag，数组flag里每个元素是对应的是tile id号，以行优先。
//    数组flag内0号/32号/64号/96号等下标对应的tile交由blockIdx.x为0的线程负责，使用for循环按顺序进行；
//    同理，1号/33号/65号/97号等由blockIdx.x为1的线程负责。
// 4）在4096*4096的目标矩阵下，tile的数量是32*32个。
//    则对应数组flag中的tile id号，如flag[0]==10，即表示为第0行第10列的tile。如为32，则表示为第1行第0列的tile。

template <typename T, int ngpus>
__global__ void disaggregated_reduce(vllm::RankData* dp, vllm::RankSignals sg, vllm::Signal* self_sg,
                                     T* __restrict__ out, int rank, int m, int n) {
  // vllm::barrier_at_start<ngpus>(sg, self_sg, rank);

#ifdef ENABLE_ALLREDUCE
  int world_size = 2;
  int target_rank = (rank+1) % world_size;
  uint32_t *flag = (uint32_t*)params_ptr->rank_signals.signals[target_rank]->_flag;
  T *rank_data = (T *)dp->ptrs[target_rank];
#else
  uint32_t *flag = (uint32_t *)dp;
#endif
                              
  const int OUT_M   = m;
  const int OUT_N   = n;
  constexpr int TILE    = 128;
  const int NTILE_M = OUT_M / TILE;   // 32
  const int NTILE_N = OUT_N / TILE;   // 32
  constexpr int THREADS = 128;
  constexpr int ELE_PER_THREAD = 8;

  const int bx = blockIdx.x;   // 0..31
  const int tx = threadIdx.x;  // 0..127

  int flagSize = NTILE_M * NTILE_N;

// #ifdef ENABLE_ALLREDUCE
  while (xop_ld_flag_volatile(&flag[0]) == 0) { __nanosleep(40); }
// #else
//   while (flag[0] == 0) { printf("0"); }
// #endif

  for (int k = bx; k < flagSize; k += gridDim.x) {
// #ifdef ENABLE_ALLREDUCE
    // while (xop_ld_flag_volatile(&flag[k+1]) == 0) { __nanosleep(40); printf("(%d,%d)", k+1, flag[k+1]); }
// #else
//     while (flag[k+1] == 0) { printf("(%d,%d)", k+1, flag[k+1]); }
// #endif

    int tileId = flag[k+1] - 1;
    
    // printf("(%d, %d, %d, %d)\n", OUT_M, OUT_N, NTILE_M, NTILE_N);
    int tile_m    = tileId / NTILE_N;   // tile 行号
    int tile_n    = tileId % NTILE_N;   // tile 列号

    // 每个 tile 128*128，起始坐标
    int base_m = tile_m * TILE;
    int base_n = tile_n * TILE;

    int global_m = base_m + tx;     // 本线程负责 tile 内的第 tx 行

    // printf("tileId %d, %d, %d, %d\n", k, tileId, tile_m, global_m);
    // 128 列 / 8 = 16 段，每段 8 个连续元素
    for (int seg = 0; seg < TILE / ELE_PER_THREAD; ++seg) {
        int colOffset = seg * ELE_PER_THREAD;   // 0,8,16,...,120
        T* ptr      = out + global_m * OUT_N + (base_n + colOffset);
#ifdef ENABLE_ALLREDUCE
        #pragma unroll
        for (int i = 0; i < ELE_PER_THREAD; ++i) { 
          ptr[i] = __hadd(ptr[i], rank_data[i]);
        }
#else
        #pragma unroll
        for (int i = 0; i < ELE_PER_THREAD; ++i) { ptr[i] = 1; }
#endif
    }
  }

  // const int kM = 128;
  // const int kN = 128;
  // const int M = 512;
  // const int N = 512;

  // int block_num_m = M/KM;
  // int block_num_n = N/KN;
  // int block_num = block_num_m * block_num_n;

  // // for 循环逐个访问下标0~block_num，每个下标存放的是已完成计算的blockid，每当一个数值置位非-1，则开始对该block做通信。
  // // 每个block的通信由整个通信kernel的所有资源进行，不再切分。
  // // 通信的一个block对应计算的一个block tile，
  // if (rank_data0[0] != 0) { // blocking

  //   int tid = threadIdx.x;
  //   int bid = blockIdx.x;

  //   for (int tileIdx = 0; tileIdx < 32; ++tileIdx) {
  //     int globalTileId = bx * 32 + tileIdx;
  //     int tileRow = globalTileId / (OUT_N / TILE);   // 0..31
  //     int tileCol = globalTileId % (OUT_N / TILE);   // 0..31

  //     // 每个 tile 128×128，按行主序
  //     // 每个线程负责 16 行 × 8 列 的小条带
  //     for (int strip = 0; strip < 16; ++strip) {
  //       int localRow = tx / 8 * 16 + strip;   // 0..127
  //       int localCol = (tx % 8) * ELE_PER_THREAD;

  //       int globalRow = tileRow * TILE + localRow;
  //       int globalCol = tileCol * TILE + localCol;

  //       int* ptr = out + globalRow * pitch + globalCol;
  //       #pragma unroll
  //       for (int k = 0; k < ELE_PER_THREAD; ++k) {
  //         if (globalCol + k < OUT_N) {
  //           ptr[k] = (globalRow << 16) | (globalCol + k);
  //         }
  //       }
  //     }
  //   }

    // for (int bid=1; bid<block_num+1; bid++) { // blocking
    //   if (rank_data0[bid] > 0) {
    //     int cur_block_id = rank_data0[bid] - 1;

    //     int bid_m = cur_block_id / block_num_n;
    //     int bid_n = cur_block_id % block_num_n;
        
    //     int pid_m = bid_m * kM;
    //     int pid_n = bid_n * kN;

    //     for (int i=pid_m; i<pid_m+kM; i++) {
    //       for (int j=pid_n; j<pid_n+kN; j++) {
    //         for (int pi=0; pi<8; pi++) {
    //           result[i*N + j] = 2;              
    //         }
    //       }
    //     }
    //     // do the actual reduction
    //     for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len/p;
    //         idx += gridDim.x * blockDim.x) {
    //       for (int i=0; i<p; i++) {
    //         result[idx*p+i] = 2; // + dp->ptrs[1][idx+i];
    //       }
    //     }          
    //   }
   
    // }    
  // }

   
  
  // vllm::barrier_at_end<ngpus, true>(sg, self_sg, rank);
}


namespace xop {

// s0: 在end_step中完成通信与相加全流程：存结果到self rank_data, 使用block_id堵塞同步，获取其他rank的data，并进行累加操作。
//     问题：每个step中都有堵塞等待操作，同步时间过长。
// s1：在end_step中完成通信：存结果到本地output内存，同时存结果到对方rank data内存上(通信)，该过程不需要任何同步操作。
//     在end_epilogue中完成累加：全局堵塞同步后，执行两份本地内存的累加。
//     问题：end_step中发起通信的sm数量过多，L2 Cache 争用，效率偏低。
// s2: 基于s1，使用flag tensor结合atomicAdd，在bidx==0下所有bidy的数据都已经写到本地buffer后，以bidx==0下的block来处理通信，
//     一次处理一整行，这样可以限制通信所使用的sm数量。
//     问题: 负责通信的sm负载严重失衡。
// s3: 限制计算所使用的sm数量，预留几个给通信。分两个kernel两个stream，二者使用对称内存互联。
//     stream0 启动计算kernel，同时新建stream1 启动通信kernel。计算ep中每回写一个tile到本地内存，置位共享内存所有GPU的对应区域标志位；
//     在通信中堵塞访问标志位，拉取该rank对应区域的数据到本地，完成reduce后，存放数据到本地输出内存。先等待stream0结束，而后等待stream1结束，得到最终结果。
struct AllReduceArguments {
  bool is_capturing;
  void *temp_input;

  int world_size;
  int rank;
  int packed_array_num;
  
  void *reg_buffer;
  vllm::RankData* rank_data;
  vllm::RankSignals rank_signals;
  vllm::Signal *self_signal;
  
  void *output;
  virtual ~AllReduceArguments() {}
};

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator,
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, 
          class ThreadblockShape, class WarpShape, class InstructionShape,
          class ThreadBlockSwizzle, int NumStages, int SplitKFactor, int AvailSms>
class GemmArV2Impl : public GemmBase  {
  using ElementCompute = ElementAccumulator;
  using ElementOutput = ElementC;

  static constexpr int EVTEpilogueStages = 1;  
  static constexpr int AlignmentC       = 128 / cutlass::sizeof_bits<ElementC>::value;
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, 
    WarpShape, 
    ElementC, 
    AlignmentC, 
    EVTEpilogueStages
  >;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, ElementC,
      cute::Stride<cute::_0, cute::_1, int32_t>  // StrideMNL
  >;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

  using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<
      Compute0, // 2
      Accum,    // 0
      Bias>;    // 1

  using D = cutlass::epilogue::threadblock::VisitorAuxStoreRs<
      OutputTileThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  >;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<
      D,
      EVTCompute0>; // EVTCompute2

  using EVTKernelStreamK =
      typename cutlass::gemm::kernel::DefaultGemmWithVisitorRs<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, 128 / cutlass::sizeof_bits<ElementA>::value,
      ElementB, LayoutB, cutlass::ComplexTransform::kNone, 128 / cutlass::sizeof_bits<ElementB>::value,
      ElementC, LayoutC, AlignmentC,
      ElementAccumulator,
      ElementCompute,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EVTD,
      ThreadBlockSwizzle,
      NumStages,
      cutlass::arch::OpMultiplyAdd,
      EVTEpilogueStages
  >::GemmKernel;

  using DeviceGemmBasic = cutlass::gemm::device::GemmUniversalAdapter<EVTKernelStreamK>;

public:
  void initialize(RtArguments *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtArgumentsV2 *rt_args = dynamic_cast<RtArgumentsV2*>(args);

    ////
    cudaEventCreate(&event_);
    cudaStreamCreate(&rs_stream_);
    m_ = rt_args->m;
    n_ = rt_args->n;
    output_len_ = rt_args->m * rt_args->n;
    ar_args_.output = rt_args->ptr_D;    
    auto cu_stream = static_cast<cudaStream_t>(stream);
    fetch_comm_args(fusion_args, cu_stream);
    ////

    gemm_dev_ = DeviceGemmBasic();
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    ImplHelper<LayoutA, LayoutB, LayoutC> helper(rt_args->m, rt_args->n, rt_args->k);
    rt_args->stride_a = helper.get_stride_a();
    rt_args->stride_b = helper.get_stride_b();
    rt_args->stride_c = rt_args->C_s == -1 ? helper.get_stride_c() : 0;
    // printf("rt_args->stride_c: %d.\n", rt_args->stride_c);
    rt_args->stride_d = helper.get_stride_c();
    auto arguments = args_from_options(rt_args);
    size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);

    // Allocate workspace memory
    void *workspace_ptr = GlobalBuffer::instance().ResizeDeviceBufferIfNeeded(workspace_size);

    // Check the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace_ptr, cu_stream));
  }

  void run(void *stream = nullptr) {

    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaEventRecord(event_, cu_stream));      // 记录计算流
    CUDA_CHECK(cudaStreamWaitEvent(rs_stream_, event_)); // 使rs流等待计算流之前的任务都结束
    
    //////////////////////////////////////////////////////////
    
    CUTLASS_CHECK(gemm_dev_.run(cu_stream));


    int max_blocks = 32;
    int threads = 128;
    int blocks = std::min(max_blocks, n_ / 128); // 一个线程8个元素，1 tile 对应 128*128，按n维度的block数量算。
#ifdef ENABLE_ALLREDUCE
    disaggregated_reduce<to_cuda_type_t<ElementOutput>, 2><<<blocks, threads, 0, rs_stream>>>(ar_args_.rank_data, ar_args_.rank_signals, ar_args_.self_signal, reinterpret_cast<to_cuda_type_t<ElementOutput>*>(ar_args_.output), ar_args_.rank, m_, n_);
#else    
    disaggregated_reduce<to_cuda_type_t<ElementOutput>, 2><<<blocks, threads, 0, rs_stream_>>>((vllm::RankData *)ar_args_.reg_buffer, ar_args_.rank_signals, ar_args_.self_signal, reinterpret_cast<to_cuda_type_t<ElementOutput>*>(ar_args_.output), ar_args_.rank, m_, n_);
#endif
    //////////////////////////////////////////////////////////
    // wait for reduce_scatter done
    CUDA_CHECK(cudaEventRecord(event_, rs_stream_)); // 记录通信流
    CUDA_CHECK(cudaStreamWaitEvent(cu_stream, event_)); // 使计算流等待event中通信流之前的任务都结束


    // CUDA_CHECK(cudaEventRecord(event_, cu_stream));
    // CUDA_CHECK(cudaStreamWaitEvent(cu_stream, event_));

    // // printf("out %p \n", ar_args_.reg_buffer);
    // int max_blocks = 32;
    // int threads = 128;
    // int blocks = std::min(max_blocks, n_ / 128); // 一个线程8个元素，1 tile 对应 128*128，按n维度的block数量算。
    // disaggregated_reduce<to_cuda_type_t<ElementOutput>, 2><<<blocks, threads, 0, rs_stream_>>>((vllm::RankData *)ar_args_.reg_buffer, ar_args_.rank_signals, ar_args_.self_signal, reinterpret_cast<to_cuda_type_t<ElementOutput>*>(ar_args_.output), ar_args_.rank, m_, n_);
    // //////////////////////////////////////////////////////////
    // CUDA_CHECK(cudaEventRecord(event_, rs_stream_));
    // CUDA_CHECK(cudaStreamWaitEvent(rs_stream_, event_));
  }

private:
  // avail_sms: Number of device SMs to use is unlimited
  //         1: Set loadbalancing width to 1 SM (no load balancing)
  //        -1: Reset loadbalancing width to unspecified SMs (i.e., the number of device SMs)
  typename DeviceGemmBasic::Arguments args_from_options(const RtArgumentsV2 *rt_args) {
    cutlass::gemm::GemmCoord problem_size = {rt_args->m, rt_args->n, rt_args->k};
    int batch_stride_C = rt_args->stride_c == 0 ? rt_args->n : problem_size.mn().product();

#ifdef ENABLE_ALLREDUCE
    ElementC *gemm_out = (ElementC *)ar_args_.temp_input;
    if (ar_args_.is_capturing == false) {
      gemm_out = (ElementC *)ar_args_.reg_buffer;
    }
#else
    ElementC *gemm_out = (ElementC *)rt_args->ptr_D;
#endif
    typename EVTD::Arguments callback_args{
      {
        {}, // Accum
        {(ElementC *)rt_args->ptr_C, ElementC(0), {cute::_0{}, cute::_1{}, int32_t(problem_size.n())}},            // Bias
        {}  // Compute0
      },        // EVTCompute2
      { 
        gemm_out, {problem_size.n(), cute::_1{}, problem_size.mn().product()}, 
        ar_args_.world_size, ar_args_.rank, ar_args_.packed_array_num, ar_args_.reg_buffer, 
        ar_args_.rank_data, ar_args_.rank_signals, ar_args_.self_signal, ar_args_.output
      },                   // D
    };   

    if constexpr (cute::is_same_v<ThreadBlockSwizzle, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>) {  
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
        callback_args, 
        rt_args->ptr_A,                   // ptr_A
        rt_args->ptr_B,                   // ptr_B
        nullptr,                   // ptr_C
        nullptr,                   // ptr_D
        problem_size.mk().product(),      // batch_stride_A
        problem_size.nk().product(),      // batch_stride_B
        0,                   // batch_stride_C
        0,      // batch_stride_D
        rt_args->stride_a,              // stride_a
        rt_args->stride_b,              // stride_b
        0,              // stride_c
        0);             // stride_d    
    }
    else {
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
        callback_args, 
        rt_args->ptr_A,                   // ptr_A
        rt_args->ptr_B,                   // ptr_B
        nullptr,                   // ptr_C
        nullptr,                   // ptr_D
        problem_size.mk().product(),      // batch_stride_A
        problem_size.nk().product(),      // batch_stride_B
        0,                   // batch_stride_C
        0,      // batch_stride_D
        rt_args->stride_a,              // stride_a
        rt_args->stride_b,              // stride_b
        0,              // stride_c
        0,              // stride_d
        AvailSms);                                // avail_sms
    }
  }

  void fetch_comm_args(void *fusion_args, cudaStream_t stream) {
#ifdef ENABLE_ALLREDUCE
    RtCommArguments *rt_args = (RtCommArguments*)(fusion_args);

    if constexpr (!(cute::is_same_v<ElementOutput, float> ||
      cute::is_same_v<ElementOutput, cutlass::half_t> ||
      cute::is_same_v<ElementOutput, cutlass::bfloat16_t>)) {
      throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }

    auto reg_buffer = reinterpret_cast<void*>(rt_args->reg_buffer);
    if (reg_buffer == 0) {
      // While capturing, reg_buffer is zero.
      // Use your own memory to open the ipc handle.
      ar_args_.is_capturing = true;
      ar_args_.reg_buffer = rt_args->gemm_out;
      ar_args_.temp_input = rt_args->gemm_out;
    }
    else {
      ar_args_.is_capturing = false;
      ar_args_.reg_buffer = reg_buffer;
      ar_args_.temp_input = rt_args->gemm_out;
    }

    auto fa = reinterpret_cast<vllm::CustomAllreduce*>(rt_args->handle);
    fa->get_ptrs<to_cuda_type_t<ElementOutput>>(
      stream, reinterpret_cast<to_cuda_type_t<ElementOutput>*>(ar_args_.reg_buffer), output_len_,
      &ar_args_.world_size, &ar_args_.rank, &ar_args_.packed_array_num,
      &ar_args_.rank_data, &ar_args_.rank_signals, &ar_args_.self_signal);

#else
    // 用于试验测试
    RtCommArguments *rt_args = (RtCommArguments*)(fusion_args);
    ar_args_.reg_buffer = reinterpret_cast<void*>(rt_args->reg_buffer);
    ar_args_.world_size = 2;
    ar_args_.rank = 1;
    printf("finish malloc.\n");
#endif
  }

private:
  DeviceGemmBasic gemm_dev_;

  AllReduceArguments ar_args_;

  cudaEvent_t event_;      // 需要跟前一次关联，不能临时创建
  cudaStream_t rs_stream_;
  int m_;
  int n_;
  int output_len_;
};

} // namespace xop