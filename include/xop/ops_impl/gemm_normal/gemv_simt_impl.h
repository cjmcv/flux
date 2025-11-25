#pragma once
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/common_cutlass.h"

#include "cutlass/gemm/kernel/gemv.h"
#include "cutlass/gemm/device/gemv.h"


// GEMV for row-major A matrix


namespace xop {

using namespace cutlass;
namespace gemm {

namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemvKernel_>
class XopGemv {
public:

  using GemvKernel = GemvKernel_;


  using ElementA = typename GemvKernel::ElementA;
  using LayoutA  = typename GemvKernel::LayoutA;
  using ElementB = typename GemvKernel::ElementB;
  using ElementC = typename GemvKernel::ElementC;

  using ElementAccumulator = typename GemvKernel::ElementAccumulator;
  using EpilogueOutputOp = typename GemvKernel::EpilogueOutputOp;

  static ComplexTransform const kTransformA = GemvKernel::kTransformA;
  static ComplexTransform const kTransformB = GemvKernel::kTransformB;

  static int const kThreadCount = GemvKernel::kThreadCount;
  static int const kThreadsPerRow = GemvKernel::kThreadsPerRow;

  using Arguments = typename GemvKernel::Arguments;
  using Params = typename GemvKernel::Params;

private:

  Params params_;
  
public:

  /// Constructs the Gemv.
  XopGemv() { }

  /// Determines whether the Gemv can execute the given problem.
  static Status can_implement(Arguments const &args) {

    return GemvKernel::can_implement(args);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    return 0;
  }

  /// Initializes Gemv state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
    params_ = Params(args);
    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {
    dim3 block = dim3(kThreadsPerRow, kThreadCount / kThreadsPerRow, 1); // (16, 8£¬ 1)
    dim3 grid = dim3((params_.problem_size.row() + (block.y - 1)) / block.y, 1, params_.splitk_num); // (m//8, 1£¬ 1)
    
    int smem_size = int(sizeof(typename GemvKernel::SharedStorage));
    
    // Launch
    cutlass::arch::synclog_setup();
    cutlass::Kernel<GemvKernel><<<grid, block, smem_size, stream>>>(params_);

    //
    // Query for errors
    //
    cudaError_t result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

} // namespace device

////////////////////////////////////////////////////////////////////////////////

namespace kernel {

template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename ElementC_,
  typename ElementAccumulator_,
  typename EpilogueOutputOp_,
  int kElementsPerAccess_ = 1,            ///< Number of elements involved in a global access.
  int kThreadCount_ = 0,                  ///< Number of threads in the thread block.
                                          ///  It will be calculated automatically if set to 0.
  int kThreadsPerRow_ = 0                 ///< Number of threads in the k dimension.
                                          ///  It will be calculated automatically if set to 0.
>
struct XopGemv;


template <
    typename ElementA_,
    typename ElementB_,
    typename ElementC_,
    typename ElementAccumulator_,
    typename EpilogueOutputOp_,
    int kElementsPerAccess_,
    int kThreadCount_,
    int kThreadsPerRow_ 
>
struct XopGemv <
    ElementA_,            
    layout::RowMajor,
    ElementB_,            
    ElementC_,
    ElementAccumulator_,
    EpilogueOutputOp_,
    kElementsPerAccess_,
    kThreadCount_,
    kThreadsPerRow_
>{
public:

  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  using ElementB = ElementB_;
  using ElementC = ElementC_;

  using ElementAccumulator = ElementAccumulator_;
  using EpilogueOutputOp = EpilogueOutputOp_;

  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  static FloatRoundStyle const Round = cutlass::FloatRoundStyle::round_to_nearest;

  // number of return elements in a global access
  static int const kElementsPerAccess = kElementsPerAccess_;
  
  using FragmentA = Array<ElementA, kElementsPerAccess>;
  using FragmentB = Array<ElementB, kElementsPerAccess>;
  using FragmentCompute = Array<ElementAccumulator, kElementsPerAccess>;

  // thread block shape (kThreadsPerRow, kThreadCount / kThreadsPerRow, 1)
  static int const kThreadCount = (kThreadCount_ <= 0) ? 128 : kThreadCount_;
  static int const kThreadsPerRow = (kThreadsPerRow_ <= 0) ?
                                  std::min(static_cast<int>(kThreadCount / (kElementsPerAccess * sizeof(ElementA))), 16)
                                  : kThreadsPerRow_;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    MatrixCoord     problem_size;
    int32_t         batch_count;
    int32_t         splitk_num;

    TensorRefA      ref_A;

    ElementB const *ptr_B;
    ElementC const *ptr_C;
    ElementC       *ptr_D;

    int64_t         batch_stride_A;
    int64_t         batch_stride_B;
    int64_t         batch_stride_C;
    int64_t         batch_stride_D;
    //
    // Methods
    //

    Arguments(): batch_count(0) { }

    Arguments(
      MatrixCoord problem_size,
      int32_t     batch_count,
      int32_t     splitk_num,
      TensorRefA  ref_A,
      void const *ptr_B,
      void const *ptr_C,
      void       *ptr_D,
      int64_t     batch_stride_A,
      int64_t     batch_stride_B,
      int64_t     batch_stride_C,
      int64_t     batch_stride_D
    ):
      problem_size(problem_size),
      batch_count(batch_count),
      splitk_num(splitk_num),
      ref_A(ref_A),
      ptr_B(static_cast<ElementB const *>(ptr_B)),
      ptr_C(static_cast<ElementC const *>(ptr_C)),
      ptr_D(static_cast<ElementC       *>(ptr_D)),
      batch_stride_A(batch_stride_A),
      batch_stride_B(batch_stride_B),
      batch_stride_C(batch_stride_C),
      batch_stride_D(batch_stride_D)
    { }

    Arguments(
      MatrixCoord problem_size,
      TensorRefA  ref_A,
      void const *ptr_B,
      void const *ptr_C,
      void       *ptr_D
    ):
      Arguments(
        problem_size,
        1,
        1,
        ref_A,
        ptr_B,
        ptr_C,
        ptr_D,
        1,
        1,
        1,
        1)
    { }
  };

  using Params = Arguments;

  /// Shared memory storage structure
  union SharedStorage {

  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  XopGemv() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(Arguments const &args) {
    if (args.problem_size.column() % kElementsPerAccess != 0) {
      return Status::kErrorMisalignedOperand;
    }
    return Status::kSuccess;
  }

  /// Executes one GEMV
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    
    // block: (16, 8, 1) => 128 threads
    // grid:  (m//8, 1, 1)
    // kElementsPerAccess: 8 => one thread for 8 data
    // kThreadsPerRow: 16

    // Loop over batch indices
    for (int batch_idx = blockIdx.z; batch_idx < params.splitk_num; batch_idx += gridDim.z) {
      // if (batch_idx == 0) return;
      // printf("batch_stride_D: %d", params.batch_stride_D);
      int idx_col_k = threadIdx.x;
      int idx_row_m = blockIdx.x * blockDim.y + threadIdx.y;

      if (idx_row_m < params.problem_size.row()) {
        // problem_size (row = m, column = k)
        // matrix A (batch, m, k)
        // vector B (batch, 1, k)
        // vector C (batch, m, 1)
        // vector D (batch, m, 1)

        // move in the batch dimension
        ElementA const *ptr_A = params.ref_A.data() + batch_idx * params.problem_size.column() / params.splitk_num;
        ElementB const *ptr_B = params.ptr_B + batch_idx * params.problem_size.column() / params.splitk_num;

        // ElementC const *ptr_C = params.ptr_C + batch_idx * params.batch_stride_C;
        ElementC *ptr_D = params.ptr_D + batch_idx * params.batch_stride_D;

        // move in the k dimension
        ptr_A += idx_col_k * kElementsPerAccess;
        ptr_B += idx_col_k * kElementsPerAccess;

        // move in the m dimension
        ptr_A += idx_row_m * params.problem_size.column();
        // ptr_C += idx_row_m;
        ptr_D += idx_row_m;

        NumericArrayConverter<ElementAccumulator, ElementA, kElementsPerAccess, Round> srcA_converter;
        NumericArrayConverter<ElementAccumulator, ElementB, kElementsPerAccess, Round> srcB_converter;

        ElementAccumulator accum = 0.f;

        FragmentB fragB;
        FragmentA fragA;

        // rows of the rolling tile
        int const tileA_k = kThreadsPerRow * kElementsPerAccess;
        
        int tile_align_col = params.problem_size.column() / params.splitk_num / tileA_k * tileA_k;
        int unroll_col_k = 0; // batch_idx * tile_align_col;

        // printf("(%d,%d, %d,%d)", unroll_col_k, tile_align_col, tileA_k, params.splitk_num);
        // for (; unroll_col_k < params.problem_size.column() / tileA_k * tileA_k; unroll_col_k += tileA_k) {
        for (; unroll_col_k < tile_align_col; unroll_col_k += tileA_k) {

          // fetch from matrix A
          arch::global_load<FragmentA,
                            sizeof(FragmentA),
                            arch::CacheOperation::LastUse>(fragA, (ptr_A + unroll_col_k), true);

          // fetch from vector B
          arch::global_load<FragmentB,
                            sizeof(FragmentB),
                            arch::CacheOperation::Always>(fragB, (ptr_B + unroll_col_k), true);

          FragmentCompute fragB_Compute = srcB_converter(fragB);
          FragmentCompute fragA_Compute = srcA_converter(fragA);

          // Math
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < kElementsPerAccess; e++) {
            accum += fragA_Compute.at(e) * fragB_Compute.at(e);
          }
        }

        // calculate the rest of K elements
        // each thread fetch 1 element each time
        // for (int k = unroll_col_k + idx_col_k; k < params.problem_size.column(); k += kThreadsPerRow) {
        for (int k = unroll_col_k + idx_col_k; k < params.problem_size.column() / params.splitk_num; k += kThreadsPerRow) {
          ElementB b = *(ptr_B - idx_col_k * kElementsPerAccess + k);
          ElementA a = *(ptr_A - idx_col_k * kElementsPerAccess + k);

          accum += ElementAccumulator(a) * ElementAccumulator(b);
        }

        for (int mask = (kThreadsPerRow >> 1); mask > 0; mask >>= 1) {
          accum += __shfl_xor_sync(0xFFFFFFFF, accum, mask, 32);
        }

        if (idx_col_k == 0) {
          *ptr_D = (ElementC)accum;
        }
      }
    }
  }
};

template <class T>
__global__
void ElementWiseAdd(T *out, const T *in, int n, int splitk_num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  T accum = T(0);
  for (int s=0; s<splitk_num; s++) {
    const T *sin = in + s * n;
    accum += sin[i];
  }
  if (i < n) 
    out[i] = accum;
}

} // namespace kernel
} // namespace gemm

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, 
          class ArchTag, int SplitKNum,
          int ElementsPerAccess, int ThreadCount, int ThreadsPerRow>
class GemvSimtImpl : public GemmBase {

  using EpilogueOpIdentity = cutlass::epilogue::thread::Identity<ElementC>;

  using Gemv = gemm::device::XopGemv<
    gemm::kernel::XopGemv<
      ElementA,               // Element A
      LayoutA,                // Layout A
      ElementB,               // Element B
      ElementC,               // Element C
      ElementAccumulator,     // Element accumulator
      EpilogueOpIdentity,     // Output operator
      ElementsPerAccess,       // Element access granularity
      ThreadCount
    >
  >;

  using TensorRefA = cutlass::TensorRef<ElementA, LayoutA>;

public:
  typename Gemv::Arguments args_from_options(RtArgumentsV2 *rt_args) {
    // gemm: A[m=1,k] * B[n,k] = C[m,n]
    // gemv: weight A[m,k] * input B[1,k] = C[1,m] row major
    // so: B[n,k] => A[m,k]
    void *t = rt_args->ptr_A;
    rt_args->ptr_A = rt_args->ptr_B;
    rt_args->ptr_B = t;
    if constexpr (cute::is_same_v<LayoutA, cutlass::layout::RowMajor>) {
      rt_args->m = rt_args->n;
      rt_args->n = 1;
    }
    else {
      assert(0); // not ready.
      rt_args->m = rt_args->k;
      rt_args->k = rt_args->n;
      rt_args->n = 1;
    }

    LayoutA A_layout(rt_args->k);
    TensorRefA ref_a = cutlass::TensorRef((ElementA *)rt_args->ptr_A, A_layout);

    // printf("mk: %d, %d, %d, %d\n", rt_args->m, rt_args->k, rt_args->l, ref_a.stride(0));
    cutlass::MatrixCoord problem_size = {rt_args->m, rt_args->k};
    return typename Gemv::Arguments {
      problem_size,
      rt_args->l,                      // batch_count
      SplitKNum,                               // splitk
      ref_a,
      (ElementB *)rt_args->ptr_B,
      (ElementC *)rt_args->ptr_C,
      (ElementC *)splitk_out_,         // rt_args->ptr_D,
      rt_args->m * rt_args->k,         // batch_stride_A
      rt_args->k,                      // batch_stride_B
      rt_args->m,                      // batch_stride_C
      rt_args->m                       // batch_stride_D   
    };
  }

  void initialize(RtArgumentsBase *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtArgumentsV2 *rt_args = static_cast<RtArgumentsV2*>(args);
    gemv_ = Gemv();
    
    size_t workspace_size = rt_args->n * SplitKNum * sizeof(ElementC);
    splitk_out_ = GlobalBuffer::instance().GetDeviceBuffer(kDevBufferPoolWorkspace, workspace_size);

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    auto arguments = args_from_options(rt_args);

    // Allocate workspace memory
    auto cu_stream = static_cast<cudaStream_t>(stream);
    // problem_size[m,k], output: [1,m]

    // Check the problem size is supported or not
    CUTLASS_CHECK(gemv_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemv_.initialize(arguments, nullptr, cu_stream));

    output_len_ = rt_args->m;
    ptr_D_ = (void *)rt_args->ptr_D; // [hardcode]
    workspace_size_ = workspace_size;
  }

  void run(void *stream = nullptr) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemv_.run(cu_stream));

    int threads = 256;
    int blocks  = (output_len_ + threads - 1) / threads;
    // printf("SplitKNum: %d.\n", SplitKNum);
    gemm::kernel::ElementWiseAdd<<<blocks, threads, 0, cu_stream>>>((ElementC*)ptr_D_, (ElementC*)splitk_out_, output_len_, SplitKNum);
    // cudaMemcpyAsync(ptr_D_, splitk_out_, workspace_size_, cudaMemcpyDeviceToDevice, cu_stream);
  }

private:
  Gemv gemv_;
  void *splitk_out_;
  void *ptr_D_;
  size_t output_len_;
  size_t workspace_size_;
};

} // namespace xop