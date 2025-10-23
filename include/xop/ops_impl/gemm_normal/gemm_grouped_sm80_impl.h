#pragma once
#include <cfloat>
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/common_cutlass.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

namespace xop {

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, 
          class ThreadblockShape, class WarpShape, class InstructionShape,  
          int NumStages> // enum class GroupScheduleMode, 
class GemmGroupedSm80Impl : public GemmBase {
public:
  using         ElementD    = ElementC;
  using         LayoutD     = LayoutC;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,               // Element type for C and D matrix operands
    128 / cutlass::sizeof_bits<ElementC>::value, // Memory access granularity of C and D matrix in units of elements
    ElementAccumulator,     // Element type from internal accumaccumulation
    ElementAccumulator>;     // Data type used to compute linear combination

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    ArchTag,           // cutlass::arch::Sm80,
    ThreadblockShape,  // cutlass::gemm::GemmShape<128, 128, 32>,
    WarpShape,         //cutlass::gemm::GemmShape<64, 64, 32>,
    InstructionShape,  // cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
    // This parameter is passed in at present to match the APIs of other kernels. The parameter
    // is unused within the kernel.
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    NumStages,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

public:
  void initialize(RtArgumentsBase *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtGroupedArguments *rt_args = static_cast<RtGroupedArguments*>(args);

    // Instantiate CUTLASS kernel depending on templates
    gemm_dev_ = Gemm();

    // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
    auto arguments = args_from_options(rt_args, stream);

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    void *workspace_ptr = GlobalBuffer::instance().ResizeDeviceBufferIfNeeded(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace_ptr));
  }

  void run(void *stream = nullptr) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemm_dev_.run(cu_stream));
  }

private:
  typename Gemm::Arguments args_from_options(const RtGroupedArguments *rt_args, void *stream)
  {
    /// 
    std::vector<int> sizes;
    std::vector<int> offsets;
    int total_size;
    get_buffer_info(rt_args, sizes, offsets, total_size);
    uint8_t *host_buffer = GlobalBuffer::instance().ResizeHostBufferIfNeeded(total_size);
    uint8_t *device_buffer = GlobalBuffer::instance().ResizeDeviceBuffer2IfNeeded(total_size);

    cpy_args2host_buffer(rt_args, sizes, offsets, host_buffer);

    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaMemcpyAsync(device_buffer, host_buffer, total_size, cudaMemcpyHostToDevice, cu_stream));

    cutlass::gemm::GemmCoord *problem_sizes = (cutlass::gemm::GemmCoord *)(device_buffer + offsets[0]);
    int64_t *stride_A = (int64_t *)(device_buffer + offsets[1]);
    int64_t *stride_B = (int64_t *)(device_buffer + offsets[2]);
    int64_t *stride_C = (int64_t *)(device_buffer + offsets[3]);
    int64_t *stride_D = (int64_t *)(device_buffer + offsets[4]);

    ElementA **ptr_A = (ElementA **)(device_buffer + offsets[5]);
    ElementB **ptr_B = (ElementB **)(device_buffer + offsets[6]);
    ElementC **ptr_C = (ElementC **)(device_buffer + offsets[7]);
    ElementD **ptr_D = (ElementD **)(device_buffer + offsets[8]);

    int threadblock_count = Gemm::sufficient(problem_sizes_host.data(), rt_args->groups);
    typename Gemm::EpilogueOutputOp::Params epilogue_op(rt_args->alpha, rt_args->beta);
    typename Gemm::Arguments arguments(
      problem_sizes,
      rt_args->groups,
      threadblock_count,
      epilogue_op,
      ptr_A,
      ptr_B,
      ptr_C,
      ptr_D,
      stride_A,
      stride_B,
      stride_C,
      stride_D,
      problem_sizes_host.data()
    );
    return arguments;
  }

  void get_buffer_info(const RtGroupedArguments *rt_args, std::vector<int> &sizes, std::vector<int> &offsets, int &total_size) {
    int groups = rt_args->groups;
    int problem_size_size = groups * 3 * sizeof(int);
    int problem_size_offset = 0;

    int stride_a_size = groups * sizeof(int64_t);
    int stride_b_size = groups * sizeof(int64_t);
    int stride_c_size = groups * sizeof(int64_t);
    int stride_d_size = groups * sizeof(int64_t);

    int stride_a_offset = problem_size_offset + problem_size_size;
    int stride_b_offset = stride_a_offset + stride_a_size;
    int stride_c_offset = stride_b_offset + stride_b_size;
    int stride_d_offset = stride_c_offset + stride_c_size;

    int ptr_a_size = groups * sizeof(ElementA *);
    int ptr_b_size = groups * sizeof(ElementB *);
    int ptr_c_size = groups * sizeof(ElementC *);
    int ptr_d_size = groups * sizeof(ElementD *);

    int ptr_a_offset = stride_d_offset + stride_d_size;
    int ptr_b_offset = ptr_a_offset + ptr_a_size;
    int ptr_c_offset = ptr_b_offset + ptr_b_size;
    int ptr_d_offset = ptr_c_offset + ptr_c_size;

    int len = 9;
    sizes.resize(len);
    sizes[0] = problem_size_size;
    sizes[1] = stride_a_size;       sizes[2] = stride_b_size;
    sizes[3] = stride_c_size;       sizes[4] = stride_d_size;
    sizes[5] = ptr_a_size;          sizes[6] = ptr_b_size;
    sizes[7] = ptr_c_size;          sizes[8] = ptr_d_size;

    offsets.resize(len);
    offsets[0] = 0;
    offsets[1] = stride_a_offset;      offsets[2] = stride_b_offset;
    offsets[3] = stride_c_offset;      offsets[4] = stride_d_offset;
    offsets[5] = ptr_a_offset;         offsets[6] = ptr_b_offset;
    offsets[7] = ptr_c_offset;         offsets[8] = ptr_d_offset; 

    total_size = 0;
    for (int i=0; i<len; i++)
      total_size += sizes[i];
  }

  void cpy_args2host_buffer(const RtGroupedArguments *rt_args, std::vector<int> &sizes, std::vector<int> &offsets, uint8_t *host_buffer) {
    problem_sizes_host.clear();
    stride_A_host.clear();
    stride_B_host.clear();
    stride_C_host.clear();
    stride_D_host.clear();
    for (int i=0; i<rt_args->groups; i++) {
      auto m = rt_args->problem_sizes[i*3+0];
      auto n = rt_args->problem_sizes[i*3+1];
      auto k = rt_args->problem_sizes[i*3+2];
      problem_sizes_host.push_back({m,n,k});

      stride_A_host.push_back(k);
      stride_B_host.push_back(k);
      stride_C_host.push_back(n);
      stride_D_host.push_back(n);
    }

    memcpy(host_buffer + offsets[0], problem_sizes_host.data(), sizes[0]);
    memcpy(host_buffer + offsets[1], stride_A_host.data(), sizes[1]);
    memcpy(host_buffer + offsets[2], stride_B_host.data(), sizes[2]);
    memcpy(host_buffer + offsets[3], stride_C_host.data(), sizes[3]);
    memcpy(host_buffer + offsets[4], stride_D_host.data(), sizes[4]);
    memcpy(host_buffer + offsets[5], rt_args->ptr_A.data(), sizes[5]);
    memcpy(host_buffer + offsets[6], rt_args->ptr_B.data(), sizes[6]);
    memcpy(host_buffer + offsets[7], rt_args->ptr_C.data(), sizes[7]);
    memcpy(host_buffer + offsets[8], rt_args->ptr_D.data(), sizes[8]);
  }

private:
  Gemm gemm_dev_;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes_host;

  std::vector<int64_t> stride_A_host;
  std::vector<int64_t> stride_B_host;
  std::vector<int64_t> stride_C_host;
  std::vector<int64_t> stride_D_host;
};

} // namespace xop