#pragma once
#include <cfloat>
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/common_cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

namespace xop {

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90GroupParams<cute::Shape<int,int,int>>::RasterOrderOptions;
template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, class TileScheduler, class TileShape,  class ClusterShape, 
          RasterOrderOptions RasterOrder, int Swizzle>
class GemmGroupedBlockScaleFp8Sm90Impl : public GemmBase {
public:
  ////
  using         ElementD    = ElementC;
  using         LayoutD     = LayoutC;
  // Auxiliary matrix configuration and other fusion types 
  using         ElementAux   = ElementC;
  using         LayoutAux    = LayoutC;
  using         ElementAmax  = float;
  using         ElementBias  = float;

  // Core kernel configurations
  using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
  using ElementCompute      = float;                                          // Element type for epilogue computation
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag                        // Threadblock-level tile size
  ////

  using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>; // <m,n,k> => scaleA[m, (k+127)//128], scaleB[(n+127)//128, (k+127)//128]
  using LayoutSFA     = decltype(ScaleConfig::deduce_layoutSFA());    // Layout type for SFA matrix operand
  using LayoutSFB     = decltype(ScaleConfig::deduce_layoutSFB());    // Layout type for SFB matrix operand

  using MainloopScheduleType    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
  using EpilogueScheduleType    = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  
  using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;
  using FusionOperation     = cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,  // Shape of the threadblocks in a cluster
      EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC *, 128 / cutlass::sizeof_bits<ElementC>::value,
      ElementD, LayoutD *, 128 / cutlass::sizeof_bits<ElementD>::value,
      EpilogueScheduleType,
      FusionOperation
    >::CollectiveOp;
  using CollectiveMainloopWithGroupWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, cute::tuple<LayoutA *, LayoutSFA *>, 128 / cutlass::sizeof_bits<ElementA>::value,
      ElementB, cute::tuple<LayoutB *, LayoutSFB *>, 128 / cutlass::sizeof_bits<ElementB>::value,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
      >,
      MainloopScheduleType
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::GroupProblemShape<cute::Shape<int,int,int>>, // <M,N,K> per group
      CollectiveMainloopWithGroupWiseScaling,
      CollectiveEpilogue
  >;

  // using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  //     cutlass::gemm::GroupProblemShape<cute::Shape<int,int,int>>; // <M,N,K> per group
  //     CollectiveMainloopWithBlockWiseScaling,
  //     CollectiveEpilogue,
  //     TileScheduler // cutlass::gemm::PersistentScheduler cutlass::gemm::StreamKScheduler
  // >;
  
  // CORE
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  //
  using ElementScalar     = typename Gemm::EpilogueOutputOp::ElementScalar;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int,int,int>>; // <M,N,K> per group
  using UlyProblemShape = typename ProblemShape::UnderlyingProblemShape;

public:
  void initialize(RtArguments *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtGroupedBlockScaleFp8ArgumentsV3 *rt_args = dynamic_cast<RtGroupedBlockScaleFp8ArgumentsV3*>(args);

    static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
      "ElementAccumulator and ElementBlockScale should be same datatype");

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
  typename Gemm::Arguments args_from_options(const RtGroupedBlockScaleFp8ArgumentsV3 *rt_args, void *stream)
  {
    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    int device_id = 0;
    cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(device_id);
  
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

    UlyProblemShape *problem_sizes = (UlyProblemShape *)(device_buffer + offsets[0]);
    StrideA *stride_A = (StrideA *)(device_buffer + offsets[1]);
    StrideB *stride_B = (StrideB *)(device_buffer + offsets[2]);
    StrideC *stride_C = (StrideC *)(device_buffer + offsets[3]);
    StrideD *stride_D = (StrideD *)(device_buffer + offsets[4]);
    LayoutSFA *layout_SFA = (LayoutSFA *)(device_buffer + offsets[5]);
    LayoutSFB *layout_SFB = (LayoutSFB *)(device_buffer + offsets[6]);

    const ElementA **ptr_A = (const ElementA **)(device_buffer + offsets[7]);
    const ElementB **ptr_B = (const ElementB **)(device_buffer + offsets[8]);
    const ElementC **ptr_C = (const ElementC **)(device_buffer + offsets[9]);
    ElementD **ptr_D = (ElementD **)(device_buffer + offsets[10]);
    const ElementBlockScale **ptr_blockscale_A = (const ElementBlockScale **)(device_buffer + offsets[11]);
    const ElementBlockScale **ptr_blockscale_B = (const ElementBlockScale **)(device_buffer + offsets[12]);

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {rt_args->groups, problem_sizes, problem_sizes_host.data()},
      {ptr_A, stride_A, ptr_B, stride_B,
       ptr_blockscale_A, layout_SFA,
       ptr_blockscale_B, layout_SFB 
      },
      {
        {}, // epilogue.thread
        ptr_C, stride_C,
        ptr_D, stride_D
      },
      kernel_hw_info
    };

    // CUDA_CHECK(cudaStreamSynchronize(cu_stream));
    // printf("hello.\n");
    // static bool is_inited = false;
    // if (!is_inited) {
    //   is_inited = true;  
    //   problem_sizes_host.reserve(rt_args->groups);
    //   for (int i=0; i<rt_args->problem_sizes.size(); i++) {
    //     auto m = rt_args->problem_sizes[i*3+0];
    //     auto n = rt_args->problem_sizes[i*3+1];
    //     auto k = rt_args->problem_sizes[i*3+2];
    //     problem_sizes_host.push_back({m,n,k});
    //   }
      
    //   problem_sizes.reset(rt_args->groups);
    //   problem_sizes.copy_from_host(problem_sizes_host.data());

    //   std::vector<StrideA> stride_A_host;
    //   std::vector<StrideB> stride_B_host;
    //   std::vector<StrideC> stride_C_host;
    //   std::vector<StrideD> stride_D_host;
    //   for (int32_t i = 0; i < rt_args->groups; ++i) {
    //     auto problem = problem_sizes_host.at(i);
    //     auto M = cute::get<0>(problem);
    //     auto N = cute::get<1>(problem);
    //     auto K = cute::get<2>(problem);
    //     stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    //     stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    //     stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    //     stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));      
    //   }
    //   stride_A.reset(rt_args->groups);
    //   stride_A.copy_from_host(stride_A_host.data());
    //   stride_B.reset(rt_args->groups);
    //   stride_B.copy_from_host(stride_B_host.data());
    //   stride_C.reset(rt_args->groups);
    //   stride_C.copy_from_host(stride_C_host.data());
    //   stride_D.reset(rt_args->groups);
    //   stride_D.copy_from_host(stride_D_host.data());
    //   ///
    //   ptr_A.reset(rt_args->groups);
    //   ptr_A.copy_from_host((const ElementA **)rt_args->ptr_A.data());
    //   ptr_B.reset(rt_args->groups);
    //   ptr_B.copy_from_host((const ElementB **)rt_args->ptr_B.data());
    //   ptr_C.reset(rt_args->groups);
    //   ptr_C.copy_from_host((const ElementC **)rt_args->ptr_C.data());
    //   ptr_D.reset(rt_args->groups);
    //   ptr_D.copy_from_host((ElementD **)rt_args->ptr_D.data());
    //   ptr_blockscale_A.reset(rt_args->groups);
    //   ptr_blockscale_A.copy_from_host((const float **)rt_args->ptr_blockscale_A.data());
    //   ptr_blockscale_B.reset(rt_args->groups);
    //   ptr_blockscale_B.copy_from_host((const float **)rt_args->ptr_blockscale_B.data());
    // }
    

    // typename Gemm::Arguments arguments{
    //   cutlass::gemm::GemmUniversalMode::kGrouped,
    //   {rt_args->groups, problem_sizes.get(), problem_sizes_host.data()},
    //   {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(),
    //    ptr_blockscale_A.get(), // blockscale_tensor_A.device_data(),
    //    ptr_blockscale_B.get(), // blockscale_tensor_B.device_data()
    //   },
    //   {
    //     {}, // epilogue.thread
    //     ptr_C.get(), stride_C.get(),
    //     ptr_D.get(), stride_D.get()
    //   },
    //   kernel_hw_info
    // };

    auto &fusion_args = arguments.epilogue.thread;
    if (rt_args->alpha != FLT_MAX && rt_args->beta != FLT_MAX) {
      // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
      fusion_args.alpha = rt_args->alpha;
      fusion_args.beta = rt_args->beta;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = nullptr;
      fusion_args.beta_ptr_array = nullptr;
      // Single alpha and beta for all groups
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    }
    // else {
    //   // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
    //   fusion_args.alpha = 0;
    //   fusion_args.beta = 0;
    //   fusion_args.alpha_ptr = nullptr;
    //   fusion_args.beta_ptr = nullptr;
    //   fusion_args.alpha_ptr_array = alpha_device.get();
    //   fusion_args.beta_ptr_array = beta_device.get();
    //   // One alpha and beta per each group
    //   fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    //   fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
    // }

    arguments.scheduler.raster_order = RasterOrder;
    // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
    arguments.scheduler.max_swizzle_size = Swizzle;

    return arguments;
  }

  void get_buffer_info(const RtGroupedBlockScaleFp8ArgumentsV3 *rt_args, std::vector<int> &sizes, std::vector<int> &offsets, int &total_size) {
    int groups = rt_args->groups;
    int problem_size_size = groups * 3 * sizeof(int);
    int problem_size_offset = 0;

    int stride_a_size = groups * sizeof(StrideA);
    int stride_b_size = groups * sizeof(StrideB);
    int stride_c_size = groups * sizeof(StrideC);
    int stride_d_size = groups * sizeof(StrideD);
    int layout_SFA_size = groups * sizeof(LayoutSFA);
    int layout_SFB_size = groups * sizeof(LayoutSFB);

    int stride_a_offset = problem_size_offset + problem_size_size;
    int stride_b_offset = stride_a_offset + stride_a_size;
    int stride_c_offset = stride_b_offset + stride_b_size;
    int stride_d_offset = stride_c_offset + stride_c_size;
    int layout_SFA_offset = stride_d_offset + stride_d_size;
    int layout_SFB_offset = layout_SFA_offset + layout_SFA_size;

    int ptr_a_size = groups * sizeof(ElementA *);
    int ptr_b_size = groups * sizeof(ElementB *);
    int ptr_c_size = groups * sizeof(ElementC *);
    int ptr_d_size = groups * sizeof(ElementD *);
    int ptr_scale_a_size = groups * sizeof(ElementBlockScale *);
    int ptr_scale_b_size = groups * sizeof(ElementBlockScale *);

    int ptr_a_offset = layout_SFB_offset + layout_SFB_size;
    int ptr_b_offset = ptr_a_offset + ptr_a_size;
    int ptr_c_offset = ptr_b_offset + ptr_b_size;
    int ptr_d_offset = ptr_c_offset + ptr_c_size;
    int ptr_scale_a_offset = ptr_d_offset + ptr_d_size;
    int ptr_scale_b_offset = ptr_scale_a_offset + ptr_scale_a_size;

    int len = 13;
    sizes.resize(len);
    sizes[0] = problem_size_size;
    sizes[1] = stride_a_size;       sizes[2] = stride_b_size;
    sizes[3] = stride_c_size;       sizes[4] = stride_d_size;
    sizes[5] = layout_SFA_size;     sizes[6] = layout_SFB_size;

    sizes[7] = ptr_a_size;          sizes[8] = ptr_b_size;
    sizes[9] = ptr_c_size;          sizes[10] = ptr_d_size;
    sizes[11] = ptr_scale_a_size;   sizes[12] = ptr_scale_b_size;

    offsets.resize(len);
    offsets[0] = 0;
    offsets[1] = stride_a_offset;      offsets[2] = stride_b_offset;
    offsets[3] = stride_c_offset;      offsets[4] = stride_d_offset;
    offsets[5] = layout_SFA_offset;    offsets[6] = layout_SFB_offset;

    offsets[7] = ptr_a_offset;         offsets[8] = ptr_b_offset;
    offsets[9] = ptr_c_offset;         offsets[10] = ptr_d_offset;
    offsets[11] = ptr_scale_a_offset;  offsets[12] = ptr_scale_b_offset;  

    total_size = 0;
    for (int i=0; i<len; i++)
      total_size += sizes[i];
  }

  void cpy_args2host_buffer(const RtGroupedBlockScaleFp8ArgumentsV3 *rt_args, std::vector<int> &sizes, std::vector<int> &offsets, uint8_t *host_buffer) {
    problem_sizes_host.clear();
    stride_A_host.clear();
    stride_B_host.clear();
    stride_C_host.clear();
    stride_D_host.clear();
    layout_SFA_host.clear();
    layout_SFB_host.clear();
    for (int i=0; i<rt_args->groups; i++) {
      auto m = rt_args->problem_sizes[i*3+0];
      auto n = rt_args->problem_sizes[i*3+1];
      auto k = rt_args->problem_sizes[i*3+2];
      problem_sizes_host.push_back({m,n,k});

      stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}));
      stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
      stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}));
      stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}));    
      
      layout_SFA_host.push_back(ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1)));
      layout_SFB_host.push_back(ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1)));
    }

    memcpy(host_buffer + offsets[0], problem_sizes_host.data(), sizes[0]);
    memcpy(host_buffer + offsets[1], stride_A_host.data(), sizes[1]);
    memcpy(host_buffer + offsets[2], stride_B_host.data(), sizes[2]);
    memcpy(host_buffer + offsets[3], stride_C_host.data(), sizes[3]);
    memcpy(host_buffer + offsets[4], stride_D_host.data(), sizes[4]);
    memcpy(host_buffer + offsets[5], layout_SFA_host.data(), sizes[5]); //
    memcpy(host_buffer + offsets[6], layout_SFB_host.data(), sizes[6]);

    memcpy(host_buffer + offsets[7], rt_args->ptr_A.data(), sizes[7]);
    memcpy(host_buffer + offsets[8], rt_args->ptr_B.data(), sizes[8]);
    memcpy(host_buffer + offsets[9], rt_args->ptr_C.data(), sizes[9]);
    memcpy(host_buffer + offsets[10], rt_args->ptr_D.data(), sizes[10]);
    memcpy(host_buffer + offsets[11], rt_args->ptr_blockscale_A.data(), sizes[11]);
    memcpy(host_buffer + offsets[12], rt_args->ptr_blockscale_B.data(), sizes[12]);
  }

private:
  Gemm gemm_dev_;

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;
  std::vector<LayoutSFA> layout_SFA_host;
  std::vector<LayoutSFB> layout_SFB_host;

  // cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

  // cutlass::DeviceAllocation<StrideA> stride_A;
  // cutlass::DeviceAllocation<StrideB> stride_B;
  // cutlass::DeviceAllocation<StrideC> stride_C;
  // cutlass::DeviceAllocation<StrideD> stride_D;

  // cutlass::DeviceAllocation<const ElementA *> ptr_A;
  // cutlass::DeviceAllocation<const ElementB *> ptr_B;
  // cutlass::DeviceAllocation<const ElementC *> ptr_C;
  // cutlass::DeviceAllocation<ElementD *> ptr_D;
  // cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_A;
  // cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_B;
};

} // namespace xop