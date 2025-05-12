#pragma once
#include "ctlop/ops_impl/global_resource.h"
#include "ctlop/ops_impl/args_util.h"

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

namespace ctlop {

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90GroupParams<cute::Shape<int,int,int>>::RasterOrderOptions;
template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, class TileScheduler, class TileShape,  class ClusterShape, 
          RasterOrderOptions RasterOrder, int Swizzle>
class GemmGroupedBlockScaleFp8Impl : public GemmBase {
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

  using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum<1, 128>;
  using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  
  using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;
  using FusionOperation   = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,  // Shape of the threadblocks in a cluster
      EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC *, 128 / cutlass::sizeof_bits<ElementC>::value,
      ElementD, LayoutD *, 128 / cutlass::sizeof_bits<ElementD>::value,
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;
  
  using CollectiveMainloopWithBlockWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA *, 128 / cutlass::sizeof_bits<ElementA>::value,
      ElementB, LayoutB *, 128 / cutlass::sizeof_bits<ElementB>::value,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
      >,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::GroupProblemShape<cute::Shape<int,int,int>>, // <M,N,K> per group
      CollectiveMainloopWithBlockWiseScaling,
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

public:
  void initialize(RtArguments *args, void *stream = nullptr) {
    RtGroupedBlockScaleFp8ArgumentsV3 *rt_args = dynamic_cast<RtGroupedBlockScaleFp8ArgumentsV3*>(args);

    static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
      "ElementAccumulator and ElementBlockScale should be same datatype");

    // Instantiate CUTLASS kernel depending on templates
    gemm_dev_ = Gemm();

    // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
    auto arguments = args_from_options(rt_args);

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    void *workspace_ptr = GlobalBuffer::instance().ResizeBufferIfNeeded(workspace_size);

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
  typename Gemm::Arguments args_from_options(const RtGroupedBlockScaleFp8ArgumentsV3 *rt_args)
  {
    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    int device_id = 0;
    cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(device_id);
  
    /// 
    static bool is_inited = false;
    if (!is_inited) {
      is_inited = true;
      problem_sizes_host.reserve(rt_args->groups);
      for (int i=0; i<rt_args->problem_sizes.size(); i++) {
        auto m = rt_args->problem_sizes[i*3+0];
        auto n = rt_args->problem_sizes[i*3+1];
        auto k = rt_args->problem_sizes[i*3+2];
        problem_sizes_host.push_back({m,n,k});
      }
      
      problem_sizes.reset(rt_args->groups);
      problem_sizes.copy_from_host(problem_sizes_host.data());

      for (int32_t i = 0; i < rt_args->groups; ++i) {
        auto problem = problem_sizes_host.at(i);
        auto M = cute::get<0>(problem);
        auto N = cute::get<1>(problem);
        auto K = cute::get<2>(problem);
        stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
        stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
        stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
        stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));      
      }
      stride_A.reset(rt_args->groups);
      stride_A.copy_from_host(stride_A_host.data());
      stride_B.reset(rt_args->groups);
      stride_B.copy_from_host(stride_B_host.data());
      stride_C.reset(rt_args->groups);
      stride_C.copy_from_host(stride_C_host.data());
      stride_D.reset(rt_args->groups);
      stride_D.copy_from_host(stride_D_host.data());
      ///
    }
    

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {rt_args->groups, problem_sizes.get(), problem_sizes_host.data()},
      {(const ElementA **)rt_args->ptr_A, stride_A.get(), (const ElementB **)rt_args->ptr_B, stride_B.get(),
        (const float **)rt_args->d_blockscale_A, // blockscale_tensor_A.device_data(),
        (const float **)rt_args->d_blockscale_B, // blockscale_tensor_B.device_data()
      },
      {
        {}, // epilogue.thread
        (const ElementC **)rt_args->ptr_C, stride_C.get(),
        (ElementD **)rt_args->ptr_D, stride_D.get()
      },
      kernel_hw_info
    };

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

private:
  Gemm gemm_dev_;

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;
};

} // namespace ctlop