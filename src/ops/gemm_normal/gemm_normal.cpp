
#include "gemm_normal.h"
#include "xop/ops_impl/global_resource.h"
#include "xop/common_cuda.h"
#include "xop/common_torch.h"
#include "xop/common_strategy.h"

#include "xop/lt_coll/gemm_lt.h"

#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <utility>

// #define GEMM_COMM_WITH_NCCL

// #ifdef GEMM_COMM_WITH_NCCL
// #include "nccl.h"
// #endif

/////////////////////////////
#include "xop/xop.h"

#define PRINTF // printf
#define NOT_TUNING_SCHEMA "" // "TORCH"
#define TUNING_WITH_CUBLASLT true

//////////////////////////////
namespace xop {
using torch::Tensor;

class GemmNormal::GemmNormalImpl {
public:
  GemmNormalImpl(
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight)
      : input_dtype(input_dtype),
        output_dtype(output_dtype),
        transpose_weight(transpose_weight) { // transpose_weight true对应的是RRR，正常的false是RCR
    // auto device_properties = torch::cuda::get_device_properties(0);
    arch_ = get_arch();

    if (TUNING_WITH_CUBLASLT) {
      CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle_));
    }
    
    // Check device memory pool
    GlobalBuffer::instance().CheckDeviceBufferAllocate(kDevBufferPoolWorkspace, 8192*20000*sizeof(short));
    GlobalBuffer::instance().CheckDeviceBufferAllocate(kDevBufferPoolAux, 8192*20000*sizeof(short));
    // GlobalBuffer::instance().CheckDeviceBufferAllocate(kDevBufferPoolOutput, 8192*20000*sizeof(short));
  } 
  ~GemmNormalImpl() {
    if (TUNING_WITH_CUBLASLT) {
      for (auto it = cublaslt_gemm_map_.begin(); it != cublaslt_gemm_map_.end(); ++it) {
        const std::vector<int32_t>& key = it->first;
        GemmLt *                    val = it->second;
        delete val;
      }
      CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle_));
    }
  }

  // tuning：tensor进入，先构建meta，依次添加序号充当key，取获取op，计算性能，并进行排序，取top5, 保留整个meta。获取不到新op时表示结束。
  //         top1的meta从cpp端写入文件，信息包括shape+序号+meta。保存时，meta信息需要按python脚本的生成方式，转为字符串。
  // python脚本根据tuning结果文件，再次生成op注册表+tuning注册表，
  //       op注册表：按第一次生成的流程再走一遍，同时检索序号+meta的字符串, 匹配者留下，不匹配的不生成。
  //       tuning注册表：key是shape+meta，value是序号，test时输入tensor，构建meta，结合shape，获取序号。组成序号+meta，充当op注册表的key，检索搜索op。
  // python1生成搜索空间op注册表，编译，python2执行tuning脚本，生成tuned表，python1生成top1的op注册表以及tuning注册表。
  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> tuning,
      bool fast_accum
    ) {
    // std::cout << "Tensor input:\n" << input << std::endl;
    std::vector<int16_t> id_meta = TorchDefaultConfig::MakeDefaultMeta(arch_, this->input_dtype, weight.scalar_type(), this->output_dtype, fast_accum, transpose_weight, false);       // id + meta
    RunModeEnum run_mode = TorchDefaultConfig::GetRunMode(tuning);
    ///////
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    torch::Tensor output;
    if (output_buf.has_value()) {
      output = output_buf.value();
    } else {
      output = create_output_tensor(input, weight, stream);
    }
    std::unique_ptr<RtArguments> rt_args = TorchDefaultConfig::GetBaseRtConf(input, weight, output, bias, input_scale, weight_scale, 
                                                                             this->input_dtype, this->output_dtype, transpose_weight, &default_schema_);

    if (run_mode == kRunWithTuning) {
      GlobalBuffer::instance().SetTuningFlag(true);
      // GlobalBuffer::instance().PrintUsedBufferSize();
      forward_tuning(input, weight, output, bias, input_scale, weight_scale, 
                      (int16_t *)tuning.value().data_ptr(), id_meta, rt_args.get(), stream);
    }
    else {
      // Misalignment case.
      // if (rt_args->n%8 != 0 || rt_args->k%8 != 0) {
      //   return RunTorch(input, weight, output, bias);
      // }
      int max_m = 16384;
      if (run_mode == kRunWithHparam) {
        int16_t *tdata = (int16_t *)tuning.value().data_ptr();
        max_m = tdata[1];
        if (tdata[2] != -1)
          id_meta[kMetaArch] = tdata[2];
        PRINTF("set max_m = %d, arch = %d.\n", max_m, id_meta[kMetaArch]);
      }
      int tuned_m = Strategy::CoarseGrainedTuningM(rt_args->m, max_m);
      PRINTF("actual_m: %d, tuned_m: %d.\n", rt_args->m, tuned_m);
      std::vector<int32_t> shape_meta = {tuned_m, rt_args->n, rt_args->k, rt_args->g};       // mnkl/g + meta
      shape_meta.insert(shape_meta.end(), id_meta.begin()+2, id_meta.end());     // skip 2 (id + schema)
      
      cublasLtMatmulAlgo_t algo;
      TunedConfigRegister& tins = TunedConfigRegister::instance();
      tins.GetSelectedConfig(shape_meta, &id_meta[kMetaId], &id_meta[kMetaSchema], algo.data);
      // tins.PrintRegistedConfig(shape_meta);
      // If the required configuration is not registered in the tuning config, directly use torch for computation.
      if (id_meta[kMetaId] == -1) {
        // if constexpr (NOT_TUNING_SCHEMA == "TORCH")
        //   return RunTorch(input, weight, output, bias);
        // else {
          id_meta[kMetaId] = 0;
          id_meta[kMetaSchema] = (int16_t)default_schema_;
        // }
      }

      id_meta[kMetaSchema] = (int16_t)UnifiedMetaEnum::GemvSimt;
      PRINTF("[runing normal] selected_id: %d, selected_schema: %d.\n", id_meta[kMetaId], id_meta[kMetaSchema]);
      if (id_meta[kMetaSchema] == (int16_t)UnifiedMetaEnum::GemmLt) {
        cudaDataType_t type_input = WarpIdMeta2CublasLtType(id_meta[kMetaTypeA]);
        cudaDataType_t type_output = WarpIdMeta2CublasLtType(id_meta[kMetaTypeCD]);
        cublasComputeType_t type_compute = WarpIdMeta2CublasLtComputeType(id_meta[kMetaTypeAcc]);
        
        GemmLt cublaslt_gemm;
        cublaslt_gemm.init(cublaslt_handle_, rt_args->n, rt_args->m, rt_args->k, type_input, type_output, type_compute, false);
        cublaslt_gemm.run(algo, weight.data_ptr(), input.data_ptr(), output.data_ptr(), stream);
      }
      else if (id_meta[kMetaSchema] == (int16_t)UnifiedMetaEnum::GemvSimt) {
        GemmConfigRegister& ins = GemmConfigRegister::instance();
        GemmBase *op = ins.GetOp(id_meta, false);

        RtArguments *base_args = rt_args.get();
        // gemv: weight A[m,k] * input B[1,k] = C[1,m]
        base_args->m = weight.size(0);
        base_args->k = weight.size(1);
        base_args->n = 1;

        void *t = base_args->ptr_A;
        base_args->ptr_A = base_args->ptr_B;
        base_args->ptr_B = t;

        op->initialize(rt_args.get(), nullptr, stream);
        op->run(stream);
      }
      else {
        GemmConfigRegister& ins = GemmConfigRegister::instance();
        GemmBase *op = ins.GetOp(id_meta, false);

        RtArguments *base_args = rt_args.get();
        std::vector<int> split_m = Strategy::SplitChunkM(base_args->m, max_m);
        // bias == out_features == N （K == in_features）
        // so bias needn't split.
        void *ptr_A = rt_args->ptr_A;
        void *ptr_D = rt_args->ptr_D;
        void *ptr_scale_A = nullptr;
        if (default_schema_ == UnifiedMetaEnum::GemmBlockScaleFp8) {
          ptr_scale_A = ((RtBlockScaleArguments *)rt_args.get())->ptr_blockscale_A;
        }
        for (int i=0; i<split_m.size(); i++) {
          rt_args->m = split_m[i];

          PRINTF("m: %d, in: %ld, out: %ld.\n", rt_args->m, at::elementSize(this->input_dtype), at::elementSize(this->output_dtype));
          rt_args->ptr_A = (void*)((char*)ptr_A + i*split_m[0]*rt_args->k*at::elementSize(this->input_dtype));
          rt_args->ptr_D = (void*)((char*)ptr_D + i*split_m[0]*rt_args->n*at::elementSize(this->output_dtype));
          if (ptr_scale_A != nullptr) {
            ((RtBlockScaleArguments *)rt_args.get())->ptr_blockscale_A = (void*)((char*)ptr_scale_A + i*split_m[0]*rt_args->k/128*sizeof(float));
          }
          op->initialize(rt_args.get(), nullptr, stream);
          op->run(stream);
        }
      }
    }
    // ins.PrintRegistered("abc");
    // PRINTF("id_meta: ");
    // for(int i=0; i<id_meta.size(); i++) {
    //   PRINTF("%d, ", id_meta[i]);
    // }
    return output;
  }

  int grouped_forward(
    std::vector<torch::Tensor> inputs,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> outputs,
    c10::optional<std::vector<torch::Tensor>> inputs_scale,
    c10::optional<std::vector<torch::Tensor>> weights_scale,
    c10::optional<torch::Tensor> tuning
  ) {

    GemmConfigRegister& ins = GemmConfigRegister::instance();
    TunedConfigRegister& tins = TunedConfigRegister::instance();

    std::vector<int16_t> id_meta = TorchDefaultConfig::MakeDefaultMeta(arch_, this->input_dtype, weights[0].scalar_type(), this->output_dtype, false, transpose_weight, true);     // id + meta
    RunModeEnum run_mode = TorchDefaultConfig::GetRunMode(tuning);

    std::unique_ptr<RtGroupedArguments> rt_args;
    if (from_torch_dtype(this->input_dtype) == (int)UnifiedMetaEnum::E4M3) {
      rt_args = std::make_unique<RtGroupedBlockScaleFp8ArgumentsV3>();
      rt_args->groups = inputs.size();
      if (inputs_scale.has_value() && weights_scale.has_value()) {
        for (int i=0; i < inputs_scale.value().size(); i++) {
          ((RtGroupedBlockScaleFp8ArgumentsV3 *)rt_args.get())->ptr_blockscale_A.push_back(inputs_scale.value()[i].data_ptr());
          ((RtGroupedBlockScaleFp8ArgumentsV3 *)rt_args.get())->ptr_blockscale_B.push_back(weights_scale.value()[i].data_ptr());
        }
      }
      default_schema_ = UnifiedMetaEnum::GemmGroupedBlockScaleFp8;
    }
    else {
      rt_args = std::make_unique<RtGroupedArguments>();
      rt_args->groups = inputs.size();
      default_schema_ = UnifiedMetaEnum::GemmGrouped;
    }

    for (int i=0; i < inputs.size(); i++) {
      rt_args->problem_sizes.push_back(inputs[i].size(0));  // m
      rt_args->problem_sizes.push_back(outputs[i].size(1)); // n
      rt_args->problem_sizes.push_back(inputs[i].size(1));  // k

      rt_args->ptr_A.push_back(inputs[i].data_ptr());
      rt_args->ptr_B.push_back(weights[i].data_ptr());
      rt_args->ptr_C.push_back(nullptr);
      rt_args->ptr_D.push_back(outputs[i].data_ptr());
    }
    rt_args->alpha = 1.0f;
    rt_args->beta = 0.0f;
    
    bool is_tuning = false;
    if (run_mode == kRunWithTuning) {
      int16_t *data = (int16_t *)tuning.value().data_ptr();
      XOP_CHECK_EQ(data[0], 1);
      id_meta[kMetaId] = data[1];
      id_meta[kMetaSchema] = data[2];
      is_tuning = true;
    }
    else {
      int32_t m = 0;
      for (int i=0; i<inputs.size(); i++) {
        m += inputs[i].size(0);
      }
      int32_t k = inputs[0].size(1); // By default, each group has the same k/n.
      int32_t n = weights[0].size(0);
      std::vector<int32_t> shape_meta = {m, n, k, rt_args->groups};       // mnkg + meta
      shape_meta.insert(shape_meta.end(), id_meta.begin()+2, id_meta.end());     // skip id and schema
      tins.GetSelectedConfig(shape_meta, &id_meta[kMetaId], &id_meta[kMetaSchema]);   
      
      if (id_meta[kMetaId] == -1) {
          id_meta[kMetaId] = 0;
          id_meta[kMetaSchema] = (int16_t)default_schema_;
      }
    }
    PRINTF("selected_id: %d, selected_schema: %d.\n", id_meta[kMetaId], id_meta[kMetaSchema]);
    GemmBase *op = ins.GetOp(id_meta, is_tuning);
    if (op == nullptr)
      return -1;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    op->initialize(rt_args.get(), nullptr, stream);
    op->run(stream);

    if (run_mode == kRunWithTuning) {
      int16_t *data = (int16_t *)tuning.value().data_ptr();
      data[0] = id_meta.size();
      for (int i=0; i<id_meta.size(); i++) {
        data[i+1] = id_meta[i];
      }
    }

    return 0;
  }

private:
  torch::Tensor
  create_output_tensor(torch::Tensor input, torch::Tensor weight, cudaStream_t stream) {
    int32_t m = input.size(0);
    int32_t n = weight.size(0);
    
    // int32_t malloc_m = 8192;
    // if (malloc_m < m) 
    //   malloc_m = m;
    
    // void *buffer_ptr = GlobalBuffer::instance().GetDeviceBuffer(kDevBufferPoolOutput, malloc_m*n*at::elementSize(output_dtype));
    // auto opts = torch::TensorOptions()
    //               .dtype(output_dtype)
    //               .device(weight.device());
    // return torch::from_blob(buffer_ptr, {m, n}, opts);

    return torch::empty({m, n}, weight.options().dtype(output_dtype));
  }

  int forward_tuning(torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor output,
                    c10::optional<torch::Tensor> bias,
                    c10::optional<torch::Tensor> input_scale,
                    c10::optional<torch::Tensor> weight_scale,
                    int16_t *tuning_data, 
                    std::vector<int16_t>& id_meta, 
                    RtArguments *rt_args,
                    cudaStream_t stream) {
    XOP_CHECK_EQ(tuning_data[0], 1); // 1 for tuning
    id_meta[kMetaId] = tuning_data[1];
    id_meta[kMetaSchema] = tuning_data[2];
    id_meta[kMetaArch] = tuning_data[3];
    
    PRINTF("[tuning normal] selected_id: %d, selected_schema: %d, selected_arch: %d.\n", id_meta[kMetaId], id_meta[kMetaSchema], id_meta[kMetaArch]);
    if (id_meta[kMetaSchema] == (int16_t)UnifiedMetaEnum::GemmLt) {
      if constexpr (TUNING_WITH_CUBLASLT == false) {
        return -1;
      }
      cudaDataType_t type_input = WarpIdMeta2CublasLtType(id_meta[kMetaTypeA]);
      cudaDataType_t type_output = WarpIdMeta2CublasLtType(id_meta[kMetaTypeCD]);
      cublasComputeType_t type_compute = WarpIdMeta2CublasLtComputeType(id_meta[kMetaTypeAcc]);

      std::vector<int32_t> lt_key{rt_args->m, rt_args->n, rt_args->k, id_meta[kMetaTypeA], id_meta[kMetaTypeCD], id_meta[kMetaTypeAcc]};
      GemmLt *gemm_lt = nullptr;
      auto it = cublaslt_gemm_map_.find(lt_key);
      if (it != cublaslt_gemm_map_.end()){
        gemm_lt = it->second;
      }
      else {
        gemm_lt = new GemmLt;
        gemm_lt->init(cublaslt_handle_, rt_args->n, rt_args->m, rt_args->k, type_input, type_output, type_compute, true);
        cublaslt_gemm_map_.emplace(lt_key, gemm_lt);
      }

      if (id_meta[kMetaId] >= gemm_lt->get_algo_num()) { 
        printf("hello run cublasLt set true\n");
        tuning_data[0] = -1; // close the tuning flag
        return -1;
      }
      cublasLtMatmulAlgo_t algo;
      gemm_lt->get_algo(id_meta[kMetaId], algo);
      gemm_lt->run(algo, weight.data_ptr(), input.data_ptr(), output.data_ptr(), stream);

      tuning_data[0] = id_meta.size();
      for (int i=0; i<id_meta.size(); i++) {
        tuning_data[i+1] = id_meta[i];
      }
      memcpy(&tuning_data[20], &algo, sizeof(algo));
    }
    else {
      GemmConfigRegister& ins = GemmConfigRegister::instance();
      GemmBase *op = ins.GetOp(id_meta, true);
      if (op == nullptr) {
        // printf("Gemm type {");
        // for (int i=0; i<id_meta.size(); i++)
        //   printf("%d-", id_meta[i]);
        // printf("} not found.\n");

        // ins.PrintRegistered("abc:");
        tuning_data[0] = -1; // close the tuning flag
        return -1;        
      }

      op->initialize(rt_args, nullptr, stream);
      op->run(stream);
      
      tuning_data[0] = id_meta.size();
      for (int i=0; i<id_meta.size(); i++) {
        tuning_data[i+1] = id_meta[i];
      }
    }
    return 0;
  }

  int RunTorch(torch::Tensor input,
                torch::Tensor weight,
                torch::Tensor output,
                c10::optional<torch::Tensor> bias) {
    PRINTF("[runing] torch\n");
    if (transpose_weight){
      if (bias.has_value()) {
        torch::addmm_out(output, bias.value(), input, weight);
      }
      else {
        torch::matmul_out(output, input, weight);
      }
    }
    else {
      if (bias.has_value()) {
        torch::addmm_out(output, bias.value(), input, weight.t());
        // output = torch::nn::functional::linear(input, weight);
      }
      else {
        torch::matmul_out(output, input, weight.t());
      }         
    }
    return 0;
  }

private:
  cublasLtHandle_t cublaslt_handle_;
  std::map<std::vector<int32_t>, GemmLt *> cublaslt_gemm_map_;

  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;

  UnifiedMetaEnum default_schema_;
  UnifiedMetaEnum arch_;

  torch::Tensor padded_input_;
};

GemmNormal::GemmNormal(
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    bool transpose_weight)
    : impl_(new GemmNormal::GemmNormalImpl(input_dtype, output_dtype, transpose_weight)) {}

GemmNormal::~GemmNormal() { delete impl_; }

torch::Tensor
GemmNormal::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    c10::optional<torch::Tensor> tuning,
    bool fast_accum) {
  // XOP_CHECK(impl_ != nullptr) << "GemmNormal is not initialized";
  return impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(output),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(tuning),
      fast_accum);
}

int GemmNormal::grouped_forward(
  std::vector<torch::Tensor> inputs,
  std::vector<torch::Tensor> weights,
  std::vector<torch::Tensor> outputs,
  c10::optional<std::vector<torch::Tensor>> input_scale,
  c10::optional<std::vector<torch::Tensor>> weight_scale,
  c10::optional<torch::Tensor> tuning) {
// XOP_CHECK(impl_ != nullptr) << "GemmNormal is not initialized";
return impl_->grouped_forward(
    std::move(inputs),
    std::move(weights),
    std::move(outputs),
    std::move(input_scale),
    std::move(weight_scale),
    std::move(tuning));
}

void print_used_size_of_device_buffer() {
  GlobalBuffer::instance().PrintUsedBufferSize();
}

}  // namespace xop