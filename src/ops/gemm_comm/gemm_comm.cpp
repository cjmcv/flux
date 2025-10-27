
#include "gemm_comm.h"
#include "xop/ops_impl/global_resource.h"
#include "xop/common_cuda.h"
#include "xop/common_torch.h"
#include "xop/common_strategy.h"

////////////////////////////////
#include "xop/../../src/ops/allreduce_normal/custom_all_reduce.h"

///////////////////////////////

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

/////////////////////////////
#include "xop/xop.h"

#define PRINTF printf

//////////////////////////////
namespace xop {
using torch::Tensor;

class GemmComm::GemmCommImpl {
public:
  GemmCommImpl(
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight)
      : input_dtype(input_dtype),
        output_dtype(output_dtype),
        transpose_weight(transpose_weight) { // transpose_weight true对应的是RRR，正常的false是RCR
    // auto device_properties = torch::cuda::get_device_properties(0);
    arch_ = get_arch();
  } 
  ~GemmCommImpl() {}

  int forward(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor output,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> tuning,
      bool fast_accum,
      bool registered,
      int64_t fa, 
      int64_t reg_buffer, 
      int64_t reg_buffer_sz_bytes
    ) {

    // torch::Tensor gemm_out = torch::zeros_like(output);

    // std::cout << "Tensor input:\n" << input << std::endl;
    std::vector<int16_t> id_meta = TorchDefaultConfig::MakeDefaultMeta(arch_, this->input_dtype, this->output_dtype, fast_accum, transpose_weight, false);       // id + meta
    id_meta[kMetaSchema] = (int16_t)UnifiedMetaEnum::GemmAllreduce;
    RunModeEnum run_mode = TorchDefaultConfig::GetRunMode(tuning);

    std::unique_ptr<RtArguments> rt_args;
    if (from_torch_dtype(this->input_dtype) == (int)UnifiedMetaEnum::E4M3) {
      rt_args = std::make_unique<RtBlockScaleArguments>();
      if (input_scale.has_value() && weight_scale.has_value()) {
        ((RtBlockScaleArguments *)rt_args.get())->ptr_blockscale_A = input_scale.value().data_ptr();
        ((RtBlockScaleArguments *)rt_args.get())->ptr_blockscale_B = weight_scale.value().data_ptr();
      }
      default_schema_ = UnifiedMetaEnum::GemmBlockScaleFp8;
    }
    else {
      rt_args = std::make_unique<RtArgumentsV2>();
      default_schema_ = UnifiedMetaEnum::GemmNormal;
    }
    TorchDefaultConfig::GetBaseRtConf(input, weight, output, bias, input_scale, weight_scale, this->input_dtype, this->output_dtype, transpose_weight, rt_args.get());
    
    if (run_mode == kRunWithTuning) {
      return forward_tuning(input, weight, output, bias, input_scale, weight_scale, 
        (int16_t *)tuning.value().data_ptr(), id_meta, rt_args.get(),
        fa, reg_buffer, reg_buffer_sz_bytes);
    }
    else {
      // Misalignment case.
      if (rt_args->n%8 != 0 || rt_args->k%8 != 0) {
        return RunTorch(input, weight, output, bias);
      }
      
      //////////////////////
      int max_m = 16384;
      if (arch_ == UnifiedMetaEnum::Sm80) {
        if (rt_args->k >= 4096) max_m = 2048;
        else if (rt_args->k >= 8192) max_m = 1024;        
      }
      if (run_mode == kRunWithHparam) {
        int16_t *tdata = (int16_t *)tuning.value().data_ptr();
        max_m = tdata[1];
        printf("set max_m = %d.\n", max_m);
      }
      //////////////////////
      int tuned_m = Strategy::CoarseGrainedTuningM(rt_args->m, max_m);
      PRINTF("actual_m: %d, tuned_m: %d.\n", rt_args->m, tuned_m);
      std::vector<int32_t> shape_meta = {tuned_m, rt_args->n, rt_args->k, 1};       // mnkl + meta
      shape_meta.insert(shape_meta.end(), id_meta.begin()+2, id_meta.end());     // skip 2 (id + schema)
      
      TunedConfigRegister& tins = TunedConfigRegister::instance();
      tins.GetCommSelectedConfig(shape_meta, &id_meta[kMetaId], &id_meta[kMetaSchema]);

      // If the required configuration is not registered in the tuning config, directly use torch for computation.
      if (id_meta[kMetaId] == -1) {
        id_meta[kMetaId] = 0;
        id_meta[kMetaSchema] = (int16_t)default_schema_;
      }
      PRINTF("[runing comm] selected_id: %d, selected_schema: %d.\n", id_meta[kMetaId], id_meta[kMetaSchema]);
      GemmConfigRegister& ins = GemmConfigRegister::instance();
      GemmBase *op = ins.GetOp(id_meta, false);

      cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

      RtCommArguments comm_args;
      comm_args.handle = fa;
      comm_args.reg_buffer = reg_buffer;
      comm_args.reg_buffer_sz_bytes = reg_buffer_sz_bytes;

      //

      RtArguments *base_args = rt_args.get();
      std::vector<int> split_m = Strategy::SplitChunkM(base_args->m, max_m);

      void *ptr_A = rt_args->ptr_A;
      void *ptr_D = rt_args->ptr_D;
      for (int i=0; i<split_m.size(); i++) {
        rt_args->m = split_m[i];

        size_t bytes = at::elementSize(this->input_dtype);
        rt_args->ptr_A = (void*)((char*)ptr_A + i*split_m[0]*rt_args->k*bytes);
        rt_args->ptr_D = (void*)((char*)ptr_D + i*split_m[0]*rt_args->n*bytes);
        op->initialize(rt_args.get(), &comm_args, stream);
        op->run(stream);
      }

      // op->initialize(rt_args.get(), &comm_args, stream);
      // op->run(stream);        
    }
    // ins.PrintRegistered("abc");
    // PRINTF("id_meta: ");
    // for(int i=0; i<id_meta.size(); i++) {
    //   PRINTF("%d, ", id_meta[i]);
    // }
    return 0;
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

    std::vector<int16_t> id_meta = TorchDefaultConfig::MakeDefaultMeta(arch_, this->input_dtype, this->output_dtype, false, transpose_weight, true);     // id + meta
    RunModeEnum run_mode = TorchDefaultConfig::GetRunMode(tuning);
    
    RtGroupedBlockScaleFp8ArgumentsV3 *rt_args = new RtGroupedBlockScaleFp8ArgumentsV3();
    // PRINTF("size: %ld, %ld, %ld, %ld, %ld.\n", inputs.size(), weights.size(), outputs.size(), inputs_scale.value().size(), weights_scale.value().size());
    rt_args->groups = inputs.size();
    if (inputs_scale.has_value() && weights_scale.has_value()) {
      for (int i=0; i < inputs_scale.value().size(); i++) {
        rt_args->ptr_blockscale_A.push_back(inputs_scale.value()[i].data_ptr());
        rt_args->ptr_blockscale_B.push_back(weights_scale.value()[i].data_ptr());
      }
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
      tins.GetCommSelectedConfig(shape_meta, &id_meta[kMetaId], &id_meta[kMetaSchema]);      
    }
    PRINTF("selected_id: %d, selected_schema: %d.\n", id_meta[kMetaId], id_meta[kMetaSchema]);
    GemmBase *op = ins.GetOp(id_meta, is_tuning);
    if (op == nullptr)
      return -1;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    op->initialize(rt_args);
    op->run(stream);

    if (run_mode == kRunWithTuning) {
      int16_t *data = (int16_t *)tuning.value().data_ptr();
      data[0] = id_meta.size();
      for (int i=0; i<id_meta.size(); i++) {
        data[i+1] = id_meta[i];
      }
    }

    delete rt_args;
    return 0;
  }

private:
  int forward_tuning(torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor output,
                    c10::optional<torch::Tensor> bias,
                    c10::optional<torch::Tensor> input_scale,
                    c10::optional<torch::Tensor> weight_scale,
                    int16_t *tuning_data, 
                    std::vector<int16_t>& id_meta, 
                    RtArguments *rt_args,
                    int64_t fa, 
                    int64_t reg_buffer, 
                    int64_t reg_buffer_sz_bytes) {
    XOP_CHECK_EQ(tuning_data[0]==1 || tuning_data[0]==2, true);
    id_meta[kMetaId] = tuning_data[1];
    id_meta[kMetaSchema] = tuning_data[2];
    
    PRINTF("[tuning comm] selected_id: %d, selected_schema: %d.\n", id_meta[kMetaId], id_meta[kMetaSchema]);
    int tuning_pass_mode = tuning_data[0];
    tuning_data[0] = id_meta.size();
    for (int i=0; i<id_meta.size(); i++) {
      tuning_data[i+1] = id_meta[i];
    }

    if (tuning_pass_mode == 2) return 0;

    GemmConfigRegister& ins = GemmConfigRegister::instance();
    GemmBase *op = ins.GetOp(id_meta, true);
    if (op == nullptr) {
      // printf("Gemm type {");
      // for (int i=0; i<id_meta.size(); i++)
      //   printf("%d-", id_meta[i]);
      // printf("} not found.\n");

      // ins.PrintRegistered("abc:");
      return -1;        
    }
    // cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // op->initialize(rt_args);
    // op->run(stream);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    RtCommArguments comm_args;
    comm_args.handle = fa;
    comm_args.reg_buffer = reg_buffer;
    comm_args.reg_buffer_sz_bytes = reg_buffer_sz_bytes;
    op->initialize(rt_args, &comm_args, stream);
    op->run(stream);
    
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
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;

  int16_t default_schema;
  UnifiedMetaEnum arch_;

  torch::Tensor padded_input_;
};

GemmComm::GemmComm(
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    bool transpose_weight)
    : impl_(new GemmComm::GemmCommImpl(input_dtype, output_dtype, transpose_weight)) {}

GemmComm::~GemmComm() { delete impl_; }

int GemmComm::forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    c10::optional<torch::Tensor> tuning,
    bool fast_accum,
    bool registered,
    int64_t fa, 
    int64_t reg_buffer, 
    int64_t reg_buffer_sz_bytes) {
  // XOP_CHECK(impl_ != nullptr) << "GemmComm is not initialized";
  return impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(output),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(tuning),
      fast_accum,
      registered,
      fa, 
      reg_buffer, 
      reg_buffer_sz_bytes);
}

int GemmComm::grouped_forward(
  std::vector<torch::Tensor> inputs,
  std::vector<torch::Tensor> weights,
  std::vector<torch::Tensor> outputs,
  c10::optional<std::vector<torch::Tensor>> input_scale,
  c10::optional<std::vector<torch::Tensor>> weight_scale,
  c10::optional<torch::Tensor> tuning) {
// XOP_CHECK(impl_ != nullptr) << "GemmComm is not initialized";
return impl_->grouped_forward(
    std::move(inputs),
    std::move(weights),
    std::move(outputs),
    std::move(input_scale),
    std::move(weight_scale),
    std::move(tuning));
}


}  // namespace xop