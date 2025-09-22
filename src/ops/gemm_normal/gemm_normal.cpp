
#include "gemm_normal.h"
#include "xop/ops_impl/global_resource.h"
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

#define PRINTF printf
#define NOT_TUNING_SCHEMA "" // "TORCH"
#define TUNING_WITH_CUBLASLT false

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
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    if (device_properties.major == 12 && device_properties.minor == 0)
      arch_ = UnifiedMetaEnum::Sm120;
    else if (device_properties.major == 10 && device_properties.minor == 0)
      arch_ = UnifiedMetaEnum::Sm100;
    else if (device_properties.major == 9 && device_properties.minor == 0)
      arch_ = UnifiedMetaEnum::Sm90;
    else if (device_properties.major == 8 && device_properties.minor == 9)
      arch_ = UnifiedMetaEnum::Sm89;
    else
      arch_ = UnifiedMetaEnum::Sm80;
    // printf("sm: %d%d.\n", device_properties.major, device_properties.minor);

    if (TUNING_WITH_CUBLASLT) {
      cublaslt_gemm_ = nullptr;
      CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle_));
    }
    
    // cuda graph里不允许有resize，1) 在创建时先按最大值分配；2）每次capture前先按对应数据规模正常推理一次。
    // GlobalBuffer::instance().ResizeDeviceBufferIfNeeded(5000000);
  } 
  ~GemmNormalImpl() {
    if (TUNING_WITH_CUBLASLT) {
      CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle_));
      if (cublaslt_gemm_ != nullptr) {
        delete cublaslt_gemm_;
      }
    }
  }
  // tuning：tensor进入，先构建meta，依次添加序号充当key，取获取op，计算性能，并进行排序，取top5, 保留整个meta。获取不到新op时表示结束。
  //         top1的meta从cpp端写入文件，信息包括shape+序号+meta。保存时，meta信息需要按python脚本的生成方式，转为字符串。
  // python脚本根据tuning结果文件，再次生成op注册表+tuning注册表，
  //       op注册表：按第一次生成的流程再走一遍，同时检索序号+meta的字符串, 匹配者留下，不匹配的不生成。
  //       tuning注册表：key是shape+meta，value是序号，test时输入tensor，构建meta，结合shape，获取序号。组成序号+meta，充当op注册表的key，检索搜索op。
  // python1生成搜索空间op注册表，编译，python2执行tuning脚本，生成tuned表，python1生成top1的op注册表以及tuning注册表。
  int forward(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor output,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> tuning,
      bool fast_accum
    ) {
    // std::cout << "Tensor input:\n" << input << std::endl;
    std::vector<int16_t> id_meta = MakeDefaultMeta(fast_accum);       // id + meta
    std::unique_ptr<RtArguments> rt_args;
    if (from_torch_dtype(this->input_dtype) == (int)UnifiedMetaEnum::E4M3) {
      rt_args = std::make_unique<RtBlockScaleFp8ArgumentsV3>();
      if (input_scale.has_value() && weight_scale.has_value()) {
        ((RtBlockScaleFp8ArgumentsV3 *)rt_args.get())->d_blockscale_A = input_scale.value().data_ptr();
        ((RtBlockScaleFp8ArgumentsV3 *)rt_args.get())->d_blockscale_B = weight_scale.value().data_ptr();
      }
      id_meta[kMetaSchema] = (int16_t)UnifiedMetaEnum::GemmBlockScaleFp8;
      id_meta[kMetaArch] = (int16_t)arch_;
      if (arch_ != UnifiedMetaEnum::Sm90 && arch_ != UnifiedMetaEnum::Sm89) {
        printf("fp8 kernel is only supported on GPUs with the sm_89 or sm_90 architecture.");
        return -1;
      }
      // PRINTF("id_meta: \n");
      // for (int i=0; i<id_meta.size(); i++) {
      //   PRINTF("%d, ", id_meta[i]);
      // }
      // PRINTF("\n");
    }
    else {
      rt_args = std::make_unique<RtArgumentsV2>();
    }
    GetBaseRtConf(input, weight, output, bias, input_scale, weight_scale, rt_args.get());
    
    if (tuning.has_value()) {
      return forward_tuning(input, weight, output, bias, input_scale, weight_scale, 
                            (int16_t *)tuning.value().data_ptr(), id_meta, rt_args.get());
    }
    else {
      // Misalignment case.
      if (rt_args->n%8 != 0 || rt_args->k%8 != 0) {
        return RunTorch(input, weight, output, bias);
      }
      
      int max_m = 16384;
      int tuned_m = Strategy::CoarseGrainedTuningM(rt_args->m, max_m);
      PRINTF("actual_m: %d, tuned_m: %d.\n", rt_args->m, tuned_m);
      std::vector<int32_t> shape_meta = {tuned_m, rt_args->n, rt_args->k, 1};       // mnkl + meta
      shape_meta.insert(shape_meta.end(), id_meta.begin()+2, id_meta.end());     // skip 2 (id + schema)
      
      cublasLtMatmulAlgo_t algo;
      TunedConfigRegister& tins = TunedConfigRegister::instance();
      tins.GetSelectedConfig(shape_meta, &id_meta[kMetaId], &id_meta[kMetaSchema], algo.data);

      // If the required configuration is not registered in the tuning config, directly use torch for computation.
      if (id_meta[kMetaId] == -1) {
        if constexpr (NOT_TUNING_SCHEMA == "TORCH")
          return RunTorch(input, weight, output, bias);
        else {
          id_meta[kMetaId] = 0;
        }
      }
      PRINTF("[runing normal] selected_id: %d, selected_schema: %d.\n", id_meta[kMetaId], id_meta[kMetaSchema]);
      if (id_meta[kMetaSchema] == (int16_t)UnifiedMetaEnum::GemmLt) {
        if constexpr (TUNING_WITH_CUBLASLT == false) {
          return RunTorch(input, weight, output, bias);
        }
        cudaDataType_t type_input = WarpIdMeta2CublasLtType(id_meta[kMetaTypeA]);
        cudaDataType_t type_output = WarpIdMeta2CublasLtType(id_meta[kMetaTypeCD]);
        cublasComputeType_t type_compute = WarpIdMeta2CublasLtComputeType(id_meta[kMetaTypeAcc]);
        
        GemmLt cublaslt_gemm;
        cublaslt_gemm.init(cublaslt_handle_, rt_args->n, rt_args->m, rt_args->k, type_input, type_output, type_compute, false);
        cublaslt_gemm.run(algo, weight.data_ptr(), input.data_ptr(), output.data_ptr());
      }
      else {
        GemmConfigRegister& ins = GemmConfigRegister::instance();
        GemmBase *op = ins.GetOp(id_meta, false);

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        // op->initialize(rt_args.get(), nullptr, stream);
        // op->run(stream);
        
        RtArguments *base_args = rt_args.get();
        std::vector<int> split_m = Strategy::SplitChunkM(base_args->m, max_m);

        // bias == out_features == N （K == in_features）
        // so bias needn't split.
        void *ptr_A = rt_args->ptr_A;
        void *ptr_D = rt_args->ptr_D;
        for (int i=0; i<split_m.size(); i++) {
          rt_args->m = split_m[i];

          size_t bytes = at::elementSize(this->input_dtype);
          rt_args->ptr_A = (void*)((char*)ptr_A + i*split_m[0]*rt_args->k*bytes);
          rt_args->ptr_D = (void*)((char*)ptr_D + i*split_m[0]*rt_args->n*bytes);
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

    std::vector<int16_t> id_meta = MakeDefaultMeta(false);     // id + meta
    id_meta[kMetaSchema] = (int16_t)UnifiedMetaEnum::GemmGroupedBlockScaleFp8;
    id_meta[kMetaArch] = (int16_t)UnifiedMetaEnum::Sm90;

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
    if (tuning.has_value()) {
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
    }
    PRINTF("selected_id: %d, selected_schema: %d.\n", id_meta[kMetaId], id_meta[kMetaSchema]);
    GemmBase *op = ins.GetOp(id_meta, is_tuning);
    if (op == nullptr)
      return -1;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    op->initialize(rt_args, nullptr, stream);
    op->run(stream);

    if (tuning.has_value()) {
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
  std::vector<int16_t> MakeDefaultMeta(bool fast_accum) {
    std::vector<int16_t> meta;
    meta.resize(8);
    meta[kMetaId] = -1;                                  // id
    // (GemmNormal / GemmNormalSimt / GemmBlockScaleFp8 / GemmGroupedBlockScaleFp8)
    meta[kMetaSchema] = (int16_t)UnifiedMetaEnum::GemmNormal; // schema type 

    meta[kMetaTypeA] = from_torch_dtype(this->input_dtype);  // type A
    meta[kMetaTypeB] = from_torch_dtype(this->input_dtype);  // type B
    meta[kMetaTypeCD] = from_torch_dtype(this->output_dtype); // type C/D

    if (fast_accum)
      meta[kMetaTypeAcc] = (int16_t)UnifiedMetaEnum::FP16;        // type acc
    else
      meta[kMetaTypeAcc] = (int16_t)UnifiedMetaEnum::FP32;

    if (transpose_weight)                           // layout
      meta[kMetaLayout] = (int16_t)UnifiedMetaEnum::RRR; 
    else
      meta[kMetaLayout] = (int16_t)UnifiedMetaEnum::RCR;
    meta[kMetaArch] = (int16_t)UnifiedMetaEnum::Sm80;        // arch

    return meta;
  }

  void GetBaseRtConf(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor output,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      RtArguments *rt_args) {
    XOP_CHECK_INPUT(input, this->input_dtype);
    XOP_CHECK_INPUT(weight, this->input_dtype);
    TORCH_CHECK(input.dim() == 2, "input shape is not 2");
    TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = transpose_weight ? weight.size(1) : weight.size(0); // true是RRR，正常使用是false，对应linear层的RCR

    rt_args->C_s = -1;
    if (bias.has_value()) {
      XOP_CHECK_INPUT(bias.value(), this->output_dtype);
      if (bias->dim() == 2) {
        XOP_CHECK_EQ(n, bias->size(1));
        XOP_CHECK((bias->size(0) == m) || (bias->size(0) == 1));
        if (bias->size(0) == 1) {
          rt_args->C_s = 0;
        }
      }
      else {
        XOP_CHECK_EQ(n, bias->size(0));
        rt_args->C_s = 0;
      }
    }
    ///////////////////////////
    // if (m < 32) {
    //   padded_input_ = torch::zeros({32, k}, input.options());
    //   padded_input_.slice(0, 0, m).slice(1, 0, k).copy_(input);
    //   m = 32;
    //   // std::cout << "Tensor padded_input_:\n" << padded_input_ << std::endl;
    // }
    ///////////////////////////
    rt_args->m = m;
    rt_args->n = n;
    rt_args->k = k;
    rt_args->l = 1;
    // rt_args->ptr_A = padded_input_.data_ptr();
    rt_args->ptr_A = input.data_ptr();
    rt_args->ptr_B = weight.data_ptr();
    rt_args->ptr_C = bias.has_value() ? bias.value().data_ptr() : nullptr;
    rt_args->ptr_D = output.data_ptr();
    rt_args->alpha = 1.0f;
    rt_args->beta = 0.0f;


    // int32_t k_remainder = k % 16;
    // if (k_remainder != 0) {
    //   // padded_input_ = torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions({0, 16-k_remainder, 0, 0}).mode(torch::kConstant).value(0.0)); // [pad_left, pad_right, pad_top, pad_bottom]
    //   int new_k = k+16-k_remainder;
    //   padded_input_ = torch::zeros({m, new_k}, input.options());
    //   padded_input_.slice(0, 0, m).slice(1, 0, k).copy_(input);
    // }
  }

  int forward_tuning(torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor output,
                    c10::optional<torch::Tensor> bias,
                    c10::optional<torch::Tensor> input_scale,
                    c10::optional<torch::Tensor> weight_scale,
                    int16_t *tuning_data, 
                    std::vector<int16_t>& id_meta, 
                    RtArguments *rt_args) {
    XOP_CHECK_EQ(tuning_data[0], 1);
    id_meta[kMetaId] = tuning_data[1];
    id_meta[kMetaSchema] = tuning_data[2];
    
    PRINTF("[tuning normal] selected_id: %d, selected_schema: %d.\n", id_meta[kMetaId], id_meta[kMetaSchema]);
    if (id_meta[kMetaSchema] == (int16_t)UnifiedMetaEnum::GemmLt) {
      if constexpr (TUNING_WITH_CUBLASLT == false) {
        return -1;
      }
      cudaDataType_t type_input = WarpIdMeta2CublasLtType(id_meta[kMetaTypeA]);
      cudaDataType_t type_output = WarpIdMeta2CublasLtType(id_meta[kMetaTypeCD]);
      cublasComputeType_t type_compute = WarpIdMeta2CublasLtComputeType(id_meta[kMetaTypeAcc]);

      // Only create in the first No.0
      if (id_meta[kMetaId] == 0 && cublaslt_gemm_ == nullptr) {
        cublaslt_gemm_ = new GemmLt;
        cublaslt_gemm_->init(cublaslt_handle_, rt_args->n, rt_args->m, rt_args->k, type_input, type_output, type_compute, true);
      }

      if (id_meta[kMetaId] >= cublaslt_gemm_->get_algo_num()) { 
        if (cublaslt_gemm_ != nullptr) {
          delete cublaslt_gemm_;
          cublaslt_gemm_ = nullptr;
        }
        printf("hello run cublasLt set true\n");
        return -1;
      }
      cublasLtMatmulAlgo_t algo;
      cublaslt_gemm_->get_algo(id_meta[kMetaId], algo);
      cublaslt_gemm_->run(algo, weight.data_ptr(), input.data_ptr(), output.data_ptr());

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
        return -1;        
      }
      cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
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
  GemmLt *cublaslt_gemm_;

  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;

  int16_t default_schema;
  UnifiedMetaEnum arch_;

  torch::Tensor padded_input_;
};

GemmNormal::GemmNormal(
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    bool transpose_weight)
    : impl_(new GemmNormal::GemmNormalImpl(input_dtype, output_dtype, transpose_weight)) {}

GemmNormal::~GemmNormal() { delete impl_; }

int GemmNormal::forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
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


}  // namespace xop