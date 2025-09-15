
#include "gemm_comm.h"
#include "xop/ops_impl/global_resource.h"
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
#define CHECK_TYPE(x, st) XOP_CHECK_EQ(x.scalar_type(), st) << "Inconsistency type of Tensor " #x
#define CHECK_CUDA(x) XOP_CHECK(x.is_cuda()) << #x << " must be a CUDA tensor"
#define CHECK_CONTIGUOUS(x) XOP_CHECK(x.is_contiguous()) << #x << " must be contiguous"
#define CHECK_INPUT(x, st) \
  CHECK_CUDA(x);           \
  CHECK_CONTIGUOUS(x);     \
  CHECK_TYPE(x, st)

#define PRINTF printf


//////////////////////////////
namespace xop {
using torch::Tensor;

enum IdMetaEnum {
  Id = 0, 
  Schema = 1,
  TypeA = 2,
  TypeB = 3,
  TypeCD = 4,
  TypeAcc = 5,
  Layout = 6,
  Arch = 7
};

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
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    if (device_properties.major == 9 && device_properties.minor == 0)
      arch_ = UnifiedMetaEnum::Sm90;
    else if (device_properties.major == 8 && device_properties.minor == 9)
      arch_ = UnifiedMetaEnum::Sm89;
    else
      arch_ = UnifiedMetaEnum::Sm80;
    // printf("sm: %d%d.\n", device_properties.major, device_properties.minor);
  } 
  ~GemmCommImpl() {}
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
      bool fast_accum,
      bool registered,
      int64_t fa, 
      int64_t reg_buffer, 
      int64_t reg_buffer_sz_bytes
    ) {

    // torch::Tensor gemm_out = torch::zeros_like(output);

    // std::cout << "Tensor input:\n" << input << std::endl;
    std::vector<int16_t> id_meta = MakeDefaultMeta(fast_accum);       // id + meta
    std::unique_ptr<RtArguments> rt_args;
    if (from_torch_dtype(this->input_dtype) == (int)UnifiedMetaEnum::E4M3) {
      rt_args = std::make_unique<RtBlockScaleFp8ArgumentsV3>();
      if (input_scale.has_value() && weight_scale.has_value()) {
        ((RtBlockScaleFp8ArgumentsV3 *)rt_args.get())->d_blockscale_A = input_scale.value().data_ptr();
        ((RtBlockScaleFp8ArgumentsV3 *)rt_args.get())->d_blockscale_B = weight_scale.value().data_ptr();
      }
      id_meta[IdMetaEnum::Schema] = (int16_t)UnifiedMetaEnum::GemmBlockScaleFp8; // TODO: 检查是否可删除？
      id_meta[IdMetaEnum::Arch] = (int16_t)arch_;
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
      id_meta[IdMetaEnum::Schema] = (int16_t)UnifiedMetaEnum::GemmAllreduce; // TODO: 检查是否可删除？
      rt_args = std::make_unique<RtArgumentsV2>();
    }
    GetBaseRtConf(input, weight, output, bias, input_scale, weight_scale, rt_args.get());
    
    if (tuning.has_value()) {
      torch::Tensor t = tuning.value();
      torch::Device dev = t.device();
      if (dev.is_cpu()) {
        return forward_tuning(input, weight, output, bias, input_scale, weight_scale, 
          (int16_t *)t.data_ptr(), id_meta, rt_args.get(),
          fa, reg_buffer, reg_buffer_sz_bytes);
      }
      else { // (dev.is_cuda()) 
        torch::Tensor t_cpu = t.cpu();
        int ret = forward_tuning(input, weight, output, bias, input_scale, weight_scale, 
                              (int16_t *)t_cpu.data_ptr(), id_meta, rt_args.get(),
                              fa, reg_buffer, reg_buffer_sz_bytes);
        t = t_cpu.to(t.device()); 
        return ret;
      }
    }
    else {
      // Misalignment case.
      if (rt_args->n%8 != 0 || rt_args->k%8 != 0) {
        return RunTorch(input, weight, output, bias);
      }
      
      int tuned_m = Strategy::CoarseGrainedTuningM(rt_args->m, 1);
      PRINTF("actual_m: %d, tuned_m: %d.\n", rt_args->m, tuned_m);
      std::vector<int32_t> shape_meta = {tuned_m, rt_args->n, rt_args->k, 1};       // mnkl + meta
      shape_meta.insert(shape_meta.end(), id_meta.begin()+2, id_meta.end());     // skip 2 (id + schema)
      
      TunedConfigRegister& tins = TunedConfigRegister::instance();
      tins.GetCommSelectedConfig(shape_meta, &id_meta[IdMetaEnum::Id], &id_meta[IdMetaEnum::Schema]);

      // If the required configuration is not registered in the tuning config, directly use torch for computation.
      if (id_meta[IdMetaEnum::Id] == -1) {
        id_meta[IdMetaEnum::Id] = 0;
      }
      PRINTF("[runing] selected_id: %d, selected_schema: %d.\n", id_meta[IdMetaEnum::Id], id_meta[IdMetaEnum::Schema]);
      GemmConfigRegister& ins = GemmConfigRegister::instance();
      GemmBase *op = ins.GetOp(id_meta, false);

      cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
      
      RtCommArguments comm_args;
      comm_args.handle = fa;
      comm_args.reg_buffer = reg_buffer;
      comm_args.reg_buffer_sz_bytes = reg_buffer_sz_bytes;
      torch::Tensor gemm_out = torch::zeros_like(output);
      comm_args.gemm_out = gemm_out.data_ptr();
      op->initialize(rt_args.get(), &comm_args, stream);
      op->run(stream);        
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
    id_meta[IdMetaEnum::Schema] = (int16_t)UnifiedMetaEnum::GemmGroupedBlockScaleFp8;
    id_meta[IdMetaEnum::Arch] = (int16_t)UnifiedMetaEnum::Sm90;

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
      id_meta[IdMetaEnum::Id] = data[1];
      id_meta[IdMetaEnum::Schema] = data[2];
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
      tins.GetCommSelectedConfig(shape_meta, &id_meta[IdMetaEnum::Id], &id_meta[IdMetaEnum::Schema]);      
    }
    PRINTF("selected_id: %d, selected_schema: %d.\n", id_meta[IdMetaEnum::Id], id_meta[IdMetaEnum::Schema]);
    GemmBase *op = ins.GetOp(id_meta, is_tuning);
    if (op == nullptr)
      return -1;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    op->initialize(rt_args);
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
    meta[IdMetaEnum::Id] = -1;                                  // id
    // (GemmComm / GemmNormalSimt / GemmBlockScaleFp8 / GemmGroupedBlockScaleFp8)
    meta[IdMetaEnum::Schema] = (int16_t)UnifiedMetaEnum::GemmAllreduce; // schema type 

    meta[IdMetaEnum::TypeA] = from_torch_dtype(this->input_dtype);  // type A
    meta[IdMetaEnum::TypeB] = from_torch_dtype(this->input_dtype);  // type B
    meta[IdMetaEnum::TypeCD] = from_torch_dtype(this->output_dtype); // type C/D

    if (fast_accum)
      meta[IdMetaEnum::TypeAcc] = (int16_t)UnifiedMetaEnum::FP16;        // type acc
    else
      meta[IdMetaEnum::TypeAcc] = (int16_t)UnifiedMetaEnum::FP32;

    if (transpose_weight)                           // layout
      meta[IdMetaEnum::Layout] = (int16_t)UnifiedMetaEnum::RRR; 
    else
      meta[IdMetaEnum::Layout] = (int16_t)UnifiedMetaEnum::RCR;
    meta[IdMetaEnum::Arch] = (int16_t)UnifiedMetaEnum::Sm80;        // arch

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
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);
    TORCH_CHECK(input.dim() == 2, "input shape is not 2");
    TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = transpose_weight ? weight.size(1) : weight.size(0); // true是RRR，正常使用是false，对应linear层的RCR

    rt_args->C_s = -1;
    if (bias.has_value()) {
      CHECK_INPUT(bias.value(), this->output_dtype);
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
                    RtArguments *rt_args,
                    int64_t fa, 
                    int64_t reg_buffer, 
                    int64_t reg_buffer_sz_bytes) {
    XOP_CHECK_EQ(tuning_data[0], 1);
    id_meta[IdMetaEnum::Id] = tuning_data[1];
    id_meta[IdMetaEnum::Schema] = tuning_data[2];
    
    PRINTF("[tuning] selected_id: %d, selected_schema: %d.\n", id_meta[IdMetaEnum::Id], id_meta[IdMetaEnum::Schema]);
    
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
    torch::Tensor gemm_out = torch::zeros_like(output);
    comm_args.gemm_out = gemm_out.data_ptr();
    op->initialize(rt_args, &comm_args, stream);
    op->run(stream);

    tuning_data[0] = id_meta.size();
    for (int i=0; i<id_meta.size(); i++) {
      tuning_data[i+1] = id_meta[i];
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