
#include "gemm_normal.h"
#include "ctlop/ops_impl/global_resource.h"
#include "ctlop/common_torch.h"

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
#include "ctlop/ctlop.h"
#define CHECK_TYPE(x, st) CTLOP_CHECK_EQ(x.scalar_type(), st) << "Inconsistency type of Tensor " #x
#define CHECK_CUDA(x) CTLOP_CHECK(x.is_cuda()) << #x << " must be a CUDA tensor"
#define CHECK_CONTIGUOUS(x) CTLOP_CHECK(x.is_contiguous()) << #x << " must be contiguous"
#define CHECK_INPUT(x, st) \
  CHECK_CUDA(x);           \
  CHECK_CONTIGUOUS(x);     \
  CHECK_TYPE(x, st)
//////////////////////////////
namespace ctlop {
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

class GemmNormal::GemmNormalImpl {
public:
  GemmNormalImpl(
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight)
      : input_dtype(input_dtype),
        output_dtype(output_dtype),
        transpose_weight(transpose_weight) {} // true对应的是RRR，正常的false是RCR

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

    GemmConfigRegister& ins = GemmConfigRegister::instance();
    TunedConfigRegister& tins = TunedConfigRegister::instance();

    std::vector<int16_t> id_meta = MakeDefaultMeta();     // id + meta
    RtArguments *rt_args;
    if (from_torch_dtype(this->input_dtype) == (int)UnifiedMetaEnum::E4M3) {
      rt_args = new RtBlockScaleFp8ArgumentsV3();
      if (input_scale.has_value() && weight_scale.has_value()) {
        ((RtBlockScaleFp8ArgumentsV3 *)rt_args)->d_blockscale_A = input_scale.value().data_ptr();
        ((RtBlockScaleFp8ArgumentsV3 *)rt_args)->d_blockscale_B = weight_scale.value().data_ptr();
      }
      id_meta[IdMetaEnum::Schema] = (int16_t)UnifiedMetaEnum::GemmBolckScaleFp8;
      id_meta[IdMetaEnum::Arch] = (int16_t)UnifiedMetaEnum::Sm90;
      // printf("id_meta: \n");
      // for (int i=0; i<id_meta.size(); i++) {
      //   printf("%d, ", id_meta[i]);
      // }
      // printf("\n");
    }
    else {
      id_meta[IdMetaEnum::Schema] = (int16_t)UnifiedMetaEnum::GemmNormal;
      rt_args = new RtArgumentsV2();
    }
    GetBaseRtConf(input, weight, output, bias, input_scale, weight_scale, rt_args);
    

    bool is_tuning = false;
    if (tuning.has_value()) {
      int16_t *data = (int16_t *)tuning.value().data_ptr();
      CTLOP_CHECK_EQ(data[0], 1);
      id_meta[IdMetaEnum::Id] = data[1];
      id_meta[IdMetaEnum::Schema] = data[2];
      is_tuning = true;
    }
    else {
      std::vector<int32_t> shape_meta = {rt_args->m, rt_args->n, rt_args->k, 1};       // mnkl + meta
      shape_meta.insert(shape_meta.end(), id_meta.begin()+2, id_meta.end());     // skip 2 (id + schema)
      tins.GetSelectedConfig(shape_meta, &id_meta[IdMetaEnum::Id], &id_meta[IdMetaEnum::Schema]);      
    }
    printf("selected_id: %d, selected_schema: %d.\n", id_meta[IdMetaEnum::Id], id_meta[IdMetaEnum::Schema]);
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

    std::vector<int16_t> id_meta = MakeDefaultMeta();     // id + meta
    id_meta[IdMetaEnum::Schema] = (int16_t)UnifiedMetaEnum::GemmGroupedBlockScaleFp8;
    id_meta[IdMetaEnum::Arch] = (int16_t)UnifiedMetaEnum::Sm90;

    RtGroupedBlockScaleFp8ArgumentsV3 *rt_args = new RtGroupedBlockScaleFp8ArgumentsV3();
    // printf("size: %ld, %ld, %ld, %ld, %ld.\n", inputs.size(), weights.size(), outputs.size(), inputs_scale.value().size(), weights_scale.value().size());
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
      CTLOP_CHECK_EQ(data[0], 1);
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
      tins.GetSelectedConfig(shape_meta, &id_meta[IdMetaEnum::Id], &id_meta[IdMetaEnum::Schema]);      
    }
    printf("selected_id: %d, selected_schema: %d.\n", id_meta[IdMetaEnum::Id], id_meta[IdMetaEnum::Schema]);
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
  std::vector<int16_t> MakeDefaultMeta(int16_t id = 0) {
    std::vector<int16_t> meta;
    meta.resize(8);
    meta[IdMetaEnum::Id] = id;                                  // id
    // (GemmNormal / GemmNormalSimt / GemmBolckScaleFp8 / GemmGroupedBlockScaleFp8)
    meta[IdMetaEnum::Schema] = (int16_t)UnifiedMetaEnum::GemmNormal; // schema type 

    meta[IdMetaEnum::TypeA] = from_torch_dtype(this->input_dtype);  // type A
    meta[IdMetaEnum::TypeB] = from_torch_dtype(this->input_dtype);  // type B
    meta[IdMetaEnum::TypeCD] = from_torch_dtype(this->output_dtype); // type C/D
    meta[IdMetaEnum::TypeAcc] = (int16_t)UnifiedMetaEnum::FP32;        // type acc
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

    if (bias.has_value()) {
      CHECK_INPUT(bias.value(), this->output_dtype);
      CTLOP_CHECK_EQ(bias->dim(), 2);
      CTLOP_CHECK_EQ(m, bias->size(0));
      CTLOP_CHECK_EQ(n, bias->size(1));
    }
    int32_t wk = transpose_weight ? weight.size(0) : weight.size(1);
    CTLOP_CHECK_EQ(wk, k) << "weight k-dim mismatch";

    rt_args->m = m;
    rt_args->n = n;
    rt_args->k = k;
    rt_args->l = 1;
    rt_args->ptr_A = input.data_ptr();
    rt_args->ptr_B = weight.data_ptr();
    rt_args->ptr_C = nullptr;
    rt_args->ptr_D = output.data_ptr();
    rt_args->alpha = 1.0f;
    rt_args->beta = 0.0f;
  }
  
private:
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;

  int16_t default_schema;
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
  // CTLOP_CHECK(impl_ != nullptr) << "GemmNormal is not initialized";
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
// CTLOP_CHECK(impl_ != nullptr) << "GemmNormal is not initialized";
return impl_->grouped_forward(
    std::move(inputs),
    std::move(weights),
    std::move(outputs),
    std::move(input_scale),
    std::move(weight_scale),
    std::move(tuning));
}


}  // namespace ctlop