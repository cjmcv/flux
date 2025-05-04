
#include "gemm_normal.h"
#include "flux/ops_impl/normal/gemm_base.h"
#include "flux/common_torch.h"

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
#include "flux/flux.h"
#define CHECK_TYPE(x, st) FLUX_CHECK_EQ(x.scalar_type(), st) << "Inconsistency type of Tensor " #x
#define CHECK_CUDA(x) FLUX_CHECK(x.is_cuda()) << #x << " must be a CUDA tensor"
#define CHECK_CONTIGUOUS(x) FLUX_CHECK(x.is_contiguous()) << #x << " must be contiguous"
#define CHECK_INPUT(x, st) \
  CHECK_CUDA(x);           \
  CHECK_CONTIGUOUS(x);     \
  CHECK_TYPE(x, st)
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
        transpose_weight(transpose_weight) {}

  // tuning：tensor进入，先构建meta，依次添加序号充当key，取获取op，计算性能，并进行排序，取top5, 保留整个meta。获取不到新op时表示结束。
  //         top1的meta从cpp端写入文件，信息包括shape+序号+meta。保存时，meta信息需要按python脚本的生成方式，转为字符串。
  // python脚本根据tuning结果文件，再次生成op注册表+tuning注册表，
  //       op注册表：按第一次生成的流程再走一遍，同时检索序号+meta的字符串, 匹配者留下，不匹配的不生成。
  //       tuning注册表：key是shape+meta，value是序号，test时输入tensor，构建meta，结合shape，获取序号。组成序号+meta，充当op注册表的key，检索搜索op。
  // python1生成搜索空间op注册表，编译，python2执行tuning脚本，生成tuned表，python1生成top1的op注册表以及tuning注册表。
  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> tuning,
      bool fast_accum
    ) {

    GemmConfigRegister& ins = GemmConfigRegister::instance();
    TunedConfigRegister& tins = TunedConfigRegister::instance();

    RtParams rt_params;
    torch::Tensor output = get_rt_conf(input, weight, bias, output_buf, input_scale, weight_scale, rt_params);
    std::vector<int8_t> id_meta = MakeMeta();     // id + meta

    bool is_tuning = false;
    int8_t selected_id = 0;
    if (tuning.has_value()) {
      int8_t *data = (int8_t *)tuning.value().data_ptr();
      FLUX_CHECK_EQ(data[0], 1);
      selected_id = data[1];
      is_tuning = true;
    }
    else {
      std::vector<int32_t> shape_meta = {rt_params.m, rt_params.n, rt_params.k};       // mnk + meta
      shape_meta.insert(shape_meta.end(), id_meta.begin()+1, id_meta.end());
      selected_id = tins.GetSelectedId(shape_meta);      
    }
    printf("selected_id: %d.\n", selected_id);
    id_meta[0] = selected_id;
    GemmBase *op = ins.getGemm(id_meta, is_tuning);
    if (op == nullptr)
      return torch::Tensor();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    op->initialize(rt_params);
    op->run(stream);

    if (tuning.has_value()) {
      int8_t *data = (int8_t *)tuning.value().data_ptr();
      data[0] = id_meta.size();
      for (int i=0; i<id_meta.size(); i++) {
        data[i+1] = id_meta[i];
      }
    }
    return output;
  }

private:
  std::vector<int8_t> MakeMeta(int8_t id = 0) {
    std::vector<int8_t> meta;
    meta.resize(8);
    meta[0] = id;                              // id
    meta[1] = (int8_t)UnifiedMetaEnum::Normal; // meta type

    meta[2] = from_torch_dtype(this->input_dtype);  // type A
    meta[3] = from_torch_dtype(this->input_dtype);  // type B
    meta[4] = from_torch_dtype(this->output_dtype); // type C/D
    meta[5] = (int8_t)UnifiedMetaEnum::FP32;        // type acc

    meta[6] = (int8_t)UnifiedMetaEnum::RCR;         // layout
    meta[7] = (int8_t)UnifiedMetaEnum::Sm80;        // arch

    return meta;
  }

  torch::Tensor get_rt_conf(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      RtParams &rt_params) {
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);
    TORCH_CHECK(input.dim() == 2, "input shape is not 2");
    TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = transpose_weight ? weight.size(1) : weight.size(0);

    if (bias.has_value()) {
      CHECK_INPUT(bias.value(), this->output_dtype);
      FLUX_CHECK_EQ(bias->dim(), 2);
      FLUX_CHECK_EQ(m, bias->size(0));
      FLUX_CHECK_EQ(n, bias->size(1));
    }
    torch::Tensor output;
    if (output_buf.has_value()) {
      CHECK_INPUT(output_buf.value(), this->output_dtype);
      FLUX_CHECK_EQ(output_buf->dim(), 2);
      FLUX_CHECK_EQ(m, output_buf->size(0));
      FLUX_CHECK_EQ(n, output_buf->size(1));
      output = output_buf.value();
    }
    else {
      output = torch::empty({m, n}, weight.options().dtype(output_dtype));
    }
    int32_t wk = transpose_weight ? weight.size(0) : weight.size(1);
    FLUX_CHECK_EQ(wk, k) << "weight k-dim mismatch";

    rt_params.m = m;
    rt_params.n = n;
    rt_params.k = k;
    rt_params.ptr_A = input.data_ptr();
    rt_params.ptr_B = weight.data_ptr();
    rt_params.ptr_C = nullptr;
    rt_params.ptr_D = output.data_ptr();
    rt_params.alpha = 1.0f;
    rt_params.beta = 0.0f;

    return output;
  }

  void lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
    if (buffer_size <= 0)
      return;
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
      this->gemm_buffer = torch::empty({buffer_size}, input.options().dtype(at::ScalarType::Byte));
    }
  }
  
private:
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;
  torch::Tensor gemm_buffer;
};

GemmNormal::GemmNormal(
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    bool transpose_weight)
    : impl_(new GemmNormal::GemmNormalImpl(input_dtype, output_dtype, transpose_weight)) {}

GemmNormal::~GemmNormal() { delete impl_; }

torch::Tensor GemmNormal::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output_buf,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    c10::optional<torch::Tensor> tuning,
    bool fast_accum) {
  // FLUX_CHECK(impl_ != nullptr) << "GemmNormal is not initialized";
  return impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output_buf),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(tuning),
      fast_accum);
}

}  // namespace xop