
#include "gemm_normal.h"
#include "flux/ops_impl/normal/gemm_base.h"

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

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale) {

    GemmConfigRegister& ins = GemmConfigRegister::instance();
    GemmBase *op = ins.getGemm({3, int(DataTypeEnum::BF16), int(ArchEnum::Sm89), int(GemmLayoutEnum::RCR)});

    RtParams rt_params;
    get_rt_conf(input, weight, bias, output_buf, input_scale, weight_scale, rt_params);

    torch::Tensor output;
    if (output_buf.has_value()) {
      output = output_buf.value();
    } else {
      output = torch::empty({rt_params.m, rt_params.n}, weight.options().dtype(output_dtype));
    }

    op->initialize(rt_params);
    op->run();

    return output;
  }

  torch::Tensor profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale) {

    torch::Tensor output;
    return output;

    // auto meta =
    //     unify_type(this->get_gemm_meta(/*has_bias=*/bias.has_value(), /*fast_accum=*/fast_accum));
    // auto rt_conf = this->get_rt_conf(input, weight, bias, output_buf, input_scale, weight_scale);

    // ProfilingContext tmp_ctx("__tmp__");
    // ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    // //

    // int m = rt_conf.m();
    // int n = rt_conf.n();
    // int k = rt_conf.k();

    // int cache_size = 100 * 1024 * 1024; // 100MB 
    // int total_bytes = (m*k + k*n) * 4; //torch.finfo(dtype).bits // 8 # + M*N
    // int problem_count = 1 + int((3 * cache_size) / total_bytes);

    // std::vector<torch::Tensor> inputs;
    // std::vector<torch::Tensor> weights;
    // for (int i=0; i<problem_count; i++) {
    //   inputs.push_back(input.clone());
    //   weights.push_back(weight.clone());      
    // }
    // // for (int i=0; i<problem_count; i++) {
    // //   printf("in: %p, ", inputs[i].data_ptr());
    // // }
    // // for (int i=0; i<problem_count; i++) {
    // //   printf("w: %p, ", weights[i].data_ptr());
    // // }
    // printf("problem_count: %d, %d, %d.\n", problem_count, cache_size, total_bytes);

    // OpRegistry::instance().visit_hparams(
    //     [&](UnifiedGemmHParams const &hparams) {
    //       constexpr int warm_iters = 20;
    //       constexpr int iters = 100;
    //       float total_elapsed = 0;

    //       auto stream = c10::cuda::getCurrentCUDAStream();
    //       for (int iter = 0; iter < warm_iters + iters; ++iter) {
    //         int problem_idx = iter % problem_count;
    //         GpuTimer timer;
    //         timer.start(stream);
    //         auto output [[maybe_unused]] = this->forward_impl(
    //             inputs[problem_idx],
    //             weights[problem_idx],
    //             bias,
    //             output_buf,
    //             input_scale,
    //             weight_scale,
    //             output_scale,
    //             fast_accum,
    //             hparams);
    //         timer.stop();
    //         if (iter >= warm_iters) {
    //           total_elapsed += timer.elapsed_millis();
    //         }
    //       }

    //       float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
    //       // printf("avg_elapsed: %f.\n", avg_elapsed);
    //       ctx->add(meta, rt_conf, hparams, avg_elapsed);
    //     },
    //     meta);

    // auto best_hparams = ctx->record_best(meta, rt_conf);

    // return this->forward_impl(
    //     std::move(input),
    //     std::move(weight),
    //     std::move(bias),
    //     std::move(output_buf),
    //     std::move(input_scale),
    //     std::move(weight_scale),
    //     std::move(output_scale),
    //     fast_accum,
    //     std::move(best_hparams));
  }

private:
  void get_rt_conf(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale, RtParams &rt_params) {
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
    if (output_buf.has_value()) {
      CHECK_INPUT(output_buf.value(), this->output_dtype);
      FLUX_CHECK_EQ(output_buf->dim(), 2);
      FLUX_CHECK_EQ(m, output_buf->size(0));
      FLUX_CHECK_EQ(n, output_buf->size(1));
    }
    int32_t wk = transpose_weight ? weight.size(0) : weight.size(1);
    FLUX_CHECK_EQ(wk, k) << "weight k-dim mismatch";

    rt_params.m = m;
    rt_params.n = n;
    rt_params.k = k;
    rt_params.ptr_A = input.data_ptr();
    rt_params.ptr_B = weight.data_ptr();
    rt_params.ptr_C = nullptr;
    rt_params.ptr_D = output_buf.value().data_ptr();
    rt_params.alpha = 1.0f;
    rt_params.beta = 0.0f;
  }

private:
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;
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
    bool fast_accum) {
  // FLUX_CHECK(impl_ != nullptr) << "GemmNormal is not initialized";
  return impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output_buf),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale));
}
// ,
//     bool fast_accum,
//     c10::intrusive_ptr<ProfilingContext> opt_ctx
      // fast_accum,
      // std::move(opt_ctx)
torch::Tensor GemmNormal::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output_buf,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale) {
  FLUX_CHECK(impl_ != nullptr) << "GemmNormal is not initialized";
  return impl_->profiling(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output_buf),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale));
}

}  // namespace xop