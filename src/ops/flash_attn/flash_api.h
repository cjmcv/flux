
#pragma once

#include <torch/all.h>
  
namespace xop {

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        const at::Tensor &k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        const at::Tensor &v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
        std::optional<const at::Tensor> &k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
        std::optional<const at::Tensor> &v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
        std::optional<const at::Tensor> &q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
        std::optional<at::Tensor> &out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        std::optional<const at::Tensor> &cu_seqlens_q_,  // b+1
        std::optional<const at::Tensor> &cu_seqlens_k_,  // b+1
        std::optional<const at::Tensor> &cu_seqlens_k_new_,  // b+1
        std::optional<const at::Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<const at::Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        std::optional<int> max_seqlen_q_,
        // TODO: check if we need max_seqlen_k
        std::optional<int> max_seqlen_k_,
        std::optional<const at::Tensor> &page_table_, // (b_k, max_num_pages_per_seq)
        std::optional<const at::Tensor> &kv_batch_idx_, // b. indices to index into the KV cache
        std::optional<const at::Tensor> &leftpad_k_, // b
        std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
        std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
        std::optional<const at::Tensor> &seqlens_rotary_, // b
        std::optional<at::Tensor> &q_descale_,  // (b, h_k), not (b, h)
        std::optional<at::Tensor> &k_descale_,  // (b, h_k)
        std::optional<at::Tensor> &v_descale_,  // (b, h_k)
        float const softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        float const softcap,
        bool const is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
        std::optional<at::Tensor> &scheduler_metadata_,  // (b + 1)
        int num_splits,
        std::optional<bool> pack_gqa_,
        int const sm_margin
        );

}  // namespace xop
