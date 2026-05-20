import itertools

import tilelang
import tilelang.language as T
from xop.ops.dsl.micro_base import BaseMicroKernel, HparamSelectMode
from xop.ops.dsl.micro_config import get_arch, get_thread_num, get_target_str, is_megakernel_enabled, get_pass_configs

#####################################################################################################
#                  短序列实现	          长序列实现                  思路
# kernel架构	 单kernel一体式计算	   双kernel拆分(__0+__1)   维度拆分可以让更多sm参与
# 共享内存  	   预存全部K/V	          流式处理	           短序列可一次性加载充分利用，长序列必须分块以节省空间
# 循环粒度       每块16 tokens	       每块 32 tokens	      长序列需要更大粒度以提高计算密度，摊销访存开销
# 并行度	       单组MMA               双组MMA 	          短序列以轻巧够用为主，长序列有足够数据，可提高并行度充分利用张量核心
# 预取策略	      一次性加载	        三级流水线预取          长序列必须隐藏更长的内存延迟
# 分支预测         尽量固化               保持灵活             短序列可减少分支开销；长序列里花销占比小，需要确保覆盖范围
# 尾处理阶段	   一阶覆盖               三个阶段	           长序列块数多，需要更精细的边界处理
# 优化重点        减少固定开销           隐藏内存延迟           短序列的固定开销占比大，长序列内存问题突出
#####################################################################################################

# 双kernel拆分的实现方式又叫flashdecoding，在seq_q很小的时候使用，对seq_k的序列维度上进行拆分。
# batch/seq_q维度天然可并行，但在seq_q小时，占不满sm，拆分seq_k会很合适方式 (seq_v不需要拆分处理)。
# 对应到gemm里，就是split-k。分开算完后，需要一个额外的kernel，对k维度进行reduce汇总。

# ┌──────────────────────────────────────────┐
# │  Step 1: Q × K^T                         │
# │  - 矩阵乘法: [seq_q, dim] × [dim, seq_k]  │
# │  - 结果: [seq_q, seq_k]                   │
# │  - 缩放: 除以 √d_k                        │
# │  Step 2: Softmax                         │
# │  - 每行减最大值（数值稳定性）               │
# │  - exp(x) 计算                            │
# │  - 每行求和并归一化                        │
# │  Step 3: × V                              │
# │  - 矩阵乘法: [seq_q, seq_k] × [seq_k, dim] │
# │  - 结果: [seq_q, dim] (输出)               │
# └───────────────────────────────────────────┘

class _GqaDecodeStrategy:
    def __init__(self, batch, max_kv_seqlen, target_kv_seqlen, num_heads, num_kv_heads, dim, is_causal, dtype, accum_dtype):
        self.name = "gqa_decode_tl"+f"_{batch}_{max_kv_seqlen}_{target_kv_seqlen}_{num_heads}_{num_kv_heads}_{dim}"
        self.thread_num = 128 # get_thread_num()
            
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dim = dim
        self.batch = batch
        self.target_kv_seqlen = target_kv_seqlen
        self.max_kv_seqlen = max_kv_seqlen
        self.is_causal = is_causal
        
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        
        self.hparam_space = self._get_hparam_space()
        print(len(self.hparam_space))
        
    def _get_hparam_space(self):
        BLOCK_N = [16, 32, 64, 128]
        BLOCK_H = [64]
        num_split = [1, 2, 4, 8]
        num_stages = [1, 2, 3]
        thread_nums = [self.thread_num]
        
        res = []
        for n, h, spilt, stage, thread_num in itertools.product(
           BLOCK_N, BLOCK_H, num_split, num_stages, thread_nums):
            if (n == 16 and spilt != 1):
                continue
            res.append([n, h, spilt, stage, thread_num])
        return res 
    
    def get_heuristic_hparams(self):
        # block_N=128, block_H=64, num_split=1, num_stages=0, threads=128
        if (self.target_kv_seqlen > 64):
            return [64, 64, 2, 1, self.thread_num]
        return [16, 64, 1, 1, self.thread_num]
    
    def gen_test_data(self, selected_hparams):
        import torch
        q = torch.randn(self.batch, self.num_heads, self.dim, device="cuda", dtype=torch.bfloat16)              # [B, N=q_seqlen=1, H=num_heads,  D=dim]
        k = torch.randn(self.batch, self.max_kv_seqlen, self.num_kv_heads, self.dim, device="cuda", dtype=torch.bfloat16)  # [B, N=kv_seqlen,  H=num_kv_heads, D=dim]
        v = torch.randn(self.batch, self.max_kv_seqlen, self.num_kv_heads, self.dim, device="cuda", dtype=torch.bfloat16)
        # mask = torch.randint(0, 2, (batch, kv_seqlen, num_kv_heads), device="cuda", dtype=torch.uint8) # Only 0/1
        edge = torch.empty(10, device="cuda", dtype=torch.int32)
        edge[0].fill_(self.target_kv_seqlen)
        mask = torch.ones(self.batch, self.max_kv_seqlen, self.num_kv_heads, device="cuda", dtype=torch.uint8)      # no mask, (batch, q_seqlen, kv_seqlen, num_kv_heads), groups维度是广播出来的，mask只跟q_seqlen, kv_seqlen有关
        # q_len = 4
        # kv_seqlen = 16
        # mask = torch.tril(torch.ones((q_len, kv_seqlen), device="cuda", dtype=torch.uint8)).unsqueeze(0).expand(batch, -1, -1) # (q_len, kv_seqlen)
        # mask.unsqueeze(2).expand(-1, -1, num_kv_heads, -1).transpose(1, 2)
        # print(mask, mask.shape)
        
        _, _, num_split, _, _ = selected_hparams
        glse = torch.empty(self.batch, self.num_heads, num_split, device="cuda", dtype=torch.bfloat16)
        Output_partial = torch.empty(self.batch, self.num_heads, num_split, self.dim, device="cuda", dtype=torch.bfloat16)
        return [q, k, v, edge, mask, glse, Output_partial]
    
    def get_torch_ref(self):
        from xop.ops.dsl.pkt_util import TorchRef
        def torch_ref(q, k, v, edge, mask, glse, Output_partial):
            k_slice = k[:, :self.target_kv_seqlen, :, :]
            v_slice = v[:, :self.target_kv_seqlen, :, :]
            return TorchRef.attention_sdpa(q, k_slice, v_slice, self.is_causal)
        return torch_ref
        
    def get_kernel(self, selected_hparams):
        # print("selected_hparams: ", selected_hparams)
        _, _, num_split, _, _ = selected_hparams
        # if (self.target_kv_seqlen < 16 and num_split == 1):
        #     print("self.target_kv_seqlen < 16")
        #     return self.kernel_main_m64(self.batch, self.num_heads, self.num_kv_heads, self.max_kv_seqlen, self.target_kv_seqlen, self.dim, self.is_causal, *selected_hparams, self.dtype, self.accum_dtype) 
        # else:
        return self.kernel_main(self.batch, self.num_heads, self.num_kv_heads, self.max_kv_seqlen, self.dim, self.is_causal, *selected_hparams, self.dtype, self.accum_dtype) 

    @tilelang.jit(out_idx=[-1], target=get_target_str(), pass_configs={"tl.disable_tma_lower": True})
    def kernel_main_m64(batch, num_heads, num_kv_heads, kv_seqlen, target_kv_seqlen, dim, is_causal, block_N, block_H, num_split, num_stages, threads, dtype="bfloat16", accum_dtype="float32"):
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
        shape_q = [batch, num_heads, dim]            # [batch, seqlen_q, num_heads, dim]
        shape_k = [batch, kv_seqlen, num_kv_heads, dim]
        shape_v = [batch, kv_seqlen, num_kv_heads, dim]
        shape_o = [batch, num_heads, dim]
        shape_mask = [batch, kv_seqlen, num_kv_heads] # [batch, seqlen_q, kv_seqlen, num_kv_heads], 因果掩码是 query 和 key 之间的关系，表示当前q能看到哪些kv，而group维度是广播出来的
        kv_group_num = num_heads // num_kv_heads  # kv_heads_num

        part_shape = [batch, num_heads, num_split, dim]
        valid_block_H = min(block_H, kv_group_num)            # 如 kv_heads_num 凑不够 block_H 时，则缩减至实际值以处理边界，但不处理kv_group_num超过block_H时的边界问题
        valid_kv_seqlen = target_kv_seqlen
        # valid_block_N = min(block_N, kv_seqlen)  # 如 kv_seqlen 凑不够 block_N 时，则缩减至实际值，但不处理kv_seqlen超过block_N时的边界问题
        
        @T.prim_func
        def flash_attn_m64(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            edge: T.Tensor([10], "int32"),
            mask: T.Tensor(shape_mask, "uint8"),
            glse: T.Tensor([batch, num_heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(batch, num_heads // valid_block_H, threads=threads) as (bx, by):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # valid_kv_seqlen = edge[0]
                # valid_block_N = min(block_N, valid_kv_seqlen)
                
                T.copy(K[bid, 0:valid_kv_seqlen, cur_kv_head, :], K_shared[:valid_kv_seqlen, :])
                if is_causal:
                    T.copy(mask[bid, 0:block_N, cur_kv_head], mask_local)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                if is_causal:
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else((mask_local[j] != 0) & (j < valid_kv_seqlen), acc_s[i, j], -T.infinity(accum_dtype))
                else:
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(j < valid_kv_seqlen, acc_s[i, j], -T.infinity(accum_dtype))
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(V[bid, 0:valid_kv_seqlen, cur_kv_head, :], V_shared[:valid_kv_seqlen, :])
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])
        return flash_attn_m64

    @tilelang.jit(out_idx=[-1], target=get_target_str(), pass_configs={"tl.disable_tma_lower": True})
    def kernel_main(batch, num_heads, num_kv_heads, kv_seqlen, dim, is_causal, block_N, block_H, num_split, num_stages, threads, dtype="bfloat16", accum_dtype="float32"):
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
        shape_q = [batch, num_heads, dim]            # [batch, seqlen_q, num_heads, dim]
        shape_k = [batch, kv_seqlen, num_kv_heads, dim]
        shape_v = [batch, kv_seqlen, num_kv_heads, dim]
        shape_o = [batch, num_heads, dim]
        shape_mask = [batch, kv_seqlen, num_kv_heads] # [batch, seqlen_q, kv_seqlen, num_kv_heads], 因果掩码是 query 和 key 之间的关系，表示当前q能看到哪些kv，而group维度是广播出来的
        kv_group_num = num_heads // num_kv_heads  # kv_heads_num

        part_shape = [batch, num_heads, num_split, dim]
        valid_block_H = min(block_H, kv_group_num)            # 如 kv_heads_num 凑不够 block_H 时，则缩减至实际值以处理边界，但不处理kv_group_num超过block_H时的边界问题
        valid_block_N = min(block_N, kv_seqlen // num_split)  # 如 kv_seqlen 凑不够 block_N 时，则缩减至实际值，但不处理kv_seqlen超过block_N时的边界问题
        
        @T.macro
        def flash_attn(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            edge: T.Tensor([10], "int32"),
            mask: T.Tensor(shape_mask, "uint8"),
            Output: T.Tensor([batch, num_heads, dim], dtype),
        ):
            with T.Kernel(batch, num_heads // valid_block_H, threads=threads) as (bx, by):
                # qkv应围绕ND分块做计算，这里布局对应的是推理用的BNHD，所以分块需要跨过H取ND。N是seqlen，H是head，d是dim。
                # flashdecoding中q的N(q_seqlen)=1, 分块是[1, dim]，数据量少，可以多份一起放到smem，smem分块取[block_H, dim]
                # K和V则正常取[N, dim]
                # smem的申请中QKV涉及计算，直接按硬件友好的固定分块 block_H/block_N，而不使用实际的 valid_block_H / valid_block_N。
                # O_shared作为输出，不再需要固定分块，则需要多少就开多少，按实际申请。
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # ceildiv 向上取整，T.copy会自动校验并截断，超出范围部分会被赋0，
                # 但输出是[batch, num_heads, dim]，即所有kv_seqlen都参与了计算，0会影响结果，正确做法是需要屏蔽掉超范围部分
                valid_kv_seqlen = edge[0] + 1
                loop_range = T.ceildiv((valid_kv_seqlen), block_N) 
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_shared)
                    if is_causal:
                        T.copy(mask[bid, k * block_N : (k + 1) * block_N, cur_kv_head], mask_local)
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    if is_causal:
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.if_then_else((mask_local[j] != 0) & (k * block_N + j < valid_kv_seqlen), acc_s[i, j], -T.infinity(accum_dtype))
                    else:
                        # 超出范围部分，需要置为-inf，不能设置为0. softmax中0会参与贡献，-inf不会，因为exp(−inf)=0，exp(0)=1
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.if_then_else(k * block_N + j<valid_kv_seqlen, acc_s[i, j], -T.infinity(accum_dtype))
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(V[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

        @T.macro
        def flash_attn_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            edge: T.Tensor([10], "int32"),
            mask: T.Tensor(shape_mask, "uint8"),
            glse: T.Tensor([batch, num_heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
        ):
            with T.Kernel(batch, num_heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                
                # valid_kv_seqlen = T.floordiv(edge[0], num_split)
                actual_kv_seqlen = edge[0]+1
                # 计算当前split的起始位置
                base_len = actual_kv_seqlen // num_split
                split_start = sid * base_len
                # 最后一个split处理剩余的所有token
                valid_kv_seqlen = T.if_then_else(
                    sid == num_split - 1,
                    actual_kv_seqlen - split_start,
                    base_len
                )
                # valid_kv_seqlen = actual_kv_seqlen // num_split
                loop_range = T.ceildiv(valid_kv_seqlen, block_N)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        K[
                            bid,
                            split_start + k * valid_block_N : split_start + (k + 1) * valid_block_N,
                            cur_kv_head,
                            :,
                        ],
                        K_shared,
                    )
                    if is_causal:
                        T.copy(
                            mask[
                                bid,
                                split_start + k * valid_block_N : split_start + (k + 1) * valid_block_N,
                                cur_kv_head,
                            ],
                            mask_local,
                        )
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    if is_causal:
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.if_then_else((mask_local[j] != 0) & (k * block_N + j < valid_kv_seqlen), acc_s[i, j], -T.infinity(accum_dtype))
                    else:
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.if_then_else(k * block_N + j < valid_kv_seqlen, acc_s[i, j], -T.infinity(accum_dtype))
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(
                        V[
                            bid,
                            split_start + k * valid_block_N : split_start + (k + 1) * valid_block_N,
                            cur_kv_head,
                            :,
                        ],
                        V_shared,
                    )
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                for i in T.Parallel(block_H):
                    if i < valid_block_H:
                        glse[bid, hid * valid_block_H + i, sid] = logsum[i]
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output_partial[bid, hid * valid_block_H : (hid + 1) * valid_block_H, sid, :])

        @T.macro
        def combine(
            glse: T.Tensor([batch, num_heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(num_heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local = T.alloc_fragment([num_split, 128], dtype)
                lse_logsum_local = T.alloc_fragment([128], accum_dtype)
                lse_max_local = T.alloc_fragment([128], accum_dtype)
                scale_local = T.alloc_fragment([128], accum_dtype)

                T.annotate_layout(
                    {
                        lse_logsum_local: T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                        lse_max_local: T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                        # lse_local: (local_id, thread_id)
                        lse_local: T.Fragment(lse_local.shape, forward_fn=lambda i, j: (j, i)),
                    }
                )

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                for k, j in T.Parallel(num_split, 128):
                    lse_local[k, j] = glse[bz, by, k]
                T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
                for k in T.serial(num_split):
                    for j in T.Parallel(128):
                        lse_logsum_local[j] += T.exp2(lse_local[k, j] - lse_max_local[j])
                for j in T.Parallel(128):
                    lse_logsum_local[j] = T.log2(lse_logsum_local[j]) + lse_max_local[j]
                for k in T.serial(num_split):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, by, k, i]
                    for j in T.Parallel(128):
                        scale_local[j] = T.exp2(lse_local[k, j] - lse_logsum_local[j])
                    # Note: Pay attention to dim and the number of threads in Parallel
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[i]
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum_local[i]

        @T.prim_func
        def flashattn_gqa_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            edge: T.Tensor([10], "int32"),
            mask: T.Tensor(shape_mask, "uint8"),
            glse: T.Tensor([batch, num_heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split(Q, K, V, edge, mask, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def flashattn_gqa_decode_no_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            edge: T.Tensor([10], "int32"),
            mask: T.Tensor(shape_mask, "uint8"),
            glse: T.Tensor([batch, num_heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn(Q, K, V, edge, mask, Output)

        if num_split > 1:
            return flashattn_gqa_decode_split
        else:
            return flashattn_gqa_decode_no_split
    
# max_kv_seqlen: 最大kv_seqlen, 每次调用该kernel所提供的kvcache的长度固定为max_kv_seqlen
# target_kv_seqlen: 表示该kernel是根据target_kv_seqlen进行tuning生成的，tuning时，会从max_kv_seqlen大小的cache里计算target_kv_seqlen的部分。
# valid_kv_seqlen: 部署推理时的实际长度，如 valid_kv_seqlen==31，会选择使用target_kv_seqlen==32的kernel，kernel内会自动处理边界到31。
class MicroGqaDecode(BaseMicroKernel):
    def __init__(self, batch, max_kv_seqlen, target_kv_seqlen, num_heads, num_kv_heads, dim, is_causal, dtype=T.bfloat16, accum_dtype=T.float32):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dim = dim
        self.batch = batch
        self.max_kv_seqlen = max_kv_seqlen
        self.target_kv_seqlen = target_kv_seqlen
        self.is_causal = is_causal
        
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.strategy = _GqaDecodeStrategy(batch, max_kv_seqlen, target_kv_seqlen, num_heads, num_kv_heads, dim, is_causal, dtype, accum_dtype)
        
    def _get_source(self, kernel, selected_hparams):
        # self.layout = "11111"
        # return kernel.get_kernel_source()
        head_str = \
'''
template <typename T,
          int THREAD_NUM,
          int SUB_KERNEL_ID,
          int M, 
          int HEAD,
          int GROUPS,
          int DIM>
__device__ __forceinline__ void flashattn_kernel_<name_suffix>(const int bx, const int by, const int bz,
                                                   const void* __restrict__ q, 
                                                   const void* __restrict__ k, 
                                                   const void* __restrict__ v,
                                                   const void* __restrict__ edge_ptr, 
                                                   const void* __restrict__ mask_ptr, 
                                                   void* __restrict__ output_ptr,
                                                   void* __restrict__ glse_ptr,
                                                   void* __restrict__ output_partial_ptr) {
  // static_assert(THREAD_NUM==<threads>);
  static_assert(M==<BATCH>); static_assert(HEAD==<HEAD>); static_assert(GROUPS==<GROUPS>); static_assert(DIM==<DIM>);
  if constexpr (SUB_KERNEL_ID == 0) { if (bx >= <gridx_0> || by >= <gridy_0> || bz >= <gridz_0> || threadIdx.x >= 128) { return; } }
  if constexpr (SUB_KERNEL_ID == 1) { if (bx >= <gridx_1> || by >= <gridy_1> || bz >= <gridz_1> || threadIdx.x >= 128) { return; } }
  const <dtype>* __restrict__ Q = static_cast<const <dtype>*>(q);
  const <dtype>* __restrict__ K = static_cast<const <dtype>*>(k);
  const <dtype>* __restrict__ V = static_cast<const <dtype>*>(v);
  const int* __restrict__ edge = static_cast<const int*>(edge_ptr);
  const uchar* __restrict__ mask = static_cast<const uchar*>(mask_ptr);
  <dtype>* __restrict__ Output = static_cast<<dtype>*>(output_ptr);
  <dtype>* __restrict__ glse = static_cast<<dtype>*>(glse_ptr);
  <dtype>* __restrict__ Output_partial = static_cast<<dtype>*>(output_partial_ptr);

'''     

        BLOCK_N, BLOCK_H, num_split, num_stages, threads = selected_hparams

        head_str = head_str.replace('<threads>', str(threads))
        head_str = head_str.replace('<BATCH>', str(self.batch))
        head_str = head_str.replace('<HEAD>', str(self.num_heads))
        head_str = head_str.replace('<GROUPS>', str(self.num_kv_heads))
        head_str = head_str.replace('<DIM>', str(self.dim))
        if num_split > 1:
            head_str = head_str.replace('<name_suffix>', f"{self.batch}_{self.max_kv_seqlen}_{self.target_kv_seqlen}_{self.num_heads}_{self.num_kv_heads}_{self.dim}__<kernel_id>")
        else:
            head_str = head_str.replace('<name_suffix>', f"{self.batch}_{self.max_kv_seqlen}_{self.target_kv_seqlen}_{self.num_heads}_{self.num_kv_heads}_{self.dim}")
        if self.dtype == T.bfloat16:
            dtype = "bfloat16_t"
        else:
            dtype = "float16_t"
        head_str = head_str.replace('<dtype>', str(dtype))
                
        origin_source = kernel.get_kernel_source()
        source = self.replace_header(origin_source, "extern \"C\" __global__", num_split, head_str)
        source = source.replace("blockIdx.x", "bx")
        source = source.replace("blockIdx.y", "by")
        source = source.replace("blockIdx.z", "bz")
        
        infos = kernel.get_launch_info()
        grid_dim, block_dim, dynamic_smem_buf, use_cooperative_groups = infos[0]
        self.layout = f"({grid_dim['blockIdx.x']}, {grid_dim['blockIdx.y']}, {grid_dim['blockIdx.z']}), ({BLOCK_N}, {BLOCK_H}, {num_split})"
        source = source.replace("<gridx_0>", str(grid_dim['blockIdx.x']))
        source = source.replace("<gridy_0>", str(grid_dim['blockIdx.y']))
        source = source.replace("<gridz_0>", str(grid_dim['blockIdx.z']))
        if (num_split > 1):
            grid_dim, block_dim, dynamic_smem_buf, use_cooperative_groups = infos[1]
            self.layout += f", ({grid_dim['blockIdx.x']}, {grid_dim['blockIdx.y']}, {grid_dim['blockIdx.z']}), ({BLOCK_N}, {BLOCK_H}, {num_split})"
            source = source.replace("<gridx_1>", str(grid_dim['blockIdx.x']))
            source = source.replace("<gridy_1>", str(grid_dim['blockIdx.y']))
            source = source.replace("<gridz_1>", str(grid_dim['blockIdx.z']))
        else:
            source = source.replace("<gridx_1>", str(0))
            source = source.replace("<gridy_1>", str(0))
            source = source.replace("<gridz_1>", str(0))
        extra_attr = f"\n// Strategy: {self.strategy.name}"
        extra_attr += f"\n// selected_hparams: {selected_hparams}."
        extra_attr += f"\n// smem: {dynamic_smem_buf} bytes."
        extra_attr += f"\n// use_cooperative_groups: {use_cooperative_groups}."
        extra_attr += f"\n// layout: {self.layout}"
        extra_attr += f"\n// block_dim=({block_dim['threadIdx.x']}, {block_dim['threadIdx.y']}, {block_dim['threadIdx.z']})."
        source += extra_attr
        
        return source
    
    def get_kernel(self, mode: HparamSelectMode):
        kernel, path = self.auto_get_kernel(self._get_source, self.strategy, mode)
        return kernel, path, self.layout

    def gen_test_data(self, selected_hparams):
        return self.strategy.gen_test_data(selected_hparams)