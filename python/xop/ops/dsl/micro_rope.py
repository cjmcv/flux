import itertools

import tilelang
import tilelang.language as T

from xop.ops.dsl.micro_base import BaseMicroKernel, HparamSelectMode, get_artifact, get_launch_info, get_dispatch_source
from xop.ops.dsl.micro_config import get_arch, get_thread_num, get_target_str, is_megakernel_enabled, get_pass_configs

######
# ROPE 是逐元素独立计算。围绕head_dim进行，batch/seq_len/num_heads 都属于可并行维度。而q和k都执行一样的操作。
# def rope_apply(x, cos, sin):
#     """
#     x:   [batch, seq_len, num_heads, d]  输入张量
#     cos: [seq_len, d]                    预计算的余弦值
#     sin: [seq_len, d]                    预计算的正弦值
#     """
#     # 分离相邻维度
#     x1, x2 = x[..., 0::2], x[..., 1::2]
#     # 应用旋转
#     out1 = x1 * cos - x2 * sin
#     out2 = x1 * sin + x2 * cos
#     # 合并并还原形状
#     return torch.stack([out1, out2], dim=-1).flatten(-2)

#####################################################################################################################
#      overlap版本                                        #            parallel版本 (good)                                 #
#####################################################################################################################
#   <<<layout-3D网格>>> - 索引简单                              <<<layout-1D扁平>>>，线程块更多
# ┌────────────────────────────────────────────────────┐    ┌───────────────────────────────────────────────────┐
# │ Grid: (batch, max_head_blocks, seq_blocks)         │    │ Grid: (Q_blocks + K_blocks)                       │
# │   每个线程块同时处理Q和K:                            │    │                                                   │
# │   ┌─────────────────────────────────────────────┐  │    │ [0...Q_blocks-1]          [Q_blocks...total-1]    │
# │   │ batch_id=bx, head_block=by, seq_start=bz*BS │  │    │ ┌────────────────────┐    ┌────────────────────┐  │
# │   │ ┌─────────────┐  ┌─────────────┐            │  │    │ │ 纯Q处理 blocks     │    │ 纯K处理 blocks      │  │
# │   │ │ [by*H_Q,    │  │ [by*H_K,    │            │  │    │ │ 每个处理完整heads   │    │ 每个处理完整heads   │  │
# │   │ │ (by+1)*H_Q) │  │ (by+1)*H_K) │            │  │    │ └────────────────────┘    └────────────────────┘  │
# │   │ └─────────────┘  └─────────────┘            │  │    └───────────────────────────────────────────────────┘
# │   └─────────────────────────────────────────────┘  │
# └────────────────────────────────────────────────────┘
# 
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  <<<负载情况>>>- 后半段存在空闲                                <<<负载情况>>> - 利用率更高
# by=0: Q[0-7] ████████████ K[0-3] ██████                    Q blocks: 4×16=64块, 每块 ████████████████████ 100%利用率
# by=1: Q[8-15]████████████ K[4-7] ██████                    K blocks: 2×16=32块, 每块 ████████████████████ 100%利用率
# by=2: Q[16-23]████████████ K[空] ░░░░░░  ← 50%空闲
# by=3: Q[24-31]████████████ K[空] ░░░░░░  ← 50%空闲
#
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  <<<cache访问>>> - QK交替                                     <<<cache访问>>> - 各自连续:
# 时间 → [Q] [K] [Q]  [K] [Q] [K]                            时间 → [Q][Q][Q]...[K][K][K]                         
# Q访问  ████    ████    ████                                Q访问 ████████████████       
# K访问      ████    ████    ████                            K访问                 ████████
# Cache: 频繁替换，hit率低一些                                Cache: Q复用高，K预热后hit率高
#
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  <<<Warp调度>>> - QK交替                                      <<<Warp调度>>> - 满流水
# Warp: [Q指令][K指令][Q指令][K指令]...                       Q块时: [Q指令][Q指令][Q指令]...
#       ████  ██   ████  ██                                         ████████████████████ 满流水线
#            ↑   ↑     ↑                                     K块时: [K指令][K指令][K指令]...
#                                                                   ████████████████████ 满流水线
#
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  <<<gmem访问>>> - cos/sin加载一次，同时用于Q和K                <<<gmem访问>>> - cos/sin重复加载
# ┌─────────────────────────────────────────────┐            ┌──────────────────────────────────┐
# │ 加载cos/sin到shared memory (1次全局内存访问)  │            │ Q块1: 加载cos/sin → 处理Q         │
# │                    ↓                        │            │ Q块2: 加载cos/sin → 处理Q         │
# │         ┌──────────┴──────────┐             │            │ ...                              │
# │         ↓                     ↓             │            │ K块1: 加载cos/sin → 处理K         │
# │    处理Q heads           处理K heads         │            │ K块2: 加载cos/sin → 处理K         │
# │    (复用cos/sin)         (复用cos/sin)       │            └──────────────────────────────────┘
# └─────────────────────────────────────────────┘
#  每个线程块只加载1次，Q和K共享，计算密度大                     每个线程块加载1次，但线程块更多，Q和K不共享，总次数增加
# 
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#   <<<smem占用>>> - 多                                         <<<smem占用>>> - 少
#  每个线程块同时申请Q_sh + K_sh，smem有部分闲置，压力更大       每个线程块只需要申请Q_sh或K_sh，smem压力小
# 
#####################################################################################################################
#  <<<优势总结>>>                                         #     <<<优势总结>>>                          
# 1. 计算密度大                                           #    1. 负载均衡更精准          
# 2. 索引简单                                             #    2. smem峰值低
# 3. 数据复用多,带宽压力小                                 #    3. cache命中与调度效率更高
#                                                        #    4. 线程块多，大显卡有优势
#####################################################################################################################


class _RopeStrategy:
    def __init__(self, batch, seqlen, num_heads, num_kv_heads, head_dim, dtype, accum_dtype):
        self.name_suffix = f"{batch}_{seqlen}_{num_heads}_{num_kv_heads}_{head_dim}"
        self.name = "rope_tl_"+self.name_suffix
        self.batch = batch  
        self.seqlen = seqlen  
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        
        self.thread_num = get_thread_num()
        self.hparam_space = self._get_hparam_space()
        print(len(self.hparam_space))
        
    def _get_hparam_space(self):
        KERNEL_MODE=[0,1]
        BLOCK_SEQ=[1]
        BLOCK_HEADS_Q=[1]
        BLOCK_HEADS_K=[1]
        thread_nums=[self.thread_num]
        
        res = []
        for mode, seq, head_q, head_k, thread_num in itertools.product(
           KERNEL_MODE, BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, thread_nums):
            res.append([mode, seq, head_q, head_k, thread_num])
        return res 
    
    def get_heuristic_hparams(self):
        # [BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, thread_nums]
        return [0,1,1,1,self.thread_num]
        
    def gen_test_data(self, selected_hparams):
        import torch
        q = torch.randn(self.batch, self.seqlen, self.num_heads, self.head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen=1, H=num_heads,  D=dim]
        k = torch.randn(self.batch, self.seqlen, self.num_kv_heads, self.head_dim, device="cuda", dtype=torch.bfloat16)   # [B, N=seqlen=1, H=num_kv_heads, D=dim]
        cos_half = torch.randn(self.batch, self.seqlen, self.head_dim//2, device="cuda", dtype=torch.bfloat16)
        sin_half = torch.randn(self.batch, self.seqlen, self.head_dim//2, device="cuda", dtype=torch.bfloat16)
        cos = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)
        return [q, k, cos, sin]
    
    def get_torch_ref(self):
        import torch
        from xop.ops.dsl.pkt_util import TorchRef
        def torch_ref(q, k, cos, sin):
            q_emb, k_emb = TorchRef.apply_rotary_pos_emb_triton(q, k, cos, sin, unsqueeze_dim=2)
            return q_emb, k_emb
        return torch_ref
    
    def get_kernel(self, selected_hparams):
        print("selected_hparams: ", selected_hparams)
        mode = selected_hparams[0]
        kernel_hparam = selected_hparams[1:]
        if mode == 0:
            return self.rope_qk_parallel(self.batch, self.seqlen, 
                                        self.num_heads, self.num_kv_heads, self.head_dim,
                                        *kernel_hparam, self.dtype, self.accum_dtype) 
        else:
            return self.rope_qk_overlap(self.batch, self.seqlen, 
                                        self.num_heads, self.num_kv_heads, self.head_dim,
                                        *kernel_hparam, self.dtype, self.accum_dtype) 

    @tilelang.jit(out_idx=[-2, -1], target=get_target_str(), pass_configs={"tl.disable_tma_lower": True})
    def rope_qk_overlap(batch, seqlen, num_heads, num_kv_heads, head_dim, 
                                BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, threads=128,
                                dtype="bfloat16", accum_dtype="float32"):

        @T.prim_func
        def rope(
            Q: T.Tensor((batch, seqlen, num_heads, head_dim), dtype),
            K: T.Tensor((batch, seqlen, num_kv_heads, head_dim), dtype),
            cos: T.Tensor((batch, seqlen, head_dim), dtype),
            sin: T.Tensor((batch, seqlen, head_dim), dtype),
            Q_embed: T.Tensor((batch, seqlen, num_heads, head_dim), dtype),
            K_embed: T.Tensor((batch, seqlen, num_kv_heads, head_dim), dtype),
        ):
            half_dim = head_dim // 2
            
            # 验证cos/sin的前后一半相同（可选，调试用）
            # 注意：在实际产品代码中可能不需要这个断言
            # T.Assert(T.all(cos[..., :half_dim] == cos[..., half_dim:]), "cos前后一半应该相同")
            # T.Assert(T.all(sin[..., :half_dim] == sin[..., half_dim:]), "sin前后一半应该相同")
            
            max_head_blocks = T.max(
                T.ceildiv(num_heads, BLOCK_HEADS_Q),
                T.ceildiv(num_kv_heads, BLOCK_HEADS_K)
            )
            
            with T.Kernel(batch, 
                        max_head_blocks,
                        T.ceildiv(seqlen, BLOCK_SEQ),
                        threads=threads) as (bx, by, bz):
                
                batch_id = bx
                seq_start = bz * BLOCK_SEQ

                # 优化1：只存储一半的cos/sin到shared memory
                cos_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)  # 节省50%
                sin_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)  # 节省50%
                
                # 1. 加载cos/sin（只加载前一半）
                for s, d in T.Parallel(BLOCK_SEQ, half_dim):
                    global_seq = seq_start + s
                    if global_seq < seqlen:
                        cos_sh[s, d] = cos[batch_id, global_seq, d]  # 只取前half_dim
                        sin_sh[s, d] = sin[batch_id, global_seq, d]  # 只取前half_dim
                
                # 2. 处理Q（如果当前head块对应Q）
                head_q_start = by * BLOCK_HEADS_Q
                if head_q_start < num_heads:
                    # 优化2：可以只存储Q的一半，计算时重组
                    # 但为了简单，我们先保持完整存储
                    Q_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_Q, head_dim), dtype)
                    
                    # 加载Q数据
                    for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_Q, head_dim):
                        global_seq = seq_start + s
                        global_head = head_q_start + h
                        if global_seq < seqlen and global_head < num_heads:
                            Q_sh[s, h, d] = Q[batch_id, global_seq, global_head, d]
                    
                    # 计算Q的RoPE（使用优化后的cos/sin）
                    for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_Q, half_dim):
                        global_seq = seq_start + s
                        global_head = head_q_start + h
                        
                        if global_seq < seqlen and global_head < num_heads:
                            # 读取Q数据
                            a_q = Q_sh[s, h, d].astype(accum_dtype)          # 前一半
                            b_q = Q_sh[s, h, d + half_dim].astype(accum_dtype)  # 后一半
                            
                            # 关键优化：使用相同的cos/sin值（只从cos_sh取前half_dim）
                            cos_val = cos_sh[s, d].astype(accum_dtype)
                            sin_val = sin_sh[s, d].astype(accum_dtype)
                            
                            # 前后half_dim使用相同的cos/sin值
                            # 前half_dim: a*cos - b*sin
                            # 后half_dim: b*cos + a*sin
                            out_first_q = a_q * cos_val - b_q * sin_val
                            out_second_q = b_q * cos_val + a_q * sin_val
                            
                            Q_embed[batch_id, global_seq, global_head, d] = out_first_q.astype(dtype)
                            Q_embed[batch_id, global_seq, global_head, d + half_dim] = out_second_q.astype(dtype)
                
                # 3. 处理K（如果当前head块对应K）
                head_k_start = by * BLOCK_HEADS_K
                if head_k_start < num_kv_heads:
                    K_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_K, head_dim), dtype)
                    
                    # 加载K数据
                    for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_K, head_dim):
                        global_seq = seq_start + s
                        global_head = head_k_start + h
                        if global_seq < seqlen and global_head < num_kv_heads:
                            K_sh[s, h, d] = K[batch_id, global_seq, global_head, d]
                    
                    # 计算K的RoPE（使用相同的cos/sin）
                    for s, h, d in T.Parallel(BLOCK_SEQ, BLOCK_HEADS_K, half_dim):
                        global_seq = seq_start + s
                        global_head = head_k_start + h
                        
                        if global_seq < seqlen and global_head < num_kv_heads:
                            # 读取K数据
                            a_k = K_sh[s, h, d].astype(accum_dtype)
                            b_k = K_sh[s, h, d + half_dim].astype(accum_dtype)
                            
                            # 使用相同的cos/sin值
                            cos_val = cos_sh[s, d].astype(accum_dtype)
                            sin_val = sin_sh[s, d].astype(accum_dtype)
                            
                            # K的RoPE公式
                            out_first_k = a_k * cos_val - b_k * sin_val
                            out_second_k = b_k * cos_val + a_k * sin_val
                            
                            K_embed[batch_id, global_seq, global_head, d] = out_first_k.astype(dtype)
                            K_embed[batch_id, global_seq, global_head, d + half_dim] = out_second_k.astype(dtype)
        
        return rope

    @tilelang.jit(out_idx=[-2, -1], target=get_target_str(), pass_configs={"tl.disable_tma_lower": True})
    def rope_qk_parallel(batch, seqlen, num_heads, num_kv_heads, head_dim, 
                        BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, threads=128,
                        dtype="bfloat16", accum_dtype="float32"):
        """
        使用T.macro提取公共计算逻辑的版本
        """
        half_dim = head_dim // 2
        q_shape = [batch, seqlen, num_heads, head_dim]
        k_shape = [batch, seqlen, num_kv_heads, head_dim]
        
        @T.macro
        def ComputeRoPE(
            data_ptr: T.Tensor,          # Q或K的输入张量
            cos_ptr: T.Tensor,           # cos张量
            sin_ptr: T.Tensor,           # sin张量  
            out_ptr: T.Tensor,           # 输出张量
            batch_id: T.int32,           # batch索引
            head_start: T.int32,         # head起始索引
            seq_start: T.int32,          # seq起始索引
            num_heads: T.int32,          # head总数
            block_heads: T.int32,        # 每个block的head数
            data_sh: T.SharedBuffer,     # 数据shared memory
            cos_sh: T.SharedBuffer,      # cos shared memory
            sin_sh: T.SharedBuffer,      # sin shared memory
        ):
            """RoPE计算的宏"""
            # 加载cos/sin到shared memory
            for s, d in T.Parallel(BLOCK_SEQ, half_dim):
                global_seq = seq_start + s
                if global_seq < seqlen:
                    cos_sh[s, d] = cos_ptr[batch_id, global_seq, d]
                    sin_sh[s, d] = sin_ptr[batch_id, global_seq, d]
            
            # 加载数据到shared memory
            for s, h, d in T.Parallel(BLOCK_SEQ, block_heads, head_dim):
                global_seq = seq_start + s
                global_head = head_start + h
                if global_seq < seqlen and global_head < num_heads:
                    data_sh[s, h, d] = data_ptr[batch_id, global_seq, global_head, d]
            
            # 计算RoPE并写回
            for s, h, d in T.Parallel(BLOCK_SEQ, block_heads, half_dim):
                global_seq = seq_start + s
                global_head = head_start + h
                
                if global_seq < seqlen and global_head < num_heads:
                    a = data_sh[s, h, d].astype(accum_dtype)
                    b = data_sh[s, h, d + half_dim].astype(accum_dtype)
                    
                    cos_val = cos_sh[s, d].astype(accum_dtype)
                    sin_val = sin_sh[s, d].astype(accum_dtype)
                    
                    out_first = a * cos_val - b * sin_val
                    out_second = b * cos_val + a * sin_val
                    
                    out_ptr[batch_id, global_seq, global_head, d] = out_first.astype(dtype)
                    out_ptr[batch_id, global_seq, global_head, d + half_dim] = out_second.astype(dtype)
        
        @T.prim_func
        def rope(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(k_shape, dtype),
            cos: T.Tensor((batch, seqlen, head_dim), dtype),
            sin: T.Tensor((batch, seqlen, head_dim), dtype),
            Q_embed: T.Tensor(q_shape, dtype),
            K_embed: T.Tensor(k_shape, dtype),
        ):
            # 计算总block数
            total_q_blocks = batch * T.ceildiv(num_heads, BLOCK_HEADS_Q) * T.ceildiv(seqlen, BLOCK_SEQ)
            total_k_blocks = batch * T.ceildiv(num_kv_heads, BLOCK_HEADS_K) * T.ceildiv(seqlen, BLOCK_SEQ)
            total_blocks = total_q_blocks + total_k_blocks
            
            with T.Kernel(total_blocks, threads=threads) as (bidx,):
                
                cos_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)
                sin_sh = T.alloc_shared((BLOCK_SEQ, half_dim), dtype)
                
                # Q block
                if bidx < total_q_blocks:
                    blocks_per_batch = T.ceildiv(num_heads, BLOCK_HEADS_Q) * T.ceildiv(seqlen, BLOCK_SEQ)
                    batch_id = bidx // blocks_per_batch
                    remaining = bidx % blocks_per_batch
                    head_block = remaining // T.ceildiv(seqlen, BLOCK_SEQ)
                    seq_block = remaining % T.ceildiv(seqlen, BLOCK_SEQ)
                    
                    head_start = head_block * BLOCK_HEADS_Q
                    seq_start = seq_block * BLOCK_SEQ
                    
                    if head_start < num_heads:
                        Q_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_Q, head_dim), dtype)
                        ComputeRoPE(
                            Q, cos, sin, Q_embed,
                            batch_id, head_start, seq_start, num_heads, BLOCK_HEADS_Q,
                            Q_sh, cos_sh, sin_sh
                        )
                
                # K block  
                else:
                    k_bidx = bidx - total_q_blocks
                    k_blocks_per_batch = T.ceildiv(num_kv_heads, BLOCK_HEADS_K) * T.ceildiv(seqlen, BLOCK_SEQ)
                    k_batch_id = k_bidx // k_blocks_per_batch
                    k_remaining = k_bidx % k_blocks_per_batch
                    k_head_block = k_remaining // T.ceildiv(seqlen, BLOCK_SEQ)
                    seq_block = k_remaining % T.ceildiv(seqlen, BLOCK_SEQ)
                    
                    head_start = k_head_block * BLOCK_HEADS_K
                    seq_start = seq_block * BLOCK_SEQ
                    
                    if head_start < num_kv_heads:
                        K_sh = T.alloc_shared((BLOCK_SEQ, BLOCK_HEADS_K, head_dim), dtype)
                        ComputeRoPE(
                            K, cos, sin, K_embed,
                            k_batch_id, head_start, seq_start, num_kv_heads, BLOCK_HEADS_K,
                            K_sh, cos_sh, sin_sh
                        )
        return rope

    
class MicroRope(BaseMicroKernel):
    def __init__(self, batch, seqlen, num_heads, num_kv_heads, head_dim, dtype=T.bfloat16, accum_dtype=T.float32):
        super().__init__()
        self.strategy = _RopeStrategy(batch, seqlen, num_heads, num_kv_heads, head_dim, dtype, accum_dtype)
        
    def get_source(self, kernel, selected_hparams):
        head_str = \
'''
template <typename T,
          int THREAD_NUM,
          int BATCH,
          int SEQLEN,
          int NUM_HEADS_Q,
          int NUM_HEADS_K,
          int HEAD_DIM>
__device__ __forceinline__ void rope_kernel_<name_suffix>(const int bx, const int by, const int bz,
                                                   const void* __restrict__ q, 
                                                   const void* __restrict__ k, 
                                                   const void* __restrict__ cos_ptr,
                                                   const void* __restrict__ sin_ptr, 
                                                   void* __restrict__ q_embed_ptr,
                                                   void* __restrict__ k_embed_ptr) {
  static_assert(THREAD_NUM==<threads>);
  static_assert(BATCH==<BATCH>); static_assert(SEQLEN==<SEQLEN>); 
  static_assert(NUM_HEADS_Q==<NUM_HEADS_Q>); static_assert(NUM_HEADS_K==<NUM_HEADS_K>); static_assert(HEAD_DIM==<HEAD_DIM>);
  if (bx >= <gridx_0> || by >= <gridy_0> || bz >= <gridz_0>) { return; }
  
  const <dtype>* __restrict__ Q = static_cast<const <dtype>*>(q);
  const <dtype>* __restrict__ K = static_cast<const <dtype>*>(k);
  const <dtype>* __restrict__ cos_1 = static_cast<const <dtype>*>(cos_ptr);
  const <dtype>* __restrict__ sin_1 = static_cast<const <dtype>*>(sin_ptr);
  <dtype>* __restrict__ Q_embed = static_cast<<dtype>*>(q_embed_ptr);
  <dtype>* __restrict__ K_embed = static_cast<<dtype>*>(k_embed_ptr);
  
'''     
        KERNEL_MODE, BLOCK_SEQ, BLOCK_HEADS_Q, BLOCK_HEADS_K, threads = selected_hparams
        BLOCK_K = 1
        
        head_str = head_str.replace('<threads>', str(threads))
        head_str = head_str.replace('<BATCH>', str(self.strategy.batch))
        head_str = head_str.replace('<SEQLEN>', str(self.strategy.seqlen))
        head_str = head_str.replace('<NUM_HEADS_Q>', str(self.strategy.num_heads)) 
        head_str = head_str.replace('<NUM_HEADS_K>', str(self.strategy.num_kv_heads)) 
        head_str = head_str.replace('<HEAD_DIM>', str(self.strategy.head_dim)) 
        head_str = head_str.replace('<name_suffix>', self.strategy.name_suffix)
        if self.strategy.dtype == T.bfloat16:
            dtype = "bfloat16_t"
        else:
            dtype = "float16_t"
        head_str = head_str.replace('<dtype>', str(dtype))
                
        origin_source = kernel.get_kernel_source()
        source = self.replace_header(origin_source, "extern \"C\" __global__", 1, head_str)
        source = source.replace("blockIdx.x", "bx")
        source = source.replace("blockIdx.y", "by")
        source = source.replace("blockIdx.z", "bz")
        
        artifact = get_artifact(kernel)
        grid_dim, block_dim, dynamic_smem_buf, use_cooperative_groups = get_launch_info(artifact)[0]
        self.layout = f"({grid_dim['blockIdx.x']}, {grid_dim['blockIdx.y']}, {grid_dim['blockIdx.z']}), ({1}, {1}, {1})"
        source = source.replace("<gridx_0>", str(grid_dim['blockIdx.x']))
        source = source.replace("<gridy_0>", str(grid_dim['blockIdx.y']))
        source = source.replace("<gridz_0>", str(grid_dim['blockIdx.z']))
        extra_attr = f"\n// Strategy: {self.strategy.name}"
        extra_attr += f"\n// selected_hparams: {selected_hparams}."
        extra_attr += f"\n// smem: {dynamic_smem_buf} bytes."
        extra_attr += f"\n// use_cooperative_groups: {use_cooperative_groups}."
        extra_attr += f"\n// layout: {self.layout}"
        extra_attr += f"\n// block_dim=({block_dim['threadIdx.x']}, {block_dim['threadIdx.y']}, {block_dim['threadIdx.z']})."
        source += extra_attr
        
        dispatch_source = get_dispatch_source(kernel, artifact).replace('call', "create_"+self.strategy.name)
        dispatch_source = dispatch_source.replace('LAUNCH_INFO', "LAUNCH_INFO_"+self.strategy.name)
        source += "\n\n" + dispatch_source + "\n"
        return source
    
    def get_kernel(self, mode: HparamSelectMode):
        kernel, path = self.auto_get_kernel(self.get_source, self.strategy, mode)
        return kernel, path, self.layout
    
    def gen_test_data(self, selected_hparams):
        return self.strategy.gen_test_data(selected_hparams)