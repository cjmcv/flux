import os
import math
from enum import Enum
import itertools
import json
from pathlib import Path

# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from concurrent.futures import ThreadPoolExecutor
import torch
import tvm
from tvm import DataType
from tvm.tir import stmt_functor, Block, For, PrimFunc
import tilelang
import tilelang.language as T

from xop.ops.dsl.pkt_util import TestUtil, TorchRef
from xop.ops.dsl.micro_base import BaseMicroKernel, HparamSelectMode, get_artifact, get_launch_info, get_dispatch_source
from xop.ops.dsl.micro_config import get_arch, get_thread_num, get_target_str, is_megakernel_enabled, get_pass_configs

# TODO: megakernel约束线程维度是一维128/256，而目前gemv方案是二维线程，且语法糖约束下，
# 无法对tn = T.get_thread_binding(0)进行二次操作，即无法由threadIdx.x // N, threadIdx.x % N, 来转换成二维。
# 解决方案是：在get_source里进行替换。
class _GemvStrategy:
    def __init__(self, M, N, K, dtype, accum_dtype):
        self.name = "linear_gemv_tl"+f"_{M}_{N}_{K}"
        
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        assert(self.M == 1)
        
        self.hparam_space = self._get_hparam_space()
        print(len(self.hparam_space))
        
    def _get_hparam_space(self):
        BLOCK_N = [2, 4, 8, 32, 64, 128]
        reduce_threads = [4, 8, 32]
        
        res = []
        for n, reduce_thread in itertools.product(BLOCK_N, reduce_threads):
            res.append([n, reduce_thread])
        # print(res)
        return res
    
    def get_heuristic_hparams(self):
        # [BLOCK_N, reduce_threads]
        return [2, 32]
        
    def get_kernel(self, selected_hparams):
        return self.kernel_main(self.N, self.K, *selected_hparams, self.dtype, self.accum_dtype) 
    
    @tilelang.jit(out_idx=[-1])
    def kernel_main(
        N: int,
        K: int,
        BLOCK_N: int,
        reduce_threads: int,
        dtype: T.dtype = T.bfloat16,
        accum_dtype: T.dtype = T.float,
    ):
        # splitk_gemv_vectorized_tvm
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_K = reduce_threads * TILE_K

        @T.prim_func
        def linear(
            A: T.Tensor((1, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((1, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
                tn = T.get_thread_binding(0)
                tk = T.get_thread_binding(1)
                A_local = T.alloc_local((TILE_K,), dtype)
                B_local = T.alloc_local((TILE_K,), dtype)
                C_accum = T.alloc_local((1,), accum_dtype)

                T.clear(C_accum)
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for k in T.vectorized(TILE_K):
                        A_local[k] = A[0, bk * BLOCK_K + tk * TILE_K + k]
                        B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                    for k in T.serial(TILE_K):
                        C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
                C_reduced = T.alloc_local((1,), accum_dtype)
                with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            C_accum[0],
                            True,
                            C_reduced[0],
                            tk,
                            dtype="handle",
                        )
                    )

                C[0, bn * BLOCK_N + tn] = C_reduced[0]

        return linear
    
    
class _GemmStrategy:
    def __init__(self, strategy, M, N, K, dtype, accum_dtype):
        self.strategy = strategy
        if (strategy == MicroLinearStrategy.SILU_MUL_GEMM):
            self.name = "linear_silu_mull_gemm_tl"+f"_{M}_{N}_{K}"
        elif (strategy == MicroLinearStrategy.GEMM_ADD):
            self.name = "linear_gemm_add_tl"+f"_{M}_{N}_{K}"
        else:
            self.name = "linear_gemm_tl"+f"_{M}_{N}_{K}"
            
        self.thread_num = 128 # get_thread_num() # ws下blocksize==thread_num(消费者)+128(生产者)

        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        
        self.hparam_space = self._get_hparam_space()
        print(len(self.hparam_space))
        
    def _get_hparam_space(self):
        # BLOCK_M=[16] # , 32, 64, 256
        # BLOCK_N=[64] # 64, 128, 256
        # BLOCK_K=[32, 64, 128] # 
        # splitks=[1] #, 2, 4
        BLOCK_M=[16, 32, 64, 256]
        BLOCK_N=[64, 128, 256]
        BLOCK_K=[32, 64, 128]
        splitks=[1, 2, 4] #
        num_stages=[0, 1, 2, 3]#
        policies=[T.GemmWarpPolicy.Square, T.GemmWarpPolicy.FullRow]
        enable_rasterations=[True, False]
        thread_nums=[self.thread_num]
        
        res = []
        for m, n, k, splitk, num_stage, thread_num, policy, enable_rasteration in itertools.product(
            BLOCK_M, BLOCK_N, BLOCK_K, splitks, num_stages, thread_nums, policies, enable_rasterations):
            res.append([m, n, k, splitk, num_stage, thread_num, policy, enable_rasteration])

        return res 
    
    def get_heuristic_hparams(self):
        # [BLOCK_M, BLOCK_N, BLOCK_K, splitk, num_stages, threads, policy, enable_rasteration]
        if self.strategy == MicroLinearStrategy.SILU_MUL_GEMM:
            return [64, 128, 64, 1, 2, self.thread_num, 0, False]
        else:
            # return [16, 64, 64, 1, 3, self.thread_num, 0, False]
            return [16, 64, 128, 1, 0, self.thread_num, 0, False]
    
    def gen_test_data(self, selected_hparams):
        import torch
        # a = torch.rand((self.M, self.K), dtype=torch.bfloat16).cuda() * 2 - 1
        # b = torch.rand((self.N, self.K), dtype=torch.bfloat16).cuda() * 2 - 1
        a = torch.randn((self.M, self.K), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((self.N, self.K), dtype=torch.bfloat16, device="cuda")
        if self.strategy == MicroLinearStrategy.GEMM_ADD:
            r = torch.randn((self.M, self.N), dtype=torch.bfloat16, device="cuda")
            return [a, b, r]
        else:
            return [a, b]
        
    def get_torch_ref(self):
        from xop.ops.dsl.pkt_util import TorchRef
        def torch_ref(a, b):
            return TorchRef.linear(a, b) 
        def torch_add_ref(a, b, r):
            return TorchRef.linear(a, b) + r
        
        if self.strategy == MicroLinearStrategy.GEMM_ADD:
            return torch_add_ref
        else:
            return torch_ref
    
    def get_kernel(self, selected_hparams):
        splitk = selected_hparams[3]
        if splitk == 1:
            if self.strategy == MicroLinearStrategy.SILU_MUL_GEMM:
                return self.kernel_silu_mul_main(self.M, self.N, self.K, *selected_hparams, self.dtype, self.accum_dtype) 
            elif self.strategy == MicroLinearStrategy.GEMM_ADD:
                return self.kernel_add_main(self.M, self.N, self.K, *selected_hparams, self.dtype, self.accum_dtype) 
            else:
                return self.kernel_main(self.M, self.N, self.K, *selected_hparams, self.dtype, self.accum_dtype) 
        else:
            return self.kernel_splitk_main(self.M, self.N, self.K, *selected_hparams, self.dtype, self.accum_dtype) 

    @tilelang.jit(out_idx=[-1], target=get_target_str(), pass_configs=get_pass_configs())
    def kernel_silu_mul_main(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, split_k, num_stages, thread_num, policy, enable_rasteration, dtype=T.float16, accum_dtype=T.float32):
        
        @T.prim_func
        def linear(
            A: T.Tensor((M, 2*K), dtype),
            B: T.Tensor((N, K),   dtype),
            C: T.Tensor((M, N),   dtype)
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=thread_num) as (bx, by):
                A_sh  = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
                B_sh  = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
                C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
                C_sh = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)

                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.annotate_layout({C_sh: tilelang.layout.make_swizzled_layout(C_sh)})
                T.clear(C_local)
                
                # 把 silu-mul 也做进 pipeline
                for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                    left  = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
                    right = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
                    T.copy(A[by * BLOCK_M, k * BLOCK_K], left)
                    T.copy(A[by * BLOCK_M, K + k * BLOCK_K], right)

                    # 2. 就地 silu-mul，结果写回 A_sh
                    for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                        x   = left[i, j].astype(accum_dtype)
                        sig = 1.0 / (1.0 + T.exp(-x))
                        A_sh[i, j] = (x * sig * right[i, j]).astype(dtype)

                    T.copy(B[bx * BLOCK_N, k * BLOCK_K], B_sh)
                    T.gemm(A_sh, B_sh, C_local, transpose_B=True, policy=policy)

                T.copy(C_local, C_sh)
                T.copy(C_sh, C[by * BLOCK_M, bx * BLOCK_N])

        return linear
        
    @tilelang.jit(out_idx=[-1], target=get_target_str(), pass_configs=get_pass_configs())
    def kernel_add_main(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, split_k, num_stages, thread_num, policy, enable_rasteration, dtype=T.float16, accum_dtype=T.float32):
        @T.prim_func
        def linear(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            R: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype)
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
                B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
                C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
                C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
                
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.annotate_layout({C_shared: tilelang.layout.make_swizzled_layout(C_shared)})
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                    T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                    T.copy(B[bx * BLOCK_N, k * BLOCK_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True, policy=policy)
                
                # note: 使用smem，在禁用ws+开启tma时，无法通过，会触发release[i].empty()，需要改用local。
                #       因为在ws逻辑中，smem 写入后可以被其他线程读取, 则需要建立 Producer-Consumer 依赖。
                #       WarpSpecializedRoleMarker 会追踪跨语句的 shared memory 访问，所以需要 acquire/release barrier 来同步。
                #       而Fragment，写入后只被同一个线程读取，不跨线程依赖，不触发 Producer-Consumer 分析，无需 barrier 同步。
                # 依赖链：Producer 加载 R 到 shared memory，Consumer 从 shared memory 读取 R_sh。但是 T.Parallel 里的 R_sh[i,j] 读取没有被正确识别为 R_sh 的消费者，所以出现了消费者缺失的情况。
                # R_sh = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
                # T.copy(R[by * BLOCK_M, bx * BLOCK_N], R_sh)
                R_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
                T.copy(R[by * BLOCK_M, bx * BLOCK_N], R_local)
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    c_val = C_local[i, j]
                    r_val = R_local[i, j].astype(accum_dtype)
                    C_shared[i, j] = (c_val + r_val).astype(dtype)
                    
                # T.copy(C_local, C_sh)
                T.copy(C_shared, C[by * BLOCK_M, bx * BLOCK_N])

        return linear
    
    @tilelang.jit(out_idx=[-1], target=get_target_str(), pass_configs=get_pass_configs())
    def kernel_main(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, split_k, num_stages, thread_num, policy, enable_rasteration, dtype=T.float16, accum_dtype=T.float32):
        
        @T.prim_func
        def linear(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
                B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
                C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
                C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.annotate_layout({C_shared: tilelang.layout.make_swizzled_layout(C_shared)})
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                    T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                    T.copy(B[bx * BLOCK_N, k * BLOCK_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True, policy=policy)
                    
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * BLOCK_M, bx * BLOCK_N])

        return linear

    @tilelang.jit(out_idx=[-1], target=get_target_str())
    def kernel_splitk_main(M, N, K, block_M, block_N, block_K, split_k, num_stages, thread_num, policy, enable_rasteration, dtype=T.float16, accum_dtype=T.float32):
        splitK = K // split_k

        @T.prim_func
        def linear(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=thread_num) as (bx, by, bz):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.annotate_layout({C_shared: tilelang.layout.make_swizzled_layout(C_shared)})
                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                    T.copy(B[bx * block_N, bz * splitK + ko * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True, policy=policy)

                T.copy(C_local, C_shared)
                T.atomic_add(C[by * block_M, bx * block_N], C_shared)

        return linear
    
class MicroLinearStrategy(Enum):
    GEMV = 0
    GEMM = 1
    SILU_MUL_GEMM = 2
    GEMM_ADD = 3
    
class MicroLinear(BaseMicroKernel):
    def __init__(self, strategy:MicroLinearStrategy, M, N, K, dtype=T.bfloat16, accum_dtype=T.float32):
        super().__init__()
        
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        if strategy == MicroLinearStrategy.GEMV:
            self.strategy = _GemvStrategy(M, N, K, dtype, accum_dtype)
        else:
            self.strategy = _GemmStrategy(strategy, M, N, K, dtype, accum_dtype)
        
    def get_source(self, kernel, selected_hparams):
        head_str = \
'''
template <typename T,
    int THREAD_NUM,
    int TILE_DIM_X, 
    int TILE_DIM_Y, 
    int TILE_DIM_Z,
    int M,
    int N,
    int K>
    __device__ __forceinline__ void <kernel_name>(const int bx, const int by, const int bz,
                                                <io_params>) {
  // static_assert(THREAD_NUM==<threads>);
  static_assert(TILE_DIM_X==<BLOCK_N>); static_assert(TILE_DIM_Y==<BLOCK_M>); static_assert(TILE_DIM_Z==<BLOCK_K>);
  static_assert(M==<M>); static_assert(N==<N>); static_assert(K==<K>);
  if (bx >= <gridx_0> || by >= <gridy_0> || bz >= <gridz_0>) { return; }
'''
        sm89_io_str = \
'''const void* __restrict__ input_ptr, const void* __restrict__ weight_ptr, const void* __restrict__ residual_ptr, void* __restrict__ output_ptr '''
        sm90_io_str = \
'''uint64_t* mbarrier_mem, const CUtensorMap *A_desc, const CUtensorMap *B_desc, const void* __restrict__ residual_ptr, const CUtensorMap *C_desc '''
        sm89_io_warp_str = \
'''
  const <dtype>* __restrict__ A = static_cast<const <dtype>*>(input_ptr);
  const <dtype>* __restrict__ B = static_cast<const <dtype>*>(weight_ptr);
  const <dtype>* __restrict__ R = static_cast<const <dtype>*>(residual_ptr);
  <dtype>* __restrict__ C = static_cast<<dtype>*>(output_ptr);
'''
        sm90_io_warp_str = \
'''
  const <dtype>* __restrict__ R = static_cast<const <dtype>*>(residual_ptr);
'''

        if (isinstance(self.strategy, _GemvStrategy)):
            BLOCK_N, reduce_threads = selected_hparams
            BLOCK_M, BLOCK_K, threads = 1, 1, 128 # todo
        else:
            BLOCK_M, BLOCK_N, BLOCK_K, splitk, num_stages, threads, policy, enable_rasteration = selected_hparams
        # gridDim_x, gridDim_y = self.get_grid_dims(self.M, self.N, BLOCK_M, BLOCK_N)
        # print(f"gridDim: ({gridDim_x}, {gridDim_y})")
        
        head_str = head_str.replace('<threads>', str(threads))
        head_str = head_str.replace('<BLOCK_M>', str(BLOCK_M))
        head_str = head_str.replace('<BLOCK_N>', str(BLOCK_N)) 
        head_str = head_str.replace('<BLOCK_K>', str(BLOCK_K)) 
        head_str = head_str.replace('<M>', str(self.M))
        head_str = head_str.replace('<N>', str(self.N)) 
        head_str = head_str.replace('<K>', str(self.K)) 
        head_str = head_str.replace('<kernel_name>', self.strategy.name)

        source = kernel.get_kernel_source()
        artifact = get_artifact(kernel)
        grid_dim, block_dim, dynamic_smem_buf, use_cooperative_groups = get_launch_info(artifact)[0]
        self.layout = f"({grid_dim['blockIdx.x']}, {grid_dim['blockIdx.y']}, {grid_dim['blockIdx.z']}), ({BLOCK_N}, {BLOCK_M}, {BLOCK_K})"        
        print("layout", self.layout)
        
        if self.dtype == T.bfloat16:
            dtype = "bfloat16_t"
        else:
            dtype = "float16_t"
        if (get_arch() == "sm_89"):
            head_str += sm89_io_warp_str.replace('<dtype>', str(dtype))
            source = self.replace_header(source, "extern \"C\" __global__", 1, head_str)
            source = source.replace("<io_params>", sm89_io_str)
        else:
            head_str += sm90_io_warp_str.replace('<dtype>', str(dtype))
            source = self.replace_header(source, "extern \"C\" __global__", 1, head_str)
            source = source.replace("A_desc", "*A_desc")
            source = source.replace("B_desc", "*B_desc")
            source = source.replace("R_desc", "*R_desc")
            source = source.replace("C_desc", "*C_desc")
            source = source.replace("<io_params>", sm90_io_str)
            # 
            source = source.replace("__shared__ uint64_t mbarrier_mem", "// __shared__ uint64_t mbarrier_mem")
            
        source = source.replace("blockIdx.x", "bx")
        source = source.replace("blockIdx.y", "by")
        source = source.replace("blockIdx.z", "bz")
    
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