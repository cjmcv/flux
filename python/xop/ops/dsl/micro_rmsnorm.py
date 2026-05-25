import itertools

import tilelang
import tilelang.language as T

from xop.ops.dsl.micro_base import BaseMicroKernel, HparamSelectMode, get_artifact, get_launch_info, get_dispatch_source
from xop.ops.dsl.micro_config import get_arch, get_thread_num, get_target_str, is_megakernel_enabled, get_pass_configs

class _RmsNormStrategy:
    def __init__(self, M, M2, N, dtype, accum_dtype):
        self.name = "rms_norm_tl"+f"_{M+M2}_{N}"
            
        self.thread_num = 128 # get_thread_num()
        self.M = M
        self.M2 = M2
        self.N = N
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        
        self.hparam_space = self._get_hparam_space()
        print(len(self.hparam_space))
        
    def _get_hparam_space(self):
        BLOCK_M=[1] #, 256
        BLOCK_N=[1] # 
        thread_nums=[self.thread_num]#
        
        res = []
        for m, n, thread_num in itertools.product(
           BLOCK_M, BLOCK_N, thread_nums):
            res.append([m, n, thread_num])
        return res 
    
    def get_heuristic_hparams(self):
        # [threads]
        return [1,1,self.thread_num]
        
    def gen_test_data(self, selected_hparams):
        import torch
        if (self.M2 == 0):
            a = torch.randn(self.M, self.N, dtype=torch.bfloat16, device="cuda")
            b = torch.randn(1, self.N, dtype=torch.bfloat16, device="cuda")
            return [a, b]
        else:
            a = torch.randn(self.M+self.M2, self.N, dtype=torch.bfloat16, device="cuda")
            b = torch.randn(2, self.N, dtype=torch.bfloat16, device="cuda")
            return [a, b]
        
    def get_torch_ref(self):
        from xop.ops.dsl.pkt_util import TorchRef
        if (self.M2 == 0):
            def torch_ref(a, b):
                return TorchRef.rms_norm(a, b) 
            return torch_ref
        else:
            import torch
            def torch_ref(a, b):
                a1 = a[ : self.M, :]
                a2 = a[self.M : self.M+self.M2, :]
                b1 = b[0 : 1, :]
                b2 = b[1 : 2, :]
                # print(a1.shape, a2.shape, b1.shape,b2.shape)
                c1 = TorchRef.rms_norm(a1, b1) 
                c2 = TorchRef.rms_norm(a2, b2) 
                return torch.cat([c1, c2], dim=0)
            return torch_ref
        
    def get_kernel(self, selected_hparams):
        print("selected_hparams: ", selected_hparams)
        if (self.M2 == 0):
            return self.kernel_main(self.M, self.N, *selected_hparams, 1e-12, self.dtype, self.accum_dtype) 
        else:
            return self.kernel_merge_main(self.M, self.M2, self.N, *selected_hparams, 1e-12, self.dtype, self.accum_dtype) 
    
    @tilelang.jit(out_idx=[-1], target=get_target_str(), pass_configs={"tl.disable_tma_lower": True})
    def kernel_main(M, N, BLOCK_M, BLOCK_N, threads, eps=1e-12, dtype="bfloat16", accum_dtype="float32"):
        @T.prim_func
        def rms_norm(A: T.Tensor((M, N), dtype), B: T.Tensor((1, N), dtype), C: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(M, BLOCK_M), threads=threads) as bx:
                A_shared = T.alloc_shared((BLOCK_M, N), dtype)
                A_pow_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
                A_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
                A_powsum = T.alloc_fragment((BLOCK_M,), accum_dtype)
                B_shared = T.alloc_shared((1, N), dtype)
                B_local = T.alloc_fragment((1, N), accum_dtype)
                
                T.copy(A[bx * BLOCK_M : (bx + 1) * BLOCK_M, :], A_shared)
                T.copy(B[0:1, :], B_shared)
                
                # note: 在sm120, M=BLOCK_M=1, N=1024, threads=256时，下面的A拷贝需要加上“coalesced_width=1”转为标量处理才能正常生成kernel。
                # 因为向量化读取使用 float4 = 4 × 32位 = 8 x bfloat16，N = 1024 x bfloat16，所需线程数 = 1024 / 8 = 128 个线程。
                # coalesced_width=1时，转用uint32 = 2 x bfloat16，1024 / 2 = 512 可以满足。
                T.copy(A_shared, A_local) #, coalesced_width=1
                T.copy(B_shared, B_local)
                
                for i, j in T.Parallel(BLOCK_M, N):
                    A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
                T.reduce_sum(A_pow_local, A_powsum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    A_powsum[i] = T.rsqrt(A_powsum[i] / N + eps)
                for i, j in T.Parallel(BLOCK_M, N):
                    A_local[i, j] *= A_powsum[i] * B_local[0, j]
                T.copy(A_local, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, :])

        return rms_norm
        
    @tilelang.jit(out_idx=[-1], target=get_target_str(), pass_configs={"tl.disable_tma_lower": True})
    def kernel_merge_main(M1, M2, N, BLOCK_M, BLOCK_N, threads, eps=1e-12, dtype="bfloat16", accum_dtype="float32"):
        @T.prim_func
        def rms_norm(A: T.Tensor((M1+M2, N), dtype), B: T.Tensor((2, N), dtype), C: T.Tensor((M1+M2, N), dtype)):
            with T.Kernel(T.ceildiv(M1+M2, BLOCK_M), threads=threads) as bx:
                A_shared = T.alloc_shared((BLOCK_M, N), dtype)
                A_pow_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
                A_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
                A_powsum = T.alloc_fragment((BLOCK_M,), accum_dtype)
                B_shared = T.alloc_shared((1, N), dtype)
                B_local = T.alloc_fragment((1, N), accum_dtype)
                
                T.copy(A[bx * BLOCK_M : (bx + 1) * BLOCK_M, :], A_shared)
                
                if bx < M1:
                    T.copy(B[0:1, :], B_shared)
                else:
                    T.copy(B[1:2, :], B_shared)
                T.copy(A_shared, A_local)
                T.copy(B_shared, B_local)
                
                for i, j in T.Parallel(BLOCK_M, N):
                    A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
                T.reduce_sum(A_pow_local, A_powsum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    A_powsum[i] = T.rsqrt(A_powsum[i] / N + eps)
                for i, j in T.Parallel(BLOCK_M, N):
                    A_local[i, j] *= A_powsum[i] * B_local[0, j]
                T.copy(A_local, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, :])

        return rms_norm
    
class MicroRmsNorm(BaseMicroKernel):
    def __init__(self, M, N, dtype=T.bfloat16, accum_dtype=T.float32, M2=0):
        super().__init__()
        self.strategy = _RmsNormStrategy(M, M2, N, dtype, accum_dtype)
        
    def get_source(self, kernel, selected_hparams):
        head_str = \
'''
template <typename T,
        int THREAD_NUM,
        int TILE_DIM_X, 
        int TILE_DIM_Y, 
        int TILE_DIM_Z,
        int M,
        int N>
__device__ __forceinline__ void rms_norm_kernel_<name_suffix>(const int bx, const int by, const int bz,
                                                            void const *input_ptr,
                                                            void const *weight_ptr,
                                                            void *output_ptr,
                                                            float eps) {
  // static_assert(THREAD_NUM==<threads>);
  static_assert(TILE_DIM_X==<BLOCK_N>); static_assert(TILE_DIM_Y==<BLOCK_M>); static_assert(TILE_DIM_Z==<BLOCK_K>);
  static_assert(M==<M>); static_assert(N==<N>);
  if (bx >= <gridx_0> || by >= <gridy_0> || bz >= <gridz_0> || threadIdx.x >= 128) { return; }
  
  const <dtype>* __restrict__ A = static_cast<const <dtype>*>(input_ptr);
  const <dtype>* __restrict__ B = static_cast<const <dtype>*>(weight_ptr);
  <dtype>* __restrict__ C = static_cast<<dtype>*>(output_ptr);
  
'''     
        BLOCK_M, BLOCK_N, threads = selected_hparams
        BLOCK_K = 1
        
        head_str = head_str.replace('<threads>', str(threads))
        head_str = head_str.replace('<BLOCK_M>', str(BLOCK_M))
        head_str = head_str.replace('<BLOCK_N>', str(BLOCK_N)) 
        head_str = head_str.replace('<BLOCK_K>', str(BLOCK_K)) 
        head_str = head_str.replace('<M>', str(self.strategy.M+self.strategy.M2))
        head_str = head_str.replace('<N>', str(self.strategy.N)) 
        if (self.strategy.M2 == 0):
            head_str = head_str.replace('<name_suffix>', str(self.strategy.M)+"_"+str(self.strategy.N))
        else:
            head_str = head_str.replace('<name_suffix>', str(self.strategy.M)+"_"+str(self.strategy.M2)+"_"+str(self.strategy.N))
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
        self.layout = f"({grid_dim['blockIdx.x']}, {grid_dim['blockIdx.y']}, {grid_dim['blockIdx.z']}), ({BLOCK_N}, {BLOCK_M}, {BLOCK_K})"
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