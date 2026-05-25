import os
import shutil
from pathlib import Path
import re

import torch
import tilelang
import tilelang.language as T

from .pkt_util import TestUtil, TorchRef
from .micro_base import HparamSelectMode
from .micro_linear import MicroLinearStrategy, MicroLinear
from .micro_rmsnorm import MicroRmsNorm
from .micro_silu_mul import MicroSiluMul
from .micro_gqa_decode import MicroGqaDecode
from .micro_rope import MicroRope
from xop.ops.dsl.micro_config import get_arch, get_target_str, is_megakernel_enabled, is_enable_profiling

class MicroAutoGen:
    def __init__(self, model_tag, batch_size, hidden_size, intermediate_size, max_kv_seqlen, num_heads, num_kv_heads, head_dim):
        print(tilelang.__version__) # 预先加载完成FFI的静态初始化，以免初始化发生在tuning的多线程场景导致崩溃
        self.model_tag = model_tag
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_kv_seqlen = max_kv_seqlen
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.dtype = T.bfloat16
        self.accum_dtype = T.float32
        
    def _save_target_info(self, micro, mode: HparamSelectMode, code_dir, config_file, name):
        kernel, file_name, info = micro.get_kernel(mode)
        print(file_name, info)
        shutil.copy2(file_name, code_dir)
        config_file.write(f"    {name} = {info}\n")
    
    def _save_target_dict_info(self, micro, mode: HparamSelectMode, code_dir, config_file, seqlen, split_point, max_longkv, max_shortkv):
        kernel, file_name, info = micro.get_kernel(mode)
        print(file_name, info)
        shutil.copy2(file_name, code_dir)
        config_file.write(f"        {seqlen}: ({info}),\n")
        
        numbers = re.findall(r'\d+', info)
        if (len(numbers) == 6):
            grid = tuple(map(int, numbers[:3]))
            tile = tuple(map(int, numbers[3:]))
            if (grid > max_shortkv[0]):
                max_shortkv[0] = grid
                max_shortkv[1] = tile
            if (seqlen > split_point):
                split_point = seqlen # 取非split的最长长度
        else:
            grid1 = tuple(map(int, numbers[:3]))
            tile1 = tuple(map(int, numbers[3:6]))
            grid2 = tuple(map(int, numbers[6:9]))
            tile2 = tuple(map(int, numbers[9:12]))
            if (grid1 > max_longkv[0]):
                max_longkv[0] = grid1
                max_longkv[1] = tile1
            if (grid2 > max_longkv[2]):
                max_longkv[2] = grid2
                max_longkv[3] = tile2
            if (seqlen < split_point): 
                print("[seqlen < split_point] Check the thread layout of GQA to determine \
                    why short sequences undergo splitting whereas long sequences do not.")
                assert(0)
        return split_point, max_longkv, max_shortkv
        
    def gen_qwen3_ops(self, layer_id: int, mode: HparamSelectMode):
        
        xop_home = os.getenv("XOP_HOME", default=None)
        if xop_home is None:
            raise EnvironmentError("The environment variable XOP_HOME is not set.")
        code_path = xop_home + f"/include/megakernel/persistent_kernel/tasks/{get_arch()}/m{self.batch_size}/"
        code_dir = Path(code_path)
        code_dir.mkdir(parents=True, exist_ok=True)
        
        dsl_home = os.getenv("DSL_HOME", default=None)
        config_path = dsl_home + f"/autogen/{get_arch()}/"
        config_dir = Path(config_path)
        config_dir.mkdir(parents=True, exist_ok=True)
        
        ## L40
        # rmsnorm_layout = (1, 1, 1), (1, 1, 1)
        # linear1_layout = (192, 1, 1), (32, 16, 256)
        # silu_mul_layout = (96, 1, 1), (32, 16, 1)
        # linear2_layout = (32, 1, 1), (32, 16, 256)        
        ## rtx4050
        # rmsnorm_layout = (1, 1, 1), (1, 1, 1)
        # linear1_layout = (96, 1, 1), (64, 16, 64)
        # silu_mul_layout = (48, 1, 1), (64, 16, 1)
        # linear2_layout = (32, 1, 1), (32, 16, 128)
    
        with open(config_path+f"{self.model_tag}_mega_config.py", "w", encoding="utf-8") as config_file:
            config_file.write(f"class Qwen3MegaConfig{self.model_tag.split('_')[1]}:\n")
            if (layer_id == 0 or layer_id == 99):
                kernel = MicroRmsNorm(self.batch_size, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target_info(kernel, mode, code_dir, config_file, "rmsnorm_layout")
            if (layer_id == 1 or layer_id == 99):
                kernel = MicroLinear(MicroLinearStrategy.GEMM, self.batch_size, self.intermediate_size*2, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target_info(kernel, mode, code_dir, config_file, "linear1_layout")
            if (layer_id == 2 or layer_id == 99):   
                kernel = MicroSiluMul(self.batch_size, self.intermediate_size, dtype=T.bfloat16, accum_dtype=T.float32)
                self._save_target_info(kernel, mode, code_dir, config_file, "silu_mul_layout")
            if (layer_id == 3 or layer_id == 99):
                kernel = MicroLinear(MicroLinearStrategy.GEMM_ADD, self.batch_size, self.hidden_size, self.intermediate_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target_info(kernel, mode, code_dir, config_file, "linear2_layout")
            if (layer_id == 4 or layer_id == 99):
                kernel = MicroLinear(MicroLinearStrategy.GEMM, self.batch_size, (self.num_heads+2*self.num_kv_heads)*self.head_dim, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target_info(kernel, mode, code_dir, config_file, "qkv_proj_layout")
            if (layer_id == 5 or layer_id == 99):
                kernel = MicroRmsNorm(self.batch_size*self.num_heads, self.head_dim, dtype=self.dtype, accum_dtype=self.accum_dtype, M2=self.batch_size*self.num_kv_heads)
                self._save_target_info(kernel, mode, code_dir, config_file, "merge_q_k_norm_layout")
            # if (layer_id == 6 or layer_id == 99):
            #     kernel = MicroRmsNorm(self.batch_size*self.num_kv_heads, self.head_dim, dtype=self.dtype, accum_dtype=self.accum_dtype)
            #     self._save_target_info(kernel, mode, code_dir, config_file, "k_norm_layout")
            if (layer_id == 6 or layer_id == 99):
                kernel = MicroRope(self.batch_size, 1, self.num_heads, self.num_kv_heads, self.head_dim, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target_info(kernel, mode, code_dir, config_file, "rope_layout")
            if (layer_id == 7 or layer_id == 99):
                # for target_kv_seqlen in list(range(1, 17)): # 
                #     kernel = MicroGqaDecode(self.batch_size, self.max_kv_seqlen, target_kv_seqlen, self.num_heads, self.num_kv_heads, self.head_dim, False, dtype=self.dtype, accum_dtype=self.accum_dtype)
                #     self._save_target_info(kernel, HparamSelectMode.HEURISTIC, code_dir, config_file, "gqa_decode_layout_"+str(target_kv_seqlen))
                
                split_point = 0
                max_shortkv = [(1, 1, 1), (1, 1, 1)]
                max_longkv = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
                config_file.write(f"    gqa_decode_layouts = {{\n")
                for target_kv_seqlen in [16, 32, 64, 128, 256, 512, 1024, 2048]: # 
                    kernel = MicroGqaDecode(self.batch_size, self.max_kv_seqlen, target_kv_seqlen, self.num_heads, self.num_kv_heads, self.head_dim, False, dtype=self.dtype, accum_dtype=self.accum_dtype)
                    split_point, max_longkv, max_shortkv = self._save_target_dict_info(kernel, mode, code_dir, config_file, target_kv_seqlen, split_point, max_longkv, max_shortkv)
                config_file.write(f"    }}\n")
                config_file.write(f"    gqa_decode_layout_shortkv = {tuple(max_shortkv)}\n")
                config_file.write(f"    gqa_decode_layout_longkv  = {tuple(max_longkv)}\n")
                config_file.write(f"    gqa_decode_layout_split_point = {split_point}\n")
            if (layer_id == 8 or layer_id == 99):
                kernel = MicroLinear(MicroLinearStrategy.GEMM_ADD, self.batch_size, self.hidden_size, self.num_heads*self.head_dim, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target_info(kernel, mode, code_dir, config_file, "o_proj_layout")
            if (layer_id == 9 or layer_id == 99):
                # vocab_size: 151936
                kernel = MicroLinear(MicroLinearStrategy.GEMM, self.batch_size, 151936, self.hidden_size, dtype=self.dtype, accum_dtype=self.accum_dtype)
                self._save_target_info(kernel, mode, code_dir, config_file, "lm_head")
            
        