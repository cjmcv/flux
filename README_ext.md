# XOP (Cutlass Operators) 交接文档

## 项目概述

XOP 是一个基于 NVIDIA CUTLASS 的高性能 CUDA 矩阵运算库，专注于 GEMM（General Matrix Multiply）操作和注意力机制的实现与优化。项目支持多种 GPU 架构（Ampere sm80、Ada sm89、Hopper sm90、Blackwell sm120），提供 FP16、BF16、FP8（E4M3/E5M2）、INT8 等多种数据类型的支持。

**版本**: 1.1.1  
**许可证**: Apache 2.0

---

## 目录结构

```
flux/
├── CMakeLists.txt              # 根 CMake 构建配置
├── build.sh                    # 构建脚本
├── setup.py                    # Python 包安装配置
├── README.md                   # 项目简述
├── pyproject.toml              # Python 项目元数据
│
├── include/xop/                # 公共头文件
│   ├── xop.h                   # 核心枚举、宏定义、UnifiedMetaEnum
│   ├── common_cuda.h           # CUDA 工具函数
│   ├── common_torch.h          # PyTorch 集成辅助函数
│   ├── common_nccl.h           # NCCL 通信工具
│   ├── common_strategy.h       # 调优策略
│   ├── dsl/                    # 领域特定语言头文件
│   ├── lt_coll/                # cuBLASlt collective 操作
│   ├── ops_impl/               # 算子实现
│   │   ├── arguments.h         # 运行时参数结构
│   │   ├── global_resource.h   # 全局资源管理（单例）
│   │   ├── gemm_normal/        # 普通 GEMM 内核
│   │   └── gemm_comm/          # 通信 GEMM 内核
│   └── megakernel/             # Megakernel 头文件
│
├── src/                        # 源代码
│   ├── ops/                    # 算子实现
│   │   ├── gemm_normal/        # 主 GEMM 算子（当前启用）
│   │   │   ├── gemm_normal.cpp # 主实现（约 18000 行）
│   │   │   ├── gemm_normal.h   # 公共 API
│   │   │   ├── common/         # 公共内核（融合激活、预处理）
│   │   │   └── tuning_config/  # 搜索空间和调优配置
│   │   ├── flash_attn/         # Flash Attention（已禁用）
│   │   ├── gemm_comm/          # 通信 GEMM
│   │   ├── allreduce_normal/   # 自定义 allreduce 内核
│   │   └── marlin/             # Marlin 量化内核
│   └── pybind/                 # Python 绑定
│       ├── ths_op.cpp          # PyTorch 扩展绑定代码
│       └── ths_op.h            # TorchClassWrapper 和注册表
│
├── python/xop/                 # Python 包
│   ├── __init__.py             # 包初始化，导入核心模块
│   ├── cpp_mod.py              # C++ 模块包装器
│   ├── common.py               # 元枚举、辅助函数
│   ├── op_util.py              # 算子工具
│   ├── util.py                 # 通用工具
│   ├── cuda_wrapper.py         # CUDA 包装器
│   ├── ops/                    # Python 侧算子接口
│   │   ├── gemm_quant.py       # 量化 GEMM
│   │   ├── gemm_comm.py        # 通信 GEMM
│   │   ├── flash_attn.py       # Flash Attention 包装器
│   │   ├── custom_all_reduce.py
│   │   ├── cuda_ipc_manager.py
│   │   └── dsl/                # DSL 内核生成
│   │       ├── micro_base.py       # 微内核基类
│   │       ├── micro_config.py     # 配置
│   │       ├── micro_gqa_decode.py # GQA 解码
│   │       ├── micro_linear.py     # 线性层
│   │       ├── micro_rmsnorm.py    # RMSNorm
│   │       ├── micro_rope.py       # RoPE
│   │       ├── micro_silu_mul.py   # SiLU 融合
│   │       ├── mk_layers.py        # Megakernel 层
│   │       └── pkt_util.py         # 数据包工具
│   ├── include/                # 安装的头文件
│   └── lib/                    # 编译的共享库
│
├── megakernel/                 # Megakernel Python 包
│   ├── core.cpython-311.so     # 编译的 Cython 核心
│   ├── persistent_kernel.py    # Persistent kernel 运行时
│   ├── kernel.py               # 内核定义
│   └── gen/                    # 生成的内核
│
├── tools/                      # 测试和调优工具
│   ├── gemm/
│   │   ├── test_gemm_normal.py      # 主 GEMM 测试
│   │   ├── test_gemm_blockscale_fp8.py
│   │   ├── test_gemm_comm.py
│   │   ├── test_gemm_mini.py
│   │   ├── gen_search_space.py       # 生成搜索空间配置
│   │   └── tuning/                   # 调优脚本
│   │       ├── tune_gemm_normal.py
│   │       ├── tune_gemm_comm.py
│   │       └── tune_common.py
│   ├── test_mega_single.py     # Megakernel 单算子测试
│   ├── test_mega_attn.py       # Megakernel attention 测试
│   ├── test_mega_mlp.py        # Megakernel MLP 测试
│   ├── test_dsl.py             # DSL 生成测试
│   └── models/                 # 模型集成测试
│
├── test/                       # C++ 测试程序
│   ├── test_gemv_only.cu
│   ├── 55_hopper_mixed_dtype_gemm/
│   └── cute_practice/          # CUTLASS 练习
│
├── 3rdparty/                   # 外部依赖
│   ├── cutlass/                # NVIDIA CUTLASS（子模块）
│   ├── json/                   # JSON 库
│   └── nccl/                   # NCCL 集合通信
│
└── draft/                      # 实验性/临时代码
```

---

## 核心 API

### 1. C++ API

#### 1.1 GemmNormal 类

主 GEMM 算子类，定义于 [gemm_normal.h](src/ops/gemm_normal/gemm_normal.h)

```cpp
namespace xop {
class GemmNormal {
public:
  GemmNormal(
      c10::ScalarType input_dtype,    // 输入数据类型
      c10::ScalarType output_dtype,   // 输出数据类型
      bool transpose_weight);         // 权重转置标志

  // 前向传播
  torch::Tensor forward(
      torch::Tensor lhs,                          // 左矩阵 [M, K]
      torch::Tensor rhs,                          // 右矩阵 [N, K] 或 [K, N]
      c10::optional<torch::Tensor> output,        // 输出缓冲区
      c10::optional<torch::Tensor> bias,          // 偏置
      c10::optional<torch::Tensor> input_scale,   // 输入缩放因子
      c10::optional<torch::Tensor> weight_scale,  // 权重缩放因子
      c10::optional<torch::Tensor> output_scale,  // 输出缩放因子
      c10::optional<torch::Tensor> tuning,        // 调优参数
      bool fast_accum);                           // 快速累加标志

  // 分组前向传播（用于批量 GEMM）
  int grouped_forward(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      std::vector<torch::Tensor> outputs,
      c10::optional<std::vector<torch::Tensor>> inputs_scale,
      c10::optional<std::vector<torch::Tensor>> weights_scale,
      c10::optional<torch::Tensor> tuning);
};
}
```

#### 1.2 核心枚举定义

定义于 [xop.h](include/xop/xop.h)

```cpp
// 算子类型
enum class UnifiedMetaEnum : int8_t {
  GemmNormal = 0, GemvSimt, GemmNormalSimt, GemmGrouped,    // gemm 类型
  GemmBlockScaleFp8, GemmGroupedBlockScaleFp8, 
  GemmW4A16, GemmLt, 
  GemmAllreduce = 20,
  Void = 50, FP16, BF16, FP32, E4M3, E5M2, S8, S32,    // 数据类型
  Sm80 = 60, Sm89, Sm90, Sm100, Sm120,                 // 架构
  RRR = 70, RCR, RCC                                   // 布局
};

// 运行模式
enum RunModeEnum {
  kRunWithNormal = 0,    // 正常运行
  kRunWithTuning = 1,    // 调优模式
  kRunWithHparam = 2,    // 超参数模式
};

// 元数据位置索引
enum MetaLocEnum {
  kMetaId = 0, kMetaSchema = 1, kMetaTypeA = 2,
  kMetaTypeB = 3, kMetaTypeCD = 4, kMetaTypeAcc = 5,
  kMetaLayout = 6, kMetaArch = 7
};
```

#### 1.3 运行时参数结构

定义于 [arguments.h](include/xop/ops_impl/arguments.h)

```cpp
// 基础运行时参数
struct RtArguments {
  int m, n, k, l, g;           // 矩阵维度 [M, N, K], batch, groups
  void *ptr_A, *ptr_B;         // 输入矩阵指针
  void *ptr_C, *ptr_D;         // 输出矩阵指针
  float alpha, beta;           // GEMM 缩放因子
  int C_s;                     // C 步长
};

// V2 版本（带步长）
struct RtArgumentsV2 : public RtArguments {
  int stride_a, stride_b, stride_c, stride_d;
};

// BlockScale 参数（FP8 量化）
struct RtBlockScaleArguments : public RtArguments {
  float scale_a, scale_b, scale_c, scale_d, scale_aux;
  void *ptr_blockscale_A, *ptr_blockscale_B;
};

// 分组 GEMM 参数
struct RtGroupedArguments : public RtArgumentsBase {
  int groups;
  std::vector<int32_t> problem_sizes;    // [m,n,k,m,n,k,...]
  std::vector<void const *> ptr_A, ptr_B;
  std::vector<void const *> ptr_C;
  std::vector<void *> ptr_D;
  float alpha, beta;
};
```

#### 1.4 全局资源管理

定义于 [global_resource.h](include/xop/ops_impl/global_resource.h)

```cpp
// 全局缓冲区单例
class GlobalBuffer {
  static GlobalBuffer& instance();
  
  void SetTuningFlag(bool is_tuning);
  void CheckDeviceBufferAllocate(DeviceBufferPoolKindEnum pool_id, size_t size);
  uint8_t* GetDeviceBuffer(DeviceBufferPoolKindEnum pool_id, size_t size);
  void PrintUsedBufferSize();
};

// GEMM 配置注册表（工厂模式）
class GemmConfigRegister {
  static GemmConfigRegister& instance();
  void add(const std::vector<int16_t> &key, GemmFactory factory);
  GemmBase* CreateOp(const std::vector<int16_t> &key, bool is_tuning = false);
  GemmBase* GetOp(const std::vector<int16_t> &key, bool is_tuning = false);
};

// 调优配置注册表
class TunedConfigRegister {
  static TunedConfigRegister& instance();
  void add(const std::vector<int32_t> &key, const std::vector<int16_t> &select_config);
  void GetSelectedConfig(const std::vector<int32_t> &key, int16_t *selected_id, 
                         int16_t *schema_id, uint64_t *cublaslt_algo = nullptr);
};
```

---

### 2. Python API

#### 2.1 GemmNormal

```python
import xop

# 创建算子
gemm = xop.GemmNormal(
    input_dtype=torch.bfloat16,  # 输入数据类型
    output_dtype=torch.bfloat16, # 输出数据类型（可选）
    transpose_weight=False       # 是否转置权重
)

# 前向传播
output = gemm.forward(
    input,           # 输入张量 [M, K]
    weight,          # 权重张量 [N, K] 或 [K, N]
    output,          # 输出缓冲区（可选）
    bias=None,       # 偏置（可选）
    input_scale=None,# 输入缩放（FP8 时需要）
    weight_scale=None,# 权重缩放（FP8 时需要）
    output_scale=None,# 输出缩放（可选）
    tuning=None,     # 调优参数（可选）
    fast_accum=False # 快速累加
)
```

#### 2.2 GemmQuant（量化 GEMM）

定义于 [gemm_quant.py](python/xop/ops/gemm_quant.py)

```python
from xop.ops.gemm_quant import GemmQuant

gemm_quant = GemmQuant(
    input_dtype=torch.bfloat16,
    output_dtype=torch.bfloat16,
    quant_bits=8,     # 8 或 4
    num_groups=128    # 分组数
)

# 权重预处理
weight_fp8, weight_scale = gemm_quant.weight_preprocess(weight, fast_accum=False)

# 前向传播
output = gemm_quant.forward(
    input,
    weight_fp8,
    output,
    bias,
    input_scale,
    weight_scale,
    tuning=None,
    fast_accum=False
)
```

#### 2.3 GemmCommRs（通信 GEMM）

定义于 [gemm_comm.py](python/xop/ops/gemm_comm.py)

```python
from xop.ops.gemm_comm import GemmCommRs

gemm_comm = GemmCommRs(
    input_dtype=torch.bfloat16,
    output_dtype=torch.bfloat16,
    transpose_weight=False,
    group=process_group,  # 进程组
    rank=rank
)

output = gemm_comm.forward(
    input, weight, output,
    bias=None,
    input_scale=None,
    weight_scale=None,
    tuning=None,
    fast_accum=False
)
```

#### 2.4 Python 端元枚举

定义于 [common.py](python/xop/common.py)

```python
from xop.common import Meta, gen_tuned_hparam

# 算子类型
Meta.GemmNormal = 0
Meta.GemmBlockScaleFp8 = 3
Meta.GemmLt = 7

# 数据类型
Meta.FP16 = 51
Meta.BF16 = 52
Meta.FP32 = 53
Meta.E4M3 = 54
Meta.E5M2 = 55
Meta.S8 = 56
Meta.S32 = 57

# 架构
Meta.Sm80 = 60
Meta.Sm89 = 61
Meta.Sm90 = 62

# 布局
Meta.RRR = 70
Meta.RCR = 71
Meta.RCC = 72

# 生成调优超参数
tuned_hparam = gen_tuned_hparam(chunk_size=8192, arch=-1)
```

---

## 设计思路

### 1. 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Layer                           │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │GemmQuant│  │ GemmCommRs   │  │  DSL (MicroKernel)   │   │
│  └────┬────┘  └──────┬───────┘  └──────────┬───────────┘   │
└───────┼──────────────┼────────────────────┼────────────────┘
        │              │                    │
        ▼              ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   PyTorch Extension                         │
│         TorchClassWrapper<T> + ThsOpsInitRegistry          │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    C++ Core Library                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │           GemmConfigRegister (Factory)              │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │    │
│  │  │ GemmNormal│ │GemmBlock │ │ GemmComm │           │    │
│  │  │          │ │ ScaleFp8 │ │          │           │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘           │    │
│  └───────┼────────────┼────────────┼─────────────────┘    │
│          │            │            │                       │
│          ▼            ▼            ▼                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │              GlobalBuffer (Singleton)               │    │
│  │   Workspace Pool │ Aux Pool │ Output Pool          │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      CUTLASS Backend                        │
│          Kernel<<<Grid, Block>>> (CUDA Kernels)            │
└─────────────────────────────────────────────────────────────┘
```

### 2. 核心设计模式

#### 2.1 单例模式 (Singleton)

全局资源使用单例模式管理：
- `GlobalBuffer::instance()` - 设备内存池
- `GemmConfigRegister::instance()` - GEMM 内核注册表
- `TunedConfigRegister::instance()` - 调优配置注册表
- `ThsOpsInitRegistry::instance()` - Python 算子注册表

#### 2.2 工厂模式 (Factory)

`GemmConfigRegister` 根据元数据键（arch + dtype + layout 等）动态创建对应的 GEMM 内核实例。

#### 2.3 注册模式 (Registry)

算子通过静态初始化注册到全局注册表：
```cpp
static int _register_gemm_normal_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_normal", [](py::module &m) {
    py::class_<GemmNormalCls>(m, "GemmNormal")...
  });
  return 0;
}();
```

#### 2.4 策略模式 (Strategy)

定义于 [common_strategy.h](include/xop/common_strategy.h)

```cpp
struct Strategy {
  // M 维度粗粒度调优
  static int CoarseGrainedTuningM(int actual_m, int max_m = 16384);
  
  // M 维度分块
  static std::vector<int> SplitChunkM(const int m, const int chunk_size);
};
```

### 3. 内存管理

`GlobalBuffer` 管理三类设备内存池：
- **Workspace Pool** - GEMM 内核工作区
- **Aux Pool** - 辅助内存（用于 FP8 量化等）
- **Output Pool** - 输出缓冲区

内存分配按 128 字节对齐，支持动态扩容。

### 4. 元数据流

```
Python Input
     │
     ▼
TorchDefaultConfig::MakeDefaultMeta()  ──► id_meta [id, schema, typeA, typeB, typeCD, typeAcc, layout, arch]
     │
     ▼
RunModeEnum GetRunMode(tuning)
     │
     ├─► kRunWithNormal ──► TunedConfigRegister::GetSelectedConfig() ──► GemmConfigRegister::GetOp()
     │
     ├─► kRunWithTuning ──► forward_tuning() ──► 测试所有搜索空间配置
     │
     └─► kRunWithHparam ──► 使用指定的超参数
```

---

## 构建系统

### 构建脚本使用

```bash
# 设置环境变量
source ./build.sh --env

# 构建（指定架构）
./build.sh --arch 80    # Ampere (sm80)
./build.sh --arch 89    # Ada Lovelace (sm89)
./build.sh --arch 90    # Hopper (sm90)
./build.sh --arch 120   # Blackwell (sm120)

# 指定并行任务数
./build.sh --arch 89 --jobs 6

# 清理
./build.sh --clean-py     # 仅清理 Python
./build.sh --clean-all    # 清理全部
```

### 依赖项

- CUDA Toolkit 11.0+
- PyTorch
- CUTLASS（子模块，位于 `3rdparty/cutlass/`）
- TileLang（用于 Megakernel DSL）：`pip install tilelang==0.1.8`
- Triton
- NCCL（可选，用于集合通信）

---

## 调优系统

### 工作流程

```
1. 生成搜索空间
   python3 tools/gen_search_space.py --schema=GemmNormal
   → 生成 search_space_gemm*.cu

2. 移动到 src/ops/gemm_normal/tuning_config/

3. 编译库
   ./build.sh --arch 89

4. 运行调优
   python3 tools/gemm/tuning/tune_gemm_normal.py --schema=GemmNormal
   → 生成 tuned_config_gemm*.cu

5. 移动到 src/ops/gemm_normal/tuning_config/

6. 重新编译
   ./build.sh --arch 89

7. 测试
   python3 tools/gemm/test_gemm_normal.py 12 12288 6144 --dtype=bfloat16
```

### 搜索空间文件格式

```cpp
// search_space_gemmnormal.cu
namespace xop {
void __global__ search_kernel(...) {
  // 配置参数
}
void register_gemm_normal_search_space() {
  // 注册所有配置变体
}
}
```

---

## DSL（领域特定语言）

DSL 用于生成微内核，定义于 `python/xop/ops/dsl/`

### 微内核类型

| 文件 | 用途 |
|------|------|
| [micro_linear.py](python/xop/ops/dsl/micro_linear.py) | 线性层/GEMV |
| [micro_gqa_decode.py](python/xop/ops/dsl/micro_gqa_decode.py) | GQA 解码 |
| [micro_rmsnorm.py](python/xop/ops/dsl/micro_rmsnorm.py) | RMSNorm |
| [micro_rope.py](python/xop/ops/dsl/micro_rope.py) | RoPE（旋转位置编码） |
| [micro_silu_mul.py](python/xop/ops/dsl/micro_silu_mul.py) | SiLU 融合 |

### TileLang 语法

DSL 使用 TileLang 进行内核编程：

```python
@tilelang.jit(out_idx=[-1])
def kernel_main(N, K, BLOCK_N, reduce_threads, dtype=T.bfloat16, accum_dtype=T.float):
    @T.prim_func
    def linear(A: T.Tensor((1, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((1, N), dtype)):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            # 内核实现
```

---

## Megakernel 系统

Megakernel 是基于 Cython 的持久化内核系统，用于融合算子。

### 构建

```bash
source ./build.sh --env
python megakernel_setup.py build_ext --inplace
```

### 测试

```bash
python tools/test_dsl.py --gen          # 生成内核
python tools/test_mega_single.py        # 单算子测试
python tools/test_mega_attn.py          # Attention 融合测试
python tools/test_mega_mlp.py           # MLP 融合测试
```

### MkLayers 类

定义于 [mk_layers.py](python/xop/ops/dsl/mk_layers.py)

```python
from xop.ops.dsl.mk_layers import MkLayers

mk = MkLayers(
    model_tag="qwen3_4b",      # 模型标签
    instance_id=0,
    kernel_num=1,
    world_size=1,
    rank=0,
    max_batch_size=8,
    trace_name="test",
    profiling=True
)

# 编译和加载
mk.compile_load(input_tensors, meta_tensors, enable_prefetch=True)
```

---

## 测试

### Python 测试

| 文件 | 用途 |
|------|------|
| `tools/gemm/test_gemm_normal.py` | 主 GEMM 正确性和性能测试 |
| `tools/gemm/test_gemm_blockscale_fp8.py` | FP8 量化测试 |
| `tools/gemm/test_gemm_comm.py` | 通信 GEMM 测试 |
| `tools/test_mega_single.py` | Megakernel 单算子测试 |
| `tools/test_mega_attn.py` | Megakernel Attention 测试 |
| `tools/test_mega_mlp.py` | Megakernel MLP 测试 |
| `tools/test_dsl.py` | DSL 生成测试 |

### 使用示例

```bash
# 主 GEMM 测试
python3 tools/gemm/test_gemm_normal.py 12 12288 6144 --dtype=bfloat16

# 显示 TFLOPS
python3 tools/gemm/test_gemm_normal.py 12 12288 6144 --dtype=bfloat16 --show_tflops

# 指定数据类型
python3 tools/gemm/test_gemm_normal.py 100 4096 4096 --dtype=float8_e4m3fn

# FP8 量化
python3 tools/gemm/test_gemm_blockscale_fp8.py 100 4096 4096
```

### 性能分析

```bash
# 使用 NCU 分析
ncu --set full --section "SpeedOfLight_RooflineChart" \
    -k "ada_blockwise_fp8_gemm_run_kernel" \
    -o my_profile \
    python3 tools/gemm/test_gemm_normal.py 100 4096 4096 --dtype=float8_e4m3fn

# 使用 Nsight Systems
nsys profile --trace=cuda,nvtx --output=my_nsys python tools/test_model_integration.py
```

---

## 目录结构详解

### `include/xop/` - 头文件

| 文件 | 内容 |
|------|------|
| `xop.h` | 核心枚举、宏定义、XOP_CHECK 错误检查 |
| `common_cuda.h` | CUDA 工具函数、get_arch()、GpuTimer |
| `common_torch.h` | PyTorch 辅助函数、Tensor 创建 |
| `common_nccl.h` | NCCL 通信辅助函数 |
| `common_strategy.h` | 调优策略（CoarseGrainedTuningM、SplitChunkM） |
| `ops_impl/arguments.h` | RtArguments 运行时参数结构 |
| `ops_impl/global_resource.h` | 全局缓冲区和注册表单例 |

### `src/ops/` - 算子实现

| 目录 | 内容 |
|------|------|
| `gemm_normal/` | 主 GEMM 算子实现 |
| `gemm_comm/` | 带集合通信的 GEMM |
| `allreduce_normal/` | 自定义 AllReduce 内核 |
| `marlin/` | Marlin 量化内核 |
| `flash_attn/` | Flash Attention（已禁用） |

### `python/xop/ops/` - Python 算子接口

| 文件 | 内容 |
|------|------|
| `gemm_quant.py` | GemmQuant 量化 GEMM 包装器 |
| `gemm_comm.py` | GemmCommRs 通信 GEMM 包装器 |
| `flash_attn.py` | Flash Attention 包装器 |
| `custom_all_reduce.py` | 自定义 AllReduce |
| `cuda_ipc_manager.py` | CUDA IPC 管理器 |
| `dsl/` | 领域特定语言微内核 |

---

## 注意事项

### 1. 数据类型支持

| 数据类型 | PyTorch dtype | 说明 |
|----------|---------------|------|
| FP16 | `torch.float16` | 半精度浮点 |
| BF16 | `torch.bfloat16` | BF16 浮点 |
| FP32 | `torch.float32` | 单精度浮点 |
| E4M3 | `torch.float8_e4m3fn` | FP8 E4M3 格式 |
| E5M2 | `torch.float8_e5m2` | FP8 E5M2 格式 |
| S8 | `torch.int8` | 8 位有符号整数 |
| S32 | `torch.int32` | 32 位有符号整数 |

### 2. 架构支持

| 架构 | sm | 说明 |
|------|-----|------|
| Sm80 | 80 | Ampere (A100 等) |
| Sm89 | 89 | Ada Lovelace (RTX 4090 等) |
| Sm90 | 90 | Hopper (H100 等) |
| Sm120 | 120 | Blackwell (B100 等) |

### 3. 布局类型

| 布局 | 说明 |
|------|------|
| RRR | Row-major × Row-major → Row-major |
| RCR | Row-major × Column-major → Row-major |
| RCC | Row-major × Column-major → Column-major |
