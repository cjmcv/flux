# Cutlass Operators

### Install from Source
```bash
git clone 

# Set env path
source ./build.sh --env
# For Ampere(sm80) GPU
./build.sh --arch 80 --jobs 6
# For Ada Lovelace(sm89) GPU
./build.sh --arch 89 --jobs 6
# For Hopper(sm90) GPU
./build.sh --arch 90 --jobs 6
# For blackwell(sm120) GPU
./build.sh --arch 120 --jobs 6

# ./build.sh --arch 89 --jobs 1 2>&1 | tee full.log

pdflatex a.tex
compute-sanitizer --tool memcheck python tools/test*.py
ncu --set full --section "SpeedOfLight_RooflineChart" -k "ada_blockwise_fp8_gemm_run_kernel" -o my_profile python3 tools/gemm/test_gemm_normal.py 100 4096 4096 --show_tflops --dtype=float8_e4m3fn
ncu --set full --section "SpeedOfLight_RooflineChart" -o my_profile python3 tools/gemm/test_gemm_mini.py --has_bias 4096 4096 4096
ncu-ui # Open file: my_profile.ncu-rep

nsys profile --trace=cuda,nvtx --output=my_nsys python tools/test_model_integration.py
nsight-sys # Open file: my_profile.ncu-rep
```


### megakernel

```bash
# 基础库编译
python megakernel_setup.py build_ext --inplace
# 生成kernel, 有HEURISTIC / TUNING / TUNED三种模式，具体看脚本
python tools/test_dsl.py --gen
# 直接测试，单算子无收益，融合以预取为主
python tools/test_mega_single.py # 单算子
python tools/test_mega_attn.py   # attn融合
python tools/test_mega_mlp.py    # mlp融合
```

#### Dependencies
1. cutlass
2. cublasLt
3. tilelang==0.1.8   pip install tilelang==0.1.8 --index-url https://pypi.org/simple
4. triton

## Quick Start

```bash
# Select the GPU you want to use: CUDA_VISIBLE_DEVICES=5,6 python...
# Generate search_space_gemmnormal.cu 
# Move it to src/ops/gemm_normal/tuning_config, and compile the library again.
python3 tools/gen_search_space.py --schema=GemmNormal

# Generate tuned_config_gemmnormal.cu
# Move it to src/ops/gemm_normal/tuning_config, and compile the library again.
python3 tools/tuning/tune_gemm_normal.py --schema=GemmNormal

# Now you can test it.
python3 tools/gemm/test_gemm_normal.py 12 12288 6144 --dtype=bfloat16
```


git clone -b pn2 --recursive https://github.com/cjmcv/flux.git
