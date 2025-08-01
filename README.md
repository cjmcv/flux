# Cutlass Operators

### Install from Source
```bash
git clone 

# For Ampere(sm80) GPU
./build.sh --arch 80 --jobs 6
# For Ada Lovelace(sm89) GPU
./build.sh --arch 89 --jobs 6
# For Hopper(sm90) GPU
./build.sh --arch 90 --jobs 6

pdflatex a.tex
compute-sanitizer --tool memcheck python tools/test*.py
ncu --set full --section "SpeedOfLight_RooflineChart" -k "ada_blockwise_fp8_gemm_run_kernel" -o my_profile python3 tools/gemm/test_gemm_normal.py 100 4096 4096 --show_tflops --dtype=float8_e4m3fn
ncu-ui # Open file£ºmy_profile.ncu-rep
```

#### Dependencies
1. cutlass
2. cublasLt
3. triton

## Quick Start

```bash
# Generate search_space_gemmnormal.cu 
# Move it to src/ops/gemm_normal/tuning_config, and compile the library again.
python3 tools/gen_search_space.py --schema=GemmNormal

# Generate tuned_config_gemmnormal.cu
# Move it to src/ops/gemm_normal/tuning_config, and compile the library again.
python3 tools/tuning/tune_gemm_normal.py --schema=GemmNormal

# Now you can test it.
python3 tools/test_gemm_normal.py 100 12288 6144 --dtype=float16
```