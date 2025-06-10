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

compute-sanitizer --tool memcheck python tools/test*.py
```

#### Dependencies
1. CUTLASS: Flux leverages CUTLASS to generate high-performance GEMM kernels. We currently use CUTLASS 3.7.0 and a tiny patch should be applied to CUTLASS.

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