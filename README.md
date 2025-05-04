# Cutlass Operators

## Getting started
Install CtlOp either from source or from PyPI.

### Install from Source
```bash
git clone 

# For Ampere(sm80) GPU
./build.sh --arch 80 --jobs 6
# For Ada Lovelace(sm89) GPU
./build.sh --arch 89 --jobs 6
# For Hopper(sm90) GPU
./build.sh --arch 90 --jobs 6
```

#### Dependencies
1. CUTLASS: Flux leverages CUTLASS to generate high-performance GEMM kernels. We currently use CUTLASS 3.7.0 and a tiny patch should be applied to CUTLASS.

## Quick Start

```bash
# gemm only
python3 tools/test_gemm_normal.py 100 12288 6144 --dtype=float16

python3 tools/gen_search_space.py

python3 tools/tune_gemm_normal.py

```