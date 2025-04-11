#!/bin/bash
script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$script_dir" 

# Patch CUTLASS
cd 3rdparty/cutlass && git checkout v3.7.0 && cd ..
patch -p1 < ./cutlass3.7.patch
