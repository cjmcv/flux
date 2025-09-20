#!/bin/bash
set -x
set -e

## Change export PATH if cuda is not at default path
export PATH=/usr/local/cuda/bin:$PATH
CMAKE=${CMAKE:-cmake}

ARCH=""
BUILD_TEST="ON"
BDIST_WHEEL="OFF"

function clean_py() {
    rm -rf build/lib.*
    rm -rf python/xop/lib
    rm -rf .eggs/
    rm -rf python/xop.egg-info
    rm -rf python/xop_pybind.*
}

function clean_all() {
    clean_py
    rm -rf 3rdparty/nccl/build
    rm -rf build/
}

# Iterate over the command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --arch)
        # Process the arch argument
        ARCH="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    --no_test)
        BUILD_TEST="OFF"
        shift # Skip the argument value
        ;;
    --jobs)
        # Process the jobs argument
        JOBS="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    --clean-py)
        clean_py
        exit 0
        ;;
    --clean-all)
        clean_all
        exit 0
        ;;
    --package)
        BDIST_WHEEL="ON"
        shift # Skip the argument key
        ;;
    *)
        # Unknown argument
        echo "Unknown argument: $1"
        shift # Skip the argument
        ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=${SCRIPT_DIR}

cd ${PROJECT_ROOT}

if [[ -n $ARCH ]]; then
    build_args=" --arch ${ARCH}"
fi

if [[ -z $JOBS ]]; then
    JOBS=$(nproc --ignore 2)
fi

function build_nccl() {
    NCCL_ROOT=$PROJECT_ROOT/3rdparty/nccl
    pushd $NCCL_ROOT
    export BUILDDIR=${NCCL_ROOT}/build
    export PREFIX=${BUILDDIR}/local

    if [[ -n $ARCH ]]; then
        NCCL_COMPILE_OPTIONS_ARCH="" # default none
        arch_list=()
        IFS=";" read -ra arch_list <<<"$ARCH"
        for arch in "${arch_list[@]}"; do
            NCCL_COMPILE_OPTIONS_ARCH="-gencode=arch=compute_${arch},code=sm_${arch} ${NCCL_COMPILE_OPTIONS_ARCH}"
        done
        make -j${JOBS} src.staticlib NVCC_GENCODE="${NCCL_COMPILE_OPTIONS_ARCH}" VERBOSE=1
    else
        make -j${JOBS} src.staticlib VERBOSE=1
    fi
    # only install static lib
    mkdir -p ${PREFIX}/lib
    cp -P -v ${BUILDDIR}/lib/lib* ${PREFIX}/lib/
    cp -P -v -r ${BUILDDIR}/include ${PREFIX}/
    popd
}


##### build xop_cuda #####
function build_xop_cuda() {
    mkdir -p build
    pushd build
    export LIBXOP_PREFIX=${PROJECT_ROOT}/python/xop
    if [ ! -f CMakeCache.txt ] || [ -z ${XOP_BUILD_SKIP_CMAKE} ]; then
        CMAKE_ARGS=(
            -DCUDAARCHS=${ARCH}
            -DCMAKE_EXPORT_COMPILE_COMMANDS=1
            -DBUILD_TEST=${BUILD_TEST}
            -DCMAKE_INSTALL_PREFIX=${LIBXOP_PREFIX}
        )
        ${CMAKE} .. ${CMAKE_ARGS[@]}
    fi
    make -j${JOBS} VERBOSE=1
    make install
    popd
}

function build_xop_py {
    LIBDIR=${PROJECT_ROOT}/python/xop/lib
    mkdir -p ${LIBDIR}

    pushd ${LIBDIR}

    popd
    ##### build xop torch bindings #####
    # MAX_JOBS=${JOBS} python3 setup.py develop --user # You should delete pyproject.toml first
    # --no-build-isolation: It reuses the torch already installed in your current conda/env, 
    #                       but you lose the "clean build" benefits of PEP 517; use with caution in CI.
    # MAX_JOBS=${JOBS} pip install --editable . --user --no-build-isolation
    MAX_JOBS=${JOBS} pip install --use-pep517 -e . --user --no-build-isolation -v
    if [ $BDIST_WHEEL == "ON" ]; then
        MAX_JOBS=${JOBS} python3 setup.py bdist_wheel
    fi
}

build_nccl
build_xop_cuda
build_xop_py
