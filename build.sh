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
    rm -rf python/ctlop/lib
    rm -rf .eggs/
    rm -rf python/ctlop.egg-info
    rm -rf python/ctlop_pybind.*
}

function clean_all() {
    clean_py
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

##### build ctlop_cuda #####
function build_ctlop_cuda() {
    mkdir -p build
    pushd build
    export LIBCTLOP_PREFIX=${PROJECT_ROOT}/python/ctlop
    if [ ! -f CMakeCache.txt ] || [ -z ${CTLOP_BUILD_SKIP_CMAKE} ]; then
        CMAKE_ARGS=(
            -DCUDAARCHS=${ARCH}
            -DCMAKE_EXPORT_COMPILE_COMMANDS=1
            -DBUILD_TEST=${BUILD_TEST}
            -DCMAKE_INSTALL_PREFIX=${LIBCTLOP_PREFIX}
        )
        ${CMAKE} .. ${CMAKE_ARGS[@]}
    fi
    make -j${JOBS} VERBOSE=1
    make install
    popd
}

function build_ctlop_py {
    LIBDIR=${PROJECT_ROOT}/python/ctlop/lib
    mkdir -p ${LIBDIR}

    pushd ${LIBDIR}

    popd
    ##### build ctlop torch bindings #####
    MAX_JOBS=${JOBS} python3 setup.py develop --user
    if [ $BDIST_WHEEL == "ON" ]; then
        MAX_JOBS=${JOBS} python3 setup.py bdist_wheel
    fi
}

build_ctlop_cuda
build_ctlop_py
