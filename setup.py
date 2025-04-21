import os
import re
import ast
from pathlib import Path
import setuptools
from torch.utils.cpp_extension import BuildExtension

# Project directory root
root_path: Path = Path(__file__).resolve().parent

PACKAGE_NAME = "byte_flux"

def get_package_version():
    with open(Path(root_path) / "python" / "flux" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    return public_version

def pathlib_wrapper(func):
    def wrapper(*kargs, **kwargs):
        include_dirs, library_dirs, libraries = func(*kargs, **kwargs)
        return map(str, include_dirs), map(str, library_dirs), map(str, libraries)

    return wrapper

@pathlib_wrapper
def cutlass_deps():
    cutlass_home = root_path / "3rdparty/cutlass"
    include_dirs = [
        cutlass_home / "include",
        cutlass_home / "tools" / "util" / "include",
        cutlass_home / "tools" / "library" / "include",
        cutlass_home / "tools" / "profiler" / "include",
    ]
    library_dirs = []
    libraries = []
    return include_dirs, library_dirs, libraries

@pathlib_wrapper
def flux_cuda_deps():
    include_dirs = [root_path / "include", root_path / "src"]
    library_dirs = [root_path / "build" / "lib"]
    libraries = ["flux_cuda_ths_op"]
    return include_dirs, library_dirs, libraries


@pathlib_wrapper
def cuda_deps():
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    include_dirs = [cuda_home / "include"]
    library_dirs = [cuda_home / "lib64", cuda_home / "lib64/stubs"]
    libraries = ["cuda", "cudart", "nvidia-ml"]
    return include_dirs, library_dirs, libraries

def setup_pytorch_extension() -> setuptools.Extension:
    """Setup CppExtension for PyTorch support"""
    include_dirs, library_dirs, libraries = [], [], []

    deps = [cutlass_deps(), flux_cuda_deps(), cuda_deps()]

    for include_dir, library_dir, library in deps:
        include_dirs += include_dir
        library_dirs += library_dir
        libraries += library

    # Compiler flags
    # too much warning from CUDA /usr/local/cuda/include/cusparse.h: "-Wdeprecated-declarations"
    cxx_flags = [
        "-O3",
        "-DTORCH_CUDA=1",
        "-fvisibility=hidden",
        "-Wno-deprecated-declarations",
        "-fdiagnostics-color=always",
    ]

    flux_ths_targets = [
        str(x.relative_to(root_path))  # relative path for include_package_data
        for x in Path(root_path / "src" / "pybind").glob("*.cc")
    ]

    from torch.utils.cpp_extension import CppExtension

    return CppExtension(
        name="flux_ths_pybind",
        sources=flux_ths_targets,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=cxx_flags,  
    )
    # extra_link_args=ld_flags,

def main():
    flux_version = get_package_version()
    packages = setuptools.find_packages(
        where="python",
        include=[
            "flux",
        ],
    )
    data_file_list = ["python/flux/lib/libflux_cuda_ths_op.so"]

    # Configure package
    setuptools.setup(
        name=PACKAGE_NAME,
        version=flux_version,
        package_dir={"": "python"},
        packages=packages,
        description="Flux library",
        ext_modules=[setup_pytorch_extension()],
        cmdclass={"build_ext": BuildExtension},
        setup_requires=["torch", "cmake", "packaging"],
        install_requires=["torch"],
        extras_require={"test": ["torch", "numpy"]},
        license_files=("LICENSE",),
        package_data={
            "python/flux/lib": ["*.so"],
            "python/flux/include": ["*.h"],
            "python/flux/share": ["*.cmake"],
        },  # only works for bdist_wheel under package
        python_requires=">=3.8",
        include_package_data=True,
        data_files=[
            (
                "lib",  # installed directory
                data_file_list,  # to installed shared libraries. only works for setup.py install/bdist_wheels
            )
        ],
    )


if __name__ == "__main__":
    main()
