# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from os import path
import sys
import sysconfig
from setuptools import find_packages

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:                                                
    from distutils.core import setup
    from distutils.extension import Extension                              
else:
    from setuptools import setup
    from setuptools.extension import Extension

def config_cython():
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize
        ret = []
        cython_path = path.join(path.dirname(__file__), "megakernel/_cython")
        megakernel_path = path.join(path.dirname(__file__), ".")
        for fn in os.listdir(cython_path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                "megakernel.%s" % fn[:-4],
                ["%s/%s" % (cython_path, fn)],
                include_dirs=[path.join(megakernel_path, "include/xop"),
                              path.join(megakernel_path, "3rdparty", "json", "include"),
                              path.join(megakernel_path, "3rdparty", "cutlass", "include"),
                              "/usr/local/cuda/include"],
                libraries=["cudadevrt", "cudart_static", "cudart", "cuda", "gomp", "rt"],
                library_dirs=[path.join(megakernel_path, "build"),
                              path.join(megakernel_path, "3rdparty", "build"),
                              "/usr/local/cuda/lib",
                              "/usr/local/cuda/lib64",
                              "/usr/local/cuda/lib64/stubs"],
                define_macros=[("MEGAKERNEL_BACKEND_USE_CUDA", None)],
                extra_compile_args=["-std=c++17", "-fopenmp"],
                extra_link_args=["-fPIC", "-fopenmp"],
                language="c++"))
        return cythonize(ret, compiler_directives={"language_level" : 3})
    except ImportError:
        print("WARNING: cython is not installed!!!")
        raise SystemExit(1)
        return []

setup_args = {}


setup(name='megakernel',
      version="0.0.1",
      description="Megakernel",
      zip_safe=False,
      install_requires=[],
      packages=find_packages(),
      url='https://github.com/',
      ext_modules=config_cython(),
      #**setup_args,
      )

print("[megakernel] Setup Success!")
# python cython_setup.py build_ext --inplace
# python -c "import megakernel"