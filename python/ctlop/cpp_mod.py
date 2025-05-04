################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import ctypes
import importlib
import logging
from pathlib import Path

CTLOP_TORCH_EXTENSION_NAME = "ctlop_pybind"


def _preload_libs(libname):
    libpath = Path(__file__).parent / "lib" / libname
    try:
        ctypes.CDLL(libpath)
    except OSError as e:
        # Try to load from the LD_LIBRARY_PATH
        logging.debug(f"failed to load {libpath}:\n {e}")
        ctypes.CDLL(libname)


def _load_deps():
    _preload_libs("libctlop.so")


_load_deps()
ctlop_mod = importlib.import_module(CTLOP_TORCH_EXTENSION_NAME)


class NotCompiled:
    pass

def _get_ctlop_member(member):
    return getattr(ctlop_mod, member, NotCompiled())

SingleGemm = _get_ctlop_member("SingleGemm")
GemmNormal = _get_ctlop_member("GemmNormal")

__all__ = [
    "GemmNormal",
]
