
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
