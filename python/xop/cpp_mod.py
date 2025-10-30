
import ctypes
import importlib
import logging
from pathlib import Path

XOP_TORCH_EXTENSION_NAME = "xop_pybind"


def _preload_libs(libname):
    libpath = Path(__file__).parent / "lib" / libname
    try:
        ctypes.CDLL(libpath)
    except OSError as e:
        # Try to load from the LD_LIBRARY_PATH
        logging.debug(f"failed to load {libpath}:\n {e}")
        ctypes.CDLL(libname)


def _load_deps():
    _preload_libs("libxop.so")


_load_deps()
xop_mod = importlib.import_module(XOP_TORCH_EXTENSION_NAME)


class NotCompiled:
    pass

def _get_xop_member(member):
    return getattr(xop_mod, member, NotCompiled())

GemmNormal = _get_xop_member("GemmNormal")
GemmW4A16Sm90ReorderWeight = _get_xop_member("GemmW4A16Sm90ReorderWeight")

marlin_fp16xint4_matmul = _get_xop_member("marlin_fp16xint4_matmul")
helloABCM = _get_xop_member("helloABCM")

GemmComm = _get_xop_member("GemmComm")

# allreduce_normal
helloABC = _get_xop_member("helloABC")
init_custom_ar = _get_xop_member("init_custom_ar")
all_reduce = _get_xop_member("all_reduce")
dispose = _get_xop_member("dispose")
meta_size = _get_xop_member("meta_size")
register_buffer = _get_xop_member("register_buffer")
get_graph_buffer_ipc_meta = _get_xop_member("get_graph_buffer_ipc_meta")
register_graph_buffers = _get_xop_member("register_graph_buffers")

# flash_attn
mha_fwd = _get_xop_member("mha_fwd")

# __all__ = [
#     "GemmNormal",
#     "helloABC",
# ]
