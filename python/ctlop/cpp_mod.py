
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

GemmNormal = _get_ctlop_member("GemmNormal")

# allreduce_normal
helloABC = _get_ctlop_member("helloABC")
init_custom_ar = _get_ctlop_member("init_custom_ar")
all_reduce = _get_ctlop_member("all_reduce")
dispose = _get_ctlop_member("dispose")
meta_size = _get_ctlop_member("meta_size")
register_buffer = _get_ctlop_member("register_buffer")
get_graph_buffer_ipc_meta = _get_ctlop_member("get_graph_buffer_ipc_meta")
register_graph_buffers = _get_ctlop_member("register_graph_buffers")

# flash_attn
mha_fwd = _get_ctlop_member("mha_fwd")

# __all__ = [
#     "GemmNormal",
#     "helloABC",
# ]
