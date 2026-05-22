import torch

import os
from typing import *

from .core import *
from .utils import *

MAX_THREADS = os.cpu_count()

# Because pip install -e . and pip install . have different directory structure,
# we need to check the directory structure to find the correct MEGAKERNEL_ROOT.
def get_key_paths():
    root_dir = os.path.join(
        os.path.dirname(__file__), "../"
    )  # Using pip install -e .
    # print("root_dir0", root_dir)
    
    if not os.path.exists(os.path.join(root_dir, "3rdparty")):  # Using pip install .
        root_dir = os.path.dirname(__file__)
    # print("root_dir1", root_dir)
    
    # If MEGAKERNEL_ROOT is not set, use the root_dir as MEGAKERNEL_ROOT
    MEGAKERNEL_ROOT = os.environ.get("MEGAKERNEL_ROOT", root_dir)
    INCLUDE_PATH = os.path.join(MEGAKERNEL_ROOT, "include/xop/")
    DEPS_PATH = os.path.join(MEGAKERNEL_ROOT, "3rdparty")

    # print("MEGAKERNEL_ROOT", MEGAKERNEL_ROOT)
    # print("INCLUDE_PATH", INCLUDE_PATH)
    # print("DEPS_PATH", DEPS_PATH)
    
    assert os.path.exists(
        MEGAKERNEL_ROOT
    ), "No MEGAKERNEL_ROOT directory found. Likely using the wrong MEGAKERNEL_ROOT."
    
    assert os.path.exists(
        INCLUDE_PATH
    ), "No /include directory found. Likely using the wrong MEGAKERNEL_ROOT."
    assert os.path.exists(
        DEPS_PATH
    ), "No /3rdparty directory found. Likely using the wrong MEGAKERNEL_ROOT."

    return MEGAKERNEL_ROOT, INCLUDE_PATH, DEPS_PATH

class TBGraph:
    def __init__(self, graph):
        self.cygraph = graph

    def new_input(
        self,
        dtensor: DTensor,
        input_map: tuple):
        return self.cygraph.new_input(dtensor, input_map)
    
class KNGraph:
    def __init__(self, graph):
        self.cygraph = graph

        self._is_compiled = False
        self.run = None
        self._valid_cuda_kernels = False
        self._cached_results = None
        self.visualizer = None

        self.backend = "cuda"

    def new_input(
        self, dims: tuple, strides: tuple = None, dtype: dtype = float16
    ) -> DTensor:
        # use the default strided layout if strides = None
        if strides is None:
            total_elements = 1
            strides = []
            for d in reversed(dims):
                strides.append(total_elements)
                total_elements *= d
            strides = reversed(strides)
        else:
            assert len(dims) == len(strides)

        return self.cygraph.new_input(dims, tuple(strides), dtype)

    def customized(self, inputs: list[DTensor], bgraph: TBGraph) -> list[DTensor]:
        return self.cygraph.customized(inputs, bgraph.cygraph)

    # Persistent Kernel functions
    def attach_torch_tensor(self, t: DTensor, torch_tensor: torch.Tensor, name: str):
        return self.cygraph.attach_torch_tensor(t, torch_tensor, name)

    def attach_cuda_tensor(self, t: DTensor, name: str):
        return self.cygraph.attach_cuda_tensor(t, name)

    def attach_nvshmem_tensor(self, t: DTensor, name: str):
        return self.cygraph.attach_nvshmem_tensor(t, name)
    
    def register_task(self, task_type: str, params: list[int] = None):
        return self.cygraph.register_task(task_type, params)

    def generate_task_graph(self, num_gpus: int, my_gpu_id: int):
        return self.cygraph.generate_task_graph(num_gpus, my_gpu_id)
