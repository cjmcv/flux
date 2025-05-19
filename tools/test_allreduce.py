
import torch

import ctlop
from ctlop.cuda_wrapper import CudaRTLibrary
# from ctlop.custom_all_reduce import CustomAllreduce
lib = CudaRTLibrary()
pointer = lib.cudaMalloc(10000)
handle = lib.cudaIpcGetMemHandle(pointer)

print(lib, pointer, handle)
ctlop.helloABC(3)
# CustomAllreduce()
