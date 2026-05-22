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

from cpython cimport array
import ctypes
import array
import numpy as np
import torch
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

# Code snippet from OpenAI Triton

ctypedef unsigned long int size_t

cdef extern from "megakernel/type.h":
    ctypedef struct dim3:
        unsigned int x
        unsigned int y
        unsigned int z
    ctypedef struct int3:
        int x
        int y
        int z

cdef extern from "megakernel/type.h" namespace "megakernel::type":
    # This must be consistent with megakernel/type.h
    cdef enum DataType:
        DT_FLOAT4 = 920,
        DT_INT4 = 925,
        DT_UINT4 = 926,
        DT_FLOAT8 = 930,
        DT_INT8 = 935,
        DT_UINT8 = 936,
        DT_FLOAT16 = 940,
        DT_BFLOAT16 = 941,
        DT_INT16 = 945,
        DT_UINT16 = 946,
        DT_FLOAT32 = 950,
        DT_INT32 = 955,
        DT_UINT32 = 956,
        DT_DOUBLE = 960,
        DT_INT64 = 965,
        DT_UINT64 = 966,
        DT_UNKNOWN = 999,
    cdef enum KNOperatorType:
        KN_INPUT_OP = 1001,
        KN_CUSTOMIZED_OP = 1999,

cdef extern from "megakernel/kernel/device_tensor.h" namespace "megakernel::kernel":
    cdef struct CppDTensor "megakernel::kernel::DTensor":
        DataType data_type
        int num_dims
        int dim[4]
        size_t guid

cdef extern from "megakernel/kernel/runtime.h" namespace "megakernel::runtime":
    ctypedef struct TaskGraphResult:
        string cuda_code
        string json_file

cdef extern from "megakernel/kernel/graph.h" namespace "megakernel::kernel":

    cdef cppclass CppKNOperator "megakernel::kernel::KNOperator":
        KNOperatorType op_type
 
    cdef cppclass CppKNCustomizedOp "megakernel::kernel::KNCustomizedOp"(CppKNOperator):
        CppTBGraph bgraph

    cdef cppclass CppKNGraph "megakernel::kernel::Graph":
        CppKNGraph(dim3 gpu_dim)
        CppDTensor* new_input_ptr(vector[int] dims,
                                  vector[size_t] strides,
                                  DataType data_type)
        void customized(vector[const CppDTensor*] inputs,
                       CppTBGraph* bgraph)

        # Persistent kernel functions
        void attach_torch_tensor(const CppDTensor *input,
                                 void *torch_data_ptr,
                                 const char *name)
        void attach_cuda_tensor(const CppDTensor *input,
                                const char *name)
        void attach_nvshmem_tensor(const CppDTensor *input,
                                   const char *name)
        void register_task(const char *task_type,
                           vector[int] params)
        TaskGraphResult generate_task_graph(int num_gpus, int my_gpu_id)

        vector[CppKNOperator*] operators

cdef extern from "megakernel/kernel/tb_graph.h" namespace "megakernel::threadblock":

    cdef cppclass CppTBOperator "megakernel::threadblock::TBOperator":
        pass

    cdef cppclass CppTBGraph "megakernel::threadblock::TBGraph":
        CppTBGraph(dim3 grid_dim,
                   dim3 block_dim,
                   int thread_num)

        CppDTensor* new_input(const CppDTensor* dtensor,
                             int3 input_map)

        dim3 grid_dim
        dim3 block_dim
        int thread_num
        vector[CppTBOperator*] operators

cdef extern from "megakernel/kernel/allreduce/custom_all_reduce.h" namespace "xop":

    cdef cppclass AllReduce "xop::AllReduce":
        AllReduce(int thread_num)

        AllReduce(dim3 grid_dim,
                   dim3 block_dim,
                   int thread_num)

        CppDTensor* new_input(const CppDTensor* dtensor,
                             int3 input_map)

        dim3 grid_dim
        dim3 block_dim
        int thread_num
        vector[CppTBOperator*] operators

cdef class PyAllReduce:
    cdef AllReduce* handle  # 持有C++实例指针

    def __cinit__(self, int thread_num):
        # 创建C++ AllReduce
        self.handle = new AllReduce(thread_num)

    def __dealloc__(self):
        # 释放内存
        if self.handle:
            del self.handle

    # 提供Python可调用的方法
    # def add_input(self, dtensor, int x, int y, int z):
    #     cdef int3 input_map = make_int3(x, y, z)
    #     cdef CppDTensor* inp = self.handle.new_input(<CppDTensor*>dtensor, input_map)
    #     return inp  # 可继续包装给Python
################################################################

class dtype:
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']

    def __init__(self, name):
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES, name

    def is_fp16(self):
        return self.name == 'fp16'

    def is_bf16(self):
        return self.name == 'bf16'

    def is_fp32(self):
        return self.name == 'fp32'

    def is_fp64(self):
        return self.name == 'fp64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def __eq__(self, other: dtype):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other: dtype):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, ))

    def __str__(self):
        return self.name

    def is_dtype(type_str):
        return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES

# data types
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')


def convert_dtype_to_ctype(type : dtype):
    if type.is_int8():
        return DT_INT8
    elif type.is_uint8():
        return DT_UINT8
    elif type.is_uint16():
        return DT_UINT16
    elif type.is_fp16():
        return DT_FLOAT16
    elif type.is_bf16():
        return DT_BFLOAT16
    elif type.is_fp32():
        return DT_FLOAT32
    elif type.is_int32():
        return DT_INT32
    elif type.is_int64():
        return DT_INT64
    elif type.is_fp64():
        return DT_DOUBLE
    else:
        raise RuntimeError(f"Unsupported dtype: {dtype}")

def convert_ctype_to_dtype(type):
    if type == DT_INT8:
        return int8
    elif type == DT_UINT16:
        return uint16
    elif type == DT_FLOAT16:
        return float16
    elif type == DT_BFLOAT16:
        return bfloat16
    elif type == DT_INT32:
        return int32
    elif type == DT_FLOAT32:
        return float32
    elif type == DT_DOUBLE:
        return float64
    else:
        return None

def convert_torch_type_to_dtype(type):
    if type is torch.int8:
        return int8
    elif type is torch.uint8:
        return uint8
    elif type is torch.uint16:
        return uint16
    elif type is torch.int32:
        return int32
    elif type is torch.float16:
        return float16
    elif type is torch.bfloat16:
        return bfloat16
    elif type is torch.float32:
        return float32
    elif type is torch.int64:
        return int64
    elif type is torch.float64:
        return float64
    else:
        raise RuntimeError(f"Unsupported dtype: {type}")




cdef class DTensor:
    cdef CppDTensor* c_ptr # Hold a Tensor instance

    cdef inline _set_tensor(self, tensor):
        cdef unsigned long long ptr
        if tensor is None:
            self.c_ptr = <CppDTensor*>(NULL)
        else:
            ptr = ctypes.cast(tensor, ctypes.c_void_p).value
            self.c_ptr = <CppDTensor*>(ptr)

    property guid:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return self.c_ptr.guid

    property tensor:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.c_ptr, ctypes.c_void_p)
        
        def __set__(self, value):
            self._set_tensor(value)

    property num_dims:
        def __get__(self):
            if self.c_ptr == NULL:
                print("Error: tensor is None in num_dims property")
                return None
            else:
                return self.c_ptr.num_dims

    property dtype:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return convert_ctype_to_dtype(self.c_ptr.data_type)

    def __cinit__(self, tensor):
        self._set_tensor(tensor)

    def dim(self, int idx):
        if (idx < self.c_ptr.num_dims):
            return self.c_ptr.dim[idx]
        else:
            assert False , "Error: index out of range"
            return None


cdef class CyKNGraph:
    cdef CppKNGraph *p_kgraph #Hold a CppKNGraph instance

    def __cinit__(self, graph = None):
        cdef unsigned long long ptr
        cdef dim3 c_gpu_dim
        if graph is None:
            c_gpu_dim.x = 1
            c_gpu_dim.y = 1
            c_gpu_dim.z = 1
            self.p_kgraph = new CppKNGraph(c_gpu_dim)
        else:
            ptr = ctypes.cast(graph, ctypes.c_void_p).value
            self.p_kgraph = <CppKNGraph*>(ptr)

    def new_input(self, tuple dims, tuple strides, dtype : dtype = float16):
        cdef vector[int] cdims
        cdef vector[size_t] cstrides
        cdims.resize(len(dims))
        for i in range(len(dims)):
            cdims[i] = dims[i]
        cstrides.resize(len(strides))
        for i in range(len(strides)):
            cstrides[i] = strides[i]

        c_type = convert_dtype_to_ctype(dtype)
        cdef CppDTensor* ptr = self.p_kgraph.new_input_ptr(cdims, cstrides, c_type)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def customized(self, list inputs, CyTBGraph bgraph):
        cdef vector[const CppDTensor*] cinputs
        cinputs.resize(len(inputs))
        cdef DTensor t
        for i in range(len(inputs)):
            if inputs[i] is None:
                cinputs[i] = NULL
            else:
                assert (type(inputs[i]) == DTensor)
                t = inputs[i]
                cinputs[i] = t.c_ptr
        self.p_kgraph.customized(cinputs, bgraph.p_bgraph)

    # Functions for ersistent kernels
    def attach_torch_tensor(self, DTensor tensor, torch_tensor, str name):
        # cdef unsigned long long torch_data_ptr = ctypes.cast(torch_tensor.data_ptr(), ctypes.c_void_p).value
        cdef unsigned long long torch_data_ptr = <unsigned long long>(torch_tensor.data_ptr())
        cdef char* cname = NULL
        if name is not None:
            py_byte_string = name.encode('UTF-8')
            cname = py_byte_string
        self.p_kgraph.attach_torch_tensor(tensor.c_ptr, <void *>torch_data_ptr, cname)

    def attach_cuda_tensor(self, DTensor tensor, str name):
        cdef char* cname = NULL
        if name is not None:
            py_byte_string = name.encode('UTF-8')
            cname = py_byte_string
        self.p_kgraph.attach_cuda_tensor(tensor.c_ptr, cname)

    def attach_nvshmem_tensor(self, DTensor tensor, str name):
        cdef char* cname = NULL
        if name is not None:
            py_byte_string = name.encode('UTF-8')
            cname = py_byte_string
        self.p_kgraph.attach_nvshmem_tensor(tensor.c_ptr, cname)

    def register_task(self, str task_type, list[int] params):
        cdef char* cname = NULL
        if task_type is not None:
            py_byte_string = task_type.encode('UTF-8')
            cname = py_byte_string
        cdef vector[int] cparams
        cparams.resize(0)
        if params is not None:
            cparams.resize(len(params))
            for i in range(len(params)):
                cparams[i] = params[i]
        self.p_kgraph.register_task(cname, cparams)

    def generate_task_graph(self, int num_gpus, int my_gpu_id):
        cdef TaskGraphResult result = self.p_kgraph.generate_task_graph(num_gpus, my_gpu_id)
        return {
            "cuda_code": result.cuda_code.decode("UTF-8"),
            "json_file": result.json_file.decode("UTF-8"),
        }
     

cdef class CyTBGraph:
    cdef CppTBGraph *p_bgraph #Hold a CppTBGraph instance

    def __cinit__(self, tuple grid_dim = (), tuple block_dim = (), int thread_num = 128, bgraph = None):
        cdef unsigned long long ptr
        cdef dim3 c_grid_dim
        cdef dim3 c_block_dim
        if bgraph is None:
            if len(grid_dim) == 0 or len(block_dim) == 0:
                assert False, "grid_dim, block_dim, thread_num must be provided"
            assert len(grid_dim) == 3, "grid_dim must include 3 dimensions"
            assert len(block_dim) == 3, "block_dim must include 3 dimensions"
            c_grid_dim.x = grid_dim[0]
            c_grid_dim.y = grid_dim[1]
            c_grid_dim.z = grid_dim[2]
            c_block_dim.x = block_dim[0]
            c_block_dim.y = block_dim[1]
            c_block_dim.z = block_dim[2]
            self.p_bgraph = new CppTBGraph(c_grid_dim, c_block_dim, thread_num)
        else:
            ptr = ctypes.cast(bgraph, ctypes.c_void_p).value
            if isinstance(bgraph, int):
                self.p_bgraph = <CppTBGraph*>(ptr)
            elif isinstance(bgraph, ctypes.c_void_p):
                self.p_bgraph = <CppTBGraph*>(ptr)
            else:
                assert False, "bgraph must be an integer or ctypes.c_void_p, but got " + str(type(bgraph))
    
    def new_input(self, DTensor dtensor, tuple input_map):
        assert len(input_map) == 3, "input_map must be of length 3"
        cdef int3 c_input_map
        c_input_map.x = input_map[0]
        c_input_map.y = input_map[1]
        c_input_map.z = input_map[2]
        cdef CppDTensor* dtensor_cptr = NULL
        if dtensor is not None:
            dtensor_cptr = dtensor.c_ptr
        cdef CppDTensor* ptr = self.p_bgraph.new_input(dtensor_cptr, c_input_map)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)


