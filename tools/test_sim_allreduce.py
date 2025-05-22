# -*- coding: utf-8 -*-

import ctypes
import multiprocessing as mp
import random
import socket
import unittest
from typing import Any, List, Optional

import os
import time
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import threading
import ctlop
from ctlop.cuda_wrapper import CudaRTLibrary

# 主线程创建内存，子线程访问。不使用进程，尽管是单卡，因为进程也需要走ipc
class SimulateBuffer:
    def __init__(self, world_size):
        self.max_size = 8196 * 1024
        self.meta_ptrs = self.create_shared_buffer(ctlop.meta_size() + self.max_size, world_size)
        self.buffer_ptrs = self.create_shared_buffer(self.max_size, world_size)
        self.ref_buffer = [None] * 8
    
    def __del__(self):
        self.free_shared_buffer(self.buffer_ptrs)
        self.free_shared_buffer(self.meta_ptrs)

    @staticmethod
    def create_shared_buffer(size_in_bytes: int, world_size: int):
        lib = CudaRTLibrary()

        pointers: List[int] = []
        for i in range(world_size):
            p = lib.cudaMalloc(size_in_bytes).value
            lib.cudaMemset(p, 0, size_in_bytes)
            pointers.append(p)

        # ipc_pointers: List[int] = []
        # handle1 = lib.cudaIpcGetMemHandle(pointer1)
        # handle2 = lib.cudaIpcGetMemHandle(pointer2)
        # ipc_pointers.append(lib.cudaIpcOpenMemHandle(handle1).value)
        # ipc_pointers.append(lib.cudaIpcOpenMemHandle(handle2).value)
        return pointers

    @staticmethod
    def free_shared_buffer(pointers: List[int]) -> None:
        lib = CudaRTLibrary()
        for p in pointers:
            lib.cudaFree(ctypes.c_void_p(p))

g_world_size = 8
g_buffer = SimulateBuffer(g_world_size)

def _run_correctness_worker(world_size, rank, test_sizes):
    try:
        print("rank: ", rank)
        device = torch.device(f"cuda:{0}")
        new_stream = torch.cuda.Stream()
        
        with torch.cuda.stream(new_stream):
            rank_data = torch.empty(8*1024 * 1024, dtype=torch.uint8, device=device)
            custom_ptr = ctlop.init_custom_ar(g_buffer.meta_ptrs, rank_data, rank, True)
            ctlop.register_buffer(custom_ptr, g_buffer.buffer_ptrs)
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            custom_kernel_time = 0

            cnt = 0
            test_loop = 20
            for sz in test_sizes:
                for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    for loop in range(test_loop):
                        inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                        # print("input", inp1, "rank: ", rank)
                        g_buffer.ref_buffer[rank] = inp1.clone()
                        out1 = torch.empty_like(inp1)
                        cnt = cnt+1
                        print("cnt:", cnt)

                        start_event.record()
                        ctlop.all_reduce(custom_ptr, inp1, out1, g_buffer.buffer_ptrs[rank], g_buffer.max_size)
                        end_event.record()
                        torch.cuda.synchronize()
                        if loop > 10:
                            custom_kernel_time += start_event.elapsed_time(end_event)
                        
                        # ref_out = g_buffer.ref_buffer[0]
                        # for i in range(1, world_size):
                        #     ref_out = ref_out + g_buffer.ref_buffer[i]
                        # torch.testing.assert_close(out1, ref_out)

                print(f"custom_kernel_time: {custom_kernel_time:.6f} ms, {sz}")

    finally:
        if custom_ptr is not None:
            ctlop.dispose(custom_ptr)

def multi_thread_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    threads = []
    for i in range(world_size):
        thread_args = (world_size, i) + target_args
        thread = threading.Thread(target=test_target, args=thread_args, name=f"Worker-{i}")
        thread.daemon = True
        thread.start()
        threads.append(thread)

    for i in range(world_size):
        threads[i].join()

def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    for i in range(world_size):
        proc_args = (world_size, i) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.daemon = True
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"

if __name__ == "__main__":
    test_sizes = [
        512,
        2560,
        4096,
        5120,
        7680,
        # 32768,
        # 262144,
        # 524288,
        # 1048576,
        # 2097152,
        # 58720256,
    ]

    # 主进程申请的空间无法在子进程上直接访问，只能通过IPC的方式，哪怕是在同一张卡上
    multi_thread_parallel(g_world_size, _run_correctness_worker, target_args=(test_sizes,))

    del g_buffer