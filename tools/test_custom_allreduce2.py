import ctypes
import multiprocessing as mp
import random
import socket
import unittest
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import time
import torch
import torch.distributed as dist
# from torch.distributed import ProcessGroup

# import ctlop
from ctlop.ops.custom_all_reduce import CustomAllreduce
# from ctlop.cuda_wrapper import CudaRTLibrary

TEST_CUDA_GRAPH = 1
def _run_correctness_worker(world_size, rank, distributed_init_port, test_sizes):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    new_group = torch.distributed.new_group(list(range(world_size)), backend="gloo")
    cop = CustomAllreduce(new_group, device)

    try:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        custom_perf = []
        nccl_perf = []
        test_loop = 20
        for sz in test_sizes:
            custom_kernel_time = 0
            nccl_kernel_time = 0
            for dtype in [torch.bfloat16]: # torch.float32, torch.float16, 
                for loop in range(test_loop):
                    if TEST_CUDA_GRAPH:
                        stream = torch.cuda.Stream()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.stream(stream):
                            with torch.cuda.graph(g):
                                inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                                inp1_ref = inp1.clone()
                                out1 = cop.custom_all_reduce(inp1)
                        torch.cuda.synchronize(stream)
                        g.replay()
                        torch.testing.assert_close(out1, inp1_ref)
                    else:
                        inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                        inp1_ref = inp1.clone()

                        if loop <= 10:
                            out1 = cop.custom_all_reduce(inp1)
                            dist.all_reduce(inp1_ref, group=group)
                        else:
                            start_event.record()
                            out1 = cop.custom_all_reduce(inp1)
                            end_event.record()
                            torch.cuda.synchronize()
                            custom_kernel_time += start_event.elapsed_time(end_event)

                            start_event.record()
                            dist.all_reduce(inp1_ref, group=group)
                            end_event.record()
                            torch.cuda.synchronize()
                            nccl_kernel_time += start_event.elapsed_time(end_event)

                        torch.testing.assert_close(out1, inp1_ref)

            custom_perf.append(custom_kernel_time)
            nccl_perf.append(nccl_kernel_time)
            print(f"custom_kernel_time: {custom_kernel_time:.6f} ms, {sz}")
            print(f"nccl_kernel_time: {nccl_kernel_time:.6f} ms, {sz}")

        # plot
        x_ticks = range(len(test_sizes))
        plt.plot(x_ticks, custom_perf, label='custom', marker='o', markersize=3)
        plt.plot(x_ticks, nccl_perf, label='nccl', marker='s', markersize=3)
        plt.xticks(x_ticks, test_sizes)

        plt.title(f'allreduce-WS-{world_size}-rank-{rank}.png')
        plt.xlabel('size')
        plt.ylabel('ms')

        plt.legend()
        plt.grid(True)

        plt.savefig('allreduce-WS-{0}-rank-{1}.png'.format(world_size, rank))

    finally:
        dist.barrier(group=group)
        cop.close()

        dist.destroy_process_group(group=group)

def get_open_port() -> int:
    return 12345
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]

def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.daemon = True
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


class TestCustomAllReduce(unittest.TestCase):
    test_sizes = [
        512,
        2560,
        4096,
        5120,
        7680,
        32768,
        262144,
        524288,
        1048576,
        2097152,
        58720256,
    ]
    world_sizes = [2, 4, 8]

    def test_correctness(self):
        for world_size in self.world_sizes:
            available_gpus = torch.cuda.device_count()
            if world_size > available_gpus:
                print(
                    f"Skipping world_size={world_size}, requires {world_size} GPUs, found {available_gpus}"
                )
                continue

            print(f"Running test for world_size={world_size}")
            multi_process_parallel(
                world_size, _run_correctness_worker, target_args=(self.test_sizes,)
            )
            print(f"custom allreduce tp = {world_size}: OK")


if __name__ == "__main__":
    unittest.main()
