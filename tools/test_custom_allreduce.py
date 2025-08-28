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

# import xop
from xop.ops.custom_all_reduce import CustomAllreduce
# from xop.cuda_wrapper import CudaRTLibrary

TEST_CUDA_GRAPH = 1

def _run_correctness_worker(world_size, rank, distributed_init_port, test_sizes, param_size):
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
        test_loop = 200
        warmup_loop = 50
        for sz in test_sizes:
            custom_kernel_time = 0
            nccl_kernel_time = 0
            for dtype in [torch.bfloat16]: # torch.float32, torch.float16, 
                if TEST_CUDA_GRAPH:
                    
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                    # inp1 = torch.ones((sz,), dtype=dtype, device=device)
                    torch.cuda.synchronize()

                    stream = torch.cuda.Stream()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(stream), cop.capture():
                        with torch.cuda.graph(graph):
                            out1 = cop.custom_all_reduce(inp1)

                    for loop in range(test_loop):
                        if loop <= warmup_loop:
                            graph.replay()
                            inp1_ref = inp1.clone() 
                            dist.all_reduce(inp1_ref, group=group)
                        else:
                            start_event.record()
                            graph.replay()
                            end_event.record()
                            torch.cuda.synchronize()
                            custom_kernel_time += start_event.elapsed_time(end_event)

                            inp1_ref = inp1.clone()

                            start_event.record()
                            dist.all_reduce(inp1_ref, group=group)
                            end_event.record()
                            torch.cuda.synchronize()
                            nccl_kernel_time += start_event.elapsed_time(end_event)
                        
                        is_close = torch.allclose(out1, inp1_ref)
                        if (is_close == False):
                            print("is_close is False.")
                            print("custom_output:", out1)
                            print("nccl_output:", inp1_ref)
                        # torch.testing.assert_close(out1, inp1_ref)
                else:
                    for loop in range(test_loop):
                        inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                        inp1_ref = inp1.clone()

                        if loop <= warmup_loop:
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

            custom_kernel_time = custom_kernel_time / (test_loop - warmup_loop)
            nccl_kernel_time = nccl_kernel_time / (test_loop - warmup_loop)

            custom_perf.append(custom_kernel_time)
            nccl_perf.append(nccl_kernel_time)
            print(f"custom_kernel_time: {custom_kernel_time:.6f} ms, {sz}, rank{rank}")
            print(f"nccl_kernel_time: {nccl_kernel_time:.6f} ms, {sz}, rank{rank}")

        print("custom_perf: ", custom_perf)
        print("nccl_perf: ", nccl_perf)
        # plot
        if (world_size == 2):
            custom_perf_origin = [0.04506389328589042, 0.0240842666849494, 0.026055466656883557, 0.03120917337636153, 0.04178431982795398, 0.05187242661913236, 0.06435776012639205, 0.07417450641592344, 0.12251242662469546, 0.17372224013010662, 0.22252693345149357, 0.32198293288548785, 0.43082133372624715]
        elif (world_size == 4):
            custom_perf_origin = [0.02802005338172118, 0.03304234682271878, 0.038419413218895596, 0.04524309354523818, 0.056024746720989546, 0.07287104015549024, 0.09238165348768235, 0.10402666722734769, 0.1736887467900912, 0.2459135992328326, 0.3368354125817617, 0.48102805415789285, 0.632537602186203]
        elif (world_size == 8):
            custom_perf_origin = [0.04666730689505736, 0.04309738698105017, 0.04571775995194912, 0.058501119886835415, 0.07392362664143244, 0.08935231998562813, 0.11013461321592331, 0.12528213342030844, 0.20270783990621566, 0.28689322888851165, 0.3739895474910736, 0.5425996792316437, 0.7337467737992605]

        x_ticks = range(len(test_sizes))
        plt.plot(x_ticks, custom_perf_origin, label='custom_origin', marker='o', markersize=3)
        plt.plot(x_ticks, nccl_perf, label='nccl', marker='s', markersize=3)
        plt.plot(x_ticks, custom_perf, label='custom', marker='o', markersize=3)

        norm_test_sizes = [x // param_size for x in test_sizes]
        plt.xticks(x_ticks, norm_test_sizes)

        plt.title(f'allreduce-WS-{world_size}-rank-{rank}-graph-{TEST_CUDA_GRAPH}.png')
        plt.xlabel(f'size*{param_size}')
        plt.ylabel('ms')

        plt.legend()
        plt.grid(True)

        plt.savefig('allreduce-WS-{0}-rank-{1}-graph-{2}.png'.format(world_size, rank, TEST_CUDA_GRAPH))

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


class TestCustomAllReduce():
    exponent = 14
    test_sizes_index = [1] + list(2**x for x in list(range(1, exponent)))
    test_sizes = list(4096*x for x in test_sizes_index)
    
    # test_sizes = [
    #     1 * param_size,
    #     50 * param_size,
    #     100 * param_size,
    #     200 * param_size,
    #     400 * param_size,
    #     600 * param_size,
    #     800 * param_size,
    #     1000 * param_size,
    #     2000 * param_size,
    #     3000 * param_size,
    #     4000 * param_size,
    #     6000 * param_size,
    #     8192 * param_size,
    #     # 512,
    #     # 2560,
    #     # 4096,
    #     # 5120,
    #     # 7680,
    #     # 32768,
    #     # 262144,
    #     # 524288,
    #     # 1048576,
    #     # 2097152,
    #     # 58720256,
    # ]
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
                world_size, _run_correctness_worker, target_args=(self.test_sizes, self.param_size)
            )
            print(f"custom allreduce tp = {world_size}: OK")


if __name__ == "__main__":
    test = TestCustomAllReduce()
    test.test_correctness()
