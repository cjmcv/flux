# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce.py

import ctypes
import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing_extensions import ParamSpec

import xop
from xop.cuda_wrapper import CudaRTLibrary
logger = logging.getLogger(__name__)

try:
    import pynvml
except ImportError as e:
    logger.warning("Failed to import pynvml with %r", e)

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


@with_nvml_context
def is_full_nvlink(physical_device_ids: List[int], world_size: int) -> bool:
    """
    query if the set of gpus are fully connected by nvlink (1 hop)
    """
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
    for i, handle in enumerate(handles):
        for j, peer_handle in enumerate(handles):
            if i < j:
                try:
                    p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                        handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                    if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                        return False
                except pynvml.NVMLError:
                    logger.exception(
                        "NVLink detection failed. This is normal if your"
                        " machine has no NVLink equipped."
                    )
                    return False
    return True


# def _can_p2p(rank: int, world_size: int) -> bool:
#     # SGLANG_SKIP_P2P_CHECK can be set to False in sglang
#     SGLANG_SKIP_P2P_CHECK = os.getenv("SGLANG_SKIP_P2P_CHECK", "0") == "1"
#     for i in range(world_size):
#         if i == rank:
#             continue
#         if SGLANG_SKIP_P2P_CHECK:
#             logger.info("Skipping P2P check and trusting the driver's P2P report.")
#             return torch.cuda.can_device_access_peer(rank, i)
#         if not gpu_p2p_access_check(rank, i):
#             return False
#     return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _MAX_CAR_SIZE = 1024 * 1024 * 600 

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=_MAX_CAR_SIZE,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True
        self.group = group

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "CustomAllreduce should be attached to a non-NCCL group."

        # if not all(in_the_same_node_as(group, source_rank=0)):
        #     # No need to initialize custom allreduce for multi-node case.
        #     logger.warning(
        #         "Custom allreduce is disabled because this process group"
        #         " spans across nodes."
        #     )
        #     return

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(torch.cuda.device_count()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        full_nvlink = is_full_nvlink(physical_device_ids, world_size)

        if world_size > 2 and not full_nvlink:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
        # # test P2P capability, this checks software/cudaruntime support
        # # this is expensive to compute at the first time
        # # then we cache the result
        # # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        # if not _can_p2p(rank, world_size):
        #     logger.warning(
        #         "Custom allreduce is disabled because your platform lacks "
        #         "GPU P2P capability or P2P test failed. To silence this "
        #         "warning, specify disable_custom_all_reduce=True explicitly."
        #     )
        #     return

        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink

        # Buffers memory are owned by this Python class and passed to C++.
        # Meta data composes of two parts: meta data for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        # <NT> meta_size是Signal的size，
        self.meta_ptrs = self.create_shared_buffer(
            xop.meta_size() + max_size, group=group
        )
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self._ptr = xop.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.full_nvlink
        )
        xop.register_buffer(self._ptr, self.buffer_ptrs)
        

        self.disabled = False

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int, group: Optional[ProcessGroup] = None
    ) -> List[int]:
        """
        Creates a shared buffer and returns a list of pointers
        representing the buffer on all processes in the group.
        """
        # <NT> 申请显存，并使用cudaIpcGetMemHandle将该显存设置为多GPU共享，得到操作这块共享内存的句柄handle
        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        # <NT> 使用allgather将各个gpu上的共享内存都收集起来，使每个gpu上都拿到所有gpu的分配的共享内存句柄。
        dist.all_gather_object(handles, handle, group=group)

        # <NT> 如果是当前GPU自己的，则直接使用普通cudaMalloc出来的显存指针pointer.value，不需要额外操作.
        #      但如果不是当前GPU的，是从其他GPU获取到的句柄，则需要额外调用cudaIpcOpenMemHandle，
        #    它允许一个进程打开另一个进程导出的设备内存句柄，并在当前进程中使用该内存。
        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:        
                pointers.append(pointer.value)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore

        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int], group: Optional[ProcessGroup] = None
    ) -> None:
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def _get_ipc_meta(self, inp: torch.Tensor):
        # _share_cuda_() doesn't accept meta buffer not allocated from
        # PyTorch cache allocator, use direct HIP call to get IPC handle
        handle = xop.get_meta_buffer_ipc_handle(inp)
        shard_data = (
            bytes(handle),  # ipc handle to base ptr
            0,  # offset of base ptr
        )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        # Note: don't use `[[None]] * self.world_size` here
        # because it will create a list of the same reference
        all_data: List[Optional[Any]] = [[None] for i in range(self.world_size)]
        all_data[self.rank][0] = shard_data

        ranks = dist.get_process_group_ranks(group=self.group)
        ranks.sort()
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )

        # we cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0][0])  # type: ignore
            offsets.append(all_data[i][0][1])  # type: ignore
        return handles, offsets

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        xop.register_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        handle, offset = xop.get_graph_buffer_ipc_meta(self._ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data = [
            [None, None] for _ in range(dist.get_world_size(group=self.group))
        ]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )
        # Unpack list of tuples to tuple of lists.
        handles = [d[0] for d in all_data]  # type: ignore
        offsets = [d[1] for d in all_data]  # type: ignore
        xop.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.

        if self.world_size == 2 or self.full_nvlink:
            return inp_size < self.max_size
        return False

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: torch.Tensor = None,
        registered: bool = False,
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            xop.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            xop.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # <NT> cuda graph的replay是已经被捕获到了，所以不会走这里。被捕获的将会是上面 registered=True 的
            return self.all_reduce(input, registered=False)

    ###########################################
    def is_capturing(self):
        return self._IS_CAPTURING
    
    def address(self):
        return self._ptr, self.buffer_ptrs[self.rank], self.max_size
    
    def custom_all_reduce_t(self, input: torch.Tensor, output: torch.Tensor):
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return 
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, out=output, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return 
        else:
            # <NT> cuda graph的replay是已经被捕获到了，所以不会走这里。被捕获的将会是上面 registered=True 的
            return self.all_reduce(input, out=output, registered=False)
    #############################################
    
    def close(self):
        if not self.disabled and self._ptr:
            xop.dispose(self._ptr)
            self.free_shared_buffer(self.meta_ptrs)
            self.free_shared_buffer(self.buffer_ptrs)
            self._ptr = 0

    def __del__(self):
        self.close()