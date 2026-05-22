from enum import Enum

import torch
from tg4perfetto import TraceGenerator

event_name_list = {
    10: "TASK_BEGIN_TASK_GRAPH",
    102: "TASK_RMS_NORM_LINEAR",
    106: "TASK_ALLREDUCE",
    107: "TASK_REDUCE",
    108: "TASK_LINEAR_WITH_RESIDUAL",
    118: "TASK_SILU_MUL",
    119: "TASK_RMS_NORM",
    120: "TASK_LINEAR",
    122: "TASK_SILU_MUL_LINEAR",
    123: "TASK_GQA_DECODE",
    124: "TASK_ROPE",
    199: "TASK_NVSHMEM_COPY",
    
    150: "TASK_HOPPER_TASK_BEGIN",
    151: "TASK_LINEAR_WITH_RESIDUAL_HOPPER",
    152: "TASK_LINEAR_HOPPER",
    160: "TASK_HOPPER_TASK_END",
    
    
}


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2


def decode_tag(tag, num_blocks, num_groups):
    event_no = tag >> 19
    block_group_tag = (tag >> 11) & 0xFF
    event_idx = (tag >> 2) & 0x1FF
    event_type = tag & 0x3
    return (
        event_no,
        block_group_tag // num_groups,
        block_group_tag % num_groups,
        event_idx,
        event_type,
    )


def export_to_perfetto_trace(
    profiler_buffer: torch.Tensor,
    file_name: str,
) -> None:

    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx}")
        for group_idx in range(num_groups):
            tid = pid.create_group(f"group_{group_idx}")
            tid_map[(block_idx, group_idx)] = tid

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue

        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        event_no, block_idx, group_idx, event_idx, event_type = decode_tag(
            tag, num_blocks, num_groups
        )

        event = event_name_list[event_idx] + f"_{event_no}"
        tid = tid_map[(block_idx, group_idx)]

        if (block_idx, group_idx, event_idx) in track_map:
            track = track_map[(block_idx, group_idx, event_idx)]
        else:
            track = tid.create_track()
            track_map[(block_idx, group_idx, event_idx)] = track

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()
