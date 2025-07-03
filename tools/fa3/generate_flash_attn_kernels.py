# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/hopper/generate_kernels.py

# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

KERNEL_BATCH = namedtuple("Kernel", ["template", "filename"])

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
    "e4m3": "cutlass::float_e4m3_t",
}

DTYPE_MAP_FWD_SM8x = {
    # "fp16": "cutlass::half_t",
    # "bf16": "cutlass::bfloat16_t",
    "e4m3": "cutlass::float_e4m3_t",
}

# sm8x
SM = [80] # [80, 90]  # Sm kernels support up to
HEAD_DIMENSIONS = [128] # [64, 96, 128, 192, 256]
PAGEDKV = [False, True]
SPLIT = [False, True]
SOFTCAP = [False, True]
PACKGQA = [False, True] # Always enable PackGQA for Sm8x to reduce compilation
# # sm90
# SM = [90]
# HEAD_DIMENSIONS = [128] # [64, 96, 128, 192, 256]
# PAGEDKV = [False, True]
# SPLIT = [False, True]
# SOFTCAP = [False, True]
# PACKGQA = [False, True]

KERNEL_IMPL_TEMPLATE_FWD = """#include "../flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_fwd_<{ARCH}, {DTYPE}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}, {SOFTCAP}, {PACKGQA}>(Flash_fwd_params &params, cudaStream_t stream);
#endif
"""

# KERNEL_IMPL_TEMPLATE_FWD_SM8x = """#include "../flash_fwd_launch_template.h"

# #ifndef FLASHATTENTION_DISABLE_SM8x
# #ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
# template void run_mha_fwd_<80, {DTYPE}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}, {SOFTCAP}, {PACKGQA}>(Flash_fwd_params &params, cudaStream_t stream);
# //template void run_mha_fwd_<86, {DTYPE}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}, {SOFTCAP}, {PACKGQA}>(Flash_fwd_params &params, cudaStream_t stream);
# #endif
# #endif
# """

@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    head_dim_v: int
    split: bool
    paged_kv: bool
    softcap: bool
    packgqa: bool

    @property
    def template(self) -> str:
        if self.sm == 90:
            # Always enable PackGQA for PagedKV or Split to reduce compilation
            packgqa = self.packgqa or self.paged_kv or self.split
        else:
            packgqa = True
        return KERNEL_IMPL_TEMPLATE_FWD.format(
            ARCH=str(self.sm), DTYPE=DTYPE_MAP[self.dtype],
            HEAD_DIM=self.head_dim, HEAD_DIM_V=self.head_dim_v,
            SPLIT=str(self.split).lower(), PAGEDKV=str(self.paged_kv).lower(),
            SOFTCAP=str(self.softcap).lower(), PACKGQA=str(packgqa).lower()
        )
        
    @property
    def filename(self) -> str:
        return f"flash_fwd_hdim{self.head_dim}{f'_{self.head_dim_v}' if self.head_dim_v != self.head_dim else ''}_{self.dtype}{'_paged' if self.paged_kv else ''}{'_split' if self.split else ''}{'_softcap' if self.softcap else ''}{'_packgqa' if self.packgqa else ''}_sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    for dtype, head_dim, split, paged_kv, softcap, packgqa, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SPLIT, PAGEDKV, SOFTCAP, PACKGQA, SM):
        # We always enable PackGQA for Sm8x or PagedKV or Split
        # so we should just pass in packgqa=False to avoid the `_packgqa` in the filename.
        if packgqa and (sm < 90 or (sm >= 90 and (paged_kv or split))):
            continue
        # print("123", packgqa)
        if sm >= 90 or dtype in DTYPE_MAP_FWD_SM8x:    
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, head_dim_v=head_dim, split=split, paged_kv=paged_kv, softcap=softcap, packgqa=packgqa)
        if sm == 90 and head_dim == 192:
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, head_dim_v=128, split=split, paged_kv=paged_kv, softcap=softcap, packgqa=packgqa)
        if sm == 90 and head_dim == 64 and dtype in ["bf16", "fp16"]:
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, head_dim_v=256, split=split, paged_kv=paged_kv, softcap=softcap, packgqa=packgqa)
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, head_dim_v=512, split=split, paged_kv=paged_kv, softcap=softcap, packgqa=packgqa)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels_all = list(get_all_kernels())
    print("hello,", kernels_all, output_dir)
    for kernel in kernels_all:
        write_kernel(kernel, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # python tools/generate_flash_attn_kernels.py -o ./tools/tmp
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        default="src/ops/flash_attn/instantiations",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
