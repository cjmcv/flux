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

KERNEL_IMPL_TEMPLATE_FWD = \
"""
#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_fwd_<{ARCH}, {DTYPE}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}, {SOFTCAP}, {PACKGQA}>(Flash_fwd_params &params, cudaStream_t stream);
#endif
"""

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
        
    # @property
    # def filename(self) -> str:
    #     return f"flash_fwd_hdim{self.head_dim}{f'_{self.head_dim_v}' if self.head_dim_v != self.head_dim else ''}_{self.dtype}{'_paged' if self.paged_kv else ''}{'_split' if self.split else ''}{'_softcap' if self.softcap else ''}{'_packgqa' if self.packgqa else ''}_sm{self.sm}.cu"


def get_all_kernels(sm) -> List[Kernel]:
    for dtype, head_dim, split, paged_kv, softcap, packgqa in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SPLIT, PAGEDKV, SOFTCAP, PACKGQA):
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


def main(output_dir: Optional[str]) -> None:
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for sm in SM:
        kernels_all = list(get_all_kernels(80))
        print("hello,", output_dir, kernels_all)

        fp = open(str(output_dir) + "/flash_fwd_sm{0}.cu".format(sm), "w")
        fp.write('#include "../flash_fwd_launch_template.h"\n\n')
        fp.write('// ARCH, DTYPE, HEAD_DIM, HEAD_DIM_V <==> SPLIT, PAGEDKV, SOFTCAP, PACKGQA\n')
        
        for kernel in kernels_all:
            # fp.write(str(kernel.head_dim))
            fp.write(kernel.template)
        fp.close()

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
