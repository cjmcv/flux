from enum import IntEnum, auto
import torch

# Sync from enum class UnifiedMetaEnum
class Meta(IntEnum):
    GemmNormal = 0           # meta type
    GemmNormalSimt = auto()
    GemmGrouped = auto()
    GemmBlockScaleFp8 = auto()
    GemmGroupedBlockScaleFp8 = auto()
    GemmW4A16 = auto()
    GemmLt = auto()
    GemmAllreduce = 20
    Void = 50                # data type
    FP16 = auto()   
    BF16 = auto()
    FP32 = auto()
    E4M3 = auto()
    E5M2 = auto()
    S8 = auto()
    S32 = auto()
    Sm80 = 60                # arch
    Sm89 = auto()
    Sm90 = auto()
    RRR = 70                 # layout
    RCR = auto()
    RCC = auto()
    def __str__(self):
        return {
            Meta.GemmNormal: "GemmNormal",
            Meta.GemmNormalSimt: "GemmNormalSimt",
            Meta.GemmGrouped: "GemmGrouped",
            Meta.GemmBlockScaleFp8: "GemmBlockScaleFp8",
            Meta.GemmGroupedBlockScaleFp8: "GemmGroupedBlockScaleFp8",
            Meta.GemmW4A16: "GemmW4A16",
            Meta.GemmLt: "GemmLt",
            Meta.GemmAllreduce: "GemmAllreduce",
            Meta.Void: "Void",
            Meta.FP16: "FP16",
            Meta.BF16: "BF16",
            Meta.FP32: "FP32",
            Meta.E4M3: "E4M3",
            Meta.E5M2: "E5M2",
            Meta.S8: "S8",
            Meta.S32: "S32",
            Meta.Sm80: "Sm80",
            Meta.Sm89: "Sm89",
            Meta.Sm90: "Sm90",
            Meta.RRR: "RRR",
            Meta.RCR: "RCR",
            Meta.RCC: "RCC",
        }.get(self, "Unknown")
        
# arch = -1: Get arch by default.
def gen_tuned_hparam(chunk_size=8192, arch=-1):
    tuned_hparam = torch.zeros(3, dtype=torch.int16, device='cpu')
    tuned_hparam[0] = 2
    tuned_hparam[1] = chunk_size # max_m
    tuned_hparam[2] = arch
    return tuned_hparam

def uupdate_tuned_hparam(tuned_hparam, chunk_size, arch):
    tuned_hparam[1] = chunk_size # max_m
    tuned_hparam[2] = arch