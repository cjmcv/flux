from enum import IntEnum, auto
from typing import Tuple

import torch
import math

# Sync from enum class UnifiedMetaEnum
class Meta(IntEnum):
    GemmNormal = 0           # meta type
    GemmNormalSimt = auto()
    GemmBlockScaleFp8 = auto()
    GemmGroupedBlockScaleFp8 = auto()
    Void = 10                # data type
    FP16 = auto()   
    BF16 = auto()
    FP32 = auto()
    E4M3 = auto()
    E5M2 = auto()
    S8 = auto()
    S32 = auto()
    Sm80 = 20                # arch
    Sm89 = auto()
    Sm90 = auto()
    RRR = 30                 # layout
    RCR = auto()
    RCC = auto()
    def __str__(self):
        return {
            Meta.GemmNormal: "GemmNormal",
            Meta.GemmNormalSimt: "GemmNormalSimt",
            Meta.GemmBlockScaleFp8: "GemmBlockScaleFp8",
            Meta.GemmGroupedBlockScaleFp8: "GemmGroupedBlockScaleFp8",
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

def ceil_div(a, b):
    return math.ceil(a / b)