
from typing import Optional, List
import torch

class GemmNormal:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
        transpose_weight: bool = False,
    ): ...
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output_buf: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        tuning: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
    ) -> torch.Tensor: ...
    def grouped_forward(
        self,
        inputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        outputs: List[torch.Tensor],
        inputs_scale: Optional[torch.Tensor] = None,
        weights_scale: Optional[torch.Tensor] = None,
        tuning: Optional[torch.Tensor] = None,
    ): ...