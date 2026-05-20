"""
Micro kernel configuration for controlling code generation behavior.
Import this module in other micro_*.py files to use these variables.
"""
import os
import torch
import tilelang


ENABLE_MEGAKERNEL = True
ENABLE_PROFILING = False

target_arch = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
if target_arch == 120:
    TARGET_ARCH = "sm_120"
elif target_arch == 90:
    TARGET_ARCH = "sm_90"
else:
    TARGET_ARCH = "sm_89"
    
# TARGET_ARCH = "sm_120" # "sm_120" / "sm_90" / "sm_89"

# =============================================================================
# Utility Functions
# =============================================================================
def get_target_str():
    """Returns the complete target string for tilelang.jit"""
    return f"cuda -arch={TARGET_ARCH}"

def get_arch():
    return TARGET_ARCH

def get_thread_num():
    if TARGET_ARCH == "sm_89":
        return 128
    else:
        return 256
    
def get_pass_configs():
    if TARGET_ARCH == "sm_89":
        return {
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    else:
        return {
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
        }

# def is_tma_enabled():
#     """Check if TMA should be enabled based on architecture"""
#     # TMA requires sm_90a (Hopper) or newer
#     tma_arches = ["sm_90a", "sm_90", "sm_100", "sm_100a"]
#     return ENABLE_TMA and (TARGET_ARCH in tma_arches)

def is_megakernel_enabled():
    """Check if megakernel mode is enabled"""
    return ENABLE_MEGAKERNEL

def is_enable_profiling():
    """Check if megakernel mode is enabled"""
    return ENABLE_PROFILING