#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpu MI300
from task import input_t, output_t
import torch
from torch.utils.cpp_extension import load_inline
import time
import os
import sys
import zlib
import base64

if "PYTORCH_ROCM_ARCH" not in os.environ:
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942:xnack-"

TESTING = os.environ.get("GPUMODE_TESTING", None)

kernel_cpp = b'@SOLUTION@'

hip_module = load_inline(
    name="fp8",
    cpp_sources="",
    cuda_sources=zlib.decompress(base64.b64decode(kernel_cpp)).decode(),
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=(["-save-temps"] if TESTING is not None else []) + ["-std=c++20", "-Werror"],
    build_directory="/workspace/build/" if TESTING == "vscode" else "/gpumode/amd/fp8/build/" if TESTING is not None else None,
    **({'no_implicit_headers': True} if TESTING != "vscode" else {}),
)

first = True

cache_m = 0
cache_n = 0
cache_k = 0
cache_i = 0

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemm
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k],
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k],
            a_scale: torch.Tensor[float32] of shape [m, k // 128],
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128],
            c: torch.Tensor[bfloat16] of shape [m, n]
    Returns:
        Tensor containing output in bf16
    """

    global first
    if first:
        print("executing on", torch.cuda.get_device_name(), file=sys.stderr)
        first = False

    a, b, a_scale, b_scale, c = data

    # a = a.contiguous()
    # b = b.contiguous()

    m, n = c.shape
    k = a.shape[1]

    global cache_m
    global cache_n
    global cache_k
    global cache_i

    if cache_m != m or cache_n != n or cache_k != k:
        cache_m = m
        cache_n = n
        cache_k = k
        cache_i = 0
    else:
        cache_i += 1

    # Don't check the performance on the first iteration to allow for some warmup
    measure = False
    if cache_i == 1:
        measure = True

    hip_module.fp8(
        a.data_ptr(),
        b.data_ptr(),
        a_scale.data_ptr(),
        b_scale.data_ptr(),
        c.data_ptr(),
        m,
        n,
        k,
        measure
    )

    return c
