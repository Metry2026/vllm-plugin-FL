"""Opt-in fixed-domain native Decode page-index reuse (task059/O3-04)."""
import hashlib
import os
from pathlib import Path

NATIVE_SHA256 = "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"


def supported(a):
    # Native IS_3D is selected only for max_seqlen_q <= 1 in the pinned caller.
    fixed = dict(IS_3D=True, BLOCK_SIZE=16, TILE_SIZE=16, BLOCK_M=16,
                 BLOCK_Q=2, num_query_heads=16, num_queries_per_kv=8,
                 HEAD_SIZE=128, HEAD_SIZE_PADDED=128, USE_TD=False,
                 USE_TD_QO=False, USE_FP8=False, Q_IS_FP8=False,
                 KV_QUANT_MODE=0, USE_ALIBI_SLOPES=False,
                 USE_ALIBI_SQRT=False, USE_QQ_BIAS=False,
                 USE_SOFTCAP=False, USE_SINKS=False, SLIDING_WINDOW=0,
                 USE_CAUSAL=True, USE_PER_SEQ_CAUSAL=False,
                 USE_MM_PREFIX=False, CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1)
    if any(a.get(k) != value for k, value in fixed.items()):
        return False
    import torch
    return all(a[k].dtype == torch.bfloat16 for k in
               ("query_ptr", "key_cache_ptr", "value_cache_ptr", "output_ptr"))


class PageReuseKernel:
    def __init__(self, original, candidate):
        self.original = original
        self.candidate = candidate

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            # The pinned native caller supplies named arguments. Positional or
            # out-of-domain callers retain their original implementation.
            kernel = self.candidate if not args and supported(kwargs) else self.original
            return kernel[grid](*args, **kwargs)
        return launch


def install():
    if os.environ.get("VLLM_FL_TASK059_SINGLE_PAGE", "0") != "1":
        return False
    from vllm.v1.attention.ops import triton_unified_attention as ops
    if isinstance(ops.kernel_unified_attention, PageReuseKernel):
        return True
    if hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest() != NATIVE_SHA256:
        raise RuntimeError("task059 native source identity mismatch")
    from .iluvatar_decode_page_kernel import kernel_single_page
    ops.kernel_unified_attention = PageReuseKernel(
        ops.kernel_unified_attention, kernel_single_page)
    return True
