"""Opt-in exact FP32 tail fusion; original RNG and all upstream work retained."""
from functools import wraps
import hashlib
import inspect
import os
from pathlib import Path
import torch

FLAG = 'VLLM_FL_TASK027_DIV_ARGMAX'
TOP_SOURCE_SHA256 = '35b9a1ffeb1e50b73b0ddd1ffc0f968910ee21cf9cc5455dd879fde547b6ddc1'
VOCAB = 130560


def eligible(probs, q):
    if not isinstance(probs, torch.Tensor) or not isinstance(q, torch.Tensor):
        return False
    if (probs.ndim != 2 or q.ndim != 2 or probs.shape != q.shape
            or probs.shape[1] != VOCAB or probs.shape[0] <= 0
            or probs.numel() > 2147483647
            or probs.dtype != torch.float32 or q.dtype != torch.float32
            or not probs.is_cuda or probs.device != q.device
            or probs.stride() != (VOCAB, 1) or q.stride() != (VOCAB, 1)):
        return False
    # The actual sampler is outside model CUDA graphs. Preserve the complete
    # original path for capture and for a different active device.
    if probs.device.index != torch.cuda.current_device():
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    # Qualified contiguous live spans: reject all alias/partial overlap without
    # reading GPU values or synchronizing. Preserve the original alias fallback.
    p0 = probs.data_ptr()
    q0 = q.data_ptr()
    span = probs.numel() * probs.element_size()
    return p0 + span <= q0 or q0 + span <= p0


def install():
    if os.environ.get(FLAG, '0') != '1':
        return False
    import flag_gems
    if str(flag_gems.vendor_name).lower() != 'iluvatar':
        return False
    import vllm.v1.sample.ops.topk_topp_sampler as top
    original = top.sample_with_exponential_noise
    if getattr(original, '_task027_div_argmax', False):
        return True
    source = inspect.getsourcefile(original)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != TOP_SOURCE_SHA256:
        return False

    @wraps(original)
    def sample_with_exponential_noise(probs, q):
        if os.environ.get(FLAG, '0') == '1' and eligible(probs, q):
            from flag_gems.fused.div_argmax import fused_div_argmax
            return fused_div_argmax(probs, q)
        return original(probs, q)

    sample_with_exponential_noise._task027_div_argmax = True
    sample_with_exponential_noise._task027_original = original
    top.sample_with_exponential_noise = sample_with_exponential_noise
    return True
