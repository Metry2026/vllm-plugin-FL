"""Opt-in private token reduction; public sampling helpers retain mutation.

Only the established task066 call context grants ownership. The original
softmax allocation and exponential noise order remain in this private scope.
"""
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import sys

import torch
from torch.overrides import _get_current_function_mode
from torch.utils._python_dispatch import _get_current_dispatch_mode

_INSTALLED = False
_RAW_NATIVE = _WRAPPED_NATIVE = None
_TOP = _UNIT = None
_RAW_APPLY = _RAW_NOISE = _RAW_TAIL = _RAW_RANDOM = None


def _eligible(self, logits, generators, k, p):
    context = _UNIT._CONTEXT.get()
    if (context is None or type(self) is not _TOP.TopKTopPSampler
            or context[0].topk_topp_sampler is not self
            or generators or k is not None or self.use_fp64_gumbel
            or self.logprobs_mode != 'raw_logprobs'
            or _TOP.TopKTopPSampler.forward_native is not _WRAPPED_NATIVE
            or getattr(self.forward, '__func__', None) is not _WRAPPED_NATIVE
            or _TOP.apply_top_k_top_p is not _RAW_APPLY
            or _TOP.empty_exponential_noise_like is not _RAW_NOISE
            or _TOP.sample_with_exponential_noise is not _RAW_TAIL
            or _TOP.random_sample is not _RAW_RANDOM):
        return False
    sampler = context[0]
    if (sampler._forward_pre_hooks or sampler._forward_hooks
            or sys.getprofile() is not None or sys.gettrace() is not None
            or _get_current_function_mode() is not None
            or _get_current_dispatch_mode() is not None
            or torch.autograd._profiler_enabled()
            or torch.cuda.is_current_stream_capturing()):
        return False
    return (type(logits) is torch.Tensor and logits.dtype == torch.float32
            and logits.device.type == 'cuda' and logits.is_inference()
            and not logits.requires_grad and logits.ndim == 2
            and logits.shape[0] >= 8 and logits.shape[1] == 130560
            and logits.is_contiguous() and logits.storage_offset() == 0
            and logits.device.index == torch.cuda.current_device())


def _private_tensors(P, Q):
    if (type(P) is not torch.Tensor or type(Q) is not torch.Tensor
            or P.dtype != torch.float32 or Q.dtype != torch.float32
            or P.device != Q.device or P.shape != Q.shape
            or P.stride() != (130560, 1) or Q.stride() != (130560, 1)
            or P.storage_offset() or Q.storage_offset()
            or not P.is_inference() or not Q.is_inference()
            or P.requires_grad or Q.requires_grad
            or P.numel() > 2147483647):
        return False
    span = P.numel() * P.element_size()
    p0, q0 = P.data_ptr(), Q.data_ptr()
    return p0 + span <= q0 or q0 + span <= p0


def _native(self, logits, generators, k, p):
    if not _eligible(self, logits, generators, k, p):
        return _RAW_NATIVE(self, logits, generators, k, p)
    # Exact stock forward_native + random_sample unseeded FP32 domain. P is
    # created here, never returned or exposed to the public helper on fast path.
    logits = _RAW_APPLY(logits, k, p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = _RAW_NOISE(probs, self.use_fp64_gumbel)
    q.exponential_()
    if not _private_tensors(probs, q):
        # Noise has already been generated once. Do not repeat the whole call.
        return _RAW_TAIL(probs, q), None
    from flag_gems.fused.iluvatar_private_div_argmax import private_div_argmax
    return private_div_argmax(probs, q), None


def install():
    global _INSTALLED, _RAW_NATIVE, _WRAPPED_NATIVE, _TOP, _UNIT
    global _RAW_APPLY, _RAW_NOISE, _RAW_TAIL, _RAW_RANDOM
    if _INSTALLED:
        return True
    from vllm_fl.patches import iluvatar_unit_temperature as unit
    if not unit._INSTALLED:
        return False
    import vllm.v1.sample.ops.topk_topp_sampler as top
    if hashlib.sha256(Path(inspect.getfile(top)).read_bytes()).hexdigest() != unit._PINS[top.__name__]:
        return False
    raw = top.TopKTopPSampler.forward_native
    functions = (raw, top.apply_top_k_top_p, top.empty_exponential_noise_like,
                 top.sample_with_exponential_noise, top.random_sample)
    if any(fn.__code__.co_filename != inspect.getfile(top) for fn in functions):
        return False
    _TOP, _UNIT, _RAW_NATIVE = top, unit, raw
    _RAW_APPLY, _RAW_NOISE = top.apply_top_k_top_p, top.empty_exponential_noise_like
    _RAW_TAIL, _RAW_RANDOM = top.sample_with_exponential_noise, top.random_sample
    _WRAPPED_NATIVE = wraps(raw)(_native)
    top.TopKTopPSampler.forward_native = _WRAPPED_NATIVE
    _INSTALLED = True
    return True
