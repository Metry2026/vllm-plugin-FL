"""Opt-in private FP32 quotient sampling for the bound MetaX native sampler.

The public vLLM sampling helpers and all noise generation remain unchanged.
Only the freshly allocated, unobserved softmax result in this private call is
eligible. Unsupported or observable calls execute the original native method.
"""

import hashlib
import importlib
import inspect
import sys
from pathlib import Path
from types import MethodType

import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from triton.compiler.compiler import CompiledKernel
from triton.runtime.jit import JITFunction

_VOCAB = 130560
_BLOCK = 4096
_TILES = 32
_SOURCE = "35b9a1ffeb1e50b73b0ddd1ffc0f968910ee21cf9cc5455dd879fde547b6ddc1"


@libentry()
@triton.jit
def _quotient_tiles(P, Q, Values, Indices, N, BLOCK: tl.constexpr, TILES: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    col = tile * BLOCK + lane
    offset = row.to(tl.int64) * N + col
    valid = col < N
    p = tl.load(P + offset, valid, other=0.0)
    q = tl.load(Q + offset, valid, other=1.0)
    # The actual parent is FP32 arith.divf, not reciprocal multiplication.
    score = p / q
    score = tl.where(valid, score, float("-inf"))
    value, index = tl.max(
        score, 0, return_indices=True, return_indices_tie_break_left=True
    )
    tl.store(Values + row * TILES + tile, value)
    tl.store(Indices + row * TILES + tile, tile * BLOCK + index)


@libentry()
@triton.jit
def _ordered_merge(Values, Indices, Out, TILES: tl.constexpr):
    row = tl.program_id(0)
    value = float("-inf")
    index = 0
    # Preserve the parent's ordered, strict comparison, including NaN cases.
    for tile in range(TILES):
        local_value = tl.load(Values + row * TILES + tile)
        local_index = tl.load(Indices + row * TILES + tile)
        update = local_value > value
        value = tl.where(update, local_value, value)
        index = tl.where(update, local_index, index)
    tl.store(Out + row, index)


def _private_quotient_argmax(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Internal only: the caller owns P, and Q has already been generated once."""
    rows = probs.shape[0]
    values = torch.empty((rows, _TILES), device=probs.device, dtype=torch.float32)
    indices = torch.empty((rows, _TILES), device=probs.device, dtype=torch.int32)
    out = torch.empty((rows,), device=probs.device, dtype=torch.int64)
    with torch.cuda.device(probs.device):
        _quotient_tiles[(rows, _TILES, 1)](
            probs, q, values, indices, _VOCAB, _BLOCK, _TILES, num_warps=8
        )
        _ordered_merge[(rows, 1, 1)](values, indices, out, _TILES, num_warps=8)
    return out.view(-1)


def bind_private_quotient(top, max_num_reqs: int) -> bool:
    """Bind one exact native instance; never patch either public helper."""
    import flag_gems
    import vllm.v1.sample.ops.topk_topp_sampler as sampling
    from vllm.platforms import current_platform
    from vllm_fl.dispatch.io_common import is_io_active

    if (
        getattr(current_platform, "vendor_name", None) != "metax"
        or type(top) is not sampling.TopKTopPSampler
        or getattr(top.forward, "__func__", None) is not sampling.TopKTopPSampler.forward_native
        or top.use_fp64_gumbel
        or type(max_num_reqs) is not int
        or max_num_reqs <= 0
    ):
        return False
    source = inspect.getsourcefile(sampling.TopKTopPSampler.forward_native)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != _SOURCE:
        return False
    for f in (
        sampling.TopKTopPSampler.forward_native, sampling.random_sample,
        sampling.sample_with_exponential_noise, sampling.empty_exponential_noise_like,
        sampling.apply_top_k_top_p_pytorch,
    ):
        if getattr(f, "__globals__", None) is not vars(sampling) or Path(f.__code__.co_filename) != Path(source):
            return False
    if (
        torch.empty_like is not torch._C._VariableFunctions.empty_like
        or torch.Tensor.softmax is not torch._C.TensorBase.softmax
    ):
        return False
    registry = flag_gems.current_work_registrar
    if registry is None or registry.reg_key != "CUDA":
        return False
    selected = {row[0]: row[1] for row in registry.config}
    div = selected.get("div_.Tensor")
    argmax = selected.get("argmax")
    softmax = selected.get("_softmax")
    if (
        getattr(div, "__module__", None) != "flag_gems.ops.div"
        or getattr(div, "__name__", None) != "true_divide_"
        or getattr(argmax, "__module__", None) != "flag_gems.ops.argmax"
        or getattr(softmax, "__module__", None) != "flag_gems.ops.softmax"
    ):
        return False

    original_forward = top.forward
    native = sampling.TopKTopPSampler.forward_native
    original_random = sampling.random_sample
    original_helper = sampling.sample_with_exponential_noise
    original_noise = sampling.empty_exponential_noise_like
    original_filter = sampling.apply_top_k_top_p
    original_filter_pytorch = sampling.apply_top_k_top_p_pytorch
    original_softmax_method = torch.Tensor.softmax
    original_empty_like = torch.empty_like
    softmax_module = importlib.import_module("flag_gems.ops.softmax")
    original_softmax_out = softmax_module.softmax_out
    original_softmax_kernel = softmax_module.softmax_kernel_inner
    original_softmax_jit = original_softmax_kernel.jit_function
    profiler_enabled = getattr(torch.autograd, "_profiler_enabled", None)
    if profiler_enabled is None:
        return False
    code_targets = (
        native, original_random, original_helper, original_noise,
        original_filter, original_filter_pytorch, div, argmax, softmax,
        original_softmax_out,
    )
    codes = tuple(f.__code__ for f in code_targets)

    def observable():
        return (
            not torch.is_inference_mode_enabled()
            or _get_current_dispatch_mode_stack()
            or torch._C._len_torch_function_stack()
            or sys.getprofile() is not None
            or sys.gettrace() is not None
            or profiler_enabled()
            or is_io_active()
            or getattr(JITFunction, "cache_hook", None) is not None
            or getattr(CompiledKernel, "launch_enter_hook", None) is not None
            or getattr(CompiledKernel, "launch_exit_hook", None) is not None
            or getattr(original_softmax_jit, "pre_run_hooks", ())
        )

    def forward(instance, logits, generators, k, p):
        # CPU-only qualification precedes every deleted private write. No
        # cached GPU-value certificate, CPU tensor readback or value guard.
        if (
            observable()
            or type(logits) is not torch.Tensor
            or logits.device.type != "cuda"
            or logits.dtype != torch.float32
            or logits.ndim != 2
            or logits.shape[1] != _VOCAB
            or not 0 < logits.shape[0] <= max_num_reqs
            or not logits.is_contiguous()
            or logits.requires_grad
            or instance.use_fp64_gumbel
            or type(generators) is not dict
            or any(type(i) is not int or type(g) is not torch.Generator
                   for i, g in generators.items())
            or (k is not None and type(k) is not torch.Tensor)
            or (p is not None and type(p) is not torch.Tensor)
            or flag_gems.current_work_registrar is not registry
            or sampling.TopKTopPSampler.forward_native is not native
            or sampling.random_sample is not original_random
            or sampling.sample_with_exponential_noise is not original_helper
            or sampling.empty_exponential_noise_like is not original_noise
            or sampling.apply_top_k_top_p is not original_filter
            or sampling.apply_top_k_top_p_pytorch is not original_filter_pytorch
            or torch.Tensor.softmax is not original_softmax_method
            or torch.empty_like is not original_empty_like
            or softmax_module.softmax_out is not original_softmax_out
            or softmax_module.softmax_kernel_inner is not original_softmax_kernel
            or original_softmax_kernel.jit_function is not original_softmax_jit
            or tuple(f.__code__ for f in code_targets) != codes
        ):
            return original_forward(logits, generators, k, p)

        logits = original_filter(logits, k, p)
        logits_to_return = None
        if instance.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif instance.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)

        # The source/observer guards establish fresh ownership. These metadata
        # checks also reject an unexpected allocator/layout/alias result.
        if (
            observable()
            or type(probs) is not torch.Tensor
            or probs.dtype != torch.float32
            or probs.shape != logits.shape
            or not probs.is_contiguous()
            or probs._base is not None
            or probs.untyped_storage()._cdata == logits.untyped_storage()._cdata
        ):
            return original_random(probs, generators, instance.use_fp64_gumbel), logits_to_return

        # Exactly the original allocation, condition, generator order and RNG
        # calls. Fallback after Q production consumes this Q without resampling.
        q = original_noise(probs, instance.use_fp64_gumbel)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
        if (
            observable()
            or type(q) is not torch.Tensor
            or q.dtype != probs.dtype
            or q.device != probs.device
            or q.shape != probs.shape
            or not q.is_contiguous()
            or q.untyped_storage()._cdata == probs.untyped_storage()._cdata
        ):
            return original_helper(probs, q), logits_to_return
        return _private_quotient_argmax(probs, q), logits_to_return

    top.forward = MethodType(forward, top)
    return True
