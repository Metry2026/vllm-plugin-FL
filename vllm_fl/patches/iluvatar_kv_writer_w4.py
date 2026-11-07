"""Opt-in fixed BI-V150 writer launch in the owned model update context.

The stock wrapper prelude executes once. Only its private JIT launch receives
four warps; the public wrapper, JIT object and launcher remain unchanged.
"""
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
from types import FunctionType
import weakref

import torch
from torch.overrides import _get_current_function_mode
from torch.utils._python_dispatch import _get_current_dispatch_mode

_PINS = {
    'vllm.v1.attention.ops.triton_reshape_and_cache_flash':
        'f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb',
    'vllm.v1.attention.backends.triton_attn':
        '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    'vllm.model_executor.layers.attention.attention':
        '3414faca4cc04c699dbb880ba1761230e1ed97035f2934604cad9a5490f4dba1',
    'vllm.forward_context':
        '1c6fa51e9a02a120fed5e7357afb1190516cbdfefd1c644dfcd860603f1527c5',
    'triton.backends.iluvatar.driver':
        '5023c317de1c2f3370d254320849a2013f68b353fd35884c60d857823c424f50',
}
_CONTEXT = ContextVar('task071_owned_writer', default=None)
_OWNER = None
_OPS = _BACKEND = _ATTENTION = _FC = None
_RAW_UPDATE = _WRAPPED_UPDATE = _RAW_WRITER = _PRIVATE_WRITER = None
_KERNEL = _WRITER_CODE = _UPDATE_CODE = _CALLER_CODE = None
_WRITER_GLOBALS = None
_INSTALLED = False


def _eligible(impl, layer, key, value, cache, slots):
    runner = None if _OWNER is None else _OWNER()
    context = _FC._forward_context
    if (runner is None or type(context) is not _FC.ForwardContext
            or context.no_compile_layers is not runner.compilation_config.static_forward_context
            or context.ubatch_slices is not None
            or type(context.slot_mapping) is not dict
            or type(layer) is not _ATTENTION.Attention
            or context.no_compile_layers.get(layer.layer_name) is not layer
            or context.slot_mapping.get(layer.layer_name) is not slots
            or layer.impl is not impl or layer.kv_cache is not cache
            or type(impl) is not _BACKEND.TritonAttentionImpl
            or impl.attn_type != _BACKEND.AttentionType.DECODER
            or impl.kv_cache_dtype != 'auto' or impl._is_per_token_head_quant
            or _BACKEND.TritonAttentionImpl.do_kv_cache_update is not _WRAPPED_UPDATE
            or _BACKEND.triton_reshape_and_cache_flash is not _entry
            or _RAW_UPDATE.__code__ is not _UPDATE_CODE
            or _RAW_WRITER.__code__ is not _WRITER_CODE
            or _ATTENTION.unified_kv_cache_update.__code__ is not _CALLER_CODE
            or _OPS.reshape_and_cache_kernel_flash is not _KERNEL
            or any(_RAW_WRITER.__globals__.get(k) is not v
                   for k, v in _WRITER_GLOBALS.items())
            or layer._forward_hooks or layer._forward_pre_hooks
            or _get_current_function_mode() is not None
            or _get_current_dispatch_mode() is not None
            or sys.getprofile() is not None or sys.gettrace() is not None
            or torch.autograd._profiler_enabled()):
        return False
    if any(type(t) is not torch.Tensor for t in (key, value, cache, slots)):
        return False
    if (key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16
            or cache.dtype != torch.bfloat16 or slots.dtype != torch.int64
            or key.device.type != 'cuda'
            or key.device != runner.device or value.device != key.device
            or cache.device != key.device or slots.device != key.device
            or key.ndim != 3 or value.shape != key.shape
            or key.shape[1:] != (2, 128)
            or key.stride() != (256, 128, 1)
            or value.stride() != (2560, 128, 1)
            or key.storage_offset() != 0
            or value.storage_offset() not in (0, 2304)
            or cache.ndim != 5 or cache.shape[1:] != (2, 16, 2, 128)
            or cache.stride() != (8192, 4096, 256, 128, 1)
            or slots.shape != (key.shape[0],) or slots.stride() != (1,)
            or any(t.requires_grad for t in (key, value, cache, slots))
            or not key.is_inference() or not value.is_inference()
            or torch._C._overlaps(cache, key)
            or torch._C._overlaps(cache, value)
            or torch._C._overlaps(cache, slots)):
        return False
    return True


def _update(impl, layer, key, value, kv_cache, slot_mapping):
    # A direct/public alias call retains the original path, even with a live
    # model context. The registered stock custom op is the only owned caller.
    if (sys._getframe(1).f_code is not _CALLER_CODE
            or not _eligible(impl, layer, key, value, kv_cache, slot_mapping)):
        return _RAW_UPDATE(impl, layer, key, value, kv_cache, slot_mapping)
    token = _CONTEXT.set((key, value, slot_mapping, _FC._forward_context))
    try:
        return _RAW_UPDATE(impl, layer, key, value, kv_cache, slot_mapping)
    finally:
        _CONTEXT.reset(token)


def _entry(*args, **kwargs):
    owned = _CONTEXT.get()
    if (owned is None or kwargs or len(args) != 8
            or args[0] is not owned[0] or args[1] is not owned[1]
            or args[4] is not owned[2] or _FC._forward_context is not owned[3]):
        return _RAW_WRITER(*args, **kwargs)
    return _PRIVATE_WRITER(*args)


class _FixedLaunch:
    def __getitem__(self, grid):
        launch = _KERNEL[grid]

        def call(*args, **kwargs):
            if (_CONTEXT.get() is not None
                    and kwargs.get('num_warps') == 16
                    and kwargs.get('num_stages') == 10
                    and kwargs.get('TILE_SIZE') == 256
                    and kwargs.get('num_heads') == 2
                    and kwargs.get('head_size') == 128
                    and kwargs.get('block_size') == 16
                    and kwargs.get('USE_HEAD_MAJOR_LAYOUT') is False
                    and kwargs.get('FP8_KV_CACHE') is False):
                kwargs['num_warps'] = 4
            return launch(*args, **kwargs)

        return call


def install(runner):
    global _INSTALLED, _OWNER, _OPS, _BACKEND, _ATTENTION, _FC
    global _RAW_UPDATE, _WRAPPED_UPDATE, _RAW_WRITER, _PRIVATE_WRITER
    global _KERNEL, _WRITER_CODE, _UPDATE_CODE, _CALLER_CODE, _WRITER_GLOBALS
    if _INSTALLED:
        return _OWNER() is runner
    from vllm_fl.worker.model_runner import ModelRunnerFL
    from vllm.platforms import current_platform
    if (type(runner) is not ModelRunnerFL
            or not runner._task069_private_sampling_installed
            or not runner._task066_unit_temperature_installed
            or runner.speculative_config is not None
            or current_platform.vendor_name != 'iluvatar'
            or runner.device.type != 'cuda'
            or torch.cuda.get_device_capability(runner.device) != (7, 1)):
        return False
    modules = {name: importlib.import_module(name) for name in _PINS}
    if any(hashlib.sha256(Path(inspect.getfile(modules[name])).read_bytes()).hexdigest() != pin
           for name, pin in _PINS.items()):
        return False
    ops, backend, attention, fc = [modules[name] for name in list(_PINS)[:4]]
    raw_update = backend.TritonAttentionImpl.do_kv_cache_update
    raw_writer = ops.triton_reshape_and_cache_flash
    if (backend.triton_reshape_and_cache_flash is not raw_writer
            or raw_update.__code__.co_filename != inspect.getfile(backend)
            or raw_writer.__code__.co_filename != inspect.getfile(ops)):
        return False
    _OWNER = weakref.ref(runner)
    _OPS, _BACKEND, _ATTENTION, _FC = ops, backend, attention, fc
    _RAW_UPDATE, _RAW_WRITER = raw_update, raw_writer
    _UPDATE_CODE, _WRITER_CODE = raw_update.__code__, raw_writer.__code__
    _CALLER_CODE = attention.unified_kv_cache_update.__code__
    _KERNEL = ops.reshape_and_cache_kernel_flash
    _WRITER_GLOBALS = {k: raw_writer.__globals__[k] for k in (
        'triton', 'torch', 'current_platform', 'is_quantized_kv_cache',
        '_is_supported_kv_cache_dtype')}
    private_globals = dict(raw_writer.__globals__)
    private_globals['reshape_and_cache_kernel_flash'] = _FixedLaunch()
    _PRIVATE_WRITER = FunctionType(_WRITER_CODE, private_globals,
                                   raw_writer.__name__, raw_writer.__defaults__,
                                   raw_writer.__closure__)
    _PRIVATE_WRITER.__kwdefaults__ = raw_writer.__kwdefaults__
    _WRAPPED_UPDATE = wraps(raw_update)(_update)
    backend.TritonAttentionImpl.do_kv_cache_update = _WRAPPED_UPDATE
    backend.triton_reshape_and_cache_flash = _entry
    _INSTALLED = True
    return True
