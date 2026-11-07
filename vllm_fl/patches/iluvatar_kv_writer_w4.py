"""Opt-in fixed BI-V150 writer launch; stock public entry and JIT stay intact."""
from functools import wraps
import hashlib
import importlib
import inspect
from pathlib import Path
from types import FunctionType

import torch
from torch.overrides import _get_current_function_mode
from torch.utils._python_dispatch import _get_current_dispatch_mode

_PINS = {
    'vllm.v1.attention.ops.triton_reshape_and_cache_flash':
        'f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb',
    'vllm.v1.attention.backends.triton_attn':
        '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    'triton.backends.iluvatar.driver':
        '5023c317de1c2f3370d254320849a2013f68b353fd35884c60d857823c424f50',
}
_INSTALLED = False
_OPS = _BACKEND = _RAW_WRITER = _PRIVATE_WRITER = _RAW_UPDATE = None
_KERNEL = _KERNEL_CODE = _KERNEL_SRC = _DEVICE = None
_CODE = _DEFAULTS = _KWDEFAULTS = _GLOBALS = _BINDINGS = None


def _source_matches():
    return (_OPS.triton_reshape_and_cache_flash is _RAW_WRITER
            and _RAW_WRITER.__code__ is _CODE
            and _RAW_WRITER.__defaults__ is _DEFAULTS
            and _RAW_WRITER.__kwdefaults__ is _KWDEFAULTS
            and _GLOBALS.get('reshape_and_cache_kernel_flash') is _KERNEL
            and _KERNEL.fn.__code__ is _KERNEL_CODE
            and _KERNEL.src is _KERNEL_SRC
            and all(_GLOBALS.get(k) is v for k, v in _BINDINGS.items()))


def _entry(*args, **kwargs):
    # Unknown signatures/types/modes execute the original prelude once. Do not
    # inspect extra tensor metadata before preserving override behavior.
    if (kwargs or len(args) != 8
            or type(args[0]) is not torch.Tensor
            or type(args[1]) is not torch.Tensor
            or type(args[2]) is not torch.Tensor
            or type(args[3]) is not torch.Tensor
            or type(args[4]) is not torch.Tensor
            or type(args[5]) is not str or args[5] != 'auto'
            or _get_current_function_mode() is not None
            or _get_current_dispatch_mode() is not None
            or not _source_matches()):
        return _RAW_WRITER(*args, **kwargs)
    return _PRIVATE_WRITER(*args)


def _eligible(a):
    # These values were already computed by the unchanged stock prelude.
    if (a['num_warps'] != 16 or a['num_stages'] != 10
            or a['TILE_SIZE'] != 256 or a['num_heads'] != 2
            or a['head_size'] != 128 or a['block_size'] != 16
            or a['USE_HEAD_MAJOR_LAYOUT'] or a['FP8_KV_CACHE']
            or a['key_stride'] != 256 or a['value_stride'] != 2560
            or a['block_stride'] != 8192 or a['page_stride'] != 256
            or a['head_stride'] != 128 or a['x'] != 1):
        return False
    key, value = a['key_ptr'], a['value_ptr']
    kc, vc, slots = a['key_cache_ptr'], a['value_cache_ptr'], a['slot_mapping_ptr']
    ks, vs = a['k_scale'], a['v_scale']
    if (key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16
            or kc.dtype != torch.bfloat16 or vc.dtype != torch.bfloat16
            or slots.dtype != torch.int64
            or type(ks) is not torch.Tensor or type(vs) is not torch.Tensor
            or ks.dtype != torch.float32 or vs.dtype != torch.float32
            or ks.ndim != 0 or vs.ndim != 0
            or key.device != _DEVICE or value.device != _DEVICE
            or kc.device != _DEVICE or vc.device != _DEVICE
            or slots.device != _DEVICE or ks.device != _DEVICE or vs.device != _DEVICE
            or key.requires_grad or value.requires_grad
            or kc.requires_grad or vc.requires_grad or ks.requires_grad or vs.requires_grad):
        return False
    t = key.shape[0]
    if (type(t) is not int or t == 0 or key.shape != (t, 2, 128)
            or value.shape != key.shape or slots.shape != (t,)
            or key.stride() != (256, 128, 1)
            or value.stride() != (2560, 128, 1) or slots.stride() != (1,)
            or kc.ndim != 4 or kc.shape[1:] != (16, 2, 128)
            or vc.shape != kc.shape or kc.shape[0] <= 0
            or kc.stride() != (8192, 256, 128, 1)
            or vc.stride() != (8192, 256, 128, 1)
            or key.storage_offset() != 0 or value.storage_offset() not in (0, 2304)):
        return False
    kp, vp, cp, dp, sp = (key.data_ptr(), value.data_ptr(), kc.data_ptr(),
                           vc.data_ptr(), slots.data_ptr())
    # Keep the validated pointer specialization, including unused scales.
    if (kp % 16 or vp % 16 or cp % 16 or dp % 16 or sp % 16
            or ks.data_ptr() % 16 or vs.data_ptr() % 16 or dp != cp + 8192):
        return False
    end = cp + kc.shape[0] * 16384
    return ((kp + t * 512 <= cp or end <= kp)
            and (vp + (t - 1) * 5120 + 512 <= cp or end <= vp)
            and (sp + t * 8 <= cp or end <= sp))


class _FixedLaunch:
    def __getitem__(self, grid):
        jit = _GLOBALS['reshape_and_cache_kernel_flash']
        launch = jit[grid]

        def call(*args, **kwargs):
            if jit is _KERNEL and _eligible(kwargs):
                kwargs['num_warps'] = 4
            return launch(*args, **kwargs)

        return call


def install(runner):
    global _INSTALLED, _OPS, _BACKEND, _RAW_WRITER, _PRIVATE_WRITER, _RAW_UPDATE
    global _KERNEL, _KERNEL_CODE, _KERNEL_SRC, _DEVICE
    global _CODE, _DEFAULTS, _KWDEFAULTS, _GLOBALS, _BINDINGS
    if _INSTALLED:
        return runner.device == _DEVICE
    from vllm_fl.worker.model_runner import ModelRunnerFL
    from vllm.platforms import current_platform
    if (type(runner) is not ModelRunnerFL
            or not runner._task069_private_sampling_installed
            or not runner._task066_unit_temperature_installed
            or current_platform.vendor_name != 'iluvatar'
            or runner.device.type != 'cuda'
            or torch.cuda.get_device_capability(runner.device) != (7, 1)):
        return False
    modules = {name: importlib.import_module(name) for name in _PINS}
    if any(hashlib.sha256(Path(inspect.getfile(modules[name])).read_bytes()).hexdigest() != pin
           for name, pin in _PINS.items()):
        return False
    ops, backend = [modules[name] for name in list(_PINS)[:2]]
    raw = ops.triton_reshape_and_cache_flash
    if backend.triton_reshape_and_cache_flash is not raw:
        return False
    _OPS, _BACKEND, _RAW_WRITER = ops, backend, raw
    _RAW_UPDATE = backend.TritonAttentionImpl.do_kv_cache_update
    _CODE, _DEFAULTS, _KWDEFAULTS = raw.__code__, raw.__defaults__, raw.__kwdefaults__
    _GLOBALS = raw.__globals__
    _KERNEL, _DEVICE = ops.reshape_and_cache_kernel_flash, runner.device
    _KERNEL_CODE, _KERNEL_SRC = _KERNEL.fn.__code__, _KERNEL.src
    _BINDINGS = {k: _GLOBALS[k] for k in ('triton', 'torch', 'current_platform',
                                       'is_quantized_kv_cache', '_is_supported_kv_cache_dtype')}
    private_globals = dict(_GLOBALS, reshape_and_cache_kernel_flash=_FixedLaunch())
    _PRIVATE_WRITER = FunctionType(_CODE, private_globals, raw.__name__, _DEFAULTS, raw.__closure__)
    _PRIVATE_WRITER.__kwdefaults__ = _KWDEFAULTS
    backend.triton_reshape_and_cache_flash = wraps(raw)(_entry)
    _INSTALLED = True
    return True
