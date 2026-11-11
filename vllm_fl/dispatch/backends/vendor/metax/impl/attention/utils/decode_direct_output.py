"""Private caller output supply for one narrow ordinary decode domain.

Unknown metadata/layout/alias domains return False before the native call.
The native call and its postconditions deliberately have no fallback catcher.
"""
import os
import sys
import types

import torch


def _storage_span(tensor):
    storage = tensor.untyped_storage()
    start, size = storage.data_ptr(), storage.nbytes()
    if start <= 0 or size <= 0:
        raise ValueError('storage range unavailable')
    return start, start + size


def _prepare_output(
    wrapper, q, k_cache, v_cache, output, block_table, cache_seqlens,
    scale, window, softcap, alibi_slopes, sinks, metadata,
    query_input, key_input, value_input, kv_cache,
):
    # Preserve the complete public collection path even when its value is empty.
    if 'MHA_INPUT_COLLECTION' in os.environ:
        return None
    if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
        return None
    interface = sys.modules.get('flash_attn.flash_attn_interface')
    if not isinstance(interface, types.ModuleType):
        return None
    public = vars(interface)
    if (type(wrapper) is not types.FunctionType
            or public.get('flash_attn_with_kvcache') is not wrapper
            or wrapper.__globals__ is not public):
        return None
    extension = public.get('flash_attn_cuda')
    if extension is not sys.modules.get('flash_attn_2_cuda'):
        return None
    if not isinstance(extension, types.ModuleType):
        return None
    native = vars(extension).get('fwd_kvcache')
    if not isinstance(native, types.BuiltinFunctionType):
        return None
    if (type(scale) is not float or scale != 128 ** -0.5
            or type(window) is not tuple or len(window) != 2
            or any(type(edge) is not int for edge in window) or window != (-1, -1)
            or type(softcap) not in (int, float) or softcap != 0
            or alibi_slopes is not None or sinks is not None):
        return None
    tensors = (q, k_cache, v_cache, output, block_table, cache_seqlens)
    if any(type(tensor) is not torch.Tensor for tensor in tensors):
        return None
    if any(tensor.layout is not torch.strided for tensor in tensors):
        return None
    if q.device.type != 'cuda' or any(tensor.device != q.device for tensor in tensors):
        return None
    if any(tensor.requires_grad or tensor.is_conj() or tensor.is_neg()
           or tensor._is_zerotensor() for tensor in tensors):
        return None
    if not output.is_inference() or not q.is_inference():
        # A normal caller's original copy_ has a visible version-counter effect.
        return None
    if any(tensor.dtype is not torch.bfloat16 for tensor in (q, k_cache, v_cache, output)):
        return None
    if q.shape != (1, 1, 16, 128) or q.stride() != (2048, 2048, 128, 1):
        return None
    if (k_cache.ndim != 4 or k_cache.shape[0] <= 0
            or k_cache.shape[1:] != (16, 2, 128)
            or v_cache.shape != k_cache.shape
            or k_cache.stride() != (4096, 256, 128, 1)
            or v_cache.stride() != (4096, 256, 128, 1)):
        return None
    if (output.ndim != 3 or output.shape[0] < 1
            or output.shape[1:] != (16, 128) or not output.is_contiguous()):
        return None
    # Preserve the aligned pointer class observed for the natural caller and
    # qualified guarded output; unaligned contiguous views use the public path.
    if output.data_ptr() % 256 != 0:
        return None
    if (block_table.dtype is not torch.int32 or block_table.ndim != 2
            or block_table.shape[0] != 1 or block_table.shape[1] < 1
            or block_table.stride(-1) != 1
            or cache_seqlens.dtype is not torch.int32
            or cache_seqlens.shape != (1,) or cache_seqlens.stride() != (1,)):
        return None
    values = vars(metadata)
    for name, expected in (('num_actual_tokens', 1), ('num_decodes', 1),
                           ('num_decode_tokens', 1), ('num_prefills', 0),
                           ('num_prefill_tokens', 0)):
        value = values.get(name)
        if type(value) is not int or value != expected:
            return None
    if values.get('use_cascade') is not False:
        return None
    # This parent's metadata has only scalar/None/direct Tensor fields. Unknown
    # containers/subclasses are outside this proposal; no recursive object scan.
    readonly = [query_input, key_input, value_input, kv_cache]
    for value in values.values():
        if value is None or type(value) in (bool, int, float):
            continue
        if type(value) is not torch.Tensor:
            return None
        readonly.append(value)
    readonly.extend((q, k_cache, v_cache, block_table, cache_seqlens))
    destination = _storage_span(output)
    for tensor in readonly:
        if tensor is None:
            continue
        if type(tensor) is not torch.Tensor or tensor.layout is not torch.strided:
            return None
        if tensor.device != output.device:
            continue
        if tensor.numel() == 0:
            continue
        # Whole backing ranges are conservative, including disjoint same-storage
        # views. Never read Tensor values or use a version/object certificate.
        source = _storage_span(tensor)
        if destination[0] < source[1] and source[0] < destination[1]:
            return None
    prefix = output[:1]
    native_out = prefix.view(1, 2, 8, 128).transpose(1, 2)
    if (native_out.shape != (1, 8, 2, 128)
            or native_out.stride() != (2048, 128, 1024, 1)
            or native_out.data_ptr() != prefix.data_ptr()):
        return None
    return native, prefix, native_out


def try_decode_output(
    wrapper, q, k_cache, v_cache, output, block_table, cache_seqlens,
    scale, window, softcap, alibi_slopes, sinks, metadata,
    query_input, key_input, value_input, kv_cache,
):
    try:
        prepared = _prepare_output(
            wrapper, q, k_cache, v_cache, output, block_table, cache_seqlens,
            scale, window, softcap, alibi_slopes, sinks, metadata,
            query_input, key_input, value_input, kv_cache,
        )
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError, OverflowError):
        # Only pre-native metadata/view failures reach the unchanged public path.
        return False
    if prepared is None:
        return False
    native, prefix, native_out = prepared
    # Exactly the original public wrapper's 21 native operands; only operand 12
    # changes from None to the qualified caller view. No retry after this point.
    returned, _softmax_lse = native(
        q, k_cache, v_cache, None, None, cache_seqlens, None, None,
        None, None, block_table, None, native_out, scale, True,
        window[0], window[1], softcap, True, 0, None,
    )
    if (returned.shape != (1, 1, 16, 128)
            or returned.dtype is not output.dtype or returned.device != output.device
            or returned.data_ptr() != prefix.data_ptr()
            or returned.untyped_storage().data_ptr() != output.untyped_storage().data_ptr()):
        raise RuntimeError('native explicit output did not preserve the caller alias contract')
    return True
