"""Task023: one fixed BF16 NeoX RoPE + persistent paged-cache writer.

The functionalization boundary owns Q/K mutation. V and the QKV source stay
read-only. Cache mutation is ordered through the original dummy dependency.
"""

import torch
import triton
import triton.language as tl
from vllm.utils.torch_utils import LayerNameType


@triton.jit
def _rope_kv(Q, K, V, P, TABLE, SLOT, CACHE,
             QS0: tl.constexpr, QS1: tl.constexpr,
             KS0: tl.constexpr, KS1: tl.constexpr,
             VS0: tl.constexpr, VS1: tl.constexpr,
             CS0: tl.constexpr, CS1: tl.constexpr,
             CS2: tl.constexpr, CS3: tl.constexpr,
             TABLE_ROWS: tl.constexpr, CACHE_BLOCKS: tl.constexpr,
             HAS_CACHE: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 64)
    pos = tl.load(P + row).to(tl.int64)
    pos = tl.where(pos < 0, pos + TABLE_ROWS, pos)
    valid_pos = (pos >= 0) & (pos < TABLE_ROWS)
    c = tl.load(TABLE + pos * 128 + d, valid_pos, other=0).to(tl.float32)
    s = tl.load(TABLE + pos * 128 + 64 + d, valid_pos, other=0).to(tl.float32)
    if head < 16:
        a = tl.load(Q + row * QS0 + head * QS1 + d).to(tl.float32)
        b = tl.load(Q + row * QS0 + head * QS1 + 64 + d).to(tl.float32)
        left = (a * c - b * s).to(tl.bfloat16)
        right = (b * c + a * s).to(tl.bfloat16)
        tl.store(Q + row * QS0 + head * QS1 + d, left)
        tl.store(Q + row * QS0 + head * QS1 + 64 + d, right)
    else:
        h = head - 16
        a = tl.load(K + row * KS0 + h * KS1 + d).to(tl.float32)
        b = tl.load(K + row * KS0 + h * KS1 + 64 + d).to(tl.float32)
        # Explicit parent K BF16 materialization point, before cache use.
        left = (a * c - b * s).to(tl.bfloat16)
        right = (b * c + a * s).to(tl.bfloat16)
        tl.store(K + row * KS0 + h * KS1 + d, left)
        tl.store(K + row * KS0 + h * KS1 + 64 + d, right)
        if HAS_CACHE:
            slot = tl.load(SLOT + row).to(tl.int64)
            valid_slot = (slot >= 0) & (slot < CACHE_BLOCKS * 16)
            block = slot // 16
            offset = slot % 16
            target = block * CS0 + offset * CS2 + h * CS3 + d
            v0 = tl.load(V + row * VS0 + h * VS1 + d, valid_slot, other=0)
            v1 = tl.load(V + row * VS0 + h * VS1 + 64 + d, valid_slot, other=0)
            tl.store(CACHE + target, left, valid_slot)
            tl.store(CACHE + target + 64, right, valid_slot)
            tl.store(CACHE + target + CS1, v0, valid_slot)
            tl.store(CACHE + target + CS1 + 64, v1, valid_slot)


def launch(query, key, value, positions, table, cache, slots):
    """Fixed grid; admission belongs to the custom-op boundary."""
    _rope_kv[(query.shape[0], 18)](
        query, key, value, positions, table,
        positions if slots is None else slots,
        query if cache is None else cache,
        query.stride(0), query.stride(1), key.stride(0), key.stride(1),
        value.stride(0), value.stride(1),
        *(cache.stride()[:4] if cache is not None else (0, 0, 0, 0)),
        table.shape[0], 0 if cache is None else cache.shape[0], slots is not None,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )


_native = None


def native_rope(positions, query, key, table):
    """Original native arithmetic compiled separately, never candidate launch."""
    global _native
    if _native is None:
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

        def original(p, q, k, t):
            return RotaryEmbedding.forward_static(p, q, k, 128, 128, t, True)

        _native = torch.compile(original, dynamic=True, fullgraph=True,
                                options={"combo_kernels": True,
                                         "benchmark_combo_kernel": True})
    return _native(positions, query, key, table)


def supported_tensors(query, key, value, positions, table, cache, slots, is_neox):
    n = query.shape[0]
    if (not is_neox or not 1 <= n <= 8
            or query.shape != (n, 16, 128) or key.shape != (n, 2, 128)
            or value.shape != key.shape or positions.shape != (n,)
            or positions.dtype != torch.int64 or positions.stride() != (1,)
            or table.shape != (131072, 128) or table.stride() != (128, 1)):
        return False
    if any(t.dtype != torch.bfloat16 or t.device != query.device
           or not t.is_cuda or t.stride(-1) != 1
           for t in (query, key, value, table)):
        return False
    if positions.device != query.device:
        return False
    for t, heads in ((query, 16), (key, 2), (value, 2)):
        if t.stride(1) < 128 or t.stride(0) < heads * t.stride(1):
            return False
    # AF clones must separate mutable Q/K from each other and the V source.
    if len({t.untyped_storage().data_ptr() for t in (query, key, value)}) != 3:
        return False
    if slots is None:
        return True
    return (cache is not None and cache.dtype == torch.bfloat16
            and cache.device == query.device and cache.ndim == 5
            and cache.shape[1:] == (2, 16, 2, 128)
            and cache.stride() == (8192, 4096, 256, 128, 1)
            and slots.shape == (n,) and slots.dtype == torch.int64
            and slots.device == query.device and slots.stride() == (1,)
            and cache.untyped_storage().data_ptr() not in
            {t.untyped_storage().data_ptr() for t in (query, key, value, table, positions, slots)})


def impl(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
         positions: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool,
         layer_name: LayerNameType) -> torch.Tensor:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention.attention import get_attention_context
    from vllm.utils.torch_utils import _resolve_layer_name
    metadata, layer, cache, slots = get_attention_context(_resolve_layer_name(layer_name))
    context = get_forward_context()
    descriptor = context.batch_descriptor
    graph_all1 = (context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                  and descriptor is not None and descriptor.uniform
                  and descriptor.num_tokens == descriptor.num_reqs == query.shape[0]
                  and not descriptor.has_lora)
    ordinary_all1 = (context.cudagraph_runtime_mode == CUDAGraphMode.NONE
                     and metadata is not None
                     and getattr(metadata, '_task023_all1', False))
    if ((graph_all1 or ordinary_all1)
            and supported_tensors(query, key, value, positions, cos_sin_cache,
                                  cache, slots, is_neox)):
        launch(query, key, value, positions, cos_sin_cache, cache, slots)
    else:
        q, k = native_rope(positions, query, key, cos_sin_cache)
        query.copy_(q)
        key.copy_(k)
        if slots is not None:
            # Same original writer, exactly once; no candidate function here.
            layer.impl.do_kv_cache_update(layer, key, value, cache, slots)
    return torch.empty(0, device=cache.device, dtype=cache.dtype)


def fake(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
         positions: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool,
         layer_name: LayerNameType) -> torch.Tensor:
    return torch.empty(0, device=query.device, dtype=query.dtype)
