# SPDX-License-Identifier: Apache-2.0
"""Default-off O4 INT8 reader dispatch for one audited Iluvatar runtime.

EngineArgs selects the format before the scheduler/worker config is created.
Disabling INT8 requires an engine restart and cache rebuild, never a BF16 reader
on live INT8 pages. No worker-only dtype or allocator override is performed.
"""
import functools
import hashlib
import logging
import math
import os
from pathlib import Path

from .iluvatar_paged_gqa import OPTIONAL, REQUIRED

logger=logging.getLogger(__name__)
RUNTIME_SOURCES={
    'vllm.engine.arg_utils':'f1a32e2207e9ed2f0d954e3e6457effca17ca907cb0cf17bba2ca8bde855a719',
    'vllm.v1.attention.backends.triton_attn':'1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    'vllm.v1.attention.ops.triton_unified_attention':'bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0',
    'vllm.v1.attention.ops.triton_reshape_and_cache_flash':'f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb',
    'vllm.v1.kv_cache_interface':'7e3c220998267190a622e216b2b125306807cf8be3ec1e9a8e84fe6e7a293003',
    'vllm.utils.torch_utils':'59a33afbe2f5e42248db0d263688131cfee26e39f9c371d1f72b61c4db4d7803',
    'vllm.model_executor.layers.attention.attention':'3414faca4cc04c699dbb880ba1761230e1ed97035f2934604cad9a5490f4dba1',
    'vllm.v1.worker.gpu.attn_utils':'a351178df66fb5364cbe663ba7f25346c6317927fcc29a8c00f8972a23440ace',
    'vllm.v1.worker.kv_connector_model_runner_mixin':'d140e79e93b8670a2100b8b365606d3af402430a59c990ef65820da987ef4974',
}


def _nonoverlap(t):
    extent = 1
    for stride, size in sorted(zip(t.stride(), t.shape)):
        if size > 1 and stride < extent:
            return False
        extent += (size - 1) * stride
    return all(s > 0 for s in t.stride())


def supported(a, torch):
    """Metadata only, no CUDA synchronization or tensor-value inspection."""
    if not REQUIRED.issubset(a) or not a.keys() <= REQUIRED | OPTIONAL:
        return False
    mode = a.get("kv_quant_mode")
    if getattr(mode, "name", None) != "INT8_PER_TOKEN_HEAD" or mode != 2:
        return False
    if a["causal"] is not True or tuple(a["window_size"]) != (-1, -1):
        return False
    if a["softcap"] not in (0, 0.0, None) or a.get("chunk_lookback", -1) != -1:
        return False
    if a.get("use_td", False) or a.get("use_alibi_sqrt", False):
        return False
    if any(a.get(x) is not None for x in (
        "alibi_slopes", "output_scale", "qq_bias", "sinks", "mm_prefix_range",
        "q_descale", "k_descale", "v_descale",
    )):
        return False
    if (type(a["softmax_scale"]) not in (int, float)
            or not math.isfinite(a["softmax_scale"]) or a["softmax_scale"] <= 0):
        return False
    if any(type(a[x]) is not int or a[x] < 0 for x in ("max_seqlen_q", "max_seqlen_k")):
        return False
    names = ("q", "k", "v", "out", "k_scale_cache", "v_scale_cache",
             "cu_seqlens_q", "seqused_k", "block_table")
    if any(a.get(x) is None for x in names):
        return False
    q, k, v, out, ks, vs, cu, lens, bt = (a[x] for x in names)
    if not q.is_cuda or any(t.device != q.device for t in (k, v, out, ks, vs, cu, lens, bt)):
        return False
    if q.ndim != 3 or tuple(q.shape[1:]) != (16, 128) or out.shape != q.shape:
        return False
    if any(t.dtype != torch.bfloat16 or t.stride(-1) != 1 or not _nonoverlap(t) for t in (q, out)):
        return False
    if k.ndim != 4 or tuple(k.shape[2:]) != (2, 132) or v.shape != k.shape:
        return False
    if k.shape[1] not in (16, 32, 64, 128):
        return False
    if (k.stride() != v.stride() or k.stride(0) != 2 * k.shape[1] * 264
            or v.data_ptr() != k.data_ptr() + k.shape[1] * 264
            or k.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()):
        return False
    if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
           for t in (q, k, v, ks, vs, cu, lens, bt)):
        return False
    for data, scales in ((k, ks), (v, vs)):
        if data.dtype != torch.int8 or data.stride(-1) != 1 or not _nonoverlap(data):
            return False
        # Only the observed NHD inline format. HND's existing scale-view code is
        # not admitted. Separate scale tensors/shadow BF16 pools are not accepted.
        if data.stride(2) != 132 or data.stride(1) != 264:
            return False
        if scales.shape != data.shape[:3] or scales.dtype != torch.float32:
            return False
        if tuple(s * 4 for s in scales.stride()) != data.stride()[:3]:
            return False
        if scales.data_ptr() != data.data_ptr() + 128:
            return False
        if scales.untyped_storage().data_ptr() != data.untyped_storage().data_ptr():
            return False
    if cu.ndim != 1 or lens.ndim != 1 or bt.ndim != 2:
        return False
    if cu.shape[0] != lens.shape[0] + 1 or bt.shape[0] != lens.shape[0]:
        return False
    if any(t.dtype not in (torch.int32, torch.int64) for t in (cu, lens, bt)):
        return False
    if cu.stride(0) != 1 or lens.stride(0) != 1 or bt.stride(1) != 1 or not _nonoverlap(bt):
        return False
    return bt.shape[1] * k.shape[1] >= a["max_seqlen_k"]


def make_dispatch(original, candidate, torch, *, block_q=8, block_n=32, audit=False):
    """Prototype seam: rejects unknown INT8 semantics, delegates BF16 to B."""
    if block_q not in (2, 4, 8) or block_n not in (32, 64):
        raise ValueError("Invalid O4 tile configuration")
    counts = {"candidate": 0, "original": 0, "rejected": 0}

    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        if args:
            # Positional calls could contain an INT8 cache; never guess format.
            raise ValueError("O4 requires the audited keyword-only backend call")
        mode = kwargs.get("kv_quant_mode", 0)
        non_bf16 = mode != 0 or any(kwargs.get(n) is not None for n in ("k_scale_cache", "v_scale_cache"))
        if supported(kwargs, torch):
            if audit:
                counts["candidate"] += 1
            candidate(*(kwargs[n] for n in (
                "q", "k", "v", "k_scale_cache", "v_scale_cache", "out",
                "cu_seqlens_q", "seqused_k", "block_table", "softmax_scale",
            )), block_q=block_q, block_n=block_n)
            return None
        # Dtype is checked independently of mode, preventing an INT8 page from
        # accidentally entering B when the caller omits/mislabels quant metadata.
        int8_pages = any(getattr(kwargs.get(n), "dtype", None) == torch.int8 for n in ("k", "v"))
        if non_bf16 or int8_pages:
            if audit:
                counts["rejected"] += 1
            raise ValueError("Unsupported O4 cache/semantics; restart and rebuild BF16 cache to roll back")
        if audit:
            counts["original"] += 1
        return original(**kwargs)

    dispatch.o4_counts = counts
    dispatch.o4_original = original
    return dispatch


def install():
    if os.environ.get("VLLM_FL_O4_KV_INT8", "0") != "1":
        return False
    if os.environ.get('VLLM_FL_W007_MIXED_GQA','0')!='0':
        raise RuntimeError('O4 cannot be combined with unresolved W007')
    if os.environ.get('VLLM_BATCH_INVARIANT','0')!='0':
        raise RuntimeError('O4 batch-invariant mode is unsupported')
    import importlib
    import torch
    from vllm.platforms import current_platform
    if getattr(current_platform,'vendor_name',None)!='iluvatar':
        raise RuntimeError('O4 requires the audited Iluvatar platform')
    modules={name:importlib.import_module(name) for name in RUNTIME_SOURCES}
    for name,module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=RUNTIME_SOURCES[name]:
            raise RuntimeError('O4 runtime source changed: '+name)
    backend=modules['vllm.v1.attention.backends.triton_attn']
    ops=modules['vllm.v1.attention.ops.triton_unified_attention']
    args_cls=modules['vllm.engine.arg_utils'].EngineArgs
    if hasattr(backend.unified_attention,'o4_original'):
        return True
    original=backend.unified_attention
    if original is not ops.unified_attention and getattr(original,'w006_original',None) is not ops.unified_attention:
        raise RuntimeError('O4 unexpected attention patch chain')
    from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_attention_int8 import paged_gqa_attention_int8
    bq=int(os.environ.get('VLLM_FL_O4_BLOCK_Q','8'))
    bn=int(os.environ.get('VLLM_FL_O4_BLOCK_N','32'))
    if not hasattr(args_cls.create_engine_config,'o4_policy_original'):
        args_cls.create_engine_config=make_cache_policy(args_cls.create_engine_config)
    if not hasattr(backend.TritonAttentionImpl._ensure_scale_caches,'o4_scale_original'):
        backend.TritonAttentionImpl._ensure_scale_caches=make_scale_ensure(backend.TritonAttentionImpl._ensure_scale_caches,torch)
    backend.unified_attention=make_dispatch(
        original,paged_gqa_attention_int8,torch,block_q=bq,block_n=bn,
        audit=os.environ.get('VLLM_FL_O4_AUDIT','0')=='1',
    )
    logger.info('O4 INT8 installed: BQ=%s BN=%s; EngineArgs early cache policy; NHD D128 only',bq,bn)
    return True


def make_cache_policy(original):
    """Hook runs after general-plugin loading, before CacheConfig construction."""
    @functools.wraps(original)
    def create(self,*args,**kwargs):
        requested=self.kv_cache_dtype
        if requested not in ('auto','int8_per_token_head'):
            raise ValueError('O4 conflicts with explicitly requested cache dtype '+str(requested))
        self.kv_cache_dtype='int8_per_token_head'
        config=original(self,*args,**kwargs)
        if config.cache_config.cache_dtype!='int8_per_token_head':
            raise RuntimeError('O4 cache policy did not propagate into engine config')
        if getattr(config,'kv_transfer_config',None) is not None:
            raise ValueError('O4 KV connectors/cross-layer packing are not admitted')
        if getattr(config.cache_config,'kv_cache_dtype_skip_layers',[]):
            raise ValueError('O4 mixed cache dtype layers are unsupported')
        parallel=config.parallel_config
        if parallel.tensor_parallel_size!=1 or parallel.pipeline_parallel_size!=1:
            raise ValueError('O4 supports the audited single-device configuration only')
        hf=config.model_config.hf_config
        if (hf.num_attention_heads!=16 or hf.num_key_value_heads!=2
                or getattr(hf,'head_dim',hf.hidden_size//hf.num_attention_heads)!=128):
            raise ValueError('O4 requires 16 query / 2 KV heads, head dimension 128')
        logger.info('O4 cache policy requested=%s effective=%s before scheduler/worker creation',requested,config.cache_config.cache_dtype)
        return config
    create.o4_policy_original=original
    return create


def make_scale_ensure(original,torch):
    """Rebind cached scale views when the pool changes, outside graph capture."""
    @functools.wraps(original)
    def ensure(self,cache):
        if getattr(getattr(self,'_kv_quant_mode',None),'name',None)!='INT8_PER_TOKEN_HEAD':
            return original(self,cache)
        if (cache.ndim!=5 or tuple(cache.shape[1:])!=(2,cache.shape[2],2,132)
                or cache.dtype!=torch.int8 or not cache.is_contiguous() or cache.storage_offset()!=0):
            raise ValueError('O4 scale views require an independent contiguous NHD INT8 pool')
        key=(cache.data_ptr(),tuple(cache.shape),tuple(cache.stride()),cache.device)
        if getattr(self,'_o4_pool_key',None)!=key:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('O4 cannot initialize/rebind scale views inside graph capture')
            self._k_scale_cache=None;self._v_scale_cache=None
            original(self,cache)
            self._o4_pool_key=key
    ensure.o4_scale_original=original
    return ensure
