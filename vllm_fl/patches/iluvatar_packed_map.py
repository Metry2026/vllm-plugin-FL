"""Default-off task025 exact packed program map; parent attention stays intact."""
from contextvars import ContextVar
import functools
import hashlib
import importlib
import logging
import os
from pathlib import Path

_INSTALLED = False
_CURRENT = ContextVar('task025_current_attention_metadata', default=None)
_INFLIGHT_HOST = []
_LOG = logging.getLogger(__name__)
_KERNEL_SHA = 'a5a8c5aa66361c653d2b3af78b9fbfdbea1131f9dca5e47b161aa0f6ad0f7e8f'
_CORE_PINS = {
    'v1/attention/backend.py': 'c524988a2223ec556bdb128a31c255075f0d07e7643e803b447975cd9c30a6b8',
    'v1/attention/backends/triton_attn.py': '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    'v1/attention/ops/triton_reshape_and_cache_flash.py': 'f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb',
    'model_executor/layers/attention/attention.py': '3414faca4cc04c699dbb880ba1761230e1ed97035f2934604cad9a5490f4dba1',
}
_PLUGIN_PINS = {
    'patches/iluvatar_paged_gqa.py': 'f45aeaf440d7cfd985936da05b533073b75da03769c093b81a4402f73e7b9341',
    'worker/model_runner.py': 'fe1679644f096ad3559277468fe2607ac44ddaee000c75f71ae78ac1aa151117',
    'worker/worker.py': 'a8de1780d1b41bf4889c26f3d593e6ec54e91a5f61996ce64b4d79f2a97efa24',
    'ops/single_token_input_prep.py': 'be10e17891d37f87b776477d8cd01863f8fd4dafe948118a68cf6c0f42be4dc3',
    'ops/fused_int_input_prep.py': '503188fdd00b0e6f191ff3298fbab14b0227627c78a8ec65c65d706f8215171f',
    'ops/constant_logits_indices.py': '6733a247bb3d9b6e7a3a0d80a5e73420306e99374099450653d64ebde0a96a0b',
}


def _configuration_scope(builder, torch):
    from vllm.config import CUDAGraphMode, CompilationMode
    c = builder.vllm_config
    h = c.model_config.hf_config
    cc = c.compilation_config
    p = c.parallel_config
    return (
        c.model_config.dtype == torch.bfloat16
        and h.architectures == ['LlamaForCausalLM']
        and (h.hidden_size, h.num_attention_heads, h.num_key_value_heads,
             h.num_hidden_layers, h.max_position_embeddings) == (2048, 16, 2, 42, 131072)
        and p.tensor_parallel_size == p.pipeline_parallel_size == p.data_parallel_size == 1
        and not p.use_ubatching and c.speculative_config is None
        and c.lora_config is None and c.model_config.quantization is None
        and not c.model_config.is_encoder_decoder
        and not c.model_config.is_multimodal_model
        and not c.model_config.enable_prompt_embeds and not c.model_config.use_mla
        and c.cache_config.block_size == 16 and c.cache_config.cache_dtype == 'auto'
        and not c.cache_config.kv_sharing_fast_prefill
        and c.scheduler_config.max_num_batched_tokens == 2048
        and cc.mode == CompilationMode.VLLM_COMPILE
        and cc.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        and cc.custom_ops == ['none'] and not cc.use_inductor_graph_partition
        and cc.compile_ranges_endpoints == [2048]
        and 'vllm::unified_kv_cache_update' in cc.splitting_ops
        and 'vllm::unified_attention_with_output' in cc.splitting_ops
    )


def exact_pairs(common, torch):
    """Return a new immutable logical map from this call's existing CPU facts."""
    n, total = common.num_reqs, common.num_actual_tokens
    if (type(n) is not int or type(total) is not int
            or not 0 < n <= 256 or not 0 < total <= 2048
            or common.max_query_len <= 1 or common.causal is not True
            or common.mm_req_doc_ranges is not None):
        return None
    cpu = (common.query_start_loc_cpu, common._seq_lens_cpu,
           common._num_computed_tokens_cpu)
    for t, size in zip(cpu, (n + 1, n, n)):
        if (not isinstance(t, torch.Tensor) or t.device.type != 'cpu'
                or t.dtype not in (torch.int32, torch.int64)
                or t.ndim != 1 or t.shape[0] != size or t.stride(0) != 1):
            return None
    cu, seq, computed = [t.tolist() for t in cpu]
    if cu[0] != 0 or cu[-1] != total:
        return None
    query = [cu[i + 1] - cu[i] for i in range(n)]
    if (any(q <= 0 for q in query) or max(query) != common.max_query_len
            or any(h < 0 or s != h + q for q, s, h in zip(query, seq, computed))):
        return None
    return tuple((s, qb) for s, q in enumerate(query) for qb in range((q + 7) // 8))


def prepare_map(builder, prefix, common):
    import torch
    if prefix != 0 or not _configuration_scope(builder, torch):
        return None
    pairs = exact_pairs(common, torch)
    if pairs is None:
        return None
    # Reap only completed source copies. No synchronization and no source reuse.
    _INFLIGHT_HOST[:] = [(event, host) for event, host in _INFLIGHT_HOST
                         if not event.query()]
    stream = torch.cuda.current_stream(common.query_start_loc.device)
    host = torch.tensor(pairs, dtype=torch.int32, pin_memory=True)
    gpu = host.to(common.query_start_loc.device, non_blocking=True)
    copied = torch.cuda.Event()
    copied.record(stream)
    _INFLIGHT_HOST.append((copied, host))
    # New storage for every call, never overwritten for a subsequent request.
    # CUDA creation-stream order protects reuse after all enqueued consumers.
    return (gpu, common.query_start_loc, common.seq_lens, common.block_table_tensor,
            common.num_reqs, common.num_actual_tokens, stream.cuda_stream)


def try_mapped(bundle, a, torch, supported, operation):
    if bundle is None or not supported(a, torch):
        return False
    gpu, cu, seq, bt, n, total, stream = bundle
    if (a['q'].shape[0] != total or a['seqused_k'].shape[0] != n
            or a['k'].shape[1] != 16
            or a['cu_seqlens_q'] is not cu or a['seqused_k'] is not seq
            or a['block_table'] is not bt
            or gpu.device != a['q'].device or gpu.dtype != torch.int32
            or gpu.ndim != 2 or gpu.shape[1] != 2 or gpu.stride() != (2, 1)
            or not 0 < gpu.shape[0] <= total
            or torch.cuda.current_stream(gpu.device).cuda_stream != stream):
        return False
    operation(*(a[x] for x in ('q', 'k', 'v', 'out', 'cu_seqlens_q',
                              'seqused_k', 'block_table', 'softmax_scale')), gpu)
    return True


def install():
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get('VLLM_FL_TASK025_EXACT_PACKED_MAP', '0') != '1':
        return False
    if any(os.environ.get(k, '0') != '1' for k in (
            'VLLM_FL_W006_PAGED_GQA', 'VLLM_FL_PAGED_GQA_SCALAR_PAGES',
            'VLLM_FL_PAGED_GQA_SINGLE_STAGE', 'VLLM_FL_TASK018_SINGLE_TOKEN_PREP',
            'VLLM_FL_TASK020_FUSED_INT_PREP', 'VLLM_FL_TASK022_CONST_LOGITS_INDICES')):
        return False
    if (os.environ.get('VLLM_FL_W006_BLOCK_Q', '8') != '8'
            or os.environ.get('VLLM_FL_W006_BLOCK_N', '32') != '32'):
        return False
    import torch
    import vllm
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends import triton_attn as backend
    from vllm_fl.patches.iluvatar_paged_gqa import supported
    if (getattr(current_platform, 'vendor_name', None) != 'iluvatar'
            or not hasattr(backend.unified_attention, 'w006_original')):
        return False
    try:
        for base, pins in ((Path(vllm.__file__).resolve().parent, _CORE_PINS),
                           (Path(__file__).resolve().parents[1], _PLUGIN_PINS)):
            if any(hashlib.sha256((base / rel).read_bytes()).hexdigest() != pin
                   for rel, pin in pins.items()):
                return False
        parent = importlib.import_module(
            'flag_gems.runtime.backend._iluvatar.fused.paged_gqa_single_stage')
        if hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest() != 'd56590191021a2dd96b88d772966d17008c8168a3589fa6ce0d2fab9e4200fa8':
            return False
        mapped = importlib.import_module(
            'flag_gems.runtime.backend._iluvatar.fused.mapped_paged_gqa_single_stage')
        if hashlib.sha256(Path(mapped.__file__).read_bytes()).hexdigest() != _KERNEL_SHA:
            return False
    except (OSError, ImportError):
        _LOG.warning('task025 exact source unavailable; retain parent attention')
        return False
    original_build = backend.TritonAttentionMetadataBuilder.build
    original_forward = backend.TritonAttentionImpl.forward
    original_dispatch = backend.unified_attention

    @functools.wraps(original_build)
    def build(builder, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = original_build(builder, common_prefix_len, common_attn_metadata,
                                  fast_build=fast_build)
        metadata._task025_program_map = prepare_map(
            builder, common_prefix_len, common_attn_metadata)
        return metadata

    @functools.wraps(original_forward)
    def forward(obj, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        token = _CURRENT.set(attn_metadata)
        try:
            return original_forward(obj, layer, query, key, value, kv_cache,
                                    attn_metadata, output, output_scale,
                                    output_block_scale)
        finally:
            _CURRENT.reset(token)

    @functools.wraps(original_dispatch)
    def dispatch(*args, **kwargs):
        metadata = _CURRENT.get()
        bundle = getattr(metadata, '_task025_program_map', None)
        if not args and try_mapped(bundle, kwargs, torch, supported,
                                   mapped.mapped_paged_gqa_attention):
            return None
        return original_dispatch(*args, **kwargs)

    backend.TritonAttentionMetadataBuilder.build = build
    backend.TritonAttentionImpl.forward = forward
    backend.unified_attention = dispatch
    _INSTALLED = True
    _LOG.info('task025 exact packed map installed; original fallback and Graph retained')
    return True
