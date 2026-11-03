"""Default-off task024 direct current-K/V prefill reader, retaining KV writer."""
import functools
import hashlib
import importlib
import logging
import os
from pathlib import Path

_INSTALLED = False
_LOG = logging.getLogger(__name__)
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
_KERNEL_SHA = 'c5c9ca4741438d5c3de54130285e3de50c8508b3348996b8d376911829bf56e0'


def _configuration_scope(builder):
    import torch
    from vllm.config import CUDAGraphMode, CompilationMode
    config = builder.vllm_config
    m, p, c, cc = (config.model_config, config.parallel_config,
                   config.cache_config, config.compilation_config)
    h = m.hf_config
    return (
        m.dtype == torch.bfloat16
        and getattr(h, 'architectures', []) == ['LlamaForCausalLM']
        and (h.hidden_size, h.num_attention_heads, h.num_key_value_heads,
             h.num_hidden_layers, h.max_position_embeddings) == (2048, 16, 2, 42, 131072)
        and p.tensor_parallel_size == p.pipeline_parallel_size == p.data_parallel_size == 1
        and not p.use_ubatching
        and config.speculative_config is None and config.lora_config is None
        and m.quantization is None and not m.is_encoder_decoder
        and not m.is_multimodal_model and not m.enable_prompt_embeds and not m.use_mla
        and c.block_size == 16 and c.cache_dtype == 'auto'
        and not c.kv_sharing_fast_prefill
        and config.scheduler_config.max_num_batched_tokens == 2048
        and cc.mode == CompilationMode.VLLM_COMPILE
        and cc.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        and cc.custom_ops == ['none'] and not cc.use_inductor_graph_partition
        and cc.compile_ranges_endpoints == [2048]
        and 'vllm::unified_kv_cache_update' in cc.splitting_ops
        and 'vllm::unified_attention_with_output' in cc.splitting_ops
    )


def qualify(builder, common_prefix_len, common):
    """Read only current authoritative CPU fields; never use lazy D2H properties."""
    import torch
    if (common_prefix_len != 0 or common.max_query_len <= 1
            or common.causal is not True or common.mm_req_doc_ranges is not None
            or not _configuration_scope(builder)):
        return None
    n, total = common.num_reqs, common.num_actual_tokens
    if type(n) is not int or type(total) is not int or n <= 0 or total <= 0:
        return None
    cu = common.query_start_loc_cpu
    seq = common._seq_lens_cpu
    computed = common._num_computed_tokens_cpu
    # None means unavailable/non-authoritative (including async speculation).
    for tensor, size in ((cu, n + 1), (seq, n), (computed, n)):
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                or tensor.dtype not in (torch.int32, torch.int64)
                or tensor.ndim != 1 or tensor.shape[0] != size
                or tensor.stride(0) != 1):
            return None
    boundaries, lengths, history = cu.tolist(), seq.tolist(), computed.tolist()
    if boundaries[0] != 0 or boundaries[-1] != total:
        return None
    query = [boundaries[i + 1] - boundaries[i] for i in range(n)]
    if (any(x <= 0 for x in query) or any(x != 0 for x in history)
            or query != lengths or max(query) != common.max_query_len):
        return None
    # Immutable call-domain facts only. Never retain a K/V view or request state.
    return n, total


def _span(tensor):
    start = tensor.data_ptr()
    end = start + (sum((size - 1) * stride for size, stride in
                       zip(tensor.shape, tensor.stride())) + 1) * tensor.element_size()
    return start, end


def _disjoint(output, inputs):
    a, b = _span(output)
    return all(b <= c or d <= a for c, d in map(_span, inputs))


def try_current(obj, layer, query, key, value, cache, md, output,
                output_scale, output_block_scale, parent_supported, operation):
    import torch
    from vllm.v1.attention.backend import AttentionType
    domain = getattr(md, '_task024_current_kv_domain', None)
    if (domain is None or obj.attn_type != AttentionType.DECODER
            or output_scale is not None or output_block_scale is not None
            or key is None or value is None or cache is None or cache.ndim != 5
            or obj.kv_cache_dtype != 'auto' or obj._is_per_token_head_quant
            or md.use_cascade or domain != (md.seq_lens.shape[0], md.num_actual_tokens)):
        return False
    n, total = domain
    if (key.ndim != 3 or value.ndim != 3
            or tuple(key.shape[1:]) != (2, 128) or tuple(value.shape[1:]) != (2, 128)
            or key.shape[0] < total or value.shape[0] < total
            or query.shape[0] < total or output.shape[0] < total
            or md.query_start_loc.shape != (n + 1,)):
        return False
    for tensor in (key, value):
        if (tensor.device != query.device or tensor.dtype != torch.bfloat16
                or tensor.stride(2) != 1 or tensor.stride(1) < 128
                or tensor.stride(0) < 2 * tensor.stride(1)):
            return False
    # Preserve the native BF16 preprocessing and every original supported guard.
    key_cache, value_cache = cache.unbind(1)
    descale_shape = (md.query_start_loc.shape[0] - 1, key_cache.shape[2])
    q, k, v, out = query[:total], key[:total], value[:total], output[:total]
    args = dict(q=q, k=key_cache, v=value_cache, out=out,
                cu_seqlens_q=md.query_start_loc, max_seqlen_q=md.max_query_len,
                seqused_k=md.seq_lens, max_seqlen_k=md.max_seq_len,
                softmax_scale=obj.scale, causal=md.causal,
                alibi_slopes=obj.alibi_slopes, use_alibi_sqrt=obj.use_alibi_sqrt,
                window_size=obj.sliding_window, block_table=md.block_table,
                softcap=obj.logits_soft_cap, q_descale=None,
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
                seq_threshold_3D=md.seq_threshold_3D,
                num_par_softmax_segments=md.num_par_softmax_segments,
                softmax_segm_output=md.softmax_segm_output,
                softmax_segm_max=md.softmax_segm_max,
                softmax_segm_expsum=md.softmax_segm_expsum,
                sinks=obj.sinks, output_scale=output_scale,
                mm_prefix_range=md.mm_prefix_range_tensor,
                kv_quant_mode=obj._kv_quant_mode, k_scale_cache=None,
                v_scale_cache=None, chunk_lookback=obj.chunk_lookback, use_td=obj.use_td)
    if (key_cache.shape[1] != 16 or not parent_supported(args, torch)
            or not _disjoint(out, (q, k, v))):
        return False
    operation(q, k, v, out, md.query_start_loc, md.seq_lens, obj.scale)
    return True


def install():
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get('VLLM_FL_TASK024_CURRENT_KV_PREFILL', '0') != '1':
        return False
    if any(os.environ.get(name, '0') != '1' for name in (
            'VLLM_FL_W006_PAGED_GQA', 'VLLM_FL_PAGED_GQA_SCALAR_PAGES',
            'VLLM_FL_PAGED_GQA_SINGLE_STAGE', 'VLLM_FL_TASK018_SINGLE_TOKEN_PREP',
            'VLLM_FL_TASK020_FUSED_INT_PREP', 'VLLM_FL_TASK022_CONST_LOGITS_INDICES')):
        return False
    if (os.environ.get('VLLM_FL_W006_BLOCK_Q', '8') != '8'
            or os.environ.get('VLLM_FL_W006_BLOCK_N', '32') != '32'):
        return False
    import vllm
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends import triton_attn as backend
    from vllm_fl.patches.iluvatar_paged_gqa import supported
    if (getattr(current_platform, 'vendor_name', None) != 'iluvatar'
            or not hasattr(backend.unified_attention, 'w006_original')):
        return False
    core_root = Path(vllm.__file__).resolve().parent
    plugin_root = Path(__file__).resolve().parents[1]
    try:
        for base, pins in ((core_root, _CORE_PINS), (plugin_root, _PLUGIN_PINS)):
            if any(hashlib.sha256((base / rel).read_bytes()).hexdigest() != pin
                   for rel, pin in pins.items()):
                return False
        parent_operation = importlib.import_module(
            'flag_gems.runtime.backend._iluvatar.fused.paged_gqa_single_stage')
        if hashlib.sha256(Path(parent_operation.__file__).read_bytes()).hexdigest() != 'd56590191021a2dd96b88d772966d17008c8168a3589fa6ce0d2fab9e4200fa8':
            return False
        operation = importlib.import_module(
            'flag_gems.runtime.backend._iluvatar.fused.current_kv_single_stage')
        if hashlib.sha256(Path(operation.__file__).read_bytes()).hexdigest() != _KERNEL_SHA:
            return False
    except (OSError, ImportError):
        _LOG.warning('task024 source unavailable; retain exact original reader')
        return False
    original_build = backend.TritonAttentionMetadataBuilder.build
    original_forward = backend.TritonAttentionImpl.forward

    @functools.wraps(original_build)
    def build(builder, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = original_build(builder, common_prefix_len, common_attn_metadata,
                                  fast_build=fast_build)
        metadata._task024_current_kv_domain = qualify(
            builder, common_prefix_len, common_attn_metadata)
        return metadata

    @functools.wraps(original_forward)
    def forward(obj, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if try_current(obj, layer, query, key, value, kv_cache, attn_metadata,
                       output, output_scale, output_block_scale,
                       supported, operation.current_kv_attention):
            return output
        return original_forward(obj, layer, query, key, value, kv_cache,
                                attn_metadata, output, output_scale,
                                output_block_scale)

    build.task024_original = original_build
    forward.task024_original = original_forward
    backend.TritonAttentionMetadataBuilder.build = build
    backend.TritonAttentionImpl.forward = forward
    _INSTALLED = True
    return True
