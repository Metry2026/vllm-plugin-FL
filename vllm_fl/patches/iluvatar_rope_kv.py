"""Default-off task023 compiler bridge, with unchanged attention split/Graph key."""

import hashlib
import os
from pathlib import Path

_INSTALLED = False
_PINS = {
    'compilation/passes/utility/noop_elimination.py': '8abf85d28c13f95059f2acbc0275b9b9ed2ad4990f7c5763bbe3067a4aa629ce',
    'config/compilation.py': '1eab106378e99ff9de668e30e5e1af02eac0d9ded13c2579ca3969a22705c110',
    'compilation/passes/fusion/rope_kvcache_fusion.py': 'faed3153db71f91bddcccf99cfab47dca7881933769d262dec52e22b5bfad163',
    'compilation/passes/fusion/matcher_utils.py': 'ff3b1c07d51fd676a452a40b0722e0d78952676cfae030787140bbf163f9e037',
    'compilation/passes/pass_manager.py': '8ccc54639c7ba1db7752bdace340937f8bc6fa6a9ff1d38474306eb49291d12a',
    'model_executor/layers/attention/attention.py': '3414faca4cc04c699dbb880ba1761230e1ed97035f2934604cad9a5490f4dba1',
    'model_executor/layers/rotary_embedding/base.py': 'c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5',
    'model_executor/layers/rotary_embedding/common.py': '39bf5bb498ae45ca7a38269e85aa8b9b257a00c919e2d4d14a7ceb44ac7e9c98',
    'model_executor/models/llama.py': '1e1003454985bf86d3868a7755bcf5324676881548b3895809edf0b76698d384',
    'v1/attention/backends/triton_attn.py': '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    'v1/attention/ops/triton_reshape_and_cache_flash.py': 'f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb',
    'forward_context.py': '1c6fa51e9a02a120fed5e7357afb1190516cbdfefd1c644dfcd860603f1527c5',
}


def scope(config):
    import vllm
    import torch
    from vllm.config import CUDAGraphMode, CompilationMode
    from vllm.platforms import current_platform
    if os.environ.get('VLLM_FL_TASK023_ROPE_KV_FUSION', '0') != '1':
        return False
    m, p, c = config.model_config, config.parallel_config, config.cache_config
    h = m.hf_config
    cc = config.compilation_config
    if (getattr(current_platform, 'vendor_name', None) != 'iluvatar'
            or m.dtype != torch.bfloat16
            or getattr(h, 'architectures', []) != ['LlamaForCausalLM']
            or (h.hidden_size, h.num_attention_heads, h.num_key_value_heads,
                h.num_hidden_layers, h.max_position_embeddings) != (2048, 16, 2, 42, 131072)
            or p.tensor_parallel_size != 1 or p.pipeline_parallel_size != 1
            or p.data_parallel_size != 1 or p.use_ubatching
            or config.speculative_config is not None or config.lora_config is not None
            or m.quantization is not None or m.is_encoder_decoder or m.is_multimodal_model
            or m.enable_prompt_embeds or m.use_mla or c.block_size != 16
            or c.cache_dtype != 'auto' or c.kv_sharing_fast_prefill
            or config.scheduler_config.max_num_batched_tokens != 2048
            or cc.mode != CompilationMode.VLLM_COMPILE
            or cc.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
            or cc.custom_ops != ['none'] or cc.use_inductor_graph_partition):
        return False
    root = Path(vllm.__file__).resolve().parent
    return all(hashlib.sha256((root / name).read_bytes()).hexdigest() == pin
               for name, pin in _PINS.items())


def configure(config):
    if not scope(config):
        return
    cc = config.compilation_config
    if cc.splitting_ops is None:
        cc.splitting_ops = list(cc._attention_ops) + [
            'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update']
    cc.splitting_ops = [op for op in cc.splitting_ops
                        if op != 'vllm::unified_kv_cache_update']
    endpoints = set(cc.compile_ranges_endpoints or [2048])
    endpoints.update((8, 2048))
    cc.compile_ranges_endpoints = sorted(endpoints)


def install():
    global _INSTALLED
    if _INSTALLED or os.environ.get('VLLM_FL_TASK023_ROPE_KV_FUSION', '0') != '1':
        return
    from vllm.compilation.passes.pass_manager import PostGradPassManager
    original = PostGradPassManager.configure

    def configure_passes(manager, config):
        original(manager, config)
        if scope(config):
            assert 'vllm::unified_attention_with_output' in config.compilation_config.splitting_ops
            assert 'vllm::unified_kv_cache_update' not in config.compilation_config.splitting_ops
            manager.add(_make_pass(config))

    PostGradPassManager.configure = configure_passes
    _INSTALLED = True



def _canonical_native_pattern(graph):
    import torch
    # Worker-side tracing can retain chunk/flatten while the real AOT graph
    # has already decomposed them. Normalize only the pinned whole-head case.
    for node in list(graph.nodes):
        if node.op != 'call_function':
            continue
        if node.target == torch.ops.aten.chunk.default:
            value = node.args[0].meta.get('val')
            if (len(node.args) == 3 and node.args[1:] == (2, -1)
                    and not node.kwargs and value is not None
                    and type(value.shape[-1]) is int and value.shape[-1] == 128):
                node.target = torch.ops.aten.split.Tensor
                node.args = (node.args[0], 64, -1)
        elif node.target == torch.ops.aten.flatten.using_ints:
            value = node.args[0].meta.get('val')
            start = node.args[1] if len(node.args) > 1 else node.kwargs.get('start_dim', 0)
            end = node.args[2] if len(node.args) > 2 else node.kwargs.get('end_dim', -1)
            if value is not None and value.ndim == 1 and start == 0 and end == -1:
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)

def _make_pass(config):
    import torch
    import torch._inductor.pattern_matcher as pm
    from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
    from vllm.compilation.passes.inductor_pass import enable_fake_mode
    from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
    from vllm.compilation.passes.fusion.rope_kvcache_fusion import RopeReshapeKVCachePattern
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.utils.torch_utils import direct_register_custom_op, _USE_LAYERNAME, _encode_layer_name
    from vllm_fl.ops import rope_kv_fusion as operation
    if not hasattr(torch.ops.vllm, 'task023_rope_kv'):
        direct_register_custom_op(op_name='task023_rope_kv', op_func=operation.impl,
                                  mutates_args=['query', 'key'], fake_impl=operation.fake)

    class Pattern(RopeReshapeKVCachePattern):
        FUSED_OP = torch.ops.vllm.task023_rope_kv.default

        def register(self, matcher):
            from torch._inductor.fx_passes.post_grad import view_to_reshape
            from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
            name = _encode_layer_name(self.layer_name)
            factory = (self._mk_pattern_with_layer_name_input if _USE_LAYERNAME
                       else self._mk_pattern_with_layer_name_closure)
            pattern, replacement = factory(name)

            def canonical_forward(*args, **kwargs):
                gm = pm.fwd_only(*args, **kwargs)
                view_to_reshape(gm)
                # The actual graph has already passed the same native no-op
                # canonicalization. Apply it to the pattern trace as well.
                NoOpEliminationPass(config)(gm.graph)
                _canonical_native_pattern(gm.graph)
                gm.graph.eliminate_dead_code()
                gm.recompile()
                return gm

            pm.register_replacement(pattern, replacement, self.get_inputs(),
                                    canonical_forward, matcher)

    class FusionPass(VllmPatternMatcherPass):
        @enable_fake_mode
        def __init__(self, cfg):
            super().__init__(cfg)
            self.patterns = pm.PatternMatcherPass(pass_name='task023_rope_kv')
            from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
            with set_current_vllm_config(cfg, check_compile=False):
                for layer in get_layers_from_vllm_config(cfg, Attention).values():
                    if (type(layer.impl) is TritonAttentionImpl
                            and (layer.num_heads, layer.num_kv_heads,
                                 layer.head_size, layer.head_size_v) == (16, 2, 128, 128)
                            and layer.impl.kv_cache_dtype == 'auto'):
                        Pattern(layer, is_neox=True).register(self.patterns)
                        if _USE_LAYERNAME:
                            break

        def __call__(self, graph):
            self.matched_count = self.patterns.apply(graph)

        def is_applicable_for_range(self, compile_range):
            return compile_range.end <= 8

        def uuid(self):
            return VllmInductorPass.hash_source(
                FusionPass, Pattern, Path(__file__).read_text(),
                Path(operation.__file__).read_text())

    return FusionPass(config)

