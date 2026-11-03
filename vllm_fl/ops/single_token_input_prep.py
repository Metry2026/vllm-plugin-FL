"""Default-off task018 guard for the pinned single-token integer preparation.

Only stable configuration/source checks are cached. Scheduling is checked each
call. No prepared data, GPU constants, replacement buffers or custom kernels.
"""
from pathlib import Path
import hashlib
import importlib
import inspect
import os

import numpy as np
import torch

_SOURCES = {
    'vllm.v1.worker.gpu_input_batch': 'de1ea4fc9d43fc6079dd375e407c2b6d93f5e1a11f49732d13654c41eaf30a65',
    'vllm.v1.worker.block_table': 'b3a6649b968fba0b48c333d7ac96e65cc713a86133cc7eef0a8399f5339ec600',
    'vllm.v1.utils': '4e4cda209e10d4477d251003779dec26ea5ef47541c3310995c007c7ee5fdcdb',
    'vllm.v1.attention.backends.triton_attn': '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    'vllm.v1.worker.lora_model_runner_mixin': '6e1c0d7aca2eaea43f5e04ebf506a598816e6663aef9eab5709cba7df8844e99',
    'vllm.v1.worker.kv_connector_model_runner_mixin': 'd140e79e93b8670a2100b8b365606d3af402430a59c990ef65820da987ef4974',
    'vllm.v1.worker.ec_connector_model_runner_mixin': '4cab531525c37d9549cff0df4bf15f4432ccacde4e130235c492a4c83c5fa0c6',
}


def _scope(r):
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.block_table import MultiGroupBlockTable, BlockTable
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
    if (type(r).__module__ != "vllm_fl.worker.model_runner" or type(r).__name__ != "ModelRunnerFL"
            or hashlib.sha256(Path(inspect.getsourcefile(type(r))).read_bytes()).hexdigest()
            != "e7e8d4003c57e0de8a1ecc1e720ef55eb6f8f54cd325d98666c0190b5c05b977"):
        return False
    p=r.parallel_config
    if (r.model_config.dtype != torch.bfloat16 or p.tensor_parallel_size != 1
            or p.pipeline_parallel_size != 1 or p.data_parallel_size != 1 or p.use_ubatching
            or r.speculative_config is not None or r.lora_config is not None
            or r.uses_mrope or r.uses_xdrope_dim > 0 or r.enable_prompt_embeds
            or r.model_config.is_encoder_decoder or r.model_config.is_multimodal_model
            or r.is_mm_prefix_lm or r.is_pooling_model or r.prepare_inputs_event is None
            or getattr(r.model_config.hf_config,'architectures',[]) != ['LlamaForCausalLM']
            or type(r.input_batch) is not InputBatch
            or type(r.input_batch.block_table) is not MultiGroupBlockTable
            or len(r.input_batch.block_table.block_tables) != 1
            or len(r.kv_cache_config.kv_cache_groups) != 1
            or type(r.kv_cache_config.kv_cache_groups[0].kv_cache_spec) is not FullAttentionSpec
            or np.dtype(np.intp) != np.dtype(np.int64) or r.arange_np.dtype != np.int64):
        return False
    b=r.input_batch.block_table[0]
    builders=[ag.get_metadata_builder(0) for group in r.attn_groups for ag in group]
    if (type(b) is not BlockTable or b.block_size != 16 or b.use_hybrid_blocks
            or b.pcp_world_size != 1 or b.dcp_world_size != 1
            or not builders or any(type(x) is not TritonAttentionMetadataBuilder for x in builders)
            or r.positions.dtype != torch.int64 or r.seq_lens.dtype != torch.int32
            or r.num_computed_tokens.dtype != torch.int32
            or r.input_batch.num_computed_tokens_cpu.dtype != np.int32
            or any(k.startswith('VLLM_FL_IO_DUMP') for k in os.environ)):
        return False
    return all(hashlib.sha256(Path(inspect.getsourcefile(importlib.import_module(name))).read_bytes()).hexdigest()==sha
               for name,sha in _SOURCES.items())


def eligible(r, scheduler_output, scheduled, n, total):
    # These are per-call invariants, never inferred from a bucket or last step.
    if (n <= 0 or total != n or not isinstance(scheduled,np.ndarray)
            or scheduled.shape != (n,) or scheduled.dtype != np.int32
            or not np.all(scheduled == 1)
            or scheduler_output.scheduled_spec_decode_tokens or r.use_async_spec_decode
            or r.input_batch.req_prompt_embeds):
        return False
    cached=getattr(r,'_task018_config_guard',None)
    if cached is None or cached[0] is not r.input_batch:
        try:supported=_scope(r)
        except (AttributeError,OSError,TypeError):supported=False
        cached=(r.input_batch,supported)
        r._task018_config_guard=cached
    return cached[1]
