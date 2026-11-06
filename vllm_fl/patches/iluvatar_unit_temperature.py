"""Skip unit-temperature div only for a qualified private native pivot path.

The proof is created beside the existing CPU->device metadata update, never
from a device readback. A fresh or changed batch uses the original path once.
Process-local opt-in; disabling requires a fresh worker, as with other patches.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import weakref

import numpy as np
import torch

_CONTEXT = ContextVar('iluvatar_task066_temperature', default=None)
_INSTALLED = False
_RAW_FORWARD = _RAW_TEMPERATURE = _RAW_METADATA = None
_PINS = {
    'vllm.v1.sample.sampler': '1a2f18ddc91c89f3744ab5a93624be056bf20b7a8b2e6e13d0236182718a2253',
    'vllm.v1.worker.gpu_input_batch': 'de1ea4fc9d43fc6079dd375e407c2b6d93f5e1a11f49732d13654c41eaf30a65',
    'vllm.v1.sample.metadata': '6582cfe1a240c7ddb975f0ef6dd7dcd28dcf0fc7670394390250013977aa7ffd',
    'vllm.v1.sample.logits_processor.builtin': 'd8967b2358debb3ba34e0c5fc49b43c3a880e3d408d83f517575485701ad5cc9',
    'vllm.v1.sample.ops.topk_topp_sampler': '35b9a1ffeb1e50b73b0ddd1ffc0f968910ee21cf9cc5455dd879fde547b6ddc1',
    'vllm.v1.sample.ops.topk_topp_triton': '64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8',
    'flag_gems.ops.div': 'e8e80ec353f76619fb10881d2dfa5c0b77a131adc64c0207da54ffe2cd17f6d8',
    'flag_gems.ops.softmax': 'd7c6eef628fba956403c1f43f4a2c94dea1142ec8fa532359de5016e3bf9488d',
    'flag_gems.ops.argmax': '646a6e1cf1a8b1b9d24fb772b3e09e1f7bdea2f0449167ba9abfafaaf23576c8',
    'flag_gems.ops.exponential_': '2222bab5ecb28e66a7d9499c5611ca9dac73181cc094e0853fe3cdc5d0869728',
}


@dataclass(slots=True)
class _Proof:
    owner: object
    epoch: int
    temperature: object
    top_p: object
    rows: int
    unit: bool
    used: bool = False


def _metadata(owner):
    meta = _RAW_METADATA(owner)
    owner._task066_epoch = epoch = getattr(owner, '_task066_epoch', 0) + 1
    n = owner.num_reqs
    # These are the exact FP32 host values used by the existing copy_slice.
    ids = owner.req_ids
    mapping = owner.req_id_to_index
    unit = (type(owner) is InputBatch and 0 < n <= owner.max_num_reqs
            and len(ids) == n and len(set(ids)) == n and None not in ids
            and all(mapping.get(req) == i for i, req in enumerate(ids))
            and meta.all_random and not meta.all_greedy
            and meta.temperature is not None and meta.top_p is not None
            and meta.top_k is None
            and bool(np.all(owner.temperature_cpu[:n].view(np.uint32) == 0x3f800000))
            and bool(np.all(owner.top_p_cpu[:n].view(np.uint32) == 0x3f733333)))
    meta._task066_proof = _Proof(weakref.ref(owner), epoch, meta.temperature,
                                meta.top_p, n, unit)
    return meta


def _eligible(sampler, logits, meta, proof, fresh):
    if (type(proof) is not _Proof or fresh or not proof.unit
            or type(meta) is not SamplingMetadata or type(sampler) is not Sampler):
        return False
    owner = proof.owner()
    if (owner is None or owner.sampling_metadata is not meta
            or getattr(owner, '_task066_epoch', None) != proof.epoch
            or meta.temperature is not proof.temperature or meta.top_p is not proof.top_p
            or not meta.all_random or meta.all_greedy or meta.top_k is not None
            or meta.generators or meta.max_num_logprobs is not None
            or meta.logprob_token_ids or not meta.no_penalties
            or meta.allowed_token_ids_mask is not None or meta.bad_words_token_ids
            or meta.thinking_budget_state_holder is not None):
        return False
    if (type(logits) is not torch.Tensor or logits.dtype != torch.bfloat16
            or logits.device.type != 'cuda' or logits.ndim != 2
            or tuple(logits.shape) != (proof.rows, 130560) or proof.rows < 8
            or not logits.is_contiguous() or logits.storage_offset() != 0
            or logits.requires_grad or not logits.is_inference()
            or not torch.is_inference_mode_enabled()):
        return False
    top = sampler.topk_topp_sampler
    module_hooks = torch.nn.modules.module
    if (type(top) is not TopKTopPSampler or sampler.logprobs_mode != 'raw_logprobs'
            or sampler.use_fp64_gumbel or top.use_fp64_gumbel
            or top._forward_pre_hooks or top._forward_hooks
            or module_hooks._global_forward_pre_hooks or module_hooks._global_forward_hooks
            or getattr(top.forward, '__func__', None) is not TopKTopPSampler.forward_native):
        return False
    # Unknown and active processors retain the complete original call chain.
    non = meta.logitsprocs.non_argmax_invariant
    arg = meta.logitsprocs.argmax_invariant
    if (len(non) != 2 or type(non[0]) is not MinTokensLogitsProcessor
            or type(non[1]) is not LogitBiasLogitsProcessor
            or non[0].min_toks or non[1].biases
            or len(arg) != 1 or type(arg[0]) is not MinPLogitsProcessor
            or arg[0].min_p_count):
        return False
    return True


def _forward(self, logits, sampling_metadata, predict_bonus_token=False,
             logprobs_mode_override=None):
    proof = getattr(sampling_metadata, '_task066_proof', None)
    fresh = True
    if type(proof) is _Proof:
        fresh = not proof.used
        proof.used = True
    supported = (predict_bonus_token is False and logprobs_mode_override is None
                 and _eligible(self, logits, sampling_metadata, proof, fresh))
    if not supported:
        return _RAW_FORWARD(self, logits, sampling_metadata, predict_bonus_token,
                            logprobs_mode_override)
    token = _CONTEXT.set((self, proof.temperature, logits.untyped_storage().data_ptr()))
    try:
        return _RAW_FORWARD(self, logits, sampling_metadata, predict_bonus_token,
                            logprobs_mode_override)
    finally:
        _CONTEXT.reset(token)


def _temperature(logits, temp, all_random):
    context = _CONTEXT.get()
    if (context is not None and all_random and temp is context[1]
            and type(logits) is torch.Tensor and logits.dtype == torch.float32
            and logits.is_inference() and not logits.requires_grad
            and logits.untyped_storage().data_ptr() != context[2]):
        return logits
    return _RAW_TEMPERATURE(logits, temp, all_random)


def install():
    global _INSTALLED, _RAW_FORWARD, _RAW_TEMPERATURE, _RAW_METADATA
    global InputBatch, SamplingMetadata, Sampler, TopKTopPSampler
    global MinTokensLogitsProcessor, LogitBiasLogitsProcessor, MinPLogitsProcessor
    if _INSTALLED:
        return True
    import importlib
    for name, expected in _PINS.items():
        module = importlib.import_module(name)
        if hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest() != expected:
            return False
    if torch.cuda.get_device_name() != 'Iluvatar BI-V150':
        return False
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
    from vllm.v1.sample.logits_processor.builtin import (
        MinTokensLogitsProcessor, LogitBiasLogitsProcessor, MinPLogitsProcessor)
    _RAW_FORWARD, _RAW_TEMPERATURE = Sampler.forward, Sampler.apply_temperature
    _RAW_METADATA = InputBatch._make_sampling_metadata
    InputBatch._make_sampling_metadata = wraps(_RAW_METADATA)(_metadata)
    Sampler.forward = wraps(_RAW_FORWARD)(_forward)
    Sampler.apply_temperature = staticmethod(wraps(_RAW_TEMPERATURE)(_temperature))
    _INSTALLED = True
    return True
