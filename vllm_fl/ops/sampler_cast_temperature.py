"""Default-off R=1 deterministic preprocessing; original native sampler/RNG."""
import os
import torch
import triton
import triton.language as tl
from vllm.v1.sample.sampler import Sampler
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.logits_processor.builtin import (
    MinTokensLogitsProcessor, LogitBiasLogitsProcessor, MinPLogitsProcessor,
)


@triton.jit
def _cast_temperature(X, T, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < N, other=0).to(tl.float32)
    temp = tl.load(T).to(tl.float32)
    result = tl.div_rn(value, temp)
    tl.store(Y + i, result, i < N)


def cast_temperature(logits, temp):
    out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    _cast_temperature[(triton.cdiv(logits.numel(), 1024),)](
        logits, temp, out, logits.numel(), BLOCK=1024,
        num_warps=4, enable_fp_fusion=False,
    )
    return out


class CastTemperatureSampler(Sampler):
    def eligible(self, x, m, predict_bonus_token=False, logprobs_mode_override=None):
        if os.getenv('VLLM_FL_SAMPLER_CAST_TEMP', '0') != '1':
            return False
        if (not self._task015_model_guard or predict_bonus_token
                or logprobs_mode_override is not None
                or self.logprobs_mode != 'raw_logprobs' or self.use_fp64_gumbel
                or torch.cuda.is_current_stream_capturing()):
            return False
        if (x.device.type != 'cuda' or x.dtype != torch.bfloat16
                or tuple(x.shape) != (1, 130560) or tuple(x.stride()) != (130560, 1)
                or x.storage_offset() != 0):
            return False
        if (not m.all_random or m.all_greedy or not m.no_penalties
                or m.max_num_logprobs is not None or m.logprob_token_ids
                or m.allowed_token_ids_mask is not None or m.bad_words_token_ids
                or m.thinking_budget_state_holder is not None or m.generators):
            return False
        temp = m.temperature
        if (temp is None or temp.dtype != torch.float32 or temp.device != x.device
                or tuple(temp.shape) != (1,) or tuple(temp.stride()) != (1,)
                or temp.storage_offset() != 0):
            return False
        # Exact builtin types and CPU activity fields. No D2H/value scan.
        for p in m.logitsprocs.non_argmax_invariant:
            if type(p) is MinTokensLogitsProcessor:
                if p.min_toks:
                    return False
            elif type(p) is LogitBiasLogitsProcessor:
                if p.biases:
                    return False
            else:
                return False
        for p in m.logitsprocs.argmax_invariant:
            if type(p) is not MinPLogitsProcessor or p.min_p_count:
                return False
        backend = self.topk_topp_sampler
        return (type(backend) is TopKTopPSampler
                and getattr(backend.forward, '__func__', None) is TopKTopPSampler.forward_native)

    def forward(self, logits, sampling_metadata, predict_bonus_token=False,
                logprobs_mode_override=None):
        m = sampling_metadata
        if not self.eligible(logits, m, predict_bonus_token, logprobs_mode_override):
            return super().forward(logits, m, predict_bonus_token,
                                   logprobs_mode_override)
        scaled = cast_temperature(logits, m.temperature)
        # Preserve all original processor calls; guard proved these exact
        # implementations return their input without touching values/state.
        scaled = self.apply_logits_processors(scaled, m, predict_bonus_token)
        for processor in m.logitsprocs.argmax_invariant:
            scaled = processor.apply(scaled)
        sampled, processed_logprobs = self.topk_topp_sampler(
            scaled, m.generators, m.top_k, m.top_p,
        )
        # Guard restricts raw mode and no requested logprobs. Native backend
        # returns None; keep the assertion rather than hide a changed backend.
        assert processed_logprobs is None
        sampled = sampled.long().to(torch.int32)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1),
                             logprobs_tensors=None)


def install(sampler, config):
    if os.getenv('VLLM_FL_SAMPLER_CAST_TEMP', '0') != '1':
        return sampler
    # Reuse the initialized native backend and object state; no RNG creation,
    # global class patch, shared installation mutation, or persistent scratch.
    if type(sampler) is not Sampler:
        return sampler
    p = config.parallel_config
    model = config.model_config
    hf = model.hf_config
    guard = (p.tensor_parallel_size == 1 and p.pipeline_parallel_size == 1
             and config.speculative_config is None and model.dtype == torch.bfloat16
             and getattr(hf, 'vocab_size', None) == 130560)
    sampler.__class__ = CastTemperatureSampler
    sampler._task015_model_guard = guard
    return sampler
