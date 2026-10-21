# SPDX-License-Identifier: Apache-2.0
"""Default-off W007 dispatch; disabling W007 restores the exact W006 B wrapper."""
import functools
import logging
import os

from vllm_fl.patches.iluvatar_paged_gqa import supported

logger = logging.getLogger(__name__)


def make_dispatch(baseline, candidate, torch, segments=16, audit=False):
    counts = {"candidate": 0, "fallback": 0, "capture_fallback": 0}

    @functools.wraps(baseline)
    def dispatch(*args, **kwargs):
        # Cheap scope checks precede the full guard, so B does not pay it twice.
        # A single request cannot be mixed. Small prefill chunks (one Q8 tile)
        # keep B: measured split overhead dominates this low-parallelism case.
        lens = kwargs.get("seqused_k")
        in_scope = (not args and lens is not None and lens.ndim == 1
                    and lens.shape[0] > 1 and kwargs.get("max_seqlen_q", 0) > 8)
        hit = in_scope and supported(kwargs, torch)
        capture = hit and torch.cuda.is_current_stream_capturing()
        if audit:
            counts["capture_fallback" if capture else "candidate" if hit else "fallback"] += 1
        if not hit or capture:
            return baseline(*args, **kwargs)
        candidate(*(kwargs[x] for x in (
            "q", "k", "v", "out", "cu_seqlens_q", "seqused_k", "block_table", "softmax_scale"
        )), segments=segments)
        return None

    dispatch.w007_counts = counts
    dispatch.w007_baseline = baseline
    return dispatch


def install():
    if os.environ.get("VLLM_FL_W007_MIXED_GQA", "0") != "1":
        return False
    # Requiring B makes the rollback and unsupported semantics unambiguous.
    from vllm_fl.patches.iluvatar_paged_gqa import install as install_b
    if not install_b():
        logger.warning("W007 requires a supported W006 B installation")
        return False
    import torch
    from vllm.v1.attention.backends import triton_attn
    if hasattr(triton_attn.unified_attention, "w007_baseline"):
        return True
    if (os.environ.get("VLLM_FL_W006_BLOCK_Q", "8") != "8"
            or os.environ.get("VLLM_FL_W006_BLOCK_N", "32") != "32"):
        raise ValueError("W007 requires frozen Q8/KV32 B")
    segments = int(os.environ.get("VLLM_FL_W007_SEGMENTS", "16"))
    if segments not in (1, 4, 8, 16):
        raise ValueError("W007 segments must be 1/4/8/16")
    from flag_gems.runtime.backend._iluvatar.fused.mixed_paged_gqa_attention import (
        mixed_paged_gqa_attention,
    )
    triton_attn.unified_attention = make_dispatch(
        triton_attn.unified_attention, mixed_paged_gqa_attention, torch, segments,
        audit=os.environ.get("VLLM_FL_W007_AUDIT", "0") == "1",
    )
    logger.info("W007 mixed indexed GQA installed: J=%s, Q8/KV32, pure decode unchanged", segments)
    return True
