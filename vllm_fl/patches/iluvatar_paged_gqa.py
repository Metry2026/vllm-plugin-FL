# SPDX-License-Identifier: Apache-2.0
"""Opt-in experimental paged GQA dispatch; preserve the selected backend.

Installed by the existing `fl` general-plugin entry point. Disable with
VLLM_FL_W006_PAGED_GQA=0 (the default) and restart the engine to roll back.
"""

import functools
import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
SOURCE_SHA256 = "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"
REQUIRED = frozenset(
    (
        "q",
        "k",
        "v",
        "out",
        "cu_seqlens_q",
        "max_seqlen_q",
        "seqused_k",
        "max_seqlen_k",
        "softmax_scale",
        "causal",
        "window_size",
        "block_table",
        "softcap",
        "q_descale",
        "k_descale",
        "v_descale",
    )
)
OPTIONAL = frozenset(
    (
        "seq_threshold_3D",
        "num_par_softmax_segments",
        "softmax_segm_output",
        "softmax_segm_max",
        "softmax_segm_expsum",
        "alibi_slopes",
        "output_scale",
        "qq_bias",
        "sinks",
        "mm_prefix_range",
        "use_alibi_sqrt",
        "kv_quant_mode",
        "k_scale_cache",
        "v_scale_cache",
        "chunk_lookback",
        "use_td",
    )
)


def supported(a, torch):
    """Metadata-only guard. Never inspect tensor values or synchronize a GPU."""
    if not REQUIRED.issubset(a) or not a.keys() <= REQUIRED | OPTIONAL:
        return False
    if a["max_seqlen_q"] <= 1 or a["causal"] is not True:
        return False
    if tuple(a["window_size"]) != (-1, -1) or a["softcap"] not in (0, 0.0, None):
        return False
    if a.get("kv_quant_mode", 0) != 0 or a.get("chunk_lookback", -1) != -1:
        return False
    if a.get("use_td", False) or a.get("use_alibi_sqrt", False):
        return False
    if any(
        a.get(x) is not None
        for x in (
            "alibi_slopes",
            "output_scale",
            "qq_bias",
            "sinks",
            "mm_prefix_range",
            "k_scale_cache",
            "v_scale_cache",
            "q_descale",
        )
    ):
        return False
    q, k, v, out = (a[x] for x in ("q", "k", "v", "out"))
    if q.ndim != 3 or tuple(q.shape[1:]) != (16, 128) or out.shape != q.shape:
        return False
    if k.ndim != 4 or tuple(k.shape[2:]) != (2, 128) or v.shape != k.shape:
        return False
    if k.shape[1] not in (16, 32, 64, 128):
        return False
    tensors = (q, k, v, out, a["cu_seqlens_q"], a["seqused_k"], a["block_table"])
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        return False
    if any(t.dtype != torch.bfloat16 or t.stride(-1) != 1 for t in (q, k, v, out)):
        return False
    # Accept padded token/head strides, but not overlapping/expanded outputs.
    for t in (q, out):
        if t.stride(1) < 128 or t.stride(0) < 16 * t.stride(1):
            return False
    for t in (k, v):
        axes = sorted(zip(t.stride()[:-1], t.shape[:-1]))
        extent = 128
        for stride, size in axes:
            if stride < extent:
                return False
            extent = stride * size
    cu, lens, bt = (a[x] for x in ("cu_seqlens_q", "seqused_k", "block_table"))
    if cu.ndim != 1 or lens.ndim != 1 or bt.ndim != 2:
        return False
    if cu.shape[0] != lens.shape[0] + 1 or bt.shape[0] != lens.shape[0]:
        return False
    if cu.stride(0) != 1 or lens.stride(0) != 1 or bt.stride(1) != 1:
        return False
    if bt.shape[1] * k.shape[1] < a["max_seqlen_k"]:
        return False
    return all(t.dtype in (torch.int32, torch.int64) for t in (cu, lens, bt))


def make_dispatch(original, candidate, torch, block_q=8, block_n=32, audit=False):
    counts = {"candidate": 0, "fallback": 0}

    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        hit = not args and supported(kwargs, torch)
        if audit:
            counts["candidate" if hit else "fallback"] += 1
        if not hit:
            return original(*args, **kwargs)
        candidate(
            *(
                kwargs[x]
                for x in (
                    "q",
                    "k",
                    "v",
                    "out",
                    "cu_seqlens_q",
                    "seqused_k",
                    "block_table",
                    "softmax_scale",
                )
            ),
            block_q=block_q,
            block_n=block_n,
        )
        return None

    dispatch.w006_counts = counts
    dispatch.w006_original = original
    return dispatch


def install():
    if os.environ.get("VLLM_FL_W006_PAGED_GQA", "0") != "1":
        return False
    import torch

    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        return False
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
        return False
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.ops import triton_unified_attention as ops

    if hasattr(triton_attn.unified_attention, "w006_original"):
        return True
    if (
        hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest() != SOURCE_SHA256
        or triton_attn.unified_attention is not ops.unified_attention
    ):
        logger.warning("W006 unsupported Attention source identity; using original")
        return False
    bq = int(os.environ.get("VLLM_FL_W006_BLOCK_Q", "8"))
    bn = int(os.environ.get("VLLM_FL_W006_BLOCK_N", "32"))
    if bq not in (2, 4, 8) or bn not in (32, 64):
        raise ValueError("Invalid W006 tile configuration")
    if os.environ.get("VLLM_FL_PAGED_GQA_SCALAR_PAGES", "0") == "1":
        if os.environ.get("VLLM_FL_PAGED_GQA_SINGLE_STAGE", "0") == "1":
            from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_single_stage import paged_gqa_attention
        else:
            from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_page_supply import paged_gqa_attention
    else:
        from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_attention import paged_gqa_attention

    triton_attn.unified_attention = make_dispatch(
        ops.unified_attention,
        paged_gqa_attention,
        torch,
        bq,
        bn,
        audit=os.environ.get("VLLM_FL_W006_AUDIT", "0") == "1",
    )
    logger.info(
        "W006 paged GQA installed: BQ=%s BN=%s; original decode/fallback retained",
        bq,
        bn,
    )
    return True
