# SPDX-License-Identifier: Apache-2.0
"""Narrow MetaX task008 fusion; scheduling and source uploads stay in the runner."""

import torch
import triton
import triton.language as tl


def integer_output_layout_supported(computed, req_indices, query_pos, scheduled,
                                    positions, seq_lens, n, total):
    tensors = (computed, req_indices, query_pos, scheduled, positions, seq_lens)
    dtypes = (torch.int32, torch.int64, torch.int64, torch.int32,
              torch.int64, torch.int32)
    sizes = (n, total, total, n, total, n)
    return all(t.is_cuda and t.device == computed.device and t.dtype == dtype
               and t.ndim == 1 and t.stride(0) == 1 and t.numel() >= size
               for t, dtype, size in zip(tensors, dtypes, sizes))


@triton.jit
def _integer_outputs(computed, req_indices, query_pos, scheduled, positions,
                     seq_lens, n, total, SEQ_CAP: tl.constexpr,
                     BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token_mask = lane < total
    req = tl.load(req_indices + lane, token_mask, other=0)
    base = tl.load(computed + req, token_mask, other=0).to(tl.int64)
    query = tl.load(query_pos + lane, token_mask, other=0)
    tl.store(positions + lane, base + query, token_mask)

    active = lane < n
    count = tl.load(computed + lane, active, other=0)
    increment = tl.load(scheduled + lane, active, other=0)
    length = tl.where(active, count + increment, 0).to(tl.int32)
    tl.store(seq_lens + lane, length, lane < SEQ_CAP)


def fuse_integer_outputs(computed, req_indices, query_pos, scheduled,
                         positions, seq_lens, n, total):
    # Fixed C500 implementation: no autotuning or inherited platform config.
    block = 256
    return _integer_outputs[(triton.cdiv(max(total, seq_lens.numel()), block),)](
        computed, req_indices, query_pos, scheduled, positions, seq_lens,
        n, total, seq_lens.numel(), BLOCK=block, num_warps=4)
