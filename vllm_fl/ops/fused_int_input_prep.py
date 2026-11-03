"""Task020: fixed integer output fusion inside the existing task018 domain.

Caller keeps the original current-step H2D, stream, event and consumers.
No GPU values are read back and no prepared value is cached across steps.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_int_outputs(X, Positions, SeqLens, n, capacity, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, mask=i < n, other=0)
    tl.store(Positions + i, x.to(tl.int64), mask=i < n)
    length = tl.where(i < n, x + 1, 0)
    tl.store(SeqLens + i, length, mask=i < capacity)


def try_fuse(x, positions, seq_lens, n):
    # Every dynamic extent/metadata/alias check is included in affected-path cost.
    tensors = (x, positions, seq_lens)
    if (type(n) is not int or n <= 0
            or any(type(t) is not torch.Tensor or t.ndim != 1
                   or t.stride() != (1,) or not t.is_cuda for t in tensors)
            or x.dtype != torch.int32 or positions.dtype != torch.int64
            or seq_lens.dtype != torch.int32
            or x.device != positions.device or x.device != seq_lens.device
            or n > min(x.numel(), positions.numel(), seq_lens.numel())):
        return False
    intervals = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors]
    for index, (lo, hi) in enumerate(intervals):
        for other_lo, other_hi in intervals[index + 1:]:
            if lo < other_hi and other_lo < hi:
                return False
    capacity = seq_lens.numel()
    _fused_int_outputs[(triton.cdiv(capacity, 256),)](
        x, positions, seq_lens, n, capacity, BLOCK=256, num_warps=4,
    )
    return True
