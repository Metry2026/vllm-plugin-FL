"""Task021: fuse full single-token CU output with the task020 integers.

Preflight runs before the original CU copy. Accepted plans launch later on the
same stream after the original x update; there is no late silent refusal.
"""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_cu_outputs(X, Positions, SeqLens, CU, n, capacity, cu_capacity,
                      BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, mask=i < n, other=0)
    tl.store(Positions + i, x.to(tl.int64), mask=i < n)
    tl.store(SeqLens + i, tl.where(i < n, x + 1, 0), mask=i < capacity)
    tl.store(CU + i, tl.minimum(i, n).to(tl.int32), mask=i < cu_capacity)


def preflight(x, positions, seq_lens, cu, n, cu_cpu, cu_np):
    tensors = (x, positions, seq_lens, cu)
    if (type(n) is not int or n <= 0
            or any(type(t) is not torch.Tensor or t.ndim != 1
                   or t.stride() != (1,) or not t.is_cuda for t in tensors)
            or x.dtype != torch.int32 or positions.dtype != torch.int64
            or seq_lens.dtype != torch.int32 or cu.dtype != torch.int32
            or any(t.device != x.device for t in tensors[1:])
            or n > min(x.numel(), positions.numel(), seq_lens.numel())):
        return None
    capacity, cu_capacity = seq_lens.numel(), cu.numel()
    if (cu_capacity != capacity + 1 or cu_capacity > 2147483647
            or type(cu_cpu) is not torch.Tensor or cu_cpu.device.type != "cpu"
            or cu_cpu.dtype != torch.int32 or cu_cpu.shape != (cu_capacity,)
            or cu_cpu.stride() != (1,) or not isinstance(cu_np, np.ndarray)
            or cu_np.dtype != np.int32 or cu_np.shape != (cu_capacity,)
            or cu_np.strides != (4,) or cu_cpu.data_ptr() != cu_np.ctypes.data):
        return None
    intervals = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                 for t in tensors]
    for index, (lo, hi) in enumerate(intervals):
        for other_lo, other_hi in intervals[index + 1:]:
            if lo < other_hi and other_lo < hi:
                return None
    return (x, positions, seq_lens, cu, n, capacity, cu_capacity)


def launch(plan):
    x, positions, seq_lens, cu, n, capacity, cu_capacity = plan
    _fused_cu_outputs[(triton.cdiv(max(capacity, cu_capacity), 256),)](
        x, positions, seq_lens, cu, n, capacity, cu_capacity,
        BLOCK=256, num_warps=4,
    )
    return True
