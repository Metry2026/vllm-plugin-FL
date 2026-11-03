"""Task022: a read-only prefix of a runner-owned immutable GPU arange.

Construction and strong ownership belong to the runner. No GPU value reads,
new stream/event, per-step refill, clone, or dependency on mutable CU storage.
"""
import torch


def prefix(indices, n, capacity, device):
    if (type(n) is not int or type(capacity) is not int
            or not 0 < n <= capacity <= 2147483647
            or type(indices) is not torch.Tensor or indices.ndim != 1
            or not indices.is_cuda or indices.device != device
            or indices.dtype != torch.int32 or indices.shape != (capacity,)
            or indices.stride() != (1,) or indices.storage_offset() != 0):
        return None
    return indices[:n]
