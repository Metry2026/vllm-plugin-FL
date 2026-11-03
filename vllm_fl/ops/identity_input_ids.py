"""Default-off task019 early selection of the parent's direct token copy.

The existing task018 scope guard checks source/config, while each invocation
checks its current CU, freshly computed mapping, and sampled-token metadata.
No prepared values or dynamic predicates are cached.
"""
import numpy as np
import torch


def eligible(r, scheduler, n, total, cu):
    cached = getattr(r, "_task018_config_guard", None)
    if (not r._task018_single_token_prep or cached is None
            or cached[0] is not r.input_batch or not cached[1]
            or n <= 0 or total != n
            or scheduler.scheduled_spec_decode_tokens or r.use_async_spec_decode
            or not isinstance(cu, np.ndarray) or cu.shape != (n,)
            or cu.dtype not in (np.dtype(np.int32), np.dtype(np.int64))):
        return False
    sampled = r.input_batch.prev_sampled_token_ids
    if (type(sampled) is not torch.Tensor or sampled.ndim != 2
            or sampled.shape[0] < n or sampled.shape[1] < 1
            or sampled.dtype not in (torch.int32, torch.int64)
            or sampled.layout != torch.strided
            or sampled.device != r.input_ids.gpu.device):
        return False
    return (np.array_equal(cu, r.arange_np[1:n + 1])
            and np.array_equal(r.prev_positions.np[:n], r.arange_np[:n]))
