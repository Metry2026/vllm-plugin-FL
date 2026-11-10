"""Runner-private page-table supply; public commit still covers every column.

The original profile_run preallocates once inside the native memory profile.
The two input-preparation calls remain in synchronize_input_prep. Completion
protects content reuse; record_stream protects GPU storage retirement.
"""
import os
import weakref

import numpy as np
import torch

from vllm.platforms import current_platform


class _DeltaState:
    def __init__(self, identities, input_event, stream_key, cpu, gpu):
        # Bind identity without retaining the entire old InputBatch. These weak
        # references are prepared before allocating any device storage.
        self.identities = tuple(weakref.ref(obj) for obj in identities)
        self.input_event = input_event
        self.pid = os.getpid()
        self.allocation_stream_key = stream_key
        self.last_stream_key = None
        self.device = gpu.device
        self.shape = tuple(cpu.shape)
        self.stride = tuple(cpu.stride())
        self.pending = False
        self.failed = False
        self.phase = "preallocating"
        self.pending_target = None
        self.last_n = 0
        self.last_dirty_rows = np.empty(0, dtype=np.int64)
        self.valid = np.zeros(self.shape[0], dtype=np.bool_)
        self.last = np.empty(self.shape, dtype=np.int32)
        self.changed = np.empty(self.shape, dtype=np.bool_)
        # These are mutable runtime buffers, including when profile_run itself
        # is inside inference_mode. Match the original CpuGpuBuffer convention.
        with torch.inference_mode(False):
            self.staging = torch.empty(
                self.shape, dtype=torch.int32, device="cpu", pin_memory=True
            )
            self.staging_np = self.staging.numpy()
            # A first real snapshot initializes each valid row, never zeros.
            self.shadow = torch.empty(self.shape, dtype=torch.int32, device=gpu.device)
        self.ready = torch.Event()

    def matches(self, identities, input_event):
        return (
            self.pid == os.getpid()
            and self.input_event is input_event
            and len(self.identities) == len(identities)
            and all(ref() is obj for ref, obj in zip(self.identities, identities))
        )

    def rebind(self, identities, input_event):
        self.identities = tuple(weakref.ref(obj) for obj in identities)
        self.input_event = input_event
        self.valid[:] = False


def _eligible_table(runner, batch, multi, num_reqs):
    if type(num_reqs) is not int or num_reqs <= 0:
        return None
    if runner.prepare_inputs_event is None or runner.use_async_scheduling is not True:
        return None
    if (
        runner.parallel_config.tensor_parallel_size != 1
        or runner.parallel_config.pipeline_parallel_size != 1
    ):
        return None
    tables = multi.block_tables
    if type(tables) is not list or len(tables) != 1:
        return None
    table = tables[0]
    if (
        table.use_hybrid_blocks
        or table.blocks_per_kv_block != 1
        or table.block_size != 16
        or table.pcp_world_size != 1
        or table.dcp_world_size != 1
    ):
        return None
    buf = table.block_table
    cpu, gpu = buf.cpu, buf.gpu
    if type(cpu) is not torch.Tensor or type(gpu) is not torch.Tensor:
        return None
    if cpu.device.type != "cpu" or gpu.device.type != "cuda" or gpu.device.index != 0:
        return None
    if cpu.dtype is not torch.int32 or gpu.dtype is not torch.int32:
        return None
    shape = tuple(cpu.shape)
    if (
        len(shape) != 2
        or any(type(dim) is not int or dim <= 0 for dim in shape)
        or tuple(gpu.shape) != shape
        or num_reqs > shape[0]
        or tuple(cpu.stride()) != (shape[1], 1)
        or tuple(gpu.stride()) != (shape[1], 1)
        or not cpu.is_pinned()
    ):
        return None
    return table, buf, cpu, gpu


def _uncaptured_api():
    api = current_platform.torch_device_fn
    query = getattr(api, "is_current_stream_capturing", None)
    if query is None or query() is not False:
        return None
    return api


def _stream_key(api, gpu):
    stream = api.current_stream()
    ptr = getattr(stream, "cuda_stream", None)
    device = getattr(stream, "device_index", None)
    if type(ptr) is not int or device != gpu.device.index:
        return None
    return stream, (device, ptr)


def _automatic_kv_budget(runner):
    config = getattr(runner, "cache_config", None)
    return config is not None and getattr(config, "kv_cache_memory_bytes", "unknown") is None


def _preallocate_block_table_delta(runner):
    """Called only at the original profile_run entry, before native peak sampling.

    The current native automatic-budget caller places profile_run inside
    memory_profiling_fl. An unsupported or repeated profile never allocates
    a later replacement pool. No page-table values are read here.
    """
    previous = getattr(runner, "_metax_block_table_delta_state", None)
    if previous is not None and previous.pid == os.getpid() and previous.failed:
        raise RuntimeError("private page-table state failed; worker must exit")
    if getattr(runner, "_metax_block_table_delta_profile_attempted", False):
        return
    runner._metax_block_table_delta_profile_attempted = True
    if not _automatic_kv_budget(runner):
        return
    api = _uncaptured_api()
    if api is None:
        return
    batch = runner.input_batch
    multi = batch.block_table
    eligible = _eligible_table(runner, batch, multi, 1)
    if eligible is None:
        return
    table, buf, cpu, gpu = eligible
    current = _stream_key(api, gpu)
    if current is None:
        return
    stream, key = current
    identities = (runner, batch, multi, table, buf, cpu, gpu)
    try:
        tuple(weakref.ref(obj) for obj in identities)
    except TypeError:
        return
    state = _DeltaState(identities, runner.prepare_inputs_event, key, cpu, gpu)
    runner._metax_block_table_delta_state = state
    # Instantiate the event handle within the same original profile interval.
    # This records no public writes and marks no shadow row as initialized.
    state.pending = True
    state.phase = "initial_event_record"
    try:
        state.ready.record()
        state.phase = "pending_initialization"
    except BaseException:
        state.failed = True
        raise


def _commit_block_table_delta(runner, num_reqs, *, for_cudagraph_capture=False):
    state = getattr(runner, "_metax_block_table_delta_state", None)
    if state is not None and state.pid == os.getpid() and state.failed:
        raise RuntimeError("private page-table state failed; worker must exit")
    batch = runner.input_batch
    multi = batch.block_table

    def original():
        multi.commit_block_table(num_reqs)

    # All capture/unknown-domain guards precede private event queries and any
    # private allocation or write. Do not infer capture from the method flag.
    if for_cudagraph_capture or not _automatic_kv_budget(runner):
        original()
        return
    api = _uncaptured_api()
    if api is None:
        original()
        return
    eligible = _eligible_table(runner, batch, multi, num_reqs)
    if eligible is None:
        original()
        return
    table, buf, cpu, gpu = eligible
    current = _stream_key(api, gpu)
    if current is None:
        original()
        return
    stream, key = current
    identities = (runner, batch, multi, table, buf, cpu, gpu)
    # Unsupported weak-reference domains are rejected before device allocation.
    try:
        tuple(weakref.ref(obj) for obj in identities)
    except TypeError:
        original()
        return
    state = getattr(runner, "_metax_block_table_delta_state", None)
    if state is None:
        # Never cold-build after the native KV budget has already been sized.
        original()
        return
    if (
        state.pid != os.getpid()
        or state.device != gpu.device
        or state.shape != tuple(cpu.shape)
        or state.stride != tuple(cpu.stride())
    ):
        original()
        return
    if state.failed:
        raise RuntimeError("private page-table state failed; worker must exit")
    if state.pending:
        if not state.ready.query():
            original()
            return
        state.pending = False
        state.pending_target = None
    if not state.matches(identities, runner.prepare_inputs_event):
        state.rebind(identities, runner.prepare_inputs_event)

    n = num_reqs
    try:
        state.phase = "snapshot"
        # Use the true original CPU Tensor, never an attribute-level .np alias
        # replacement or a CPU version certificate. Do not retain this view.
        np.copyto(state.staging_np[:n], cpu.numpy()[:n])
        np.not_equal(state.staging_np[:n], state.last[:n], out=state.changed[:n])
        dirty = np.flatnonzero(np.any(state.changed[:n], axis=1) | ~state.valid[:n])
        state.last_n = n
        state.last_dirty_rows = dirty
        state.last_stream_key = key
        state.pending_target = gpu
        state.pending = True
        state.phase = "record_stream"
        state.shadow.record_stream(stream)
        gpu.record_stream(stream)
        state.phase = "private_H2D"
        for row in dirty:
            i = int(row)
            state.shadow[i].copy_(state.staging[i], non_blocking=True)
        state.phase = "public_D2D"
        gpu[:n].copy_(state.shadow[:n], non_blocking=True)
        state.phase = "ready_record"
        state.ready.record()
        np.copyto(state.last[:n], state.staging_np[:n])
        state.valid[:n] = True
        state.phase = "pending"
    except BaseException:
        # Preserve the original exception. There is no silent native fallback
        # after registration/enqueue and no future sampling of failed state.
        state.failed = True
        raise
