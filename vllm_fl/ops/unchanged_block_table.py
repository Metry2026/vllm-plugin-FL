"""Default-off task016 experiment for the pinned task010 input-prep contract.

No per-step data comparison. Unknown sources, aliases or lifecycles use the
original whole-table copy. Source closure is deliberately part of the guard:
NumPy stores do not update the CPU tensor version counter.
"""
from pathlib import Path
import hashlib
import inspect
import types

import torch

_SOURCES = {
    "vllm.v1.worker.block_table": "b3a6649b968fba0b48c333d7ac96e65cc713a86133cc7eef0a8399f5339ec600",
    "vllm.v1.worker.gpu_input_batch": "de1ea4fc9d43fc6079dd375e407c2b6d93f5e1a11f49732d13654c41eaf30a65",
    "vllm.v1.utils": "4e4cda209e10d4477d251003779dec26ea5ef47541c3310995c007c7ee5fdcdb",
    "vllm.v1.attention.backends.triton_attn": "1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605",
}


class _TableState:
    def __init__(self, runner):
        self.runner = runner
        self.batch = runner.input_batch
        self.mg = self.batch.block_table
        self.table = self.mg.block_tables[0]
        self.buffer = self.table.block_table
        self.clean = self.pending = None
        self.disabled = False
        self.active = False
        self.generation = 0
        self.bindings = []
        b = self.buffer
        self.structure = (id(b.cpu), id(b.gpu), id(b.np), b.cpu.data_ptr(), b.gpu.data_ptr(),
                          tuple(b.cpu.shape), tuple(b.gpu.shape), tuple(b.cpu.stride()), tuple(b.gpu.stride()),
                          b.cpu.storage_offset(), b.gpu.storage_offset(), b.cpu.dtype, b.gpu.dtype)
        for name in ("append_row", "add_row", "clear_row", "move_row", "swap_row", "clear", "commit_block_table"):
            original = getattr(self.table, name)
            def wrapped(obj, *args, _name=name, _original=original, **kwargs):
                # Exact installed append_row returns without writing for [].
                if not (_name == "append_row" and not (args[0] if args else kwargs["block_ids"])):
                    self.invalidate()
                return _original(*args, **kwargs)
            self.bind(self.table, name, wrapped)
        for obj, name in ((self.table, "get_cpu_tensor"), (self.table, "get_numpy_array"), (self.buffer, "copy_to_cpu")):
            original = getattr(obj, name)
            def exposed(obj, *args, _original=original, **kwargs):
                self.disabled = True  # A writable alias can outlive this call.
                self.invalidate()
                return _original(*args, **kwargs)
            self.bind(obj, name, exposed)

    def bind(self, obj, name, fn):
        method = types.MethodType(fn, obj)
        setattr(obj, name, method)
        self.bindings.append((obj, name, method, getattr(type(obj), name)))

    def invalidate(self):
        self.generation += 1
        self.clean = self.pending = None

    def key(self, n):
        b = self.buffer
        if (self.disabled or not self.active or self.runner.input_batch is not self.batch
                or self.batch.block_table is not self.mg or self.mg.block_tables != [self.table]
                or self.table.block_table is not b or torch.cuda.is_current_stream_capturing()
                or not 0 < n <= b.cpu.shape[0]
                or any(getattr(obj, name) is not method or getattr(type(obj), name) is not cls_method
                       for obj, name, method, cls_method in self.bindings)):
            return None
        structure = (id(b.cpu), id(b.gpu), id(b.np), b.cpu.data_ptr(), b.gpu.data_ptr(),
                tuple(b.cpu.shape), tuple(b.gpu.shape), tuple(b.cpu.stride()), tuple(b.gpu.stride()),
                b.cpu.storage_offset(), b.gpu.storage_offset(), b.cpu.dtype, b.gpu.dtype)
        if structure != self.structure:
            return None
        return (n, tuple(self.batch.req_ids[:n]), self.generation, structure,
                b.cpu._version, b.gpu._version, id(self.runner.prepare_inputs_event),
                torch.cuda.current_stream().cuda_stream)


def _supported(runner):
    from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.utils import CpuGpuBuffer
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    import sys
    if (type(runner).__module__ != "vllm_fl.worker.model_runner"
            or type(runner).__name__ != "ModelRunnerFL"
            or hashlib.sha256(Path(inspect.getsourcefile(type(runner))).read_bytes()).hexdigest()
            != "1fe27804f3c7ab0d8b8247c8757c5c61dc1e9a354c3c58c8b38768ed54982837"):
        return False
    p = runner.parallel_config
    batch = runner.input_batch
    mg = batch.block_table
    if (runner.model_config.dtype != torch.bfloat16 or p.tensor_parallel_size != 1
            or p.pipeline_parallel_size != 1 or p.data_parallel_size != 1
            or p.use_ubatching or runner.speculative_config is not None
            or runner.prepare_inputs_event is None or type(batch) is not InputBatch
            or runner.lora_config is not None or runner.uses_mrope or runner.uses_xdrope_dim > 0
            or runner.model_config.is_encoder_decoder
            or type(mg) is not MultiGroupBlockTable or len(mg.block_tables) != 1
            or len(runner.kv_cache_config.kv_cache_groups) != 1
            or type(runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec) is not FullAttentionSpec):
        return False
    builders = [ag.get_metadata_builder(0) for groups in runner.attn_groups for ag in groups]
    if not builders or any(type(x) is not TritonAttentionMetadataBuilder for x in builders):
        return False
    b = mg.block_tables[0]
    v = b.block_table
    if (type(b) is not BlockTable or type(v) is not CpuGpuBuffer
            or b.block_size != 16 or b.use_hybrid_blocks
            or b.pcp_world_size != 1 or b.dcp_world_size != 1
            or v.cpu.dtype != torch.int32 or v.gpu.dtype != torch.int32
            or v.cpu.shape != (256, 8192) or v.gpu.shape != v.cpu.shape
            or not v.cpu.is_pinned() or not v.cpu.is_contiguous() or not v.gpu.is_contiguous()
            or v.gpu.device.type != "cuda" or v.cpu.is_inference() or v.gpu.is_inference()):
        return False
    return all(hashlib.sha256(Path(inspect.getsourcefile(sys.modules[name])).read_bytes()).hexdigest() == sha
               for name, sha in _SOURCES.items())


def begin(runner):
    state = getattr(runner, "_task016_table_state", None)
    # A recreated input batch gets a fresh, initially dirty tracker.
    if state is None or state.batch is not runner.input_batch:
        try:
            state = _TableState(runner) if _supported(runner) else None
        except (AttributeError, OSError, TypeError, RuntimeError):
            state = None
        runner._task016_table_state = state
    if state is not None:
        if state.active:  # Nested/dummy context: do not inherit clean state.
            state.invalidate()
        state.pending = None
        state.active = True


def commit(runner, num_reqs):
    state = getattr(runner, "_task016_table_state", None)
    if state is None:
        runner.input_batch.block_table.commit_block_table(num_reqs)
        return True
    key = state.key(num_reqs)
    if key is not None and key == state.clean:
        state.pending = key
        return False
    state.invalidate()
    runner.input_batch.block_table.commit_block_table(num_reqs)
    state.pending = state.key(num_reqs)  # Copy has now been queued on this stream.
    return True


def padding(runner, table, n, padded, operation, capturing=False):
    state = getattr(runner, "_task016_table_state", None)
    before = state.key(n) if state is not None else None
    result = operation()  # Preserve exact original zeroing, including empty views.
    if state is not None:
        if (not capturing and table is state.table and 0 < n <= padded <= state.buffer.gpu.shape[0]
                and before is not None and before == state.pending
                and state.buffer.gpu._version == before[-3] + 1):
            # This one source-pinned fill only touched rows outside [0:n].
            state.pending = state.key(n)
        else:
            state.invalidate()
    return result


def finish(runner, succeeded):
    state = getattr(runner, "_task016_table_state", None)
    if state is not None:
        pending = state.pending
        # Called only after the original prepare_inputs_event.record().
        if succeeded and pending is not None and state.key(pending[0]) == pending:
            state.clean = pending
        else:
            state.invalidate()
        state.pending = None
        state.active = False
