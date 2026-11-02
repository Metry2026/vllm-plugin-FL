"""Default-off task017: copy each step's proven reader rectangle, same pitch.

The unmodified input-prep event protects pinned CPU reuse and orders consumers.
No dirty/clean history, packing, row loop, temporary table or GPU readback.
"""
from pathlib import Path
import ctypes
import hashlib
import importlib
import inspect
import os
import time

import numpy as np
import torch

_SOURCES = {
    'vllm.v1.worker.block_table': 'b3a6649b968fba0b48c333d7ac96e65cc713a86133cc7eef0a8399f5339ec600',
    'vllm.v1.worker.gpu_input_batch': 'de1ea4fc9d43fc6079dd375e407c2b6d93f5e1a11f49732d13654c41eaf30a65',
    'vllm.v1.utils': '4e4cda209e10d4477d251003779dec26ea5ef47541c3310995c007c7ee5fdcdb',
    'vllm.v1.attention.backends.triton_attn': '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    'vllm.v1.attention.ops.triton_unified_attention': 'bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0',
    'vllm.v1.attention.ops.triton_attention_helpers': '59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1',
    'vllm_fl.patches.iluvatar_paged_gqa': 'f45aeaf440d7cfd985936da05b533073b75da03769c093b81a4402f73e7b9341',
    'flag_gems.runtime.backend._iluvatar.fused.paged_gqa_single_stage': 'd56590191021a2dd96b88d772966d17008c8168a3589fa6ce0d2fab9e4200fa8',
}
_RUNTIME = '/usr/local/corex-4.5.0/lib64/libcudart.so.10.2.89'


def choose_width(counts, computed, scheduled, n, capacity=8192):
    """Pinned ordinary BF16 readers: max tile32, PAGE16, causal/no-MM.

    GPU positions/seq_lens equal CPU computed+scheduled in this non-spec path.
    Original fallback issues unmasked table loads for the whole last tile.
    Therefore use 2*ceil(seq_len/32), even when effective block count is odd.
    """
    arrays = (counts[:n], computed[:n], scheduled)
    if n <= 0 or any(a.shape != (n,) or a.dtype != np.int32 for a in arrays):
        return None
    c, p, s = arrays
    if (np.any(c < 0) or np.any(c > capacity) or np.any(p < 0)
            or np.any(p > capacity*16) or np.any(s <= 0) or np.any(s > capacity*16)):
        return None
    lengths = p + s
    if np.any(lengths > c * 16):
        return None
    width = max(1, int(c.max()), 2*((int(lengths.max())+31)//32))
    return width if width < capacity else None


class RectAPI:
    def __init__(self):
        self.library = ctypes.CDLL(_RUNTIME)
        version = ctypes.c_int()
        get_version = self.library.cudaRuntimeGetVersion
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
        rc = get_version(ctypes.byref(version))
        if rc:
            raise RuntimeError(f'task017 runtime version query failed rc={rc}; no copy fallback')
        if version.value != 13010:
            raise ValueError('Unverified task017 runtime version')
        self.fn = self.library.cudaMemcpy2DAsync
        self.fn.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,
                            ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p]
        self.fn.restype = ctypes.c_int
        self.error = self.library.cudaGetErrorString
        self.error.argtypes = [ctypes.c_int]
        self.error.restype = ctypes.c_char_p

    def enqueue(self, buffer, n, width):
        stream = torch.cuda.current_stream(buffer.gpu.device).cuda_stream
        rc = self.fn(buffer.gpu.data_ptr(), buffer.gpu.stride(0)*4,
                     buffer.cpu.data_ptr(), buffer.cpu.stride(0)*4, width*4, n, 1, stream)
        if rc:
            # The API may report prior async errors or leave queue state unknown.
            # Never attempt a second copy as recovery after this call.
            raise RuntimeError(f'task017 cudaMemcpy2DAsync rc={rc}: {self.error(rc).decode()}; no fallback')


def _supported(r):
    from vllm.v1.worker.block_table import MultiGroupBlockTable, BlockTable
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.utils import CpuGpuBuffer
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
    if (type(r).__module__ != "vllm_fl.worker.model_runner" or type(r).__name__ != "ModelRunnerFL"
            or hashlib.sha256(Path(inspect.getsourcefile(type(r))).read_bytes()).hexdigest()
            != "681e4a03e383186fd6efe12156f8f179928644d3a590c5cf56dd10cb96d05b5c"):
        return False
    p = r.parallel_config
    batch = r.input_batch
    if (r.model_config.dtype != torch.bfloat16 or p.tensor_parallel_size != 1
            or p.pipeline_parallel_size != 1 or p.data_parallel_size != 1 or p.use_ubatching
            or r.speculative_config is not None or r.prepare_inputs_event is None
            or r.lora_config is not None or r.uses_mrope or r.uses_xdrope_dim > 0
            or r.model_config.is_encoder_decoder or r.model_config.is_multimodal_model
            or r.is_mm_prefix_lm or r.use_alibi or r.num_query_heads != 16
            or type(batch) is not InputBatch or type(batch.block_table) is not MultiGroupBlockTable
            or len(batch.block_table.block_tables) != 1
            or len(r.kv_cache_config.kv_cache_groups) != 1
            or type(r.kv_cache_config.kv_cache_groups[0].kv_cache_spec) is not FullAttentionSpec):
        return False
    spec = r.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    if (spec.head_size != 128 or spec.num_kv_heads != 2 or spec.dtype != torch.bfloat16
            or spec.sliding_window is not None or spec.attention_chunk_size is not None):
        return False
    builders = [ag.get_metadata_builder(0) for group in r.attn_groups for ag in group]
    if not builders or any(type(x) is not TritonAttentionMetadataBuilder for x in builders):
        return False
    if any(os.getenv(k) != v for k,v in [('VLLM_FL_W006_PAGED_GQA','1'),('VLLM_FL_PAGED_GQA_SINGLE_STAGE','1'),('VLLM_FL_PAGED_GQA_SCALAR_PAGES','1')]):
        return False
    if os.getenv('VLLM_BATCH_INVARIANT', '0') != '0':
        return False
    b = batch.block_table[0];v = b.block_table
    if (type(b) is not BlockTable or type(v) is not CpuGpuBuffer or b.block_size != 16
            or b.use_hybrid_blocks or b.pcp_world_size != 1 or b.dcp_world_size != 1
            or v.cpu.shape != (256,8192) or v.gpu.shape != v.cpu.shape
            or v.cpu.dtype != torch.int32 or v.gpu.dtype != torch.int32
            or not v.cpu.is_pinned() or not v.cpu.is_contiguous() or not v.gpu.is_contiguous()
            or v.cpu.storage_offset() != 0 or v.gpu.storage_offset() != 0
            or v.gpu.device.type != 'cuda' or v.cpu.requires_grad or v.gpu.requires_grad):
        return False
    return all(hashlib.sha256(Path(inspect.getsourcefile(importlib.import_module(name))).read_bytes()).hexdigest()==sha
               for name,sha in _SOURCES.items())


class _CopyState:
    def __init__(self, r):
        start = time.perf_counter_ns()
        self.batch = r.input_batch
        self.mg = self.batch.block_table
        self.table = self.mg[0]
        self.buffer = self.table.block_table
        self.cpu = self.buffer.cpu;self.gpu = self.buffer.gpu
        self.cpu_ptr = self.cpu.data_ptr();self.gpu_ptr = self.gpu.data_ptr()
        self.api = RectAPI()
        self.setup_ns = time.perf_counter_ns()-start

    def same(self, r):
        b = self.buffer
        return (r.input_batch is self.batch and self.batch.block_table is self.mg
                and self.mg.block_tables == [self.table] and self.table.block_table is b
                and b.cpu is self.cpu and b.gpu is self.gpu
                and self.cpu.data_ptr() == self.cpu_ptr and self.gpu.data_ptr() == self.gpu_ptr
                and self.cpu.shape == self.gpu.shape == (256,8192)
                and self.cpu.stride() == self.gpu.stride() == (8192,1))


def commit(r, n, scheduled):
    mg = r.input_batch.block_table
    if torch.cuda.is_current_stream_capturing():
        mg.commit_block_table(n)
        return None
    cached = getattr(r, '_task017_copy_state', None)
    if cached is None or cached[0] is not r.input_batch:
        try:
            state = _CopyState(r) if _supported(r) else None
        except (AttributeError, OSError, ValueError):
            state = None  # No asynchronous copy has been attempted here.
        cached = (r.input_batch, state)
        r._task017_copy_state = cached
    state = cached[1]
    if state is not None and state.same(r):
        width = choose_width(state.table.num_blocks_per_row,
                             r.input_batch.num_computed_tokens_cpu, scheduled, n)
        if width is not None:
            state.api.enqueue(state.buffer, n, width)
            return width
    mg.commit_block_table(n)
    return None
